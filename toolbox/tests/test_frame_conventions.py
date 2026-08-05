"""Round-trip tests for the georef.db → EUS/OpenGL frame conversion.

`georef_source` composes three independent flips (a transpose, an inverse, and two
axis-permutation matrices). Drop any one and the result is wrong but plausible:
objects land mirrored or 180° off, with no exception anywhere. These tests pin the
composition down from both ends — the orientation path
(`headings_from_orientations`) and the position path (`observation_heading_deg`)
must agree on where a camera is looking.

`test_conversion_is_not_accidentally_symmetric` is the important one: it proves the
others would actually fail if a flip were dropped, rather than passing for a
degenerate reason.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from toolbox.bricks.georef_source import (
    ROT_OPENGL_TO_OPENCV,
    ROT_WDS_TO_EUS,
    load_geo_transform,
    load_keyframe_poses,
)
from toolbox.bricks.localize import observation_heading_deg
from toolbox.bricks.vendored.maths import vector3
from toolbox.bricks.vendored.viewer360_headings import headings_from_orientations

ORIGIN_LAT = 48.8566
ORIGIN_LNG = 2.3522
ORIGIN_ALT = 35.0


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings, in degrees.

    Needed because 0 and 360 are the same heading: a correct North bearing comes
    back as 359.999… from `atan2`, and a naive comparison would call that a miss.
    """
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _pose_blob(rotation_c2w_wds: np.ndarray, translation_wds: np.ndarray) -> bytes:
    """Encode a camera-to-world WDS pose the way `GeoRefKeyframe.pose` stores it.

    The reader does `frombuffer(...).reshape(4, 4).T` and then inverts, so we
    invert first and write the transpose — i.e. exactly undo the reader.
    """
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation_c2w_wds
    camera_to_world[:3, 3] = translation_wds
    world_to_camera = np.linalg.inv(camera_to_world)
    return np.ascontiguousarray(world_to_camera.T, dtype=np.float64).tobytes()


def _camera_to_world_wds_for_eus(
    rotation_opengl_to_eus: np.ndarray, position_eus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert `georef_source`'s conversion, to build a fixture from a desired result.

    Forward:  R_eus = ROT_WDS_TO_EUS @ R_c2w @ ROT_OPENGL_TO_OPENCV
              p_eus = ROT_WDS_TO_EUS @ p_wds
    Both matrices are diagonal ±1, hence their own inverses.
    """
    rotation_c2w = ROT_WDS_TO_EUS @ rotation_opengl_to_eus @ ROT_OPENGL_TO_OPENCV
    translation_wds = ROT_WDS_TO_EUS @ np.asarray(position_eus, dtype=np.float64)
    return rotation_c2w, translation_wds


def _write_georef_db(
    path: Path, keyframes: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Minimal georef.db: an origin, one level band, and the given keyframes.

    `keyframes` maps id → (rotation_opengl_to_eus, position_eus).
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE GeoRef (latitude REAL, longitude REAL, altitude REAL)")
    conn.execute(
        "INSERT INTO GeoRef VALUES (?, ?, ?)", (ORIGIN_LAT, ORIGIN_LNG, ORIGIN_ALT)
    )
    conn.execute(
        "CREATE TABLE Level (id INTEGER, min_altitude REAL, max_altitude REAL, "
        "geometry TEXT)"
    )
    # Bands are heights above the origin, not absolute altitudes.
    conn.execute("INSERT INTO Level VALUES (0, -2.0, 4.0, NULL)")
    conn.execute("INSERT INTO Level VALUES (1, 4.0, 10.0, NULL)")
    conn.execute("CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB)")
    for keyframe_id, (rotation_eus, position_eus) in keyframes.items():
        rotation_c2w, translation_wds = _camera_to_world_wds_for_eus(
            rotation_eus, position_eus
        )
        conn.execute(
            "INSERT INTO GeoRefKeyframe VALUES (?, ?)",
            (keyframe_id, _pose_blob(rotation_c2w, translation_wds)),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def georef_db(tmp_path: Path) -> Path:
    """One keyframe at the origin, facing North (identity OpenGL→EUS rotation).

    In EUS (+X East, +Y Up, +Z South) the OpenGL camera forward is -Z, which is
    North — so an identity rotation means "looking North", heading 0.
    """
    path = tmp_path / "georef.db"
    _write_georef_db(
        path, {1: (np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.0]))}
    )
    # lru_cache in the georef reader is keyed on the path; tmp_path is unique per
    # test, so no cross-test bleed.
    return path


def test_position_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "georef.db"
    expected = np.array([12.5, 3.0, -7.25])
    _write_georef_db(path, {42: (np.eye(3), expected)})
    poses = load_keyframe_poses(path)
    assert set(poses) == {42}
    np.testing.assert_allclose(poses[42].position_eus, expected, atol=1e-9)


def test_identity_orientation_round_trips_as_heading_north(georef_db: Path) -> None:
    poses = load_keyframe_poses(georef_db)
    heading = headings_from_orientations(poses[1].orientation_wxyz[None, :])[0]
    assert _angular_diff_deg(heading, 0.0) < 1e-6


@pytest.mark.parametrize(
    "yaw_deg, expected_heading",
    [(0.0, 0.0), (90.0, 90.0), (180.0, 180.0), (270.0, 270.0)],
)
def test_yaw_maps_to_compass_heading(
    tmp_path: Path, yaw_deg: float, expected_heading: float
) -> None:
    """A clockwise-from-North yaw must come back as that compass heading.

    Compass convention: NORTH = 0, EAST = 90, increasing clockwise seen from above.
    Rotating the camera about the EUS up axis (+Y) by -yaw turns -Z toward +X (East).
    """
    path = tmp_path / "georef.db"
    angle = np.radians(-yaw_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]], dtype=np.float64
    )
    _write_georef_db(path, {1: (rotation, np.zeros(3))})

    poses = load_keyframe_poses(path)
    heading = headings_from_orientations(poses[1].orientation_wxyz[None, :])[0]
    assert _angular_diff_deg(heading, expected_heading) < 1e-6


@pytest.mark.parametrize(
    "offset_eus, expected_bearing",
    [
        (np.array([0.0, 0.0, -10.0]), 0.0),  # -Z is North
        (np.array([10.0, 0.0, 0.0]), 90.0),  # +X is East
        (np.array([0.0, 0.0, 10.0]), 180.0),  # +Z is South
        (np.array([-10.0, 0.0, 0.0]), 270.0),  # -X is West
    ],
)
def test_position_bearing_agrees_with_orientation_heading(
    tmp_path: Path, offset_eus: NDArray[np.float64], expected_bearing: float
) -> None:
    """The two independent paths from EUS to a compass bearing must agree.

    Path A: keyframe orientation → `headings_from_orientations` (rotation-based).
    Path B: keyframe/target lat-lng → `observation_heading_deg` (geodesy-based).

    They share no code, so agreement means both the rotation composition and the
    EUS axis convention are right. This is the check that catches a dropped flip.
    """
    path = tmp_path / "georef.db"
    _write_georef_db(path, {1: (np.eye(3), np.zeros(3))})
    geo_transform = load_geo_transform(path)

    wgs84 = geo_transform.local_positions_to_wgs84(
        vector3.cast_batch(np.array([[0.0, 0.0, 0.0], offset_eus], dtype=np.float64))
    )
    bearing = observation_heading_deg(
        keyframe_lat=wgs84[0, 1],
        keyframe_lng=wgs84[0, 0],
        target_lat=wgs84[1, 1],
        target_lng=wgs84[1, 0],
    )
    assert _angular_diff_deg(bearing, expected_bearing) < 0.01


def test_conversion_is_not_accidentally_symmetric() -> None:
    """Guard the guards: dropping a flip must change the answer.

    If either matrix were the identity (or the two cancelled), every test above
    would pass for the wrong reason. Assert the composition is genuinely
    load-bearing.
    """
    assert not np.allclose(ROT_WDS_TO_EUS, np.eye(3))
    assert not np.allclose(ROT_OPENGL_TO_OPENCV, np.eye(3))
    assert not np.allclose(ROT_WDS_TO_EUS @ ROT_OPENGL_TO_OPENCV, np.eye(3))
    # Both are proper rotations (det +1), not reflections — a reflection here would
    # mirror the world and still "look fine" on a map.
    assert np.linalg.det(ROT_WDS_TO_EUS) == pytest.approx(1.0)
    assert np.linalg.det(ROT_OPENGL_TO_OPENCV) == pytest.approx(1.0)


def test_levels_use_height_above_origin_not_absolute_altitude(tmp_path: Path) -> None:
    """georef.db level bands are heights above the origin.

    Feeding `levels_for_altitudes` a WGS84 altitude instead of the EUS up
    coordinate silently returns None for every candidate, which disables the
    level-compatibility guard in clustering and merges objects across floors.
    """
    path = tmp_path / "georef.db"
    _write_georef_db(path, {1: (np.eye(3), np.zeros(3))})
    geo_transform = load_geo_transform(path)

    eus_up = np.array([0.0, 6.0])  # ground level, then upper level
    levels = geo_transform.levels_for_altitudes(eus_up)
    np.testing.assert_array_equal(levels, np.array([0.0, 1.0]))

    # The absolute altitude of the same points is ~35 m and ~41 m: outside every
    # band, so everything would come back NaN.
    absolute = eus_up + ORIGIN_ALT
    assert np.all(np.isnan(geo_transform.levels_for_altitudes(absolute)))


def test_keyframe_with_null_pose_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "georef.db"
    _write_georef_db(path, {1: (np.eye(3), np.zeros(3))})
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO GeoRefKeyframe VALUES (2, NULL)")
    conn.commit()
    conn.close()
    assert set(load_keyframe_poses(path)) == {1}
