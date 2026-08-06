"""Frame and level conventions on the v2 manifest path.

The manifest stores poses in the frames production stores, so there is no
conversion left to get wrong — the three-flip composition these tests used to
guard died with `georef.db`. What survives is the pair of conventions that are
still ours to hold, and that still fail silently:

1. **The EUS axis convention.** `local_positions_to_wgs84` and
   `headings_from_orientations` reach a compass bearing by paths that share no
   code — geodesy on one side, rotation composition on the other. If the manifest
   axes were read in the wrong order they would disagree.
2. **The level datum.** `levels_for_altitudes` wants the EUS *up* coordinate, not
   the WGS84 altitude. Feed it the wrong one and every level comes back NaN, which
   disables the level-compatibility guard in clustering and merges objects across
   floors with no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from toolbox.bricks.georef_source import load_pose_source
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


def _write_manifest(
    map_path: Path,
    keyframes: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> Path:
    """Minimal manifest: an origin, two level bands, and the given keyframes.

    `keyframes` is a list of `(orientation_wxyz, position_eus)`, in the order the
    keyframe ids will follow — ids are `geo_keyframes` indices.
    """
    map_path.mkdir(parents=True, exist_ok=True)
    path = map_path / "frames_1_20260101_000000.json"
    path.write_text(
        json.dumps(
            {
                # PointField order: longitude first.
                "local_origin": [ORIGIN_LNG, ORIGIN_LAT, ORIGIN_ALT],
                "map": {"name": "frames", "uuid": "u", "venue_type": "rail"},
                "geo_levels": [
                    {
                        "value": 0,
                        "min_altitude": -2.0,
                        "max_altitude": 3.0,
                        "geo_ref": 1,
                        "geometry": None,
                    },
                    {
                        "value": 1,
                        "min_altitude": 3.0,
                        "max_altitude": 9.0,
                        "geo_ref": 1,
                        "geometry": None,
                    },
                ],
                "geo_keyframes": [
                    {
                        "image_url": f"https://example.test/images/{index}.jpg",
                        "depth_url": f"https://example.test/depths/{index}.tif",
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "orientation": [float(v) for v in orientation],
                    }
                    for index, (orientation, position) in enumerate(keyframes)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _yaw_quaternion(yaw_deg: float) -> NDArray[np.float64]:
    """`[w, x, y, z]` for a clockwise-from-North yaw about the EUS up axis (+Y).

    Compass convention: NORTH = 0, EAST = 90, increasing clockwise seen from
    above. Rotating the camera about +Y by -yaw turns its -Z forward toward +X.
    """
    half = np.radians(-yaw_deg) / 2.0
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


def test_identity_orientation_is_heading_north(tmp_path: Path) -> None:
    """In EUS the OpenGL camera forward (-Z) is North, so identity means heading 0."""
    _write_manifest(tmp_path, [(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))])
    source = load_pose_source(tmp_path)

    heading = headings_from_orientations(source.poses[0].orientation_wxyz[None, :])[0]
    assert _angular_diff_deg(heading, 0.0) < 1e-6


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 180.0, 270.0])
def test_yaw_maps_to_compass_heading(tmp_path: Path, yaw_deg: float) -> None:
    _write_manifest(tmp_path, [(_yaw_quaternion(yaw_deg), np.zeros(3))])
    source = load_pose_source(tmp_path)

    heading = headings_from_orientations(source.poses[0].orientation_wxyz[None, :])[0]
    assert _angular_diff_deg(heading, yaw_deg) < 1e-6


@pytest.mark.parametrize(
    "offset_eus, expected_bearing",
    [
        (np.array([0.0, 0.0, -10.0]), 0.0),  # -Z is North
        (np.array([10.0, 0.0, 0.0]), 90.0),  # +X is East
        (np.array([0.0, 0.0, 10.0]), 180.0),  # +Z is South
        (np.array([-10.0, 0.0, 0.0]), 270.0),  # -X is West
    ],
)
def test_position_bearing_agrees_with_axis_convention(
    tmp_path: Path, offset_eus: NDArray[np.float64], expected_bearing: float
) -> None:
    """Geodesy must place the EUS axes where the rotation path expects them.

    `local_positions_to_wgs84` (ellipsoid math) and `headings_from_orientations`
    (rotation composition) share no code. Agreement here means the manifest's
    x/y/z really are East/Up/South — which nothing else checks.
    """
    _write_manifest(tmp_path, [(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))])
    geo_transform = load_pose_source(tmp_path).geo_transform

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


def test_levels_use_height_above_origin_not_absolute_altitude(tmp_path: Path) -> None:
    """Manifest level bands are heights above the origin, not WGS84 altitudes."""
    _write_manifest(tmp_path, [(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))])
    geo_transform = load_pose_source(tmp_path).geo_transform

    eus_up = np.array([0.0, 6.0])  # ground level, then upper level
    levels = geo_transform.levels_for_altitudes(eus_up)
    np.testing.assert_array_equal(levels, np.array([0.0, 1.0]))

    # The absolute altitude of the same points is ~35 m and ~41 m: outside every
    # band, so everything would come back NaN.
    absolute = eus_up + ORIGIN_ALT
    assert np.all(np.isnan(geo_transform.levels_for_altitudes(absolute)))
