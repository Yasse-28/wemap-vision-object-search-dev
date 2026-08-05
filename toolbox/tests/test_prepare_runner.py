"""Tests for keyframe-id resolution before `prepare` runs.

This is the step that stops `prepare` output from being attached to the wrong poses.
The mirrored `prepare` CLI numbers keyframes positionally (`enumerate`), so if that
numbering ever reached `ingest_cli` the candidates would land on whichever keyframes
happened to share those ids — silently, and with every object in the wrong place.
`collect_image_entries` is what prevents it, so its failure modes are pinned here.

Both pose formats are covered: the **v2 manifest** (the current one) and the legacy
**`georef.db`**. No detection runs — `run_prepare` needs a GPU and model weights.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from toolbox.bricks.georef_source import (
    ROT_OPENGL_TO_OPENCV,
    ROT_WDS_TO_EUS,
    load_pose_source,
)
from toolbox.bricks.prepare_runner import collect_image_entries, resolve_images_dir

ORIGIN_LNG = 2.3668
ORIGIN_LAT = 48.8170


# ------------------------------------------------------------------ v2 manifest maps


def _make_v2_map(
    tmp_path: Path,
    *,
    keyframes: list[tuple[str, tuple[float, float, float]]],
    images: list[str] | None = None,
    venue_type: str | None = "hotel",
    geo_ref_id: int | None = 149,
    images_dirname: str = "images_360",
    name: str = "bbhotel-choisy_2_20260805_083206.json",
    query: str = "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef",
) -> Path:
    """A v2 map directory.

    `keyframes` is `[(image_filename, (x, y, z)), …]` in manifest order — so the
    keyframe id of each is its index. `images` defaults to exactly those filenames;
    pass it to make the directory disagree with the manifest.

    Asset URLs carry a presigned `query` by default, because real manifest URLs do.
    Keeping it on every case means the whole v2 suite fails if the reader ever goes
    back to splitting on "/" instead of parsing the URL.
    """
    map_path = tmp_path / "map"
    images_dir = map_path / images_dirname
    images_dir.mkdir(parents=True)
    for filename in images if images is not None else [name for name, _ in keyframes]:
        (images_dir / filename).write_bytes(b"not-a-real-jpeg")

    manifest = {
        "local_origin": [ORIGIN_LNG, ORIGIN_LAT, 0.0],
        "map": {"name": "bbhotel-choisy", "uuid": "u", "venue_type": venue_type},
        "geo_levels": [
            {
                "value": 0.0,
                "min_altitude": -2.0,
                "max_altitude": 4.0,
                "geometry": None,
                **({"geo_ref": geo_ref_id} if geo_ref_id is not None else {}),
            }
        ],
        "geo_keyframes": [
            {
                "x": x,
                "y": y,
                "z": z,
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "image_url": f"https://example/uuid/images/{filename}{query}",
                "depth_url": (
                    f"https://example/uuid/depths/{Path(filename).stem}.tif{query}"
                ),
            }
            for filename, (x, y, z) in keyframes
        ],
    }
    (map_path / name).write_text(json.dumps(manifest), encoding="utf-8")
    return map_path


def test_v2_ids_are_the_manifest_index(tmp_path: Path) -> None:
    """Ids come from manifest order, not from the images' sort order.

    Alphabetically the files are a, b, c; in the manifest they are c, a, b. Sorting
    the directory instead of reading the manifest would give the wrong pose to all
    three.
    """
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[
            ("c.jpg", (0.0, 0.0, 0.0)),
            ("a.jpg", (1.0, 0.0, 0.0)),
            ("b.jpg", (2.0, 0.0, 0.0)),
        ],
    )

    entries = collect_image_entries(map_path)

    assert {path.name: kf_id for kf_id, path in entries} == {
        "c.jpg": 0,
        "a.jpg": 1,
        "b.jpg": 2,
    }
    assert [kf_id for kf_id, _ in entries] == [0, 1, 2], "sorted by keyframe id"


def test_v2_uuid_filenames_resolve(tmp_path: Path) -> None:
    """v2 names images by UUID, so only the manifest can identify them."""
    uuid_name = "a83273f2-6a23-4ab0-a8fc-d0b07a71b404.jpg"
    map_path = _make_v2_map(tmp_path, keyframes=[(uuid_name, (0.0, 0.0, 0.0))])
    entries = collect_image_entries(map_path)
    assert entries == [(0, map_path / "images_360" / uuid_name)]


@pytest.mark.parametrize(
    "query", ["", "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef"]
)
def test_v2_presigned_urls_resolve_to_the_filename_on_disk(
    tmp_path: Path, query: str
) -> None:
    """Asset URLs are presigned, so the query string must not reach the filename.

    `retrieve-map-data` writes each asset under the URL's path basename. Reading the
    manifest by splitting on "/" would instead yield `a.jpg?X-Amz-Signature=…`, which
    matches nothing on disk: no image would resolve to a keyframe id and no depth map
    would be found. Both forms must give the same answer.
    """
    map_path = _make_v2_map(
        tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))], query=query
    )

    source = load_pose_source(map_path)

    assert source.image_filename_to_keyframe_id == {"a.jpg": 0}
    assert source.depth_filename_by_keyframe_id == {0: "a.tif"}
    assert collect_image_entries(map_path) == [(0, map_path / "images_360" / "a.jpg")]


def test_v2_images_directory_may_be_named_images(tmp_path: Path) -> None:
    """v2 mirrors the S3 layout (`images/`); v1 used `images_360/`. Both work."""
    map_path = _make_v2_map(
        tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))], images_dirname="images"
    )
    assert resolve_images_dir(map_path).name == "images"
    assert [kf_id for kf_id, _ in collect_image_entries(map_path)] == [0]


def test_v2_keyframes_without_an_image_on_disk_are_dropped(tmp_path: Path) -> None:
    """A partial download must index what is there, not renumber it."""
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[
            ("a.jpg", (0.0, 0.0, 0.0)),
            ("b.jpg", (1.0, 0.0, 0.0)),
            ("c.jpg", (2.0, 0.0, 0.0)),
        ],
        images=["c.jpg"],  # only the third was downloaded
    )
    assert collect_image_entries(map_path) == [(2, map_path / "images_360" / "c.jpg")]


def test_v2_extra_images_not_in_the_manifest_are_skipped(tmp_path: Path) -> None:
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[("known.jpg", (0.0, 0.0, 0.0))],
        images=["known.jpg", "stranger.jpg"],
    )
    expected = [(0, map_path / "images_360" / "known.jpg")]
    assert collect_image_entries(map_path) == expected


def test_v2_raises_when_no_image_matches(tmp_path: Path) -> None:
    map_path = _make_v2_map(
        tmp_path, keyframes=[("known.jpg", (0.0, 0.0, 0.0))], images=["stranger.jpg"]
    )
    with pytest.raises(ValueError, match="resolved to a keyframe id"):
        collect_image_entries(map_path)


def test_v2_pose_source_exposes_venue_geo_ref_and_needs_no_frame_conversion(
    tmp_path: Path,
) -> None:
    """The manifest stores what api_geokeyframe stores — poses pass through as-is."""
    map_path = _make_v2_map(
        tmp_path, keyframes=[("a.jpg", (1.5, 2.5, -3.5))], venue_type="rail"
    )

    source = load_pose_source(map_path)

    assert source.kind == "manifest"
    assert source.venue_type == "rail"
    assert source.geo_ref_id == 149
    np.testing.assert_allclose(source.poses[0].position_eus, [1.5, 2.5, -3.5])
    np.testing.assert_allclose(source.poses[0].orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert source.depth_filename_by_keyframe_id == {0: "a.tif"}


def test_v2_manifest_wins_over_a_legacy_georef_db(tmp_path: Path) -> None:
    """A map carrying both must use the manifest — the v1 poses need conversion."""
    map_path = _make_v2_map(tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))])
    _write_georef_db(map_path, rows=[(99, "a.jpg", (0.0, 0.0, 0.0))])

    source = load_pose_source(map_path)

    assert source.kind == "manifest"
    assert set(source.poses) == {0}, "the georef.db id 99 must not appear"


def test_newest_manifest_wins(tmp_path: Path) -> None:
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[("a.jpg", (0.0, 0.0, 0.0))],
        name="m_2_20260101_000000.json",
    )
    # A newer export listing two keyframes.
    newer = json.loads((map_path / "m_2_20260101_000000.json").read_text())
    newer["geo_keyframes"].append(
        {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "image_url": "https://example/uuid/images/b.jpg",
            "depth_url": "https://example/uuid/depths/b.tif",
        }
    )
    (map_path / "m_2_20260607_120000.json").write_text(json.dumps(newer))
    (map_path / "images_360" / "b.jpg").write_bytes(b"x")

    source = load_pose_source(map_path)

    assert source.path.name == "m_2_20260607_120000.json"
    assert set(source.poses) == {0, 1}


# ------------------------------------------------------------------ v1 georef.db maps


def _pose_blob(position_eus: tuple[float, float, float]) -> bytes:
    """A `GeoRefKeyframe.pose` blob that decodes to `position_eus`, identity rotation.

    Undoes the v1 reader: EUS → WDS, camera-to-world → world-to-camera, transposed.
    """
    rotation = ROT_WDS_TO_EUS @ np.eye(3) @ ROT_OPENGL_TO_OPENCV
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = ROT_WDS_TO_EUS @ np.asarray(position_eus, dtype=np.float64)
    return np.ascontiguousarray(
        np.linalg.inv(camera_to_world).T, dtype=np.float64
    ).tobytes()


def _write_georef_db(
    map_path: Path,
    *,
    rows: list[tuple[int, str | None, tuple[float, float, float]]],
    filename_column: str | None = "filename",
) -> None:
    """A complete legacy georef.db: origin, one level band, and keyframes."""
    conn = sqlite3.connect(map_path / "georef.db")
    conn.execute("CREATE TABLE GeoRef (latitude REAL, longitude REAL, altitude REAL)")
    conn.execute("INSERT INTO GeoRef VALUES (?, ?, ?)", (ORIGIN_LAT, ORIGIN_LNG, 0.0))
    conn.execute(
        "CREATE TABLE Level (id INTEGER, min_altitude REAL, max_altitude REAL, "
        "geometry TEXT)"
    )
    conn.execute("INSERT INTO Level VALUES (0, -2.0, 4.0, NULL)")
    if filename_column is None:
        conn.execute("CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB)")
        for keyframe_id, _filename, position in rows:
            conn.execute(
                "INSERT INTO GeoRefKeyframe VALUES (?, ?)",
                (keyframe_id, _pose_blob(position)),
            )
    else:
        conn.execute(
            f"CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB, "
            f'"{filename_column}" TEXT)'
        )
        for keyframe_id, filename, position in rows:
            conn.execute(
                "INSERT INTO GeoRefKeyframe VALUES (?, ?, ?)",
                (keyframe_id, _pose_blob(position), filename),
            )
    conn.commit()
    conn.close()


def _make_v1_map(
    tmp_path: Path,
    *,
    rows: list[tuple[int, str | None, tuple[float, float, float]]],
    images: list[str],
    filename_column: str | None = "filename",
) -> Path:
    map_path = tmp_path / "v1map"
    images_dir = map_path / "images_360"
    images_dir.mkdir(parents=True)
    for filename in images:
        (images_dir / filename).write_bytes(b"not-a-real-jpeg")
    _write_georef_db(map_path, rows=rows, filename_column=filename_column)
    return map_path


def test_v1_ids_come_from_georef_db_not_from_position(tmp_path: Path) -> None:
    """Sorted the files are a, b, c — positional numbering would give 0, 1, 2."""
    map_path = _make_v1_map(
        tmp_path,
        rows=[
            (907, "a.jpg", (0.0, 0.0, 0.0)),
            (42, "b.jpg", (1.0, 0.0, 0.0)),
            (108, "c.jpg", (2.0, 0.0, 0.0)),
        ],
        images=["a.jpg", "b.jpg", "c.jpg"],
    )

    entries = collect_image_entries(map_path)

    assert [kf_id for kf_id, _ in entries] == [42, 108, 907]
    assert {kf_id: path.name for kf_id, path in entries} == {
        42: "b.jpg",
        108: "c.jpg",
        907: "a.jpg",
    }


def test_v1_legacy_image_filename_column_is_supported(tmp_path: Path) -> None:
    map_path = _make_v1_map(
        tmp_path,
        rows=[(5, "kf.jpg", (0.0, 0.0, 0.0))],
        images=["kf.jpg"],
        filename_column="image_filename",
    )
    assert [kf_id for kf_id, _ in collect_image_entries(map_path)] == [5]


def test_v1_falls_back_to_the_stem_without_a_filename_column(tmp_path: Path) -> None:
    """`{id}.jpg` maps: the stem *is* the keyframe id."""
    map_path = _make_v1_map(
        tmp_path,
        rows=[(12, None, (0.0, 0.0, 0.0)), (7, None, (1.0, 0.0, 0.0))],
        images=["12.jpg", "7.jpg"],
        filename_column=None,
    )
    assert [kf_id for kf_id, _ in collect_image_entries(map_path)] == [7, 12]


def test_v1_pose_source_is_flagged_as_the_legacy_path(tmp_path: Path) -> None:
    map_path = _make_v1_map(
        tmp_path, rows=[(3, "a.jpg", (1.5, 2.5, -3.5))], images=["a.jpg"]
    )

    source = load_pose_source(map_path)

    assert source.kind == "georef_db"
    # No venue and no georef id in v1 — they must be passed on the command line.
    assert source.venue_type is None
    assert source.geo_ref_id is None
    # And the three frame flips must round-trip the position the fixture encoded.
    np.testing.assert_allclose(
        source.poses[3].position_eus, [1.5, 2.5, -3.5], atol=1e-9
    )


def test_v1_duplicate_ids_raise(tmp_path: Path) -> None:
    """`04.jpg` and `4.jpg` both parse to 4 via the stem fallback."""
    map_path = _make_v1_map(
        tmp_path,
        rows=[(4, None, (0.0, 0.0, 0.0))],
        images=["04.jpg", "4.jpg"],
        filename_column=None,
    )
    with pytest.raises(ValueError, match="already used"):
        collect_image_entries(map_path)


# ------------------------------------------------------------------------- both / none


def test_no_pose_source_at_all_raises(tmp_path: Path) -> None:
    map_path = tmp_path / "bare"
    (map_path / "images_360").mkdir(parents=True)
    (map_path / "images_360" / "a.jpg").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="neither a v2 manifest"):
        collect_image_entries(map_path)


def test_missing_images_directory_raises(tmp_path: Path) -> None:
    map_path = _make_v2_map(tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))])
    with pytest.raises(FileNotFoundError, match="No ERP images directory"):
        collect_image_entries(map_path, images_dirname="does-not-exist")


def test_empty_images_directory_raises(tmp_path: Path) -> None:
    """An empty run would otherwise write an empty index and look successful."""
    map_path = _make_v2_map(tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))], images=[])
    with pytest.raises(FileNotFoundError, match="images in"):
        collect_image_entries(map_path)


def test_non_image_files_are_ignored(tmp_path: Path) -> None:
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[("a.jpg", (0.0, 0.0, 0.0)), ("b.PNG", (1.0, 0.0, 0.0))],
    )
    (map_path / "images_360" / "notes.txt").write_text("ignore me")
    assert [kf_id for kf_id, _ in collect_image_entries(map_path)] == [0, 1]
