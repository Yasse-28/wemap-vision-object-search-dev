"""Tests for keyframe-id resolution before `prepare` runs.

This is the step that stops `prepare` output from being attached to the wrong poses.
The mirrored `prepare` CLI numbers keyframes positionally (`enumerate`), so if that
numbering ever reached `ingest_cli` the candidates would land on whichever keyframes
happened to share those ids — silently, and with every object in the wrong place.
`collect_image_entries` is what prevents it, so its failure modes are pinned here.

No detection runs — `run_prepare` needs a GPU and model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toolbox.bricks.georef_source import load_pose_source
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
    images_dirname: str = "images",
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
    assert entries == [(0, map_path / "images" / uuid_name)]


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
    assert collect_image_entries(map_path) == [(0, map_path / "images" / "a.jpg")]


def test_images_directory_mirrors_the_s3_layout(tmp_path: Path) -> None:
    """`images/` is the only convention; a differently named one must be passed."""
    map_path = _make_v2_map(tmp_path, keyframes=[("a.jpg", (0.0, 0.0, 0.0))])
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
    assert collect_image_entries(map_path) == [(2, map_path / "images" / "c.jpg")]


def test_v2_extra_images_not_in_the_manifest_are_skipped(tmp_path: Path) -> None:
    map_path = _make_v2_map(
        tmp_path,
        keyframes=[("known.jpg", (0.0, 0.0, 0.0))],
        images=["known.jpg", "stranger.jpg"],
    )
    expected = [(0, map_path / "images" / "known.jpg")]
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

    assert source.venue_type == "rail"
    assert source.geo_ref_id == 149
    np.testing.assert_allclose(source.poses[0].position_eus, [1.5, 2.5, -3.5])
    np.testing.assert_allclose(source.poses[0].orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert source.depth_filename_by_keyframe_id == {0: "a.tif"}


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
    (map_path / "images" / "b.jpg").write_bytes(b"x")

    source = load_pose_source(map_path)

    assert source.path.name == "m_2_20260607_120000.json"
    assert set(source.poses) == {0, 1}


def test_no_manifest_raises(tmp_path: Path) -> None:
    map_path = tmp_path / "bare"
    (map_path / "images").mkdir(parents=True)
    (map_path / "images" / "a.jpg").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="No v2 map manifest"):
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
    (map_path / "images" / "notes.txt").write_text("ignore me")
    assert [kf_id for kf_id, _ in collect_image_entries(map_path)] == [0, 1]
