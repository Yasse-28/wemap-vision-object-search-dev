"""The two pivot-export fields the download path depends on.

Both failures are silent and both cost a full re-download:

1. **The version.** Every S3 path for the 360-viewer artefacts is built from it. It
   used to come from the filename only, so a renamed export pointed the trajectory
   download at another version — which `build_keyframe_id_map` then rejects for
   "coordinates that do not match", a message that reads like a join bug.
2. **`videos[]`.** It says which prod captures this version was built from. Absent,
   the dump falls back to the whole S3 listing and quietly pulls captures whose
   keyframes are in no manifest. Absent *from the file*, it must stay "unknown" —
   an empty tuple read as "no captures" would make the dump fetch nothing at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolbox.bricks.map_manifest import load_map_manifest

ORIGIN = [2.35, 48.85, 0.0]


def _write_manifest(
    map_path: Path,
    *,
    filename: str = "m_8_20260101_000000.json",
    map_block: dict | None = None,
    videos: object = None,
) -> Path:
    map_path.mkdir(parents=True, exist_ok=True)
    document: dict = {
        "local_origin": ORIGIN,
        "map": {"name": "m", "uuid": "u", "venue_type": "airport", **(map_block or {})},
        "geo_levels": [
            {
                "value": 0,
                "min_altitude": -2.0,
                "max_altitude": 4.0,
                "geo_ref": 30,
                "geometry": None,
            }
        ],
        "geo_keyframes": [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "image_url": "https://example.test/images/a.jpg",
                "depth_url": "https://example.test/depths/a.tif",
            }
        ],
    }
    if videos is not None:
        document["videos"] = videos
    path = map_path / filename
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_georef_version_wins_over_the_filename(tmp_path: Path) -> None:
    """The field is the exported value; the filename is a convention a rename breaks."""
    _write_manifest(
        tmp_path, filename="m_8_20260101_000000.json", map_block={"georef_version": 9}
    )
    assert load_map_manifest(tmp_path).map_version == 9


def test_version_falls_back_to_the_filename(tmp_path: Path) -> None:
    """Exports predating `georef_version` must keep working unchanged."""
    _write_manifest(tmp_path, filename="m_8_20260101_000000.json")
    assert load_map_manifest(tmp_path).map_version == 8


def test_videos_are_sorted_and_deduplicated(tmp_path: Path) -> None:
    _write_manifest(tmp_path, videos=[151, 80, 80, 100])
    assert load_map_manifest(tmp_path).video_capture_ids == (80, 100, 151)


def test_a_manifest_without_videos_says_unknown_not_none(tmp_path: Path) -> None:
    """Empty means "the export predates the field", never "this map has no capture"."""
    _write_manifest(tmp_path)
    assert load_map_manifest(tmp_path).video_capture_ids == ()


def test_videos_of_the_wrong_shape_is_an_error(tmp_path: Path) -> None:
    """A string would iterate into characters and produce a capture list of digits."""
    _write_manifest(tmp_path, videos="80,81")
    with pytest.raises(ValueError, match="videos"):
        load_map_manifest(tmp_path)
