"""Exporting a drawn region as a new map.

Three things here fail silently if they are wrong, which is why each has its own
test rather than being covered by one end-to-end pass:

1. **The level datum.** `select_keyframes` must feed `levels_for_altitudes` the EUS
   up coordinate. Fed the WGS84 altitude instead, every level comes back NaN and
   the selection quietly empties or, worse, collapses two floors into one.
2. **The renumbering.** Keyframe ids are `geo_keyframes` indices, so a subset
   renumbers. If the parquet's `video_keyframe_id` is not remapped through the same
   table, candidates attach to whichever keyframes happen to sit at those indices.
3. **The row alignment.** `embeddings.npy` is a headerless dump whose row *i* is
   parquet row *i*. Subset one without the other and every embedding is off by
   however many rows were dropped before it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolbox.bricks.export_roi import (
    PROVENANCE_FILENAME,
    Roi,
    build_manifest_document,
    export_roi,
    select_keyframes,
)
from toolbox.bricks.ingest_cli import EMBEDDING_DIM
from toolbox.bricks.map_manifest import find_manifest, load_manifest

ORIGIN_LNG = 2.3522
ORIGIN_LAT = 48.8566
ORIGIN_ALT = 35.0

# Ground floor sits in [-2, 3], first floor in [3, 9] — the EUS up coordinate of a
# keyframe is what decides, so `y` is what these fixtures vary.
GROUND_Y = 1.0
FIRST_Y = 5.0


def _write_map(map_path: Path, keyframes: list[tuple[float, float, float]]) -> Path:
    """A minimal map: manifest, images/ and depths/, one file per keyframe."""
    map_path.mkdir(parents=True, exist_ok=True)
    manifest_path = map_path / "src_2_20260101_000000.json"
    manifest_path.write_text(
        json.dumps(
            {
                "local_origin": [ORIGIN_LNG, ORIGIN_LAT, ORIGIN_ALT],
                "map": {"name": "src", "uuid": "source-uuid", "venue_type": "rail"},
                "geo_levels": [
                    {
                        "value": 0,
                        "min_altitude": -2.0,
                        "max_altitude": 3.0,
                        "geo_ref": 7,
                        "geometry": None,
                    },
                    {
                        "value": 1,
                        "min_altitude": 3.0,
                        "max_altitude": 9.0,
                        "geo_ref": 7,
                        "geometry": None,
                    },
                ],
                "geo_keyframes": [
                    {
                        "image_url": f"https://example.test/images/kf{index}.jpg",
                        "depth_url": f"https://example.test/depths/kf{index}.tif",
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                    }
                    for index, (x, y, z) in enumerate(keyframes)
                ],
            }
        ),
        encoding="utf-8",
    )
    (map_path / "images").mkdir(exist_ok=True)
    (map_path / "depths").mkdir(exist_ok=True)
    for index in range(len(keyframes)):
        (map_path / "images" / f"kf{index}.jpg").write_bytes(b"jpeg" + bytes([index]))
        (map_path / "depths" / f"kf{index}.tif").write_bytes(b"tiff" + bytes([index]))
    return manifest_path


def _ring_around(map_path: Path, keyframe_ids: list[int], pad_m: float = 3.0):
    """A lng/lat ring enclosing exactly the given keyframes' ground positions.

    Built by converting their EUS positions to WGS84 and taking a padded bounding
    box — a hand-written ring would encode a guess about the projection, which is
    the thing under test.
    """
    from toolbox.bricks.vendored.geo_transform import Vector3Batch

    manifest = load_manifest(find_manifest(map_path))
    transform = manifest.geo_transform()
    positions = np.stack(
        [manifest.keyframes[i].position_eus for i in keyframe_ids]
    ).astype(np.float64)
    padded = np.concatenate(
        [positions + [pad_m, 0.0, pad_m], positions - [pad_m, 0.0, pad_m]]
    )
    wgs84 = np.asarray(transform.local_positions_to_wgs84(Vector3Batch(padded)))
    lng_min, lng_max = wgs84[:, 0].min(), wgs84[:, 0].max()
    lat_min, lat_max = wgs84[:, 1].min(), wgs84[:, 1].max()
    return (
        (lng_min, lat_min),
        (lng_max, lat_min),
        (lng_max, lat_max),
        (lng_min, lat_max),
    )


def _write_artifacts(map_path: Path, rows: list[tuple[int, str]]) -> None:
    """`object-search/` with one parquet row per (keyframe id, thumbnail key)."""
    outputs = map_path / "object-search"
    (outputs / "thumbnails").mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(rows))), type=pa.int64()),
            "video_keyframe_id": pa.array([r[0] for r in rows], type=pa.int64()),
            "theta_center": pa.array([0.1] * len(rows), type=pa.float16()),
            "thumbnail_key": pa.array([r[1] for r in rows], type=pa.string()),
            "depth": pa.array([2.5] * len(rows), type=pa.float64()),
        }
    )
    pq.write_table(table, outputs / "metadata.parquet")
    # Row i of the embeddings carries i in every component, so a misalignment after
    # the subset is visible rather than merely plausible.
    embeddings = np.tile(
        np.arange(len(rows), dtype=np.float16).reshape(-1, 1), (1, EMBEDDING_DIM)
    )
    embeddings.tofile(outputs / "embeddings.npy")
    for index, (_, key) in enumerate(rows):
        if key.startswith("object-search/thumbnails/"):
            (map_path / key).write_bytes(b"thumb" + bytes([index]))


# --- selection ---------------------------------------------------------------


def test_selection_keeps_only_the_right_floor(tmp_path: Path) -> None:
    """Two keyframes at the same place on different floors; the ROI takes one."""
    map_path = tmp_path / "src"
    _write_map(map_path, [(0.0, GROUND_Y, 0.0), (0.0, FIRST_Y, 0.0)])
    manifest = load_manifest(find_manifest(map_path))

    ring = _ring_around(map_path, [0, 1])
    ground = select_keyframes(manifest, [Roi(ring=ring, level=0.0)])
    first = select_keyframes(manifest, [Roi(ring=ring, level=1.0)])

    assert ground.keyframe_ids == [0]
    assert first.keyframe_ids == [1]


def test_selection_excludes_keyframes_outside_the_ring(tmp_path: Path) -> None:
    map_path = tmp_path / "src"
    _write_map(
        map_path,
        [(0.0, GROUND_Y, 0.0), (5.0, GROUND_Y, 0.0), (500.0, GROUND_Y, 0.0)],
    )
    manifest = load_manifest(find_manifest(map_path))

    selection = select_keyframes(
        manifest, [Roi(ring=_ring_around(map_path, [0, 1]), level=0.0)]
    )

    assert selection.keyframe_ids == [0, 1]
    assert selection.per_level == {"0": 2}


def test_selection_unions_regions_across_floors(tmp_path: Path) -> None:
    """The multi-floor session: several regions, deduplicated union."""
    map_path = tmp_path / "src"
    _write_map(
        map_path,
        [
            (0.0, GROUND_Y, 0.0),
            (50.0, GROUND_Y, 0.0),
            (0.0, FIRST_Y, 0.0),
        ],
    )
    manifest = load_manifest(find_manifest(map_path))

    selection = select_keyframes(
        manifest,
        [
            Roi(ring=_ring_around(map_path, [0]), level=0.0),
            # Overlapping second region on the same floor: the union must not
            # count keyframe 0 twice.
            Roi(ring=_ring_around(map_path, [0]), level=0.0),
            Roi(ring=_ring_around(map_path, [2]), level=1.0),
        ],
    )

    assert selection.keyframe_ids == [0, 2]
    assert selection.total == 2
    assert selection.per_level == {"0": 1, "1": 1}


# --- the exported directory --------------------------------------------------


def test_export_renumbers_densely_and_links_assets(tmp_path: Path) -> None:
    map_path = tmp_path / "src"
    _write_map(
        map_path,
        [
            (0.0, GROUND_Y, 0.0),
            (500.0, GROUND_Y, 0.0),  # outside, so ids 0 and 2 survive
            (5.0, GROUND_Y, 0.0),
        ],
    )
    destination = tmp_path / "out"

    result = export_roi(
        source_map=map_path,
        destination=destination,
        map_id="src-roi",
        rois=[Roi(ring=_ring_around(map_path, [0, 2]), level=0.0)],
        geo_ref_id=42,
        parent_map="src",
    )

    assert result["keyframe_count"] == 2
    exported = load_manifest(find_manifest(destination))
    assert [kf.keyframe_id for kf in exported.keyframes] == [0, 1]
    assert [kf.image_filename for kf in exported.keyframes] == ["kf0.jpg", "kf2.jpg"]
    assert exported.geo_ref_id == 42
    assert exported.venue_type == "rail"

    # Hardlinked, not copied: same inode as the source.
    source_image = map_path / "images" / "kf0.jpg"
    exported_image = destination / "images" / "kf0.jpg"
    assert exported_image.read_bytes() == source_image.read_bytes()
    assert exported_image.stat().st_ino == source_image.stat().st_ino
    assert not (destination / "images" / "kf1.jpg").exists()
    assert (destination / "depths" / "kf2.tif").is_file()

    provenance = json.loads((destination / PROVENANCE_FILENAME).read_text())
    assert provenance["keyframe_id_map"] == {"0": 0, "2": 1}
    assert provenance["source"]["geo_ref_id"] == 7
    assert provenance["geo_ref_id"] == 42
    assert provenance["parent_map"] == "src"
    assert len(provenance["rois"]) == 1


def test_export_gives_the_new_map_its_own_identity(tmp_path: Path) -> None:
    """A fresh uuid and version 1: the export starts a lineage, it is not a re-dump."""
    map_path = tmp_path / "src"
    manifest_path = _write_map(map_path, [(0.0, GROUND_Y, 0.0)])
    source_document = json.loads(manifest_path.read_text())

    document = build_manifest_document(source_document, [0], "src-roi", 42)

    assert document["map"]["name"] == "src-roi"
    assert document["map"]["uuid"] != "source-uuid"
    assert document["map"]["venue_type"] == "rail"
    assert [level["geo_ref"] for level in document["geo_levels"]] == [42, 42]
    assert source_document["map"]["uuid"] == "source-uuid"  # untouched


def test_export_refuses_an_empty_selection(tmp_path: Path) -> None:
    map_path = tmp_path / "src"
    _write_map(map_path, [(0.0, GROUND_Y, 0.0)])

    with pytest.raises(ValueError, match="No keyframe"):
        export_roi(
            source_map=map_path,
            destination=tmp_path / "out",
            map_id="src-roi",
            rois=[Roi(ring=_ring_around(map_path, [0]), level=1.0)],
            geo_ref_id=42,
        )
    assert not (tmp_path / "out").exists()


def test_a_failed_export_leaves_no_half_written_map(tmp_path: Path) -> None:
    """A partial directory would still satisfy `find_manifest`; it must not survive."""
    map_path = tmp_path / "src"
    _write_map(map_path, [(0.0, GROUND_Y, 0.0)])
    destination = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        export_roi(
            source_map=map_path,
            destination=destination,
            map_id="src-roi",
            rois=[Roi(ring=_ring_around(map_path, [0]), level=0.0)],
            geo_ref_id=42,
            include_artifacts=True,  # there are none, so this raises mid-flight
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".out.partial-*"))


# --- the optional object-search artifacts ------------------------------------


def test_artifacts_are_remapped_and_stay_row_aligned(tmp_path: Path) -> None:
    map_path = tmp_path / "src"
    _write_map(
        map_path,
        [
            (0.0, GROUND_Y, 0.0),
            (500.0, GROUND_Y, 0.0),  # dropped
            (5.0, GROUND_Y, 0.0),
        ],
    )
    _write_artifacts(
        map_path,
        [
            (0, "object-search/thumbnails/000000.jpg"),
            (1, "object-search/thumbnails/000001.jpg"),  # dropped with keyframe 1
            (2, "object-search/thumbnails/000002.jpg"),
            (2, "object-search/thumbnails/000003.jpg"),
        ],
    )
    destination = tmp_path / "out"

    export_roi(
        source_map=map_path,
        destination=destination,
        map_id="src-roi",
        rois=[Roi(ring=_ring_around(map_path, [0, 2]), level=0.0)],
        geo_ref_id=42,
        include_artifacts=True,
    )

    metadata = pq.read_table(
        destination / "object-search" / "metadata.parquet"
    ).to_pandas()
    assert metadata["row_index"].tolist() == [0, 1, 2]
    # Keyframe 0 stays 0; keyframe 2 becomes 1. Not remapping would leave a 2 here
    # pointing at a keyframe the exported map does not have.
    assert metadata["video_keyframe_id"].tolist() == [0, 1, 1]

    embeddings = np.fromfile(
        destination / "object-search" / "embeddings.npy", dtype=np.float16
    ).reshape(-1, EMBEDDING_DIM)
    assert embeddings.shape[0] == 3
    # Source rows 0, 2 and 3 survived, and each row carries its source index.
    assert embeddings[:, 0].tolist() == [0.0, 2.0, 3.0]

    assert metadata["thumbnail_key"].tolist() == [
        "object-search/thumbnails/000000.jpg",
        "object-search/thumbnails/000001.jpg",
        "object-search/thumbnails/000002.jpg",
    ]
    # The renamed thumbnail must carry the *source* row's pixels, not its neighbour's.
    assert (destination / "object-search/thumbnails/000001.jpg").read_bytes() == (
        map_path / "object-search/thumbnails/000002.jpg"
    ).read_bytes()


def test_virtual_thumbnail_keys_are_renumbered_too(tmp_path: Path) -> None:
    """A v1-converted index has no crops; its key is virtual but still row-indexed."""
    map_path = tmp_path / "src"
    _write_map(map_path, [(0.0, GROUND_Y, 0.0), (500.0, GROUND_Y, 0.0)])
    _write_artifacts(
        map_path,
        [
            (1, "object-search/rows/0.png"),
            (0, "object-search/rows/1.png"),
            (0, "object-search/rows/2.png"),
        ],
    )
    destination = tmp_path / "out"

    export_roi(
        source_map=map_path,
        destination=destination,
        map_id="src-roi",
        rois=[Roi(ring=_ring_around(map_path, [0]), level=0.0)],
        geo_ref_id=42,
        include_artifacts=True,
    )

    metadata = pq.read_table(
        destination / "object-search" / "metadata.parquet"
    ).to_pandas()
    assert metadata["thumbnail_key"].tolist() == [
        "object-search/rows/0.png",
        "object-search/rows/1.png",
    ]
    assert metadata["video_keyframe_id"].tolist() == [0, 0]
