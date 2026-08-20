"""Tests for folding a per-capture prod dump into the single-file layout.

The invariant that matters is positional: after the merge, `row_index` must be a
dense 0-based counter *and* index the merged `embeddings.npy` — the explorer relies
on both, and getting it wrong silently pairs a row with another row's embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolbox.bricks.ingest_cli import EMBEDDING_DIM
from toolbox.bricks.map_manifest import load_map_manifest
from toolbox.bricks.prod_dump_merge import merge_dump
from toolbox.bricks.prod_dump_remap import REMAP_METADATA_KEY, REMAP_METADATA_VALUE
from toolbox.bricks.vendored.geo_transform import Vector3Batch

ORIGIN = [2.35, 48.85, 0.0]
# Local keyframe index → (prod VideoKeyframe.id, video_capture_id).
KEYFRAMES = ((5001, 100), (5002, 100), (5003, 101))
POSITIONS = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, -10.0))


def _write_manifest(map_path: Path) -> None:
    map_path.mkdir(parents=True, exist_ok=True)
    (map_path / "m_8_20260101_000000.json").write_text(
        json.dumps(
            {
                "local_origin": ORIGIN,
                "map": {"name": "m", "uuid": "u", "venue_type": "rail"},
                "geo_levels": [
                    {
                        "value": 0,
                        "min_altitude": -2.0,
                        "max_altitude": 4.0,
                        "geo_ref": 1,
                        "geometry": None,
                    }
                ],
                "geo_keyframes": [
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "image_url": f"https://example/images/kf{i}.jpg",
                        "depth_url": f"https://example/depths/kf{i}.tif",
                    }
                    for i, (x, y, z) in enumerate(POSITIONS)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_trajectory(map_path: Path) -> None:
    manifest = load_map_manifest(map_path)
    wgs84 = np.asarray(
        manifest.geo_transform().local_positions_to_wgs84(
            Vector3Batch(np.asarray([kf.position_eus for kf in manifest.keyframes]))
        )
    )
    captures: dict[int, list[dict]] = {}
    for i, (prod_id, vc_id) in enumerate(KEYFRAMES):
        captures.setdefault(vc_id, []).append(
            {
                "id": prod_id,
                "coords": {
                    "lng": float(wgs84[i, 0]),
                    "lat": float(wgs84[i, 1]),
                    "alt": float(wgs84[i, 2]),
                },
            }
        )
    (map_path / "trajectory.json").write_text(
        json.dumps(
            {
                "captures": [
                    {"video_capture_id": vc, "keyframes": kfs}
                    for vc, kfs in sorted(captures.items())
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_capture(
    capture_dir: Path, local_keyframe_ids: list[int], *, remapped: bool = True
) -> None:
    """One capture's pair, with each row's embedding filled with its own row_index."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    n = len(local_keyframe_ids)
    table = pa.table(
        {
            "row_index": pa.array(range(n), type=pa.int64()),
            "video_keyframe_id": pa.array(local_keyframe_ids, type=pa.int64()),
            "theta_center": pa.array([0.25] * n, type=pa.float16()),
            "thumbnail_key": pa.array(
                [f"{capture_dir.name}/thumbnails/{i:06d}.jpg" for i in range(n)]
            ),
        }
    )
    if remapped:
        table = table.replace_schema_metadata(
            {REMAP_METADATA_KEY: REMAP_METADATA_VALUE}
        )
    pq.write_table(table, capture_dir / "metadata.parquet")
    # Row i's embedding is a constant vector of value (capture number + i/100), so a
    # mispaired row is visible in the merged file.
    base = float(int(capture_dir.name))
    rows = np.stack(
        [np.full(EMBEDDING_DIM, base + i / 100.0, dtype=np.float16) for i in range(n)]
    )
    rows.tofile(capture_dir / "embeddings.npy")


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    map_path = tmp_path / "map"
    _write_manifest(map_path)
    _write_trajectory(map_path)
    _write_capture(map_path / "object-search" / "100", [0, 1])
    _write_capture(map_path / "object-search" / "101", [2])
    return map_path


def test_merges_into_one_dense_pair(dump: Path) -> None:
    result = merge_dump(dump)
    assert (result.rows, result.captures) == (3, 2)

    root = dump / "object-search"
    table = pq.read_table(root / "metadata.parquet")
    assert table.column("row_index").to_pylist() == [0, 1, 2]
    assert table.column("video_keyframe_id").to_pylist() == [0, 1, 2]

    embeddings = np.fromfile(root / "embeddings.npy", dtype=np.float16)
    assert embeddings.reshape(-1, EMBEDDING_DIM).shape == (3, EMBEDDING_DIM)


def test_row_index_indexes_the_merged_embeddings(dump: Path) -> None:
    """The whole point: merged row i must carry capture-row i's embedding."""
    merge_dump(dump)
    root = dump / "object-search"
    table = pq.read_table(root / "metadata.parquet")
    embeddings = np.fromfile(root / "embeddings.npy", dtype=np.float16).reshape(
        -1, EMBEDDING_DIM
    )
    # Capture 100 rows 0,1 then capture 101 row 0 → 100.0, 100.01, 101.0.
    assert [float(embeddings[i, 0]) for i in range(3)] == pytest.approx(
        [100.0, 100.01, 101.0], abs=0.02
    )
    for row_index, key in enumerate(table.column("thumbnail_key").to_pylist()):
        expected = "100" if row_index < 2 else "101"
        assert key.startswith(expected), "thumbnail_key must not be renumbered"


def test_carries_geokeyframe_id_and_image_path(dump: Path) -> None:
    merge_dump(dump)
    table = pq.read_table(dump / "object-search" / "metadata.parquet")
    assert table.column("geokeyframe_id").to_pylist() == [0, 1, 2]
    assert table.column("vk_image_path").to_pylist() == [
        "kf0.jpg",
        "kf1.jpg",
        "kf2.jpg",
    ]


def test_unremapped_capture_is_refused(tmp_path: Path) -> None:
    map_path = tmp_path / "map"
    _write_manifest(map_path)
    _write_trajectory(map_path)
    _write_capture(map_path / "object-search" / "100", [0, 1], remapped=False)
    with pytest.raises(RuntimeError, match="prod keyframe ids"):
        merge_dump(map_path)
    assert not (map_path / "object-search" / "metadata.parquet").exists()


def test_capture_with_no_manifest_keyframe_is_skipped(dump: Path) -> None:
    _write_capture(dump / "object-search" / "102", [-1, -1])
    result = merge_dump(dump)
    assert result.skipped_captures == ("102",)
    assert result.rows == 3


def test_partial_write_leaves_nothing_behind(dump: Path, monkeypatch) -> None:
    """A crash mid-merge must not leave a parquet without its embeddings."""
    import toolbox.bricks.prod_dump_merge as module

    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.pd, "concat", boom)
    with pytest.raises(KeyboardInterrupt):
        merge_dump(dump)
    root = dump / "object-search"
    assert not (root / "metadata.parquet").exists()
    assert not (root / "embeddings.npy").exists()
    assert not list(root.glob("*.tmp"))


def test_manifest_json_records_the_provenance(dump: Path) -> None:
    merge_dump(dump)
    manifest = json.loads(
        (dump / "object-search" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["georef_id"] == 1
    assert manifest["thin_distance"] is None
    assert manifest["captures"] == ["100", "101"]
