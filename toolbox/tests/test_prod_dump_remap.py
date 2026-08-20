"""Tests for re-keying a prod dump to local keyframe ids.

The failure this guards against is silent: prod `VideoKeyframe.id`s and the local
`geo_keyframes` indices share a numeric range, so an unremapped — or twice-remapped
— dump attaches candidates to the wrong keyframes with no error anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolbox.bricks.map_manifest import load_map_manifest
from toolbox.bricks.prod_dump_remap import (
    UNMAPPED_KEYFRAME_ID,
    build_keyframe_id_map,
    remap_capture,
    remap_dump,
)
from toolbox.bricks.vendored.geo_transform import Vector3Batch

ORIGIN = [2.35, 48.85, 0.0]
# Local keyframe index → prod VideoKeyframe.id, deliberately overlapping 0..2.
PROD_IDS: tuple[int, ...] = (1, 70000, 2)
POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (0.0, 0.0, -10.0),
)


def _write_manifest(
    map_path: Path, positions: tuple[tuple[float, float, float], ...] = POSITIONS
) -> None:
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
                        "image_url": f"https://example/images/{i}.jpg",
                        "depth_url": f"https://example/depths/{i}.tif",
                    }
                    for i, (x, y, z) in enumerate(positions)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_trajectory(
    map_path: Path, *, prod_ids: tuple[int, ...] = PROD_IDS, shift: float = 0.0
) -> Path:
    """A trajectory whose keyframes sit at the manifest's positions, in WGS84.

    `shift` offsets every point, which is how a trajectory from another map version
    looks: same shape, no position in common.
    """
    manifest = load_map_manifest(map_path)
    eus = np.asarray([kf.position_eus for kf in manifest.keyframes]) + shift
    wgs84 = np.asarray(
        manifest.geo_transform().local_positions_to_wgs84(Vector3Batch(eus))
    )
    path = map_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "captures": [
                    {
                        "video_capture_id": 100,
                        "keyframes": [
                            {
                                "id": int(prod_id),
                                "frame_time": float(i),
                                "coords": {
                                    "lng": float(wgs84[i, 0]),
                                    "lat": float(wgs84[i, 1]),
                                    "alt": float(wgs84[i, 2]),
                                },
                            }
                            for i, prod_id in enumerate(prod_ids)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_capture(capture_dir: Path, video_keyframe_ids: list[int]) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    path = capture_dir / "metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "row_index": pa.array(range(len(video_keyframe_ids)), type=pa.int64()),
                "video_keyframe_id": pa.array(video_keyframe_ids, type=pa.int64()),
                "theta_center": pa.array(
                    [0.0] * len(video_keyframe_ids), type=pa.float16()
                ),
            }
        ),
        path,
    )
    return path


@pytest.fixture
def map_path(tmp_path: Path) -> Path:
    path = tmp_path / "map"
    _write_manifest(path)
    return path


def test_maps_prod_ids_to_manifest_indices(map_path: Path) -> None:
    trajectory = _write_trajectory(map_path)
    id_map = build_keyframe_id_map(load_map_manifest(map_path), trajectory)

    got = id_map.apply(np.asarray([70000, 1, 2, 999], dtype=np.int64))
    assert got.tolist() == [1, 0, 2, UNMAPPED_KEYFRAME_ID]


def test_coincident_positions_are_not_guessed(tmp_path: Path) -> None:
    """Two keyframes at the same point map to neither, rather than to one at random."""
    map_path = tmp_path / "map"
    _write_manifest(map_path, positions=(POSITIONS[0], POSITIONS[0], POSITIONS[2]))
    trajectory = _write_trajectory(map_path)

    id_map = build_keyframe_id_map(load_map_manifest(map_path), trajectory)
    assert id_map.ambiguous == 2
    assert id_map.apply(np.asarray([1, 70000, 2], dtype=np.int64)).tolist() == [
        UNMAPPED_KEYFRAME_ID,
        UNMAPPED_KEYFRAME_ID,
        2,
    ]


def test_trajectory_from_another_version_is_refused(map_path: Path) -> None:
    trajectory = _write_trajectory(map_path, shift=500.0)
    with pytest.raises(ValueError, match="another map version"):
        build_keyframe_id_map(load_map_manifest(map_path), trajectory)


def test_remap_dump_rewrites_every_capture(map_path: Path) -> None:
    _write_trajectory(map_path)
    first = _write_capture(map_path / "object-search" / "100", [70000, 1, 1])
    second = _write_capture(map_path / "object-search" / "101", [2, 4242])

    rows, mapped = remap_dump(map_path)
    assert (rows, mapped) == (5, 4)

    assert pq.read_table(first).column("video_keyframe_id").to_pylist() == [1, 0, 0]
    assert pq.read_table(second).column("video_keyframe_id").to_pylist() == [
        2,
        UNMAPPED_KEYFRAME_ID,
    ]


def test_remap_is_not_applied_twice(map_path: Path) -> None:
    """A second pass would re-map now-local ids onto other keyframes. It must not."""
    trajectory = _write_trajectory(map_path)
    metadata_path = _write_capture(map_path / "object-search" / "100", [70000, 1, 2])

    id_map = build_keyframe_id_map(load_map_manifest(map_path), trajectory)
    remap_capture(metadata_path, id_map)
    after_first = pq.read_table(metadata_path).column("video_keyframe_id").to_pylist()
    assert after_first == [1, 0, 2]

    remap_capture(metadata_path, id_map)
    assert (
        pq.read_table(metadata_path).column("video_keyframe_id").to_pylist()
        == after_first
    )


def test_other_columns_and_row_order_are_untouched(map_path: Path) -> None:
    """`embeddings.npy` and `thumbnail_key` are positional — nothing may move."""
    trajectory = _write_trajectory(map_path)
    metadata_path = _write_capture(map_path / "object-search" / "100", [2, 70000, 1])
    before = pq.read_table(metadata_path)

    remap_capture(
        metadata_path, build_keyframe_id_map(load_map_manifest(map_path), trajectory)
    )
    after = pq.read_table(metadata_path)

    assert after.num_rows == before.num_rows
    assert after.schema.names == before.schema.names
    assert (
        after.column("row_index").to_pylist() == before.column("row_index").to_pylist()
    )
    assert after.schema.field("theta_center").type == pa.float16()


def test_missing_trajectory_is_an_error(map_path: Path) -> None:
    _write_capture(map_path / "object-search" / "100", [1])
    with pytest.raises(FileNotFoundError, match="trajectory.json"):
        remap_dump(map_path)
