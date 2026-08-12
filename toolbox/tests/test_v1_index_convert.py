"""Tests for the v1 → v2 index conversion.

The conversion is only correct if three joins hold: v1 `keyframe_id` → manifest
index (by image filename), `bbox_spherical_coordinates` → the four angle columns,
and parquet row `i` → embedding row `i`. Each is silent when wrong — a shifted
embedding file still loads, and a wrong keyframe id still ingests — so all three
are pinned here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toolbox.bricks.v1_index_convert import EMBEDDING_DIM, convert

V1_SCHEMA = """
CREATE TABLE object (
    object_idx INTEGER PRIMARY KEY,
    keyframe_id INTEGER NOT NULL,
    bbox_spherical_coordinates BLOB,
    embedding BLOB NOT NULL,
    depth REAL,
    label TEXT,
    detection_source TEXT
);
CREATE TABLE GeoRefKeyframe (
    id INTEGER PRIMARY KEY,
    image_filename TEXT NOT NULL
);
"""


def _write_manifest(map_path: Path, filenames: list[str]) -> None:
    """A v2 manifest whose `geo_keyframes` order defines the v2 keyframe ids."""
    map_path.mkdir(parents=True, exist_ok=True)
    (map_path / "test-map_1_20260101_000000.json").write_text(
        json.dumps(
            {
                "local_origin": [-69.6, 18.4, 0.0],
                "map": {"uuid": "uuid", "name": "test-map", "venue_type": "airport"},
                "geo_levels": [
                    {
                        "value": 0.0,
                        "min_altitude": -1.0,
                        "max_altitude": 5.0,
                        "geo_ref": 30,
                    }
                ],
                "geo_keyframes": [
                    {
                        "x": float(i),
                        "y": 0.0,
                        "z": 0.0,
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "image_url": f"https://example.test/images/{name}",
                        "depth_url": f"https://example.test/depths/{Path(name).stem}.tif",
                    }
                    for i, name in enumerate(filenames)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_v1(
    v1_dir: Path,
    *,
    georef_rows: list[tuple[int, str]],
    object_rows: list[tuple],
) -> None:
    v1_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(v1_dir / "object-search.db") as objects:
        objects.executescript(V1_SCHEMA)
        objects.executemany(
            "INSERT INTO object (object_idx, keyframe_id, "
            "bbox_spherical_coordinates, embedding, depth, label, detection_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            object_rows,
        )
    with sqlite3.connect(v1_dir / "georef.db") as georef:
        georef.executescript(V1_SCHEMA)
        georef.executemany(
            "INSERT INTO GeoRefKeyframe (id, image_filename) VALUES (?, ?)",
            georef_rows,
        )


def _angles(theta: float, phi: float, fov_x: float, fov_y: float) -> bytes:
    return np.array([theta, phi, fov_x, fov_y], dtype=np.float32).tobytes()


def _embedding(value: float) -> bytes:
    return np.full(EMBEDDING_DIM, value, dtype=np.float32).tobytes()


@pytest.fixture
def converted(tmp_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """One conversion covering every skip path, read back from disk."""
    map_path = tmp_path / "map"
    # Manifest order is deliberately not the v1 id order: the join is by filename.
    _write_manifest(map_path, ["b.jpg", "a.jpg", "c.jpg"])
    _write_v1(
        tmp_path / "v1",
        georef_rows=[(7, "a.jpg"), (8, "b.jpg")],
        object_rows=[
            (1, 7, _angles(0.25, -0.5, 0.1, 0.2), _embedding(0.5), 3.5, "lamp", "yolo"),
            (2, 8, _angles(1.0, 0.0, 0.3, 0.4), _embedding(0.25), None, None, "gdino"),
            # keyframe 99 is in no georef row → unknown keyframe.
            (3, 99, _angles(0.0, 0.0, 0.1, 0.1), _embedding(1.0), 1.0, "x", "yolo"),
            # no spherical bbox → no angles to convert.
            (4, 7, None, _embedding(1.0), 1.0, "x", "yolo"),
        ],
    )

    stats = convert(v1_dir=tmp_path / "v1", map_path=map_path, batch_rows=2)

    assert (stats.read, stats.written) == (4, 2)
    assert (stats.unknown_keyframe, stats.no_spherical_bbox, stats.no_depth) == (
        1,
        1,
        1,
    )
    metadata = pd.read_parquet(map_path / "object-search" / "metadata.parquet")
    raw = np.fromfile(map_path / "object-search" / "embeddings.npy", dtype=np.float16)
    return metadata, raw.reshape(-1, EMBEDDING_DIM)


def test_keyframe_ids_come_from_the_manifest_order(
    converted: tuple[pd.DataFrame, np.ndarray],
) -> None:
    metadata, _ = converted

    # a.jpg is manifest index 1, b.jpg is index 0 — not the v1 ids 7 and 8.
    assert metadata["video_keyframe_id"].tolist() == [1, 0]
    assert metadata["geokeyframe_id"].tolist() == [1, 0]
    assert metadata["vk_image_path"].tolist() == ["a.jpg", "b.jpg"]


def test_angles_and_columns_map_across(
    converted: tuple[pd.DataFrame, np.ndarray],
) -> None:
    metadata, _ = converted

    assert metadata["row_index"].tolist() == [0, 1]
    # theta and the two extents carry over as-is; phi is negated, because v1's is
    # positive downwards and v2's is positive upwards (see the module docstring).
    np.testing.assert_allclose(
        metadata[["theta_center", "phi_center", "angular_width", "angular_height"]]
        .to_numpy()
        .astype(np.float32),
        np.array([[0.25, +0.5, 0.1, 0.2], [1.0, 0.0, 0.3, 0.4]], dtype=np.float32),
        atol=1e-3,  # the v2 columns are float16
    )
    assert metadata["detector_source"].tolist() == ["yolo", "gdino"]
    assert metadata["label"].tolist()[0] == "lamp"
    assert metadata["label"].isna().tolist() == [False, True]
    assert metadata["depth"].tolist()[0] == 3.5
    assert np.isnan(metadata["depth"].tolist()[1])
    # v1 stored no detector confidence; NULL is what ingest must see.
    assert metadata["detection_score"].isna().all()


def test_embeddings_stay_aligned_with_the_metadata_rows(
    converted: tuple[pd.DataFrame, np.ndarray],
) -> None:
    metadata, embeddings = converted

    assert embeddings.shape == (len(metadata), EMBEDDING_DIM)
    assert embeddings.dtype == np.float16
    # Row order follows object_idx, and the skipped rows leave no gap.
    np.testing.assert_array_equal(embeddings[:, 0], np.array([0.5, 0.25], np.float16))


def test_refuses_to_overwrite_an_existing_output_dir(tmp_path: Path) -> None:
    map_path = tmp_path / "map"
    _write_manifest(map_path, ["a.jpg"])
    _write_v1(
        tmp_path / "v1",
        georef_rows=[(1, "a.jpg")],
        object_rows=[
            (1, 1, _angles(0.0, 0.0, 0.1, 0.1), _embedding(1.0), 1.0, "x", "yolo")
        ],
    )
    (map_path / "object-search").mkdir()

    with pytest.raises(FileExistsError):
        convert(v1_dir=tmp_path / "v1", map_path=map_path)
