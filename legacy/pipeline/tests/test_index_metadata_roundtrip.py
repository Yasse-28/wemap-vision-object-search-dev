import json
import sqlite3
from pathlib import Path

import numpy as np

from pipeline.core.database import (
    INDEX_METADATA_PARAM_KEY,
    LEGACY_MANIFEST_PARAM_KEY,
    create_object_search_index_tables,
)
from pipeline.core.io import load_index, save_index_to_db
from pipeline.core.types import (
    OBJECT_SEARCH_INDEX_DB_FILENAME,
    ObjectSearchIndexMetadata,
    default_created_utc,
)


def test_object_search_index_metadata_json_roundtrip() -> None:
    m = ObjectSearchIndexMetadata(
        projection_dim=512,
        created_utc=default_created_utc(),
        object_detector_prompt="chair .",
        cutout_count=3,
        object_count=10,
        source_images_dir="/tmp/img",
        gdino_params_json='{"prompt": "a ."}',
    )
    payload = m.to_json()
    loaded = ObjectSearchIndexMetadata.from_json(payload)
    assert loaded.projection_dim == 512
    assert loaded.object_count == 10
    assert loaded.id_stride == 1024
    assert loaded.gdino_params_json == '{"prompt": "a ."}'
    data = json.loads(payload)
    assert data["schema_version"] == 3
    assert data["source"] == "equirect360"
    assert data["gdino_params_json"] == '{"prompt": "a ."}'


def test_load_index_reads_sqlite_index(tmp_path: Path) -> None:
    metadata = ObjectSearchIndexMetadata(
        projection_dim=2,
        created_utc=default_created_utc(),
        cutout_count=1,
    )
    state = {
        "cutout_embeddings": np.array([[1.0, 0.0]], dtype=np.float32),
        "cutout_ids": np.array([777], dtype=np.int64),
        "cutout_keyframe_ids": np.array([42], dtype=np.int64),
        "cutout_center_xy": np.array([[12.5, 33.5]], dtype=np.float32),
        "cutout_rotation_cutout_to_equirect": np.eye(4, dtype=np.float32)[None, ...],
        "object_embeddings": np.zeros((0, 2), dtype=np.float32),
        "object_keyframe_ids": np.array([], dtype=np.int64),
        "object_cutout_ids": np.array([], dtype=np.int64),
        "object_bboxes": np.zeros((0, 4), dtype=np.float32),
    }
    save_index_to_db(tmp_path / OBJECT_SEARCH_INDEX_DB_FILENAME, metadata, state)

    loaded = load_index(tmp_path)

    assert loaded.cutout_keyframe_ids.tolist() == [42]
    assert loaded.cutout_center_xy.tolist() == [[12.5, 33.5]]
    assert loaded.cutout_rotation_cutout_to_equirect.shape == (1, 4, 4)


def test_load_index_reads_ocr_fields(tmp_path: Path) -> None:
    metadata = ObjectSearchIndexMetadata(
        projection_dim=2,
        created_utc=default_created_utc(),
        cutout_count=1,
        object_count=2,
    )
    state = {
        "cutout_embeddings": np.array([[1.0, 0.0]], dtype=np.float32),
        "cutout_ids": np.array([777], dtype=np.int64),
        "cutout_keyframe_ids": np.array([42], dtype=np.int64),
        "cutout_center_xy": np.array([[12.5, 33.5]], dtype=np.float32),
        "cutout_rotation_cutout_to_equirect": np.eye(4, dtype=np.float32)[None, ...],
        "object_embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "object_keyframe_ids": np.array([42, 43], dtype=np.int64),
        "object_cutout_ids": np.array([777, 778], dtype=np.int64),
        "object_bboxes": np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32),
        "object_textness_scores": np.array([0.2, 0.4], dtype=np.float32),
        "object_ocr_texts": np.array(["repere s 14", ""], dtype="<U512"),
        "object_ocr_tokens": np.array(["repere s 14", ""], dtype="<U512"),
        "object_ocr_keys": np.array(["letters=s;numbers=14", ""], dtype="<U256"),
        "object_ocr_candidate_mask": np.array([True, False]),
        "object_ocr_assigned_mask": np.array([True, False]),
        "object_ocr_source": np.array([2, 0], dtype=np.int16),
        "object_cluster_ids": np.array([0, 0], dtype=np.int32),
        "object_detection_levels": np.array([1, 1], dtype=np.int32),
        "cluster_centroids_world": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "cluster_centroids_geo": np.array([[40.0, 50.0, 100.0]], dtype=np.float32),
        "cluster_observation_counts": np.array([2], dtype=np.int32),
        "cluster_confidence": np.array([0.9], dtype=np.float32),
        "cluster_levels": np.array([1], dtype=np.int32),
        "cluster_cutout_cluster_ids": np.array([0], dtype=np.int32),
        "cluster_cutout_ids": np.array([777], dtype=np.int64),
        "cluster_cutout_keyframe_ids": np.array([42], dtype=np.int64),
        "cluster_cutout_levels": np.array([1], dtype=np.int32),
        "cluster_cutout_observation_counts": np.array([2], dtype=np.int32),
        "cluster_ocr_texts": np.array(["repere s 14"], dtype="<U512"),
        "cluster_ocr_tokens": np.array(["repere s 14"], dtype="<U512"),
        "cluster_ocr_keys": np.array(["letters=s;numbers=14"], dtype="<U256"),
        "cluster_ocr_observation_counts": np.array([1], dtype=np.int32),
        "cluster_ocr_source": np.array([2], dtype=np.int16),
    }
    save_index_to_db(tmp_path / OBJECT_SEARCH_INDEX_DB_FILENAME, metadata, state)

    loaded = load_index(tmp_path)

    assert loaded.object_textness_scores is not None
    assert loaded.object_textness_scores.tolist() == [
        0.20000000298023224,
        0.4000000059604645,
    ]
    assert loaded.object_ocr_texts is not None
    assert loaded.object_ocr_texts.tolist() == ["repere s 14", ""]
    assert loaded.object_ocr_tokens is not None
    assert loaded.object_ocr_tokens.tolist() == ["repere s 14", ""]
    assert loaded.object_ocr_keys is not None
    assert loaded.object_ocr_keys.tolist() == ["letters=s;numbers=14", ""]
    assert loaded.object_ocr_candidate_mask is not None
    assert loaded.object_ocr_candidate_mask.tolist() == [True, False]
    assert loaded.object_ocr_assigned_mask is not None
    assert loaded.object_ocr_assigned_mask.tolist() == [True, False]
    assert loaded.object_ocr_source is not None
    assert loaded.object_ocr_source.tolist() == [2, 0]
    assert loaded.object_detection_levels is not None
    assert loaded.object_detection_levels.tolist() == [1, 1]
    assert loaded.cluster_cutout_cluster_ids is not None
    assert loaded.cluster_cutout_cluster_ids.tolist() == [0]
    assert loaded.cluster_cutout_ids is not None
    assert loaded.cluster_cutout_ids.tolist() == [777]
    assert loaded.cluster_cutout_keyframe_ids is not None
    assert loaded.cluster_cutout_keyframe_ids.tolist() == [42]
    assert loaded.cluster_cutout_levels is not None
    assert loaded.cluster_cutout_levels.tolist() == [1]
    assert loaded.cluster_cutout_observation_counts is not None
    assert loaded.cluster_cutout_observation_counts.tolist() == [2]
    assert loaded.cluster_ocr_texts is not None
    assert loaded.cluster_ocr_texts.tolist() == ["repere s 14"]
    assert loaded.cluster_ocr_tokens is not None
    assert loaded.cluster_ocr_tokens.tolist() == ["repere s 14"]
    assert loaded.cluster_ocr_keys is not None
    assert loaded.cluster_ocr_keys.tolist() == ["letters=s;numbers=14"]
    assert loaded.cluster_ocr_observation_counts is not None
    assert loaded.cluster_ocr_observation_counts.tolist() == [1]
    assert loaded.cluster_ocr_source is not None
    assert loaded.cluster_ocr_source.tolist() == [2]


def test_params_key_migration_legacy_manifest_to_index_metadata(tmp_path: Path) -> None:
    """Legacy DBs with only manifest_json are upgraded on load to
    index_metadata_json."""
    db_path = tmp_path / OBJECT_SEARCH_INDEX_DB_FILENAME
    meta = ObjectSearchIndexMetadata(
        projection_dim=2,
        created_utc=default_created_utc(),
        cutout_count=0,
    )
    conn = sqlite3.connect(str(db_path))
    try:
        create_object_search_index_tables(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
            (LEGACY_MANIFEST_PARAM_KEY, meta.to_json().encode("utf-8")),
        )
        conn.commit()
    finally:
        conn.close()

    loaded = load_index(tmp_path)
    assert loaded.metadata.projection_dim == 2

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM params WHERE key = ?", (LEGACY_MANIFEST_PARAM_KEY,))
        assert cur.fetchone() is None
        cur.execute(
            "SELECT value FROM params WHERE key = ?", (INDEX_METADATA_PARAM_KEY,)
        )
        row = cur.fetchone()
        assert row is not None
        assert (
            ObjectSearchIndexMetadata.from_json(row[0].decode("utf-8")).projection_dim
            == 2
        )
    finally:
        conn.close()
