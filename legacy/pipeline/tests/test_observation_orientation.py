import sqlite3
from pathlib import Path

import numpy as np

from pipeline.core.types import LoadedIndex, ObjectSearchIndexMetadata
from pipeline.online.observation_orientation import observation_orientation_quaternions
from pipeline.online.request_models import ObjectLocation, ObjectObservation


def _write_georef_pose(db_path: Path, keyframe_id: int, pose: np.ndarray) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB NOT NULL)"
        )
        conn.execute(
            "INSERT INTO GeoRefKeyframe (id, pose) VALUES (?, ?)",
            (
                keyframe_id,
                np.asarray(pose, dtype=np.dtype(float).newbyteorder("<")).T.tobytes(),
            ),
        )


def test_observation_orientation_quaternion_uses_wemap_tools_eus_wxyz_convention(
    tmp_path: Path,
):
    _write_georef_pose(tmp_path / "georef.db", 7, np.eye(4, dtype=np.float64))
    index = LoadedIndex(
        metadata=ObjectSearchIndexMetadata(),
        cutout_embeddings=np.zeros((1, 2), dtype=np.float32),
        cutout_ids=np.asarray([70], dtype=np.int64),
        cutout_keyframe_ids=np.asarray([7], dtype=np.int64),
        cutout_center_xy=np.zeros((1, 2), dtype=np.float32),
        cutout_rotation_cutout_to_equirect=np.eye(4, dtype=np.float32)[None, :, :],
        object_embeddings=np.zeros((1, 2), dtype=np.float32),
        object_keyframe_ids=np.asarray([7], dtype=np.int64),
        object_cutout_ids=np.asarray([70], dtype=np.int64),
        object_bboxes=np.zeros((1, 4), dtype=np.float32),
    )

    result = observation_orientation_quaternions(
        index=index,
        map_path=tmp_path,
        object_indices=[0],
    )

    assert set(result) == {0}
    quat = np.asarray(result[0], dtype=np.float64)
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)
    np.testing.assert_allclose(quat, [0.0, 0.0, 1.0, 0.0], atol=1e-6)


def test_observation_orientation_returns_empty_when_georef_missing(tmp_path: Path):
    index = LoadedIndex(
        metadata=ObjectSearchIndexMetadata(),
        cutout_embeddings=np.zeros((1, 2), dtype=np.float32),
        cutout_ids=np.asarray([70], dtype=np.int64),
        cutout_keyframe_ids=np.asarray([7], dtype=np.int64),
        cutout_center_xy=np.zeros((1, 2), dtype=np.float32),
        cutout_rotation_cutout_to_equirect=np.eye(4, dtype=np.float32)[None, :, :],
        object_embeddings=np.zeros((1, 2), dtype=np.float32),
        object_keyframe_ids=np.asarray([7], dtype=np.int64),
        object_cutout_ids=np.asarray([70], dtype=np.int64),
        object_bboxes=np.zeros((1, 4), dtype=np.float32),
    )

    result = observation_orientation_quaternions(
        index=index,
        map_path=tmp_path,
        object_indices=[0],
    )

    assert result == {}


def test_object_observation_serializes_quaternion_field():
    observation = ObjectObservation(
        object_idx=1900,
        cutout_id="1192000",
        keyframe_id="1192",
        coordinates=(45.0, 6.0, 1.0),
        bbox=(0.0, 1.0, 2.0, 3.0),
        similarity_score=0.9,
        quaternion=[1.0, 0.0, 0.0, 0.0],
        heading=0.25,
    )

    payload = observation.model_dump()
    assert payload["coordinates"] == (45.0, 6.0, 1.0)
    assert payload["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert payload["heading"] == 0.25
    assert "orientation_quaternion" not in payload


def test_object_location_serializes_coordinates_field():
    location = ObjectLocation(
        coordinates=(45.0, 6.0, 1.0),
        confidence=0.8,
        observation_count=1,
        similarity_score=0.9,
        match_score=0.95,
        keyframe_ids=["1192"],
        observations=[],
    )

    payload = location.model_dump()
    assert payload["coordinates"] == (45.0, 6.0, 1.0)
    assert "lat" not in payload
    assert "lng" not in payload
    assert "alt" not in payload
