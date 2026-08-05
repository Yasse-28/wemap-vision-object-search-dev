from __future__ import annotations

import sys
import types

import numpy as np

if "spatialmath" not in sys.modules:
    spatialmath_stub = types.ModuleType("spatialmath")

    class _SE3:
        @staticmethod
        def Rt(*args, **kwargs):
            raise NotImplementedError("SE3.Rt is not used in these tests")

    spatialmath_stub.SE3 = _SE3
    sys.modules["spatialmath"] = spatialmath_stub

if "zarr" not in sys.modules:
    zarr_stub = types.ModuleType("zarr")

    def _open_group(*args, **kwargs):
        raise NotImplementedError("zarr.open_group is not used in these tests")

    zarr_stub.open_group = _open_group
    sys.modules["zarr"] = zarr_stub

if "pyproj" not in sys.modules:
    pyproj_stub = types.ModuleType("pyproj")

    class _Transformer:
        @staticmethod
        def from_crs(*args, **kwargs):
            return _Transformer()

        def transform(self, xx, yy, zz=None, radians=False):
            if zz is None:
                zz = 0.0
            return xx, yy, zz

    pyproj_stub.Transformer = _Transformer
    sys.modules["pyproj"] = pyproj_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_CUBIC = 0
    cv2_stub.BORDER_WRAP = 0

    def _remap(image, map_x, map_y, interpolation, borderMode=None):
        raise NotImplementedError("cv2.remap is not used in these tests")

    cv2_stub.remap = _remap
    sys.modules["cv2"] = cv2_stub

from pipeline.offline.localize.localize_3d import (
    cluster_detections_3d,
    compute_cluster_statistics,
)


def _embed(*rows: tuple[float, ...]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_joint_clustering_keeps_semantically_similar_spatial_neighbors():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([True, True], dtype=bool)
    keyframes = np.array([10, 11], dtype=np.int64)
    embeddings = _embed((1.0, 0.0), (0.95, 0.05))

    cluster_ids = cluster_detections_3d(
        positions,
        valid_mask,
        keyframes,
        eps_meters=1.0,
        embeddings=embeddings,
        embedding_similarity_threshold=0.85,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [0, 0]


def test_joint_clustering_rejects_same_keyframe_cluster():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([True, True], dtype=bool)
    keyframes = np.array([10, 10], dtype=np.int64)
    embeddings = _embed((1.0, 0.0), (0.95, 0.05))

    cluster_ids = cluster_detections_3d(
        positions,
        valid_mask,
        keyframes,
        eps_meters=1.0,
        embeddings=embeddings,
        embedding_similarity_threshold=0.85,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [-1, -1]


def test_joint_clustering_splits_low_similarity_neighbors():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([True, True], dtype=bool)
    keyframes = np.array([10, 11], dtype=np.int64)
    embeddings = _embed((1.0, 0.0), (0.0, 1.0))

    cluster_ids = cluster_detections_3d(
        positions,
        valid_mask,
        keyframes,
        eps_meters=1.0,
        embeddings=embeddings,
        embedding_similarity_threshold=0.85,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [-1, -1]


def test_joint_clustering_does_not_connect_distant_neighbors():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([True, True], dtype=bool)
    keyframes = np.array([10, 11], dtype=np.int64)
    embeddings = _embed((1.0, 0.0), (0.95, 0.05))

    cluster_ids = cluster_detections_3d(
        positions,
        valid_mask,
        keyframes,
        eps_meters=1.0,
        embeddings=embeddings,
        embedding_similarity_threshold=0.85,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [-1, -1]


def test_joint_clustering_keeps_invalid_detections_out_and_relabels_compact():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.4, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([True, True, True, True, False], dtype=bool)
    keyframes = np.array([10, 11, 20, 21, 99], dtype=np.int64)
    embeddings = _embed(
        (1.0, 0.0),
        (0.95, 0.05),
        (0.0, 1.0),
        (0.05, 0.95),
        (0.5, 0.5),
    )

    cluster_ids = cluster_detections_3d(
        positions,
        valid_mask,
        keyframes,
        eps_meters=1.0,
        embeddings=embeddings,
        embedding_similarity_threshold=0.85,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [0, 0, 1, 1, -1]


def test_compute_cluster_statistics_uses_detection_levels() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cluster_ids = np.array([0, 0, 1], dtype=np.int32)
    detection_levels = np.array([2, 2, 4], dtype=np.int32)

    _, _, obs_counts, _, cluster_levels = compute_cluster_statistics(
        positions,
        cluster_ids,
        georef=None,
        detection_levels=detection_levels,
    )

    assert obs_counts.tolist() == [2, 1]
    assert cluster_levels.tolist() == [2, 4]
