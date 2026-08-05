from __future__ import annotations

import numpy as np

from pipeline.offline.localize.clustering_online import (
    cluster_detections_leader_canopy,
    select_localization_candidates,
)


def _legacy_top_candidate_indices(
    similarities: np.ndarray, candidate_count: int
) -> np.ndarray:
    count = min(max(int(candidate_count), 1), similarities.shape[0])
    top = np.argpartition(-similarities, count - 1)[:count]
    return np.asarray(top[np.argsort(-similarities[top])], dtype=np.int64)


def test_select_localization_candidates_dedup_same_cutout():
    similarities = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cutout_ids = np.array([1, 1, 1], dtype=np.int64)
    bboxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )

    selected = select_localization_candidates(
        similarities,
        cutout_ids,
        bboxes,
        candidate_count=10,
        face_dedup_iou=0.5,
    )

    assert selected.tolist() == [0, 2]


def test_select_localization_candidates_preserves_distinct_cutouts():
    similarities = np.array([0.9, 0.85], dtype=np.float32)
    cutout_ids = np.array([1, 2], dtype=np.int64)
    keyframe_ids = np.array([10, 10], dtype=np.int64)
    bboxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    del keyframe_ids  # same keyframe, different cutouts

    selected = select_localization_candidates(
        similarities,
        cutout_ids,
        bboxes,
        candidate_count=10,
        face_dedup_iou=0.5,
    )

    assert selected.tolist() == [0, 1]


def test_dedup_before_top_k_frees_budget():
    n_dup = 50
    similarities = np.zeros(n_dup + 1, dtype=np.float32)
    similarities[:n_dup] = 1.0 - np.arange(n_dup, dtype=np.float32) * 0.001
    similarities[n_dup] = 0.5

    cutout_ids = np.zeros(n_dup + 1, dtype=np.int64)
    cutout_ids[n_dup] = 1
    bboxes = np.zeros((n_dup + 1, 4), dtype=np.float32)
    bboxes[:n_dup] = [0.0, 0.0, 10.0, 10.0]
    bboxes[n_dup] = [0.0, 0.0, 10.0, 10.0]

    selected = select_localization_candidates(
        similarities,
        cutout_ids,
        bboxes,
        candidate_count=10,
        face_dedup_iou=0.5,
    )

    assert selected.tolist()[0] == 0
    assert int(n_dup) in selected.tolist()
    assert len(selected) <= 10

    legacy_topk = _legacy_top_candidate_indices(similarities, candidate_count=10)
    assert int(n_dup) not in legacy_topk.tolist()
    assert legacy_topk.size == 10


def test_leader_canopy_anti_chain():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones(3, dtype=bool)
    keyframes = np.array([10, 11, 12], dtype=np.int64)
    similarities = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    cluster_ids = cluster_detections_leader_canopy(
        positions,
        valid_mask,
        keyframes,
        similarities,
        eps_meters=1.4,
        min_keyframes_per_cluster=1,
    )

    assert len(np.unique(cluster_ids[cluster_ids >= 0])) == 3


def test_leader_canopy_merges_nearby_views():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones(2, dtype=bool)
    keyframes = np.array([10, 11], dtype=np.int64)
    similarities = np.array([0.9, 0.7], dtype=np.float32)

    cluster_ids = cluster_detections_leader_canopy(
        positions,
        valid_mask,
        keyframes,
        similarities,
        eps_meters=2.0,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [0, 0]


def test_leader_canopy_seed_order():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones(2, dtype=bool)
    keyframes = np.array([10, 11], dtype=np.int64)
    similarities = np.array([0.5, 0.9], dtype=np.float32)

    cluster_ids = cluster_detections_leader_canopy(
        positions,
        valid_mask,
        keyframes,
        similarities,
        eps_meters=2.0,
        min_keyframes_per_cluster=2,
    )

    assert cluster_ids.tolist() == [0, 0]
