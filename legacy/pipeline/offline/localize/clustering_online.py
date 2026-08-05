"""Online-only candidate selection and 3D clustering for request-time localization."""

from __future__ import annotations

import numpy as np

from pipeline.core.types import UNRESOLVED_LEVEL_SENTINEL

Bbox = tuple[float, float, float, float]


def _bbox_iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter
    )
    return inter / union if union > 0.0 else 0.0


def select_localization_candidates(
    similarities: np.ndarray,
    cutout_ids: np.ndarray,
    bboxes: np.ndarray,
    *,
    candidate_count: int,
    face_dedup_iou: float = 0.5,
) -> np.ndarray:
    """Return object indices: global similarity order, per-cutout IoU dedup,
    then top-K."""
    n = int(similarities.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)

    count = min(max(int(candidate_count), 1), n)
    if face_dedup_iou <= 0:
        order = np.argsort(-similarities)
        return np.asarray(order[:count], dtype=np.int64)

    order = np.argsort(-similarities)
    kept: list[int] = []
    kept_boxes_by_cutout: dict[int, list[Bbox]] = {}

    for idx in order:
        cutout = int(cutout_ids[idx])
        bbox: Bbox = (
            float(bboxes[idx, 0]),
            float(bboxes[idx, 1]),
            float(bboxes[idx, 2]),
            float(bboxes[idx, 3]),
        )
        prior = kept_boxes_by_cutout.get(cutout, [])
        if any(_bbox_iou(bbox, other) > face_dedup_iou for other in prior):
            continue
        kept.append(int(idx))
        prior.append(bbox)
        kept_boxes_by_cutout[cutout] = prior
        if len(kept) >= count:
            break

    kept_arr = np.asarray(kept, dtype=np.int64)
    return kept_arr[: min(count, kept_arr.size)]


def _relabel_clusters_compact(cluster_ids: np.ndarray) -> np.ndarray:
    relabeled = np.full(cluster_ids.shape, -1, dtype=np.int32)
    unique_labels = np.unique(cluster_ids)
    unique_labels = unique_labels[unique_labels >= 0]
    for new_label, old_label in enumerate(unique_labels.tolist()):
        relabeled[cluster_ids == old_label] = new_label
    return relabeled


def filter_clusters_by_min_keyframes(
    cluster_ids: np.ndarray,
    keyframe_ids: np.ndarray,
    *,
    min_keyframes: int = 3,
) -> np.ndarray:
    if min_keyframes <= 1:
        return _relabel_clusters_compact(cluster_ids)

    filtered = cluster_ids.copy()
    for cluster_id in np.unique(cluster_ids):
        if cluster_id < 0:
            continue
        cluster_keyframes = np.unique(keyframe_ids[cluster_ids == cluster_id])
        if cluster_keyframes.size < min_keyframes:
            filtered[cluster_ids == cluster_id] = -1
    return _relabel_clusters_compact(filtered)


def _levels_compatible(
    detection_levels: np.ndarray | None,
    local_a: int,
    local_b: int,
    valid_indices: np.ndarray,
) -> bool:
    if detection_levels is None:
        return True
    level_a = int(detection_levels[valid_indices[local_a]])
    level_b = int(detection_levels[valid_indices[local_b]])
    if (
        level_a != UNRESOLVED_LEVEL_SENTINEL
        and level_b != UNRESOLVED_LEVEL_SENTINEL
        and level_a != level_b
    ):
        return False
    return True


def cluster_detections_leader_canopy(
    positions_local: np.ndarray,
    valid_mask: np.ndarray,
    object_keyframe_ids: np.ndarray,
    query_similarities: np.ndarray,
    *,
    eps_meters: float,
    min_keyframes_per_cluster: int = 2,
    detection_levels: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy similarity-seeded spatial clustering (leader / canopy).

    Each cluster is seeded by the highest query-similarity unassigned detection.
    Members are unassigned detections within ``eps_meters`` of the seed (same level).
    No transitive chaining through non-seed members.

    Uses a cKDTree radius query per seed so the inner loop is O(log n + k)
    per seed (k = neighbourhood size) rather than O(n) brute-force scan.
    """
    from scipy.spatial import cKDTree

    labels = np.full(len(positions_local), -1, dtype=np.int32)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return labels

    similarities = np.asarray(query_similarities, dtype=np.float64).reshape(-1)
    if similarities.shape[0] != len(positions_local):
        raise ValueError(
            "query_similarities must have one value per detection row "
            f"(got {similarities.shape[0]}, expected {len(positions_local)})"
        )

    valid_positions = np.asarray(positions_local[valid_indices], dtype=np.float64)
    tree = cKDTree(valid_positions)

    assigned = np.zeros(valid_indices.size, dtype=bool)
    order = np.argsort(-similarities[valid_indices])
    next_label = 0

    for seed_local in order:
        if assigned[seed_local]:
            continue
        neighbors = tree.query_ball_point(
            valid_positions[seed_local], r=float(eps_meters)
        )
        for j in neighbors:
            if assigned[j]:
                continue
            if not _levels_compatible(
                detection_levels, seed_local, int(j), valid_indices
            ):
                continue
            assigned[j] = True
            labels[int(valid_indices[j])] = next_label
        next_label += 1

    return filter_clusters_by_min_keyframes(
        labels,
        object_keyframe_ids,
        min_keyframes=min_keyframes_per_cluster,
    )


def cluster_detections_for_online(
    positions_local: np.ndarray,
    valid_mask: np.ndarray,
    object_keyframe_ids: np.ndarray,
    query_similarities: np.ndarray,
    *,
    clustering_method: str,
    eps_meters: float,
    min_keyframes_per_cluster: int,
    detection_levels: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    embedding_similarity_threshold: float = 0.85,
) -> np.ndarray:
    """Dispatch online clustering: leader_canopy (default) or single_linkage (A/B)."""
    if clustering_method == "single_linkage":
        from pipeline.offline.localize.localize_3d import cluster_detections_3d

        return cluster_detections_3d(
            positions_local,
            valid_mask,
            object_keyframe_ids,
            eps_meters=eps_meters,
            embeddings=embeddings,
            embedding_similarity_threshold=embedding_similarity_threshold,
            min_keyframes_per_cluster=min_keyframes_per_cluster,
            detection_levels=detection_levels,
        )
    return cluster_detections_leader_canopy(
        positions_local,
        valid_mask,
        object_keyframe_ids,
        query_similarities,
        eps_meters=eps_meters,
        min_keyframes_per_cluster=min_keyframes_per_cluster,
        detection_levels=detection_levels,
    )
