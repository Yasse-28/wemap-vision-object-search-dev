from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.core.logging import logger
from pipeline.core.types import LoadedIndex
from pipeline.online.observation_coordinates import online_observation_coordinates
from pipeline.online.observation_orientation import observation_orientation_quaternions
from pipeline.online.ocr_scoring import best_ocr_score, extract_ocr_query
from pipeline.online.request_models import ObjectLocation, ObjectObservation

if TYPE_CHECKING:
    from pipeline.offline.localize.localize_3d import LocalizationResult


@dataclass(frozen=True)
class OnlineLocalizationParams:
    depth_dir: str = "depths"
    candidate_count: int = 1000
    max_observations_per_cluster: int = 10
    clustering_eps_m: float = 2.0
    min_depth_m: float = 0.5
    max_depth_m: float = 50.0
    embedding_similarity_threshold: float = 0.85
    min_similarity: float = 0.2
    min_keyframes_per_cluster: int = 2
    face_dedup_iou: float = 0.5
    clustering_method: str = "leader_canopy"
    use_stored_positions: bool = True
    """If True and the index contains pre-computed per-object ENU positions,
    skip depth-map loading and 2D→3D projection and re-cluster the stored
    positions with the request-time parameters. Falls back to the depth-based
    path if stored positions are unavailable."""
    robust_centroid: bool = False
    """If True, replace each cluster centroid with the geometric (L1) median
    of its member positions. Robust to ~50% outliers; recommended when
    clusters span a few well-positioned observations and a few off-position
    ones (depth contamination, transient detections, etc.)."""


@dataclass(frozen=True)
class ClusterRanking:
    cluster_id: int
    similarity_score: float
    normalized_similarity: float
    match_score: float
    ocr_score: float = 0.0


class OnlineLocalizationError(RuntimeError):
    """Raised when online localization cannot be computed for an index/map."""


def _top_candidate_indices(
    similarities: np.ndarray, candidate_count: int
) -> np.ndarray:
    """Legacy top-K without per-cutout dedup; kept for tests and comparisons."""
    if similarities.size == 0:
        return np.array([], dtype=np.int64)

    count = min(max(int(candidate_count), 1), similarities.shape[0])
    if count == similarities.shape[0]:
        order = np.argsort(-similarities)
    else:
        top = np.argpartition(-similarities, count - 1)[:count]
        order = top[np.argsort(-similarities[top])]
    return np.asarray(order, dtype=np.int64)


def _select_localization_candidate_indices(
    index: LoadedIndex,
    similarities: np.ndarray,
    params: OnlineLocalizationParams,
) -> np.ndarray:
    from pipeline.offline.localize.clustering_online import (
        select_localization_candidates,
    )

    return select_localization_candidates(
        similarities,
        index.object_cutout_ids,
        index.object_bboxes,
        candidate_count=params.candidate_count,
        face_dedup_iou=params.face_dedup_iou,
    )


def _relative_similarity_scores(
    cluster_best_sim: dict[int, float],
    *,
    min_similarity: float,
) -> dict[int, float]:
    if not cluster_best_sim:
        return {}

    values = np.asarray(list(cluster_best_sim.values()), dtype=np.float32)
    best = float(np.max(values))
    denom = best - float(min_similarity)
    if denom <= 1e-6:
        return {cluster_id: 1.0 for cluster_id in cluster_best_sim}

    return {
        cluster_id: float(
            np.clip((float(sim) - float(min_similarity)) / denom, 0.0, 1.0)
        )
        for cluster_id, sim in cluster_best_sim.items()
    }


def rank_localization_clusters(
    *,
    cluster_best_sim: dict[int, float],
    cluster_confidence: dict[int, float],
    cluster_keyframes: dict[int, set[str]],
    cluster_ocr_scores: dict[int, float] | None = None,
    min_similarity: float = 0.2,
) -> list[ClusterRanking]:
    """Filter low-similarity clusters and rank the rest with query + spatial support."""
    eligible = {
        int(cluster_id): float(sim)
        for cluster_id, sim in cluster_best_sim.items()
        if float(sim) >= float(min_similarity)
    }
    relative_scores = _relative_similarity_scores(
        eligible,
        min_similarity=float(min_similarity),
    )

    rankings: list[ClusterRanking] = []
    for cluster_id, sim in eligible.items():
        confidence = float(cluster_confidence.get(cluster_id, 0.0))
        confidence = float(
            np.clip(confidence if np.isfinite(confidence) else 0.0, 0.0, 1.0)
        )
        keyframe_count = len(cluster_keyframes.get(cluster_id, set()))
        keyframe_score = min(1.0, max(0, keyframe_count) / 3.0)
        normalized_similarity = relative_scores.get(cluster_id, 0.0)
        ocr_score = 0.0
        if cluster_ocr_scores is not None:
            ocr_score = float(cluster_ocr_scores.get(cluster_id, 0.0))
            ocr_score = float(
                np.clip(ocr_score if np.isfinite(ocr_score) else 0.0, 0.0, 1.0)
            )
            match_score = (
                0.50 * normalized_similarity
                + 0.10 * confidence
                + 0.20 * keyframe_score
                + 0.20 * ocr_score
            )
        else:
            match_score = (
                0.50 * normalized_similarity + 0.15 * confidence + 0.35 * keyframe_score
            )
        rankings.append(
            ClusterRanking(
                cluster_id=cluster_id,
                similarity_score=sim,
                normalized_similarity=normalized_similarity,
                match_score=float(match_score),
                ocr_score=ocr_score,
            )
        )

    rankings.sort(key=lambda r: (r.match_score, r.similarity_score), reverse=True)
    return rankings


def _safe_str_array_value(values: np.ndarray | None, idx: int) -> str:
    if values is None or idx < 0 or idx >= values.shape[0]:
        return ""
    return str(values[idx])


def cluster_ocr_scores_from_summaries(
    *,
    query_text: str,
    cluster_ids: set[int],
    cluster_ocr_keys: np.ndarray | None,
    cluster_ocr_tokens: np.ndarray | None,
    cluster_ocr_texts: np.ndarray | None,
) -> dict[int, float] | None:
    query = extract_ocr_query(query_text)
    if not query.has_identity:
        return None
    if (
        cluster_ocr_keys is None
        and cluster_ocr_tokens is None
        and cluster_ocr_texts is None
    ):
        return None

    return {
        cluster_id: best_ocr_score(
            query,
            [
                (
                    _safe_str_array_value(cluster_ocr_keys, cluster_id),
                    _safe_str_array_value(cluster_ocr_tokens, cluster_id),
                    _safe_str_array_value(cluster_ocr_texts, cluster_id),
                )
            ],
        )
        for cluster_id in cluster_ids
    }


def cluster_ocr_scores_from_observations(
    *,
    query_text: str,
    cluster_observations: dict[int, list[tuple[int, float]]],
    object_ocr_keys: np.ndarray | None,
    object_ocr_tokens: np.ndarray | None,
    object_ocr_texts: np.ndarray | None,
) -> dict[int, float] | None:
    query = extract_ocr_query(query_text)
    if not query.has_identity:
        return None
    if (
        object_ocr_keys is None
        and object_ocr_tokens is None
        and object_ocr_texts is None
    ):
        return None

    scores: dict[int, float] = {}
    for cluster_id, observations in cluster_observations.items():
        candidates = [
            (
                _safe_str_array_value(object_ocr_keys, obj_idx),
                _safe_str_array_value(object_ocr_tokens, obj_idx),
                _safe_str_array_value(object_ocr_texts, obj_idx),
            )
            for obj_idx, _ in observations
        ]
        scores[cluster_id] = best_ocr_score(query, candidates)
    return scores


def geometric_median(
    points: np.ndarray,
    *,
    max_iter: int = 50,
    eps: float = 1e-6,
) -> np.ndarray:
    """Weiszfeld iteration for the L1 (geometric) median of a point set.

    Outlier-robust up to ~50% breakdown point — single far-away observations
    do not pull the centroid the way an arithmetic mean does. Falls back to
    the coordinate-wise median as the initial guess and after any degenerate
    iteration step.
    """
    if points.shape[0] == 0:
        return np.zeros(points.shape[1], dtype=np.float64)
    if points.shape[0] == 1:
        return np.asarray(points[0].astype(np.float64))

    pts = points.astype(np.float64)
    y = np.median(pts, axis=0)
    for _ in range(max_iter):
        diffs = pts - y
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists > eps
        if not mask.any():
            return np.asarray(y)
        weights = 1.0 / dists[mask]
        next_y = (pts[mask] * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(next_y - y) < eps:
            return np.asarray(next_y)
        y = next_y
    return np.asarray(y)


def _apply_robust_centroid(
    *,
    localization: "LocalizationResult",
    map_path: Path,
) -> None:
    """Replace each cluster centroid with the geometric median of its members.

    Mutates ``localization.cluster_centroids_world`` and
    ``cluster_centroids_geo`` in place. Confidence and observation counts are
    left untouched (they are computed from the full cluster, not the centroid).
    Outlier members are *not* dropped — they still contribute to the cluster's
    observation_count and confidence score; the robust centroid only changes
    *where* the cluster is reported on the map.
    """
    from pipeline.offline.localize.georef import load_georef_from_db
    from pipeline.offline.localize.localize_3d import world_to_geo_with_level

    cluster_ids = localization.cluster_ids
    positions_local = localization.positions_local
    centroids_world = localization.cluster_centroids_world
    centroids_geo = localization.cluster_centroids_geo
    if centroids_world is None or centroids_world.shape[0] == 0:
        return

    georef_path = map_path / "georef.db"
    georef = load_georef_from_db(georef_path) if georef_path.exists() else None

    valid_mask = getattr(localization, "valid_mask", None)
    n_clusters = centroids_world.shape[0]
    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        if valid_mask is not None:
            mask = mask & valid_mask
        member_positions = positions_local[mask]
        # Reject NaN rows (a depth-reprojection failure leaves NaN).
        member_positions = member_positions[np.isfinite(member_positions).all(axis=1)]
        if member_positions.shape[0] == 0:
            continue
        robust = geometric_median(member_positions)
        centroids_world[cluster_id] = robust.astype(centroids_world.dtype)
        if georef is not None:
            try:
                lat, lon, alt, _ = world_to_geo_with_level(robust, georef)
                centroids_geo[cluster_id] = np.array(
                    [lat, lon, alt], dtype=centroids_geo.dtype
                )
            except Exception:
                # Leave the original geo fallback in place if conversion fails.
                continue


def _apply_online_clustering_to_depth_localization(
    *,
    localization: "LocalizationResult",
    candidate_indices: np.ndarray,
    subset_similarities: np.ndarray,
    index: LoadedIndex,
    map_path: Path,
    params: OnlineLocalizationParams,
) -> "LocalizationResult":
    """Cluster depth-projected positions with online method and fill cluster stats."""
    from pipeline.offline.localize.clustering_online import (
        cluster_detections_for_online,
    )
    from pipeline.offline.localize.georef import load_georef_from_db
    from pipeline.offline.localize.localize_3d import compute_cluster_statistics

    keyframe_ids = np.asarray(index.object_keyframe_ids[candidate_indices])
    detection_levels = localization.detection_levels

    cluster_ids = cluster_detections_for_online(
        localization.positions_local,
        localization.valid_mask,
        keyframe_ids,
        subset_similarities,
        clustering_method=params.clustering_method,
        eps_meters=params.clustering_eps_m,
        min_keyframes_per_cluster=params.min_keyframes_per_cluster,
        detection_levels=detection_levels,
        embeddings=(
            np.asarray(index.object_embeddings[candidate_indices], dtype=np.float32)
            if params.clustering_method == "single_linkage"
            else None
        ),
        embedding_similarity_threshold=params.embedding_similarity_threshold,
    )

    georef_path = map_path / "georef.db"
    georef = load_georef_from_db(georef_path) if georef_path.exists() else None

    (
        centroids_world,
        centroids_geo,
        observation_counts,
        confidence_scores,
        cluster_levels,
    ) = compute_cluster_statistics(
        positions_local=localization.positions_local,
        cluster_ids=cluster_ids,
        georef=georef,
        detection_confidences=subset_similarities,
        detection_levels=detection_levels,
        detection_keyframe_ids=keyframe_ids,
    )

    from pipeline.offline.localize.localize_3d import LocalizationResult

    return LocalizationResult(
        positions_keyframe=localization.positions_keyframe,
        positions_local=localization.positions_local,
        depths=localization.depths,
        valid_mask=localization.valid_mask,
        cluster_ids=cluster_ids,
        cluster_centroids_world=centroids_world,
        cluster_centroids_geo=centroids_geo,
        cluster_observation_counts=observation_counts,
        cluster_confidence=confidence_scores,
        detection_levels=detection_levels,
        cluster_levels=cluster_levels,
    )


def _localize_from_stored_positions(
    *,
    index: LoadedIndex,
    map_path: Path,
    candidate_indices: np.ndarray,
    subset_similarities: np.ndarray,
    params: OnlineLocalizationParams,
) -> "LocalizationResult":
    """Re-cluster pre-computed object ENU positions with request-time params.

    Skips depth-map I/O and 2D→3D projection. Reuses the same clustering and
    centroid-stats helpers as the offline build, so any fix made there applies
    here verbatim.
    """
    from pipeline.offline.localize.clustering_online import (
        cluster_detections_for_online,
    )
    from pipeline.offline.localize.georef import (
        R_ENU_TO_WDS,
        R_WDS_TO_ENU,
        load_georef_from_db,
    )
    from pipeline.offline.localize.localize_3d import (
        LocalizationResult,
        compute_cluster_statistics,
    )

    # DB format detection:
    #   New DBs: position_keyframe = camera frame, position_local = ENU local metric,
    #     position_world = geographic [lat, lon, alt].
    #   Old DBs: position_local = camera-frame XYZ, position_world = WDS local metric.
    #
    # Clustering can run in ENU because it is metric. GeoRef conversion still expects
    # WDS local-map coordinates, so convert ENU -> WDS only at that boundary.
    _kf = index.object_positions_keyframe
    _is_new_db = _kf is not None and not np.all(np.isnan(_kf))
    if _is_new_db:
        positions_enu_all = np.asarray(index.object_positions_local, dtype=np.float32)
        positions_wds_all = (
            R_ENU_TO_WDS @ np.asarray(positions_enu_all, dtype=np.float64).T
        ).T.astype(np.float32)
    else:
        # Old DB fallback: position_world column holds WDS metric positions.
        # Convert WDS → ENU so clustering radius (in metres) still works.
        _pos_wds = index.object_positions_world
        if _pos_wds is None:
            positions_wds_all = np.full(
                (index.object_embeddings.shape[0], 3),
                np.nan,
                dtype=np.float32,
            )
            positions_enu_all = positions_wds_all.copy()
        else:
            positions_wds_all = np.asarray(_pos_wds, dtype=np.float32)
            positions_enu_all = (
                R_WDS_TO_ENU @ np.asarray(positions_wds_all, dtype=np.float64).T
            ).T.astype(np.float32)
    valid_all = np.asarray(index.object_localization_valid, dtype=bool)

    positions_enu = positions_enu_all[candidate_indices]
    positions_wds = positions_wds_all[candidate_indices]
    valid_mask = (
        valid_all[candidate_indices]
        & np.isfinite(positions_enu).all(axis=1)
        & np.isfinite(positions_wds).all(axis=1)
    )
    keyframe_ids = np.asarray(index.object_keyframe_ids[candidate_indices])

    detection_levels: np.ndarray | None = None
    if index.object_detection_levels is not None:
        detection_levels = np.asarray(
            index.object_detection_levels[candidate_indices], dtype=np.int32
        )

    cluster_ids = cluster_detections_for_online(
        positions_enu,
        valid_mask,
        keyframe_ids,
        subset_similarities,
        clustering_method=params.clustering_method,
        eps_meters=params.clustering_eps_m,
        min_keyframes_per_cluster=params.min_keyframes_per_cluster,
        detection_levels=detection_levels,
        embeddings=(
            np.asarray(index.object_embeddings[candidate_indices], dtype=np.float32)
            if params.clustering_method == "single_linkage"
            else None
        ),
        embedding_similarity_threshold=params.embedding_similarity_threshold,
    )

    georef_path = map_path / "georef.db"
    georef = load_georef_from_db(georef_path) if georef_path.exists() else None

    (
        centroids_world,
        centroids_geo,
        observation_counts,
        confidence_scores,
        cluster_levels,
    ) = compute_cluster_statistics(
        positions_local=positions_wds,
        cluster_ids=cluster_ids,
        georef=georef,
        detection_confidences=subset_similarities,
        detection_levels=detection_levels,
        detection_keyframe_ids=keyframe_ids,
    )

    n = positions_wds.shape[0]
    return LocalizationResult(
        positions_keyframe=np.full((n, 3), np.nan, dtype=np.float32),
        # LocalizationResult predates the DB rename and its positions_local field
        # is consumed by GeoRef helpers as WDS local-map coordinates.
        positions_local=positions_wds.astype(np.float32),
        depths=np.full((n,), np.nan, dtype=np.float32),
        valid_mask=valid_mask,
        cluster_ids=cluster_ids,
        cluster_centroids_world=centroids_world,
        cluster_centroids_geo=centroids_geo,
        cluster_observation_counts=observation_counts,
        cluster_confidence=confidence_scores,
        cluster_levels=cluster_levels,
        detection_levels=detection_levels,
    )


def _aggregate_candidate_clusters(
    *,
    localization: LocalizationResult,
    candidate_indices: np.ndarray,
    subset_similarities: np.ndarray,
    index: LoadedIndex,
) -> tuple[dict[int, float], dict[int, set[str]], dict[int, list[tuple[int, float]]]]:
    """Group valid candidate observations by cluster.

    Returns (best similarity, keyframe-id set, observations) per cluster id.
    """
    cluster_best_sim: dict[int, float] = {}
    cluster_keyframes: dict[int, set[str]] = {}
    cluster_observations: dict[int, list[tuple[int, float]]] = {}
    for local_idx, original_idx in enumerate(candidate_indices.tolist()):
        if not localization.valid_mask[local_idx]:
            continue
        cluster_id = int(localization.cluster_ids[local_idx])
        if cluster_id < 0:
            continue
        sim = float(subset_similarities[local_idx])
        if cluster_id not in cluster_best_sim or sim > cluster_best_sim[cluster_id]:
            cluster_best_sim[cluster_id] = sim
        cluster_keyframes.setdefault(cluster_id, set()).add(
            str(int(index.object_keyframe_ids[original_idx]))
        )
        cluster_observations.setdefault(cluster_id, []).append((original_idx, sim))
    return cluster_best_sim, cluster_keyframes, cluster_observations


def _resolve_response_coordinates(
    *,
    index: LoadedIndex,
    map_path: Path,
    localization: LocalizationResult,
    candidate_indices: np.ndarray,
    response_object_indices: list[int],
) -> dict[int, tuple[float, float, float]]:
    """Resolve lat/lon/alt per response object, preferring stored world positions.

    Stored coordinates avoid loading GeoRef per request; otherwise the request-
    time WDS positions are converted via georef.db.
    """
    if index.object_positions_world is not None:
        from pipeline.online.observation_coordinates import (
            stored_observation_coordinates,
        )

        return stored_observation_coordinates(
            index=index,
            map_path=map_path,
            object_indices=response_object_indices,
        )
    response_object_index_set = set(response_object_indices)
    object_position_wds = {
        int(original_idx): np.asarray(
            localization.positions_local[local_idx], dtype=np.float64
        )
        for local_idx, original_idx in enumerate(candidate_indices.tolist())
        if int(original_idx) in response_object_index_set
        and bool(localization.valid_mask[local_idx])
    }
    return online_observation_coordinates(
        map_path=map_path,
        object_position_wds=object_position_wds,
    )


def _build_object_locations(
    *,
    ranked_clusters: list[ClusterRanking],
    localization: LocalizationResult,
    cluster_observations: dict[int, list[tuple[int, float]]],
    cluster_keyframes: dict[int, set[str]],
    index: LoadedIndex,
    coordinates_by_object_idx: dict[int, tuple[float, float, float]],
    orientation_by_object_idx: dict[int, list[float]],
    params: OnlineLocalizationParams,
) -> list[ObjectLocation]:
    """Assemble an ObjectLocation response for each ranked cluster."""
    localizations: list[ObjectLocation] = []
    for ranked in ranked_clusters:
        cluster_id = ranked.cluster_id
        if cluster_id >= localization.cluster_centroids_geo.shape[0]:
            logger.warning("Skipping out-of-range online cluster id %d", cluster_id)
            continue

        geo = localization.cluster_centroids_geo[cluster_id]
        raw_observations = cluster_observations.get(cluster_id, [])
        raw_observations.sort(key=lambda x: x[1], reverse=True)
        observations: list[ObjectObservation] = []
        for obj_idx, obj_sim in raw_observations[: params.max_observations_per_cluster]:
            observations.append(
                ObjectObservation(
                    object_idx=obj_idx,
                    cutout_id=str(int(index.object_cutout_ids[obj_idx])),
                    keyframe_id=str(int(index.object_keyframe_ids[obj_idx])),
                    coordinates=coordinates_by_object_idx.get(obj_idx),
                    bbox=tuple(float(v) for v in index.object_bboxes[obj_idx].tolist()),
                    similarity_score=float(obj_sim),
                    quaternion=orientation_by_object_idx.get(obj_idx),
                )
            )

        cluster_level: int | None = None
        if localization.cluster_levels is not None and cluster_id < len(
            localization.cluster_levels
        ):
            lv = int(localization.cluster_levels[cluster_id])
            cluster_level = lv if lv >= 0 else None

        localizations.append(
            ObjectLocation(
                coordinates=(float(geo[0]), float(geo[1]), float(geo[2])),
                level=cluster_level,
                confidence=float(localization.cluster_confidence[cluster_id]),
                observation_count=int(
                    localization.cluster_observation_counts[cluster_id]
                ),
                similarity_score=float(ranked.similarity_score),
                match_score=float(ranked.match_score),
                keyframe_ids=sorted(cluster_keyframes.get(cluster_id, set())),
                observations=observations,
            )
        )
    return localizations


def localize_online_matches(
    *,
    index: LoadedIndex,
    map_path: Path,
    query_text: str,
    similarities: np.ndarray,
    params: OnlineLocalizationParams,
) -> list[ObjectLocation]:
    """Localize top matching object detections at request time.

    This keeps offline localization optional: the online path recomputes 3D positions
    for the current query's best object candidates using the map's depth/georef files.
    """
    metadata = index.metadata
    if metadata.source != "equirect360" or metadata.geometry != "cubemap":
        raise OnlineLocalizationError(
            "Online localization currently requires equirect360 cubemap map"
            " data (see cutouts / index geometry)"
        )
    if metadata.cubemap_face_size is None or metadata.cubemap_fov_deg is None:
        raise OnlineLocalizationError(
            "Object-search index metadata is missing cubemap geometry fields"
        )

    from pipeline.offline.localize.localize_3d import (
        can_localize_3d,
        localize_detections,
    )

    t0 = time.perf_counter()
    candidate_indices = _select_localization_candidate_indices(
        index,
        similarities,
        params,
    )
    t_candidates = time.perf_counter() - t0
    if candidate_indices.size == 0:
        return []

    subset_similarities = np.asarray(similarities[candidate_indices], dtype=np.float32)

    have_stored = (
        index.object_positions_local is not None
        and index.object_localization_valid is not None
    )
    t1 = time.perf_counter()
    if params.use_stored_positions and have_stored:
        localization = _localize_from_stored_positions(
            index=index,
            map_path=map_path,
            candidate_indices=candidate_indices,
            subset_similarities=subset_similarities,
            params=params,
        )
    else:
        can_localize, reason = can_localize_3d(map_path, params.depth_dir)
        if not can_localize:
            raise OnlineLocalizationError(reason)
        localization = localize_detections(
            object_keyframe_ids=index.object_keyframe_ids[candidate_indices],
            object_cutout_ids=index.object_cutout_ids[candidate_indices],
            object_bboxes=index.object_bboxes[candidate_indices],
            depth_dir=map_path / params.depth_dir,
            georef_db_path=map_path / "georef.db",
            face_size=int(metadata.cubemap_face_size),
            fov_deg=float(metadata.cubemap_fov_deg),
            id_stride=int(metadata.id_stride),
            min_depth_m=params.min_depth_m,
            max_depth_m=params.max_depth_m,
            skip_clustering=True,
        )
        localization = _apply_online_clustering_to_depth_localization(
            localization=localization,
            candidate_indices=candidate_indices,
            subset_similarities=subset_similarities,
            index=index,
            map_path=map_path,
            params=params,
        )
    t_clustering = time.perf_counter() - t1

    t2 = time.perf_counter()
    if params.robust_centroid:
        _apply_robust_centroid(localization=localization, map_path=map_path)
    t_robust = time.perf_counter() - t2

    cluster_best_sim, cluster_keyframes, cluster_observations = (
        _aggregate_candidate_clusters(
            localization=localization,
            candidate_indices=candidate_indices,
            subset_similarities=subset_similarities,
            index=index,
        )
    )

    cluster_confidence = {
        cluster_id: float(localization.cluster_confidence[cluster_id])
        for cluster_id in cluster_best_sim
        if cluster_id < localization.cluster_confidence.shape[0]
    }
    ranked_clusters = rank_localization_clusters(
        cluster_best_sim=cluster_best_sim,
        cluster_confidence=cluster_confidence,
        cluster_keyframes=cluster_keyframes,
        cluster_ocr_scores=cluster_ocr_scores_from_observations(
            query_text=query_text,
            cluster_observations=cluster_observations,
            object_ocr_keys=index.object_ocr_keys,
            object_ocr_tokens=index.object_ocr_tokens,
            object_ocr_texts=index.object_ocr_texts,
        ),
        min_similarity=params.min_similarity,
    )
    response_object_indices = [
        obj_idx
        for ranked in ranked_clusters
        for obj_idx, _ in sorted(
            cluster_observations.get(ranked.cluster_id, []),
            key=lambda x: x[1],
            reverse=True,
        )[: params.max_observations_per_cluster]
    ]
    t3 = time.perf_counter()
    orientation_by_object_idx = observation_orientation_quaternions(
        index=index,
        map_path=map_path,
        object_indices=response_object_indices,
    )
    t_orientation = time.perf_counter() - t3

    t4 = time.perf_counter()
    coordinates_by_object_idx = _resolve_response_coordinates(
        index=index,
        map_path=map_path,
        localization=localization,
        candidate_indices=candidate_indices,
        response_object_indices=response_object_indices,
    )
    t_coordinates = time.perf_counter() - t4

    logger.info(
        "localize_online_matches timing: candidates=%.0fms clustering=%.0fms"
        " robust=%.0fms orientation=%.0fms coordinates=%.0fms"
        " n_candidates=%d n_clusters=%d",
        t_candidates * 1000,
        t_clustering * 1000,
        t_robust * 1000,
        t_orientation * 1000,
        t_coordinates * 1000,
        int(candidate_indices.size),
        len(cluster_best_sim),
    )

    return _build_object_locations(
        ranked_clusters=ranked_clusters,
        localization=localization,
        cluster_observations=cluster_observations,
        cluster_keyframes=cluster_keyframes,
        index=index,
        coordinates_by_object_idx=coordinates_by_object_idx,
        orientation_by_object_idx=orientation_by_object_idx,
        params=params,
    )
