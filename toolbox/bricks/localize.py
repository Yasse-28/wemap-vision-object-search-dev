"""Online localization logic for object-search v1.5 (livemap wire format).

Ported from `backend/object_search/v1_5_logic.py`, which itself ports
leader_canopy clustering, cluster statistics and ranking from the standalone
service. This file has no Django dependency; its deliberate dev-only divergences are
documented in ``AI_CONTEXT/bricks.md``.

Operates on Postgres ``object_position`` values loaded via
``candidates.load_enriched_candidates``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from toolbox.bricks.candidates import EnrichedCandidate, FeedbackNormalization
from toolbox.bricks.vendored.candidate_orientation import candidate_orientation
from toolbox.bricks.vendored.geo_transform import GeoTransform, Pose
from toolbox.bricks.vendored.maths import quaternion

# A level value no georef will ever declare. It was -1, which collides with the real
# basement level on every map that has one — see gotcha 5 in AI_CONTEXT/bricks.md.
UNRESOLVED_LEVEL = -9999
PLACEHOLDER_BBOX = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class LocalizationParams:
    candidate_count: int = 1000
    num_results: int = 100
    min_similarity: float = 0.2
    max_observations_per_cluster: int = 10
    clustering_eps_m: float = 2.0
    min_keyframes_per_cluster: int = 2
    # Geometric support, as *filters* rather than score terms. A cluster failing one
    # is dropped outright instead of being scored lower: measured on bbhotel-choisy,
    # blending them into `match_score` cost ranking quality (see the module docstring
    # of `rank_localization_clusters`), while a filter leaves the score interpretable.
    # Both default to off, because the same measurement found no threshold that pays
    # for itself — they exist to be swept, not to be on.
    min_observations_per_cluster: int = 1
    max_cluster_spread_m: float | None = None
    # Two-gate association (ConceptGraphs): a detection joins a cluster only if it
    # passes the spatial gate *and* a cutout↔cutout cosine gate against the seed.
    # `None` = off, which is production's geometry-only rule. Requires candidates
    # loaded with `with_embeddings=True`; the service arranges that when it is set.
    semantic_gate_threshold: float | None = None
    # Review-feedback gains. Both zero (the default) means the boosted similarity
    # is not merely equal to the raw one — it is never consulted at all. See
    # `_ranking_similarities`.
    feedback_alpha: float = 0.0
    feedback_beta: float = 0.0
    # How the prototype columns are rescaled before the gains apply. Carried here
    # only so the response and the logs can report what was measured — the rescaling
    # itself happens in `candidates.load_enriched_candidates`, which is the only
    # place that sees the whole retrieved set at once.
    feedback_normalization: FeedbackNormalization = "none"
    # How a cluster picks the floor it claims: `"seed"` is production's behaviour and
    # stays the default, so this file's default path remains the ported one. `"median"`
    # is a dev-only experiment for objects whose observers straddle two floors; see
    # `_cluster_level_from_detections`. Any change of default belongs in
    # wemap-vision-backend first.
    level_strategy: str = "seed"

    @property
    def feedback_enabled(self) -> bool:
        return bool(self.feedback_alpha) or bool(self.feedback_beta)


@dataclass(frozen=True)
class ClusterRanking:
    cluster_id: int
    similarity_score: float
    match_score: float


@dataclass(frozen=True)
class ClusterStatistics:
    centroids_eus: np.ndarray
    centroids_lat: np.ndarray
    centroids_lng: np.ndarray
    centroids_alt: np.ndarray
    observation_counts: np.ndarray
    confidence_scores: np.ndarray
    cluster_levels: np.ndarray
    # Mean per-axis standard deviation of the cluster's member positions, in metres.
    # Folded into `confidence_scores` as well, but exposed on its own because it is
    # the only purely geometric quantity here and `max_cluster_spread_m` filters on it.
    spread_m: np.ndarray


def v1_5_observation_quaternion(
    pose: Pose, theta_center: float, phi_center: float
) -> list[float]:
    """EUS observation quaternion for livemap (kiosk applies +180° client-side)."""
    direction = candidate_orientation(float(theta_center), float(phi_center))
    composed = quaternion.multiply(pose.orientation_wxyz, direction)
    return composed.tolist()


def observation_heading_deg(
    *,
    keyframe_lat: float,
    keyframe_lng: float,
    target_lat: float,
    target_lng: float,
) -> float:
    """Compass bearing from keyframe to target, degrees in [0, 360)."""
    lat1 = math.radians(float(keyframe_lat))
    lat2 = math.radians(float(target_lat))
    dlon = math.radians(float(target_lng) - float(keyframe_lng))
    east = math.sin(dlon) * math.cos(lat2)
    north = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(dlon)
    return float(math.degrees(math.atan2(east, north))) % 360.0


def select_top_candidates(
    candidates: list[EnrichedCandidate], candidate_count: int
) -> list[EnrichedCandidate]:
    if not candidates:
        return []
    count = min(max(int(candidate_count), 1), len(candidates))
    return candidates[:count]


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
    min_keyframes: int,
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


def filter_clusters_by_geometry(
    cluster_best_sim: dict[int, float],
    stats: ClusterStatistics,
    *,
    min_observations: int,
    max_spread_m: float | None,
) -> dict[int, float]:
    """Drop clusters whose geometric support is too thin, before ranking.

    Applied **before** `rank_localization_clusters` on purpose: `match_score` is a
    ratio to the query's best cluster, so the denominator must be a cluster we would
    actually return. Filtering afterwards would leave every score normalised against
    something the caller never sees.

    A dropped cluster is gone, not demoted — that is the whole point of the split
    between filtering and scoring.
    """
    if min_observations <= 1 and max_spread_m is None:
        return cluster_best_sim

    kept: dict[int, float] = {}
    for cluster_id, sim in cluster_best_sim.items():
        if cluster_id >= stats.observation_counts.shape[0]:
            continue
        if int(stats.observation_counts[cluster_id]) < int(min_observations):
            continue
        if max_spread_m is not None and float(stats.spread_m[cluster_id]) > float(
            max_spread_m
        ):
            continue
        kept[cluster_id] = sim
    return kept


def _levels_compatible(
    detection_levels: np.ndarray | None,
    local_a: int,
    local_b: int,
) -> bool:
    if detection_levels is None:
        return True
    level_a = int(detection_levels[local_a])
    level_b = int(detection_levels[local_b])
    if (
        level_a != UNRESOLVED_LEVEL
        and level_b != UNRESOLVED_LEVEL
        and level_a != level_b
    ):
        return False
    return True


def _embedding_matrix(candidates: list[EnrichedCandidate]) -> np.ndarray | None:
    """Stack the candidates' embeddings, or None if any is missing.

    All-or-nothing on purpose: a partially populated matrix would gate some pairs and
    wave others through, which is indistinguishable from a threshold that happens not
    to bite. The default path carries no embeddings at all, so this returns None.
    """
    if not candidates or any(c.embedding is None for c in candidates):
        return None
    return np.vstack([c.embedding for c in candidates])


def _semantic_gate(
    embeddings: np.ndarray | None,
    valid_indices: np.ndarray,
    threshold: float | None,
) -> Callable[[int, int], bool] | None:
    """`(seed_local, other_local) -> bool` cosine gate, or None when disabled.

    The second half of the two-gate association. Embeddings are L2-normalised by the
    pipeline (norms measured 0.999428–1.000566), so a dot product *is* the cosine —
    the same identity the prototype-similarity SQL relies on. Re-normalised here
    anyway, because a silent scale error would read as "the gate does nothing".

    Rows are reindexed to the valid subset once, so the hot loop indexes with the
    same local indices the caller already uses.
    """
    if threshold is None or embeddings is None:
        return None
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return None
    subset = matrix[valid_indices]
    norms = np.linalg.norm(subset, axis=1, keepdims=True)
    subset = subset / np.where(norms > 0, norms, 1.0)
    cut = float(threshold)

    def gate(seed_local: int, other_local: int) -> bool:
        return bool(float(subset[seed_local] @ subset[other_local]) >= cut)

    return gate


def cluster_detections_leader_canopy(
    positions_local: np.ndarray,
    valid_mask: np.ndarray,
    object_keyframe_ids: np.ndarray,
    query_similarities: np.ndarray,
    detection_levels: np.ndarray | None,
    *,
    eps_meters: float,
    min_keyframes_per_cluster: int,
    embeddings: np.ndarray | None = None,
    semantic_gate_threshold: float | None = None,
) -> np.ndarray:
    """Greedy similarity-seeded spatial clustering (leader / canopy).

    With `semantic_gate_threshold` set, a detection joins the seed's cluster only if
    it also passes a **cutout↔cutout cosine** gate against the seed — the two-gate
    (geometric AND semantic) association ConceptGraphs uses, in place of production's
    geometry-only rule. Off by default; `embeddings` must be supplied for it to apply.
    """
    labels = np.full(len(positions_local), -1, dtype=np.int32)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return labels

    similarities = np.asarray(query_similarities, dtype=np.float64).reshape(-1)
    valid_positions = np.asarray(positions_local[valid_indices], dtype=np.float64)
    assigned = np.zeros(valid_indices.size, dtype=bool)
    order = np.argsort(-similarities[valid_indices])
    next_label = 0
    eps = float(eps_meters)

    gate = _semantic_gate(embeddings, valid_indices, semantic_gate_threshold)

    for seed_local in order:
        if assigned[seed_local]:
            continue
        seed_pos = valid_positions[seed_local]
        dists = np.linalg.norm(valid_positions - seed_pos, axis=1)
        neighbors = np.where((dists <= eps) & ~assigned)[0]
        for j in neighbors:
            if not _levels_compatible(detection_levels, seed_local, int(j)):
                continue
            if gate is not None and not gate(int(seed_local), int(j)):
                continue
            assigned[j] = True
            labels[int(valid_indices[j])] = next_label
        next_label += 1

    return filter_clusters_by_min_keyframes(
        labels,
        object_keyframe_ids,
        min_keyframes=min_keyframes_per_cluster,
    )


def _cluster_level_from_detections(
    levels: np.ndarray,
    similarities: np.ndarray,
    keyframe_ids: np.ndarray,
    *,
    strategy: str,
) -> int:
    """The level a cluster claims.

    ``"seed"`` takes the level of the highest-similarity detection — production's
    behaviour, and the default.

    ``"median"`` takes the median level of the cluster's **distinct keyframes**, one
    vote each. Per keyframe and not per detection because a level *is* a property of
    the camera pose: a keyframe contributing 40 detections and one contributing 1
    observed the cluster from one floor each, and weighting by detection count would
    let a single well-covered viewpoint decide the floor on its own.

    The median is the lower of the two middle values on an even count, never their
    average: levels are ordinal, and floor 4.5 does not exist.

    ``UNRESOLVED_LEVEL`` is **not** filtered out. An unresolved keyframe therefore gets
    one vote like every other keyframe; see gotcha 5 in AI_CONTEXT/bricks.md for why
    the sentinel is distinct from every real floor value.
    """
    if strategy == "median":
        # One row per keyframe. `vkf_level` is per keyframe, so every detection of a
        # keyframe carries the same level and the first occurrence is exact — which
        # holds only because the median path never falls back to the depth-projected
        # object level (see `localize_from_enriched_candidates`).
        _, first_of_keyframe = np.unique(keyframe_ids, return_index=True)
        ordered = np.sort(levels[first_of_keyframe])
        return int(ordered[(ordered.size - 1) // 2])
    return int(levels[int(np.argmax(similarities))])


def compute_cluster_statistics(
    positions_eus: np.ndarray,
    cluster_ids: np.ndarray,
    similarities: np.ndarray,
    detection_levels: np.ndarray | None,
    detection_keyframe_ids: np.ndarray,
    geo_transform: GeoTransform,
    level_strategy: str = "seed",
) -> ClusterStatistics:
    unique_labels = np.unique(cluster_ids)
    unique_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(unique_labels)

    centroids_eus = np.zeros((n_clusters, 3), dtype=np.float64)
    centroids_lat = np.zeros(n_clusters, dtype=np.float64)
    centroids_lng = np.zeros(n_clusters, dtype=np.float64)
    centroids_alt = np.zeros(n_clusters, dtype=np.float64)
    observation_counts = np.zeros(n_clusters, dtype=np.int32)
    confidence_scores = np.zeros(n_clusters, dtype=np.float64)
    spread_values = np.zeros(n_clusters, dtype=np.float64)
    cluster_levels = np.full(n_clusters, UNRESOLVED_LEVEL, dtype=np.int32)

    level_by_value = {int(lv.value): lv for lv in geo_transform.levels}

    for i, label in enumerate(unique_labels):
        mask = cluster_ids == label
        cluster_positions = positions_eus[mask]
        weights = similarities[mask]
        weights = weights / weights.sum()
        centroid = np.average(cluster_positions, axis=0, weights=weights)
        centroids_eus[i] = centroid
        observation_counts[i] = int(mask.sum())

        spread = (
            float(np.std(cluster_positions, axis=0).mean())
            if cluster_positions.shape[0] > 1
            else 0.0
        )
        spread_values[i] = spread
        count_factor = min(1.0, observation_counts[i] / 5.0)
        spread_factor = max(0.0, 1.0 - spread / 2.0)
        confidence_scores[i] = count_factor * (0.5 + 0.5 * spread_factor)

        if detection_levels is not None:
            cluster_levels[i] = _cluster_level_from_detections(
                detection_levels[mask],
                similarities[mask],
                detection_keyframe_ids[mask],
                strategy=level_strategy,
            )

        # Level came from the (stable) keyframe pose — clamp the centroid's
        # local-up to that level's altitude band so the reported altitude stays
        # consistent with the declared level (the depth-projected centroid
        # altitude drifts across level boundaries).
        if cluster_levels[i] != UNRESOLVED_LEVEL:
            level_obj = level_by_value.get(int(cluster_levels[i]))
            if level_obj is not None:
                if level_obj.min_altitude is not None:
                    centroid[1] = max(centroid[1], level_obj.min_altitude)
                if level_obj.max_altitude is not None:
                    centroid[1] = min(centroid[1], level_obj.max_altitude)
                centroids_eus[i] = centroid

        wgs84 = geo_transform.local_positions_to_wgs84(
            centroid.reshape(1, 3).astype(np.float64)
        )
        centroids_lng[i] = float(wgs84[0, 0])
        centroids_lat[i] = float(wgs84[0, 1])
        centroids_alt[i] = float(wgs84[0, 2])

        # Last-resort resolution from the centroid's own altitude — i.e. from the
        # depth-projected object position. Skipped under `"median"`, which is defined as
        # taking the floor from the keyframe poses and nothing else: a cluster whose
        # keyframes resolve no level must stay unresolved rather than inherit a floor
        # from a depth estimate.
        if cluster_levels[i] == UNRESOLVED_LEVEL and level_strategy != "median":
            level_val = geo_transform.levels_for_altitudes(
                np.array([centroid[1]], dtype=np.float64),
                lats=np.array([centroids_lat[i]], dtype=np.float64),
                lngs=np.array([centroids_lng[i]], dtype=np.float64),
            )[0]
            if np.isfinite(level_val):
                cluster_levels[i] = int(level_val)

    return ClusterStatistics(
        centroids_eus=centroids_eus,
        centroids_lat=centroids_lat,
        centroids_lng=centroids_lng,
        centroids_alt=centroids_alt,
        observation_counts=observation_counts,
        confidence_scores=confidence_scores,
        cluster_levels=cluster_levels,
        spread_m=spread_values,
    )


def _similarity_ratio_scores(cluster_best_sim: dict[int, float]) -> dict[int, float]:
    """Each cluster's similarity as a fraction of the query's best one.

    Deliberately *not* a min-max rescale. The old denominator was
    ``best - min_similarity``, which made every score a function of a **filter
    parameter**: moving `min_similarity` from 0.2 to 0.15 moved every `match_score`
    without any evidence having changed. The ratio has no free parameter, so the two
    knobs are finally independent — `min_similarity` is the absolute floor, this is
    the relative gate.
    """
    if not cluster_best_sim:
        return {}
    best = max(float(sim) for sim in cluster_best_sim.values())
    if best <= 1e-6:
        return {int(cluster_id): 1.0 for cluster_id in cluster_best_sim}
    return {
        int(cluster_id): float(np.clip(float(sim) / best, 0.0, 1.0))
        for cluster_id, sim in cluster_best_sim.items()
    }


def rank_localization_clusters(
    *,
    cluster_best_sim: dict[int, float],
    min_similarity: float,
) -> list[ClusterRanking]:
    """``match_score = best_sim / best_sim_of_the_query`` — one term, one meaning.

    "This cluster reaches X% of the quality of this query's best match."

    It replaces production's
    ``0.50·norm_sim + 0.15·confidence + 0.35·min(1, keyframes/3)``, measured on
    bbhotel-choisy (12 prompts, 674 annotations, 1287 clusters):

    - the two size terms carry no ranking signal. `similarity_score` alone scores
      mAP 0.653 against the full mixture's 0.652 (0.713 vs 0.715 grouped);
      `min(1, kf/3)` alone scores 0.318, i.e. half the weight budget bought noise;
    - they were also *saturated*, so mostly constant: 65% of clusters have >= 3
      keyframes and 53% have >= 5 observations, and both terms cap there;
    - size was counted three times over — `kf/3`, `min(1, n_obs/5)` inside
      `confidence`, and `max` over N detections, which grows with N;
    - as a global acceptance gate the ratio transfers between prompts, which the
      mixture does not: leave-one-prompt-out macro F1 0.611 vs 0.533 (strict ground
      truth) and 0.627 vs 0.552 (grouped at 2 m).

    Geometric support did not disappear, it moved: `min_keyframes_per_cluster`,
    `min_observations_per_cluster` and `max_cluster_spread_m` are filters now, and
    `confidence` / `observation_count` / `spread_m` stay on the response so a caller
    can gate on them itself rather than receive them diluted into one number.

    **This is a deliberate divergence from `wemap-vision-backend`**, not a port
    artefact — see AI_CONTEXT/bricks.md. Production still ships the weighted mixture.
    """
    eligible = {
        int(cluster_id): float(sim)
        for cluster_id, sim in cluster_best_sim.items()
        if float(sim) >= float(min_similarity)
    }
    ratios = _similarity_ratio_scores(eligible)

    rankings = [
        ClusterRanking(
            cluster_id=cluster_id,
            similarity_score=sim,
            match_score=ratios.get(cluster_id, 0.0),
        )
        for cluster_id, sim in eligible.items()
    ]
    rankings.sort(key=lambda r: (r.match_score, r.similarity_score), reverse=True)
    return rankings


def _ranking_similarities(
    selected: list[EnrichedCandidate], params: LocalizationParams
) -> np.ndarray:
    """The similarity used to *score* clusters — boosted only when asked.

    Disablement is structural at two levels, deliberately. With both gains at
    zero this returns the raw array and `similarity_boosted` is never read, so a
    bad value written upstream cannot leak into the default path; and even with
    gains set, a candidate loaded without feedback falls back to its raw
    similarity through `effective_similarity`.
    """
    if not params.feedback_enabled:
        return np.array([c.similarity for c in selected], dtype=np.float64)
    return np.array([c.effective_similarity for c in selected], dtype=np.float64)


def _observation_feedback_fields(
    candidate: EnrichedCandidate, params: LocalizationParams
) -> dict:
    """Per-observation feedback terms, or `{}` when the feature is off.

    Emitted only when enabled so the default response shape is byte-identical to
    what it was before this feature — `toolbox/benchmark/` and the toolbox UI both
    parse this dict, and an always-present set of nulls would be a silent contract
    change for a feature nobody turned on.
    """
    if not params.feedback_enabled:
        return {}
    fields = {
        "pos_sim": round(float(candidate.pos_sim), 6),
        "neg_sim": round(float(candidate.neg_sim), 6),
        "similarity_boosted": round(float(candidate.effective_similarity), 6),
        "feedback_delta": round(
            float(candidate.effective_similarity) - float(candidate.similarity), 6
        ),
    }
    if params.feedback_normalization != "none":
        # Only under a normalisation: otherwise these duplicate `pos_sim`/`neg_sim`
        # exactly, and a field that is always a copy is a field nobody reads.
        fields["pos_sim_applied"] = round(float(candidate.pos_sim_applied), 6)
        fields["neg_sim_applied"] = round(float(candidate.neg_sim_applied), 6)
    return fields


def localize_from_enriched_candidates(
    candidates: list[EnrichedCandidate],
    geo_transform: GeoTransform,
    params: LocalizationParams | None = None,
) -> list[dict]:
    """Cluster enriched candidates and return livemap localization dicts."""
    params = params or LocalizationParams()
    selected = select_top_candidates(candidates, params.candidate_count)
    if not selected:
        return []

    positions_eus = np.array([c.eus_xyz for c in selected], dtype=np.float64)
    # RAW similarity, everywhere below except `cluster_best_sim`.
    #
    # `similarities` drives three things that are *geometry*, not ranking, and
    # boosting any of them would change what the map says rather than how it is
    # ordered:
    #   - the leader-canopy seed order, i.e. which detections group together;
    #   - the centroid weights in `compute_cluster_statistics`, i.e. where the
    #     cluster is reported to be — and those are normalised by `weights.sum()`,
    #     so a boosted value clipped to a negative number would produce a
    #     nonsensical centroid rather than an error;
    #   - the level seed (`argmax`), i.e. which floor the cluster claims.
    # Only the cluster's *score* is boosted. That is the whole intervention.
    similarities = np.array([c.similarity for c in selected], dtype=np.float64)
    ranking_similarities = _ranking_similarities(selected, params)
    keyframe_ids = np.array([c.video_keyframe_id for c in selected], dtype=np.int64)
    # Level is a keyframe-pose property (depth-independent), matching the
    # standalone: keyframe altitude poses and the depth model are both noisy, so
    # the object-position altitude is unreliable for level assignment. Use the
    # keyframe (camera) level, falling back to the object-position level only
    # when the keyframe pose resolves no level (boundary / outside polygon).
    #
    # Under `"median"` that last fallback is dropped: `c.level` is the level of the
    # depth-projected object position, and the median strategy is defined as reading the
    # floor from the keyframe poses only. Keeping it would also break the per-keyframe
    # vote in `_cluster_level_from_detections`, which assumes every detection of a
    # keyframe carries that keyframe's level.
    keyframe_levels_only = params.level_strategy == "median"
    detection_levels = np.array(
        [
            (
                c.vkf_level
                if c.vkf_level is not None
                else (
                    UNRESOLVED_LEVEL
                    if keyframe_levels_only or c.level is None
                    else c.level
                )
            )
            for c in selected
        ],
        dtype=np.int32,
    )
    valid_mask = np.ones(len(selected), dtype=bool)

    cluster_ids = cluster_detections_leader_canopy(
        positions_eus,
        valid_mask,
        keyframe_ids,
        similarities,
        detection_levels,
        eps_meters=params.clustering_eps_m,
        min_keyframes_per_cluster=params.min_keyframes_per_cluster,
        embeddings=_embedding_matrix(selected),
        semantic_gate_threshold=params.semantic_gate_threshold,
    )

    stats = compute_cluster_statistics(
        positions_eus,
        cluster_ids,
        similarities,
        detection_levels,
        keyframe_ids,
        geo_transform,
        level_strategy=params.level_strategy,
    )

    cluster_best_sim: dict[int, float] = {}
    cluster_keyframes: dict[int, set[str]] = {}
    cluster_observations: dict[int, list[tuple[int, float]]] = {}

    for local_idx, candidate in enumerate(selected):
        cid = int(cluster_ids[local_idx])
        if cid < 0:
            continue
        # The one boosted consumer: the cluster's score, which feeds
        # `rank_localization_clusters` and therefore `match_score`.
        ranking_sim = float(ranking_similarities[local_idx])
        if cid not in cluster_best_sim or ranking_sim > cluster_best_sim[cid]:
            cluster_best_sim[cid] = ranking_sim
        cluster_keyframes.setdefault(cid, set()).add(str(candidate.video_keyframe_id))
        # Observations are ordered and truncated on the RAW similarity: this
        # selects *which* observations a cluster shows, and the plan routes every
        # selection step to raw. Reordering them on the boost is a defensible
        # follow-up, not part of this change.
        cluster_observations.setdefault(cid, []).append(
            (local_idx, float(similarities[local_idx]))
        )

    ranked = rank_localization_clusters(
        cluster_best_sim=filter_clusters_by_geometry(
            cluster_best_sim,
            stats,
            min_observations=params.min_observations_per_cluster,
            max_spread_m=params.max_cluster_spread_m,
        ),
        min_similarity=params.min_similarity,
    )
    ranked = ranked[: params.num_results]

    localizations: list[dict] = []
    for ranking in ranked:
        cluster_id = ranking.cluster_id
        if cluster_id >= stats.centroids_lat.shape[0]:
            continue

        lat = float(stats.centroids_lat[cluster_id])
        lng = float(stats.centroids_lng[cluster_id])
        alt = float(stats.centroids_alt[cluster_id])
        level_val = int(stats.cluster_levels[cluster_id])
        level = level_val if level_val != UNRESOLVED_LEVEL else None

        raw_observations = cluster_observations.get(cluster_id, [])
        raw_observations.sort(key=lambda item: item[1], reverse=True)

        observations: list[dict] = []
        for local_idx, obj_sim in raw_observations[
            : params.max_observations_per_cluster
        ]:
            cand = selected[local_idx]
            observations.append(
                {
                    "object_idx": cand.id,
                    "cutout_id": str(cand.id),
                    "keyframe_id": str(cand.video_keyframe_id),
                    # The stored thumbnail is the only preview a cluster observation
                    # can have: the toolbox used to render one from the standalone
                    # SQLite index, which is gone, and pgvector does not carry
                    # `row_index`, so there is no way back to the parquet row.
                    "thumbnail": cand.thumbnail,
                    "coordinates": [cand.lat, cand.lng, cand.alt],
                    "bbox": list(PLACEHOLDER_BBOX),
                    # Raw retrieval similarity — unchanged, and what the HTTP
                    # benchmark reads. The feedback terms sit beside it rather
                    # than replacing it, so a tuning session can see both.
                    "similarity_score": float(obj_sim),
                    **_observation_feedback_fields(cand, params),
                    "heading": observation_heading_deg(
                        keyframe_lat=cand.vkf_lat,
                        keyframe_lng=cand.vkf_lng,
                        target_lat=cand.lat,
                        target_lng=cand.lng,
                    ),
                    "quaternion": v1_5_observation_quaternion(
                        cand.geokeyframe_pose,
                        cand.theta_center,
                        cand.phi_center,
                    ),
                }
            )

        localizations.append(
            {
                "coordinates": [lat, lng, alt],
                "confidence": float(stats.confidence_scores[cluster_id]),
                "observation_count": int(stats.observation_counts[cluster_id]),
                # Geometric support, reported rather than mixed into `match_score`.
                # `max_cluster_spread_m` filters on exactly this value.
                "spread_m": float(stats.spread_m[cluster_id]),
                "similarity_score": float(ranking.similarity_score),
                "match_score": float(ranking.match_score),
                "level": level,
                "keyframe_ids": sorted(cluster_keyframes.get(cluster_id, set())),
                "observations": observations,
            }
        )

    return localizations


def build_localize_response(
    candidates: list[EnrichedCandidate],
    geo_transform: GeoTransform,
    *,
    params: LocalizationParams | None = None,
    time_embedding_ms: int = 0,
    time_retrieval_ms: int = 0,
) -> dict:
    localizations = localize_from_enriched_candidates(
        candidates, geo_transform, params=params
    )
    return {
        "localizations": localizations,
        "time_embedding_ms": int(time_embedding_ms),
        "time_retrieval_ms": int(time_retrieval_ms),
    }
