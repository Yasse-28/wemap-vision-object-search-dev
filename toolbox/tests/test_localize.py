"""Tests for the ported clustering, ranking and response shape.

These tests pin the production-compatible default path plus the deliberate dev-only
divergences documented in `AI_CONTEXT/bricks.md`. They exist because the backend has
no tests for that module — the port is the first chance to add them.

Two things are asserted deliberately rather than incidentally:

- the **ranking rule** (`match_score = best_sim / best_sim_of_the_query`), so a
  well-meaning tweak here shows up as a failing test rather than as silently
  different result ordering. This is a deliberate divergence from production's
  weighted mixture; `rank_localization_clusters`' docstring carries the measurement
  that motivated it;
- the **response shape**, because `toolbox/benchmark/object_search_http_benchmark.py`
  parses `localizations[].coordinates` and `match_score` directly.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from toolbox.bricks import localize
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.localize import (
    UNRESOLVED_LEVEL,
    LocalizationParams,
    _filter_edges_by_delta_overlap,
    build_localize_response,
    cluster_detections_cdog,
    cluster_detections_incremental,
    cluster_detections_leader_canopy,
    cluster_detections_multicut,
    compute_cluster_statistics,
    filter_clusters_by_geometry,
    filter_clusters_by_min_keyframes,
    greedy_additive_edge_contraction,
    localize_from_enriched_candidates,
    rank_localization_clusters,
    ray_closest_approach,
)
from toolbox.bricks.vendored.geo_transform import Coordinates, GeoTransform, Level, Pose
from toolbox.bricks.vendored.maths import quaternion, vector3


def _cluster(
    positions: Any,
    keyframe_ids: Any,
    similarities: Any,
    levels: Any = None,
    *,
    eps: float = 2.0,
    min_keyframes: int = 1,
) -> NDArray[np.int32]:
    """Thin wrapper so the cases below can be written as plain literals."""
    positions_arr = np.asarray(positions, dtype=np.float64)
    return cluster_detections_leader_canopy(
        positions_arr,
        np.ones(len(positions_arr), dtype=bool),
        np.asarray(keyframe_ids, dtype=np.int64),
        np.asarray(similarities, dtype=np.float64),
        None if levels is None else np.asarray(levels, dtype=np.int32),
        eps_meters=eps,
        min_keyframes_per_cluster=min_keyframes,
    )


# ------------------------------------------------------------------- clustering


def test_nearby_detections_merge_and_distant_ones_do_not() -> None:
    labels = _cluster(
        [[0, 0, 0], [0.5, 0, 0], [50, 0, 0]],
        keyframe_ids=[1, 2, 3],
        similarities=[0.9, 0.8, 0.7],
        eps=2.0,
    )
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_clustering_is_deterministic_and_seeded_by_similarity() -> None:
    """Same input → same labels, and the highest-similarity point seeds first.

    The algorithm is greedy over `argsort(-similarity)`, so seeding order is part of
    the contract: change it and cluster membership shifts for tied distances.
    """
    positions = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]]
    keyframe_ids = [1, 2, 3]
    similarities = [0.5, 0.95, 0.4]

    first = _cluster(positions, keyframe_ids, similarities, eps=2.0)
    for _ in range(5):
        np.testing.assert_array_equal(
            first, _cluster(positions, keyframe_ids, similarities, eps=2.0)
        )

    # The middle point has the top similarity, so it seeds and absorbs both
    # neighbours (each within 2 m of it) into one cluster.
    assert len(set(first.tolist())) == 1


def test_min_keyframes_drops_single_keyframe_clusters() -> None:
    """Two detections of the same object from one keyframe are not corroboration."""
    labels = _cluster(
        [[0, 0, 0], [0.2, 0, 0], [50, 0, 0], [50.2, 0, 0]],
        keyframe_ids=[1, 1, 7, 8],  # first pair: one keyframe; second: two
        similarities=[0.9, 0.85, 0.8, 0.75],
        eps=2.0,
        min_keyframes=2,
    )
    assert labels[0] == -1 and labels[1] == -1
    assert labels[2] >= 0 and labels[2] == labels[3]


def test_incompatible_levels_are_not_merged() -> None:
    """Co-located detections on different floors must stay separate."""
    labels = _cluster(
        [[0, 0, 0], [0.1, 0, 0]],
        keyframe_ids=[1, 2],
        similarities=[0.9, 0.8],
        levels=[0, 1],
        eps=2.0,
    )
    assert labels[0] != labels[1]


def test_unresolved_level_stays_mergeable() -> None:
    """UNRESOLVED_LEVEL must not block merging, or unlevelled maps cluster nothing."""
    labels = _cluster(
        [[0, 0, 0], [0.1, 0, 0]],
        keyframe_ids=[1, 2],
        similarities=[0.9, 0.8],
        levels=[UNRESOLVED_LEVEL, 3],
        eps=2.0,
    )
    assert labels[0] == labels[1]


# ---------------------------------------------------------------------- multicut


def _multicut_cluster(
    positions: Any,
    *,
    levels: Any = None,
    embeddings: Any = None,
    sem_weight: float = 0.0,
) -> NDArray[np.int32]:
    """Run depth-based multicut on concise test fixtures."""
    points = np.asarray(positions, dtype=np.float64)
    origins = np.zeros_like(points)
    directions = points / np.where(
        np.linalg.norm(points, axis=1, keepdims=True) > 0.0,
        np.linalg.norm(points, axis=1, keepdims=True),
        1.0,
    )
    return cluster_detections_multicut(
        points,
        origins,
        directions,
        np.ones(len(points), dtype=bool),
        np.arange(len(points), dtype=np.int64),
        None if levels is None else np.asarray(levels, dtype=np.int32),
        pair_radius_m=10.0,
        geo_pivot=1.0,
        sem_weight=sem_weight,
        min_keyframes_per_cluster=1,
        embeddings=(
            None if embeddings is None else np.asarray(embeddings, dtype=np.float64)
        ),
    )


def test_gaec_known_four_node_partition_changes_with_bridge_sign() -> None:
    negative_bridge = greedy_additive_edge_contraction(
        4, [(0, 1, 2.0), (2, 3, 2.0), (1, 2, -1.0)]
    )
    positive_bridge = greedy_additive_edge_contraction(
        4, [(0, 1, 2.0), (2, 3, 2.0), (1, 2, 1.0)]
    )

    np.testing.assert_array_equal(negative_bridge, [0, 0, 1, 1])
    np.testing.assert_array_equal(positive_bridge, [0, 0, 0, 0])


def test_gaec_sums_parallel_costs_created_by_contraction() -> None:
    labels = greedy_additive_edge_contraction(
        3,
        [
            (0, 1, 2.0),
            (0, 2, 1.0),
            (1, 2, -1.5),
        ],
    )

    # A max-only merge would retain +1 and merge node 2. GAEC sums +1 and -1.5,
    # leaving a negative component edge and therefore exactly two clusters.
    np.testing.assert_array_equal(labels, [0, 0, 1])


def test_multicut_level_incompatibility_is_a_hard_component_cut() -> None:
    labels = _multicut_cluster(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        levels=[0, UNRESOLVED_LEVEL, 1],
    )

    assert labels[0] != labels[2]


def test_multicut_allows_same_keyframe_proposals_to_merge() -> None:
    points = np.asarray([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    labels = cluster_detections_multicut(
        points,
        np.zeros_like(points),
        points / np.linalg.norm(points, axis=1, keepdims=True),
        np.ones(2, dtype=bool),
        np.asarray([7, 7], dtype=np.int64),
        None,
        min_keyframes_per_cluster=1,
    )

    assert labels[0] == labels[1]


def test_multicut_zero_semantic_weight_ignores_embeddings() -> None:
    positions = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [3.0, 0.0, 0.0]]
    embeddings = np.eye(3, dtype=np.float64)
    first = _multicut_cluster(positions, embeddings=embeddings, sem_weight=0.0)
    second = _multicut_cluster(
        positions, embeddings=embeddings[[2, 0, 1]], sem_weight=0.0
    )

    np.testing.assert_array_equal(first, second)


def _layout_cluster(gap_rad: float, layout_weight: float) -> NDArray[np.int32]:
    """Two boxes of one keyframe, 1.5 m apart, separated by ``gap_rad`` on the sphere.

    Their combined half-extent is 0.2 rad, so ``gap_rad`` above 0.2 means disjoint
    boxes and below it means overlapping proposals.
    """
    points = np.asarray([[3.0, 0.0, 0.0], [4.5, 0.0, 0.0]], dtype=np.float64)
    return cluster_detections_multicut(
        points,
        np.zeros_like(points),
        points / np.linalg.norm(points, axis=1, keepdims=True),
        np.ones(2, dtype=bool),
        np.asarray([7, 7], dtype=np.int64),
        None,
        pair_radius_m=10.0,
        geo_pivot=1.0,
        layout_weight=layout_weight,
        min_keyframes_per_cluster=1,
        box_angles=np.asarray(
            [[0.0, 0.0, 0.2, 0.0], [gap_rad, 0.0, 0.2, 0.0]], dtype=np.float64
        ),
    )


def test_multicut_layout_term_is_attractive_below_one_and_repulsive_above() -> None:
    # Geometry alone cuts this pair (1.5 m at pivot 1 m => cost -0.5). Overlapping
    # boxes must be able to overcome that; clearly disjoint ones must not.
    overlapping = _layout_cluster(0.02, layout_weight=2.0)
    disjoint = _layout_cluster(0.6, layout_weight=2.0)

    assert overlapping[0] == overlapping[1]
    assert disjoint[0] != disjoint[1]


def test_multicut_zero_layout_weight_ignores_box_angles() -> None:
    baseline = _layout_cluster(0.02, layout_weight=0.0)
    swapped = _layout_cluster(0.6, layout_weight=0.0)

    np.testing.assert_array_equal(baseline, swapped)


def test_multicut_layout_term_needs_box_angles() -> None:
    points = np.asarray([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="box angles"):
        cluster_detections_multicut(
            points,
            np.zeros_like(points),
            points / np.linalg.norm(points, axis=1, keepdims=True),
            np.ones(2, dtype=bool),
            np.asarray([7, 7], dtype=np.int64),
            None,
            layout_weight=1.0,
        )


def test_gaec_equal_cost_ties_use_node_indices_not_edge_order() -> None:
    edges = [(1, 2, 1.0), (0, 1, 1.0), (0, 2, -1.5)]
    forward = greedy_additive_edge_contraction(3, edges)
    reversed_order = greedy_additive_edge_contraction(3, list(reversed(edges)))

    np.testing.assert_array_equal(forward, [0, 0, 1])
    np.testing.assert_array_equal(reversed_order, forward)


def test_basement_level_does_not_disable_the_clustering_veto() -> None:
    """A real basement level must not be treated as the unresolved sentinel."""
    labels = _cluster(
        [[0, 0, 0], [0.1, 0, 0]],
        keyframe_ids=[1, 2],
        similarities=[0.9, 0.8],
        levels=[-1, 0],
        eps=2.0,
    )
    assert labels[0] != labels[1]


def test_labels_are_compacted_from_zero() -> None:
    """Ranking indexes cluster statistics by label, so labels must be 0..n-1."""
    labels = _cluster(
        [[0, 0, 0], [100, 0, 0], [200, 0, 0]],
        keyframe_ids=[1, 2, 3],
        similarities=[0.9, 0.8, 0.7],
        eps=1.0,
    )
    assert sorted(labels.tolist()) == [0, 1, 2]


def test_filter_clusters_by_min_keyframes_is_a_noop_below_two() -> None:
    labels = np.array([0, 0, 5, -1], dtype=np.int32)
    out = filter_clusters_by_min_keyframes(
        labels, np.array([1, 1, 2, 3], dtype=np.int64), min_keyframes=1
    )
    # Compacted but nothing dropped: label 5 becomes 1, -1 stays -1.
    assert out.tolist() == [0, 0, 1, -1]


def test_empty_input_returns_no_labels() -> None:
    labels = _cluster(np.empty((0, 3)), [], [], eps=2.0)
    assert labels.shape == (0,)


def _cdog_cluster(
    positions: list[list[float]],
    origins: list[list[float]],
    directions: list[list[float]],
    *,
    epipolar_m: float = 0.25,
    delta: float = 0.0,
) -> NDArray[np.int32]:
    """Run C-DOG on small explicit ray fixtures without semantic gating."""
    count = len(positions)
    return cluster_detections_cdog(
        np.asarray(positions, dtype=np.float64),
        np.asarray(origins, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
        np.ones(count, dtype=bool),
        np.arange(count, dtype=np.int64),
        None,
        epipolar_m=epipolar_m,
        pair_radius_m=5.0,
        range_m=(0.3, 30.0),
        delta=delta,
        min_keyframes_per_cluster=1,
    )


def test_cdog_crossing_rays_ignore_wrong_projected_depths() -> None:
    """Normalizing point-minus-origin removes 0.5x/2x depth scale errors."""
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    origins = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
    positions = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 0.0]])
    vectors = positions - origins
    directions = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    distance, t_i, t_j = ray_closest_approach(
        origins[0], directions[0], origins[1], directions[1]
    )
    assert distance == pytest.approx(0.0, abs=1e-12)
    assert t_i == pytest.approx(1.0)
    assert t_j == pytest.approx(1.0)

    labels = _cdog_cluster(
        positions.tolist(), origins.tolist(), directions.tolist(), delta=0.0
    )
    assert labels.tolist() == [0, 0]

    stats = compute_cluster_statistics(
        positions,
        labels,
        np.ones(2, dtype=np.float64),
        None,
        np.array([0, 1], dtype=np.int64),
        _geo_transform(),
        centroid_from="rays",
        ray_origins=origins,
        ray_directions=directions,
    )
    np.testing.assert_allclose(stats.centroids_eus[0], target, atol=1e-6)


def test_cdog_rejects_a_crossing_behind_one_camera() -> None:
    labels = _cdog_cluster(
        positions=[[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        origins=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert labels[0] != labels[1]


def test_cdog_metric_ray_distance_is_sweepable() -> None:
    positions = [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    origins = [[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]]
    directions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    strict = _cdog_cluster(positions, origins, directions, epipolar_m=0.25, delta=0.0)
    permissive = _cdog_cluster(
        positions, origins, directions, epipolar_m=1.5, delta=0.0
    )
    assert strict[0] != strict[1]
    assert permissive[0] == permissive[1]


def test_cdog_delta_overlap_cuts_a_dumbbell_bridge() -> None:
    left = set(range(4))
    right = set(range(4, 8))
    adjacency = {
        node: (left - {node}) if node in left else (right - {node}) for node in range(8)
    }
    adjacency[3].add(4)
    adjacency[4].add(3)

    cut = _filter_edges_by_delta_overlap(adjacency, 0.5)
    joined = _filter_edges_by_delta_overlap(adjacency, 0.0)
    assert 4 not in cut[3]
    assert 4 in joined[3]

    def component_count(graph: dict[int, set[int]]) -> int:
        remaining = set(graph)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                for neighbor in graph[stack.pop()] & remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        return count

    assert component_count(cut) == 2
    assert component_count(joined) == 1


def test_ray_centroid_parallel_rays_fall_back_to_depth(
    caplog: pytest.LogCaptureFixture,
) -> None:
    positions = np.array([[2.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    similarities = np.array([0.75, 0.25])
    stats = compute_cluster_statistics(
        positions,
        np.zeros(2, dtype=np.int32),
        similarities,
        None,
        np.array([0, 1], dtype=np.int64),
        _geo_transform(),
        centroid_from="rays",
        ray_origins=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ray_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    expected = np.average(positions, axis=0, weights=similarities)
    np.testing.assert_allclose(stats.centroids_eus[0], expected)
    assert "fell back to the depth centroid for 1/1 cluster(s)" in caplog.text


# ---------------------------------------------------------------------- ranking


def test_match_score_is_the_ratio_to_the_best_cluster() -> None:
    """match_score = best_sim / best_sim_of_the_query. The top cluster always gets 1."""
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.30, 1: 0.24, 2: 0.21},
        min_similarity=0.2,
    )
    by_id = {r.cluster_id: r.match_score for r in rankings}
    assert by_id[0] == pytest.approx(1.0)
    assert by_id[1] == pytest.approx(0.24 / 0.30)
    assert by_id[2] == pytest.approx(0.21 / 0.30)


def test_match_score_does_not_depend_on_min_similarity() -> None:
    """The whole point of dropping the min-max rescale.

    Under the old ``(sim - min_similarity) / (best - min_similarity)`` these two calls
    returned different scores for identical evidence.
    """

    def scores(min_similarity: float) -> list[float]:
        return [
            r.match_score
            for r in rank_localization_clusters(
                cluster_best_sim={0: 0.30, 1: 0.24},
                min_similarity=min_similarity,
            )
        ]

    assert scores(0.20) == pytest.approx(scores(0.15))


def test_match_score_ignores_cluster_size() -> None:
    """Keyframe count and observation count are filters now, never score terms."""
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.30, 1: 0.30},
        min_similarity=0.2,
    )
    assert [r.match_score for r in rankings] == pytest.approx([1.0, 1.0])


def test_clusters_below_min_similarity_are_dropped() -> None:
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.9, 1: 0.05},
        min_similarity=0.2,
    )
    assert [r.cluster_id for r in rankings] == [0]


def test_min_similarity_drops_before_the_ratio_is_taken() -> None:
    """A filtered-out cluster must not set the denominator it is excluded from."""
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.24, 1: 0.12},
        min_similarity=0.2,
    )
    assert [(r.cluster_id, r.match_score) for r in rankings] == [(0, 1.0)]


def test_rankings_are_sorted_by_match_score_descending() -> None:
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.5, 1: 0.9, 2: 0.7},
        min_similarity=0.2,
    )
    scores = [r.match_score for r in rankings]
    assert scores == sorted(scores, reverse=True)
    assert rankings[0].cluster_id == 1


# ------------------------------------- two-gate association (semantic gate)


def _cluster_gated(
    positions: Any,
    keyframe_ids: Any,
    similarities: Any,
    embeddings: Any,
    threshold: float | None,
    *,
    eps: float = 2.0,
) -> NDArray[np.int32]:
    positions_arr = np.asarray(positions, dtype=np.float64)
    return cluster_detections_leader_canopy(
        positions_arr,
        np.ones(len(positions_arr), dtype=bool),
        np.asarray(keyframe_ids, dtype=np.int64),
        np.asarray(similarities, dtype=np.float64),
        None,
        eps_meters=eps,
        min_keyframes_per_cluster=1,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        semantic_gate_threshold=threshold,
    )


def _cluster_incremental(
    positions: Any,
    similarities: Any,
    embeddings: Any,
    *,
    combination: Literal["conjunctive", "sum"] = "conjunctive",
    semantic_gate_threshold: float | None = None,
    association_sim_threshold: float = 1.1,
    descriptor: Literal["seed", "running_mean"] = "running_mean",
    eps: float = 2.0,
) -> NDArray[np.int32]:
    """Build aligned arrays for focused incremental-association cases."""
    positions_arr = np.asarray(positions, dtype=np.float64)
    return cluster_detections_incremental(
        positions_arr,
        np.ones(len(positions_arr), dtype=bool),
        np.arange(len(positions_arr), dtype=np.int64),
        np.asarray(similarities, dtype=np.float64),
        None,
        eps_meters=eps,
        min_keyframes_per_cluster=1,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        semantic_gate_threshold=semantic_gate_threshold,
        combination=combination,
        association_sim_threshold=association_sim_threshold,
        descriptor=descriptor,
    )


def test_semantic_gate_splits_two_objects_the_spatial_gate_merges() -> None:
    """The case the whole experiment is about: two objects inside one 2 m ball.

    Orthogonal embeddings, so the cosine between the two groups is 0 and any positive
    threshold must separate them — while geometry alone cannot.
    """
    positions = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0), (1.2, 0.0, 0.0)]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    similarities = [0.9, 0.8, 0.7, 0.6]

    merged = _cluster_gated(positions, [1, 2, 3, 4], similarities, embeddings, None)
    split = _cluster_gated(positions, [1, 2, 3, 4], similarities, embeddings, 0.5)

    assert len(set(merged.tolist())) == 1, "geometry alone merges all four"
    assert split.tolist() == [0, 0, 1, 1]


def test_semantic_gate_keeps_one_object_together() -> None:
    """Guard the guard: a gate that splits everything would also pass the test above."""
    positions = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)]
    embeddings = [[1.0, 0.0], [0.99, 0.141], [0.98, 0.199]]

    labels = _cluster_gated(positions, [1, 2, 3], [0.9, 0.8, 0.7], embeddings, 0.9)

    assert len(set(labels.tolist())) == 1


def test_semantic_gate_is_off_without_embeddings() -> None:
    """A threshold with no embeddings must be inert, not a silent no-op filter."""
    positions = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)]
    labels = cluster_detections_leader_canopy(
        np.asarray(positions, dtype=np.float64),
        np.ones(2, dtype=bool),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.9, 0.8], dtype=np.float64),
        None,
        eps_meters=2.0,
        min_keyframes_per_cluster=1,
        embeddings=None,
        semantic_gate_threshold=0.99,
    )
    assert labels.tolist() == [0, 0]


def test_semantic_gate_normalises_before_comparing() -> None:
    """Un-normalised rows must not sneak past a threshold by being long."""
    positions = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)]
    # Cosine is 0, but the raw dot product of these two is 0 as well — scale the first
    # so a missing normalisation would show up as a dot product far above 1.
    embeddings = [[10.0, 0.0], [7.07, 7.07]]

    labels = _cluster_gated(positions, [1, 2], [0.9, 0.8], embeddings, 0.9)

    assert labels.tolist() == [0, 1], "cos = 0.707 must fail a 0.9 gate"


# ----------------------------------------------- incremental association


def test_incremental_best_match_beats_leader_first_catch() -> None:
    """The third detection is in both catchments but strictly nearer cluster 1."""
    positions = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.8, 0.0, 0.0)]
    similarities = [0.9, 0.8, 0.7]
    embeddings = [[1.0, 0.0]] * 3

    leader = _cluster_gated(
        positions, [1, 2, 3], similarities, embeddings, threshold=None, eps=2.0
    )
    incremental = _cluster_incremental(
        positions, similarities, embeddings, combination="conjunctive", eps=2.0
    )

    assert leader.tolist() == [0, 1, 0]
    assert incremental.tolist() == [0, 1, 1]


def test_incremental_running_mean_moves_the_descriptor() -> None:
    """A 50° cutout clears the moved 15° descriptor, but not the 0° seed."""
    angles = np.deg2rad([0.0, 30.0, 50.0])
    embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
    positions = [(0.0, 0.0, 0.0)] * 3
    similarities = [0.9, 0.8, 0.7]
    seed = _cluster_incremental(
        positions,
        similarities,
        embeddings,
        combination="conjunctive",
        semantic_gate_threshold=0.7,
        descriptor="seed",
    )
    running_mean = _cluster_incremental(
        positions,
        similarities,
        embeddings,
        combination="conjunctive",
        semantic_gate_threshold=0.7,
        descriptor="running_mean",
    )

    assert seed.tolist() == [0, 0, 1]
    assert running_mean.tolist() == [0, 0, 0]


def test_incremental_sum_can_trade_semantics_against_geometry() -> None:
    """A near-perfect cosine misses a strict gate but clears the combined score."""
    cosine = 0.98
    embeddings = [[1.0, 0.0], [cosine, math.sqrt(1.0 - cosine**2)]]
    positions = [(0.0, 0.0, 0.0), (1.6, 0.0, 0.0)]
    similarities = [0.9, 0.8]

    conjunctive = _cluster_incremental(
        positions,
        similarities,
        embeddings,
        combination="conjunctive",
        semantic_gate_threshold=0.99,
        eps=2.0,
    )
    summed = _cluster_incremental(
        positions,
        similarities,
        embeddings,
        combination="sum",
        association_sim_threshold=1.1,
        eps=2.0,
    )

    assert conjunctive.tolist() == [0, 1]
    assert summed.tolist() == [0, 0]


def test_incremental_sum_rejects_two_mediocre_terms() -> None:
    labels = _cluster_incremental(
        [(0.0, 0.0, 0.0), (1.6, 0.0, 0.0)],
        [0.9, 0.8],
        [[1.0, 0.0], [0.0, 1.0]],
        combination="sum",
        association_sim_threshold=1.1,
        eps=2.0,
    )

    assert labels.tolist() == [0, 1]


def test_incremental_missing_embeddings_raises() -> None:
    with pytest.raises(ValueError, match="requires embeddings"):
        cluster_detections_incremental(
            np.asarray([(0.0, 0.0, 0.0)], dtype=np.float64),
            np.ones(1, dtype=bool),
            np.asarray([1], dtype=np.int64),
            np.asarray([0.9], dtype=np.float64),
            None,
            eps_meters=2.0,
            min_keyframes_per_cluster=1,
            embeddings=None,
        )


# ------------------------------------------------------- geometry as a filter


def _stats(observation_counts: Any, spreads: Any) -> Any:
    from toolbox.bricks.localize import ClusterStatistics

    n = len(observation_counts)
    return ClusterStatistics(
        centroids_eus=np.zeros((n, 3)),
        centroids_lat=np.zeros(n),
        centroids_lng=np.zeros(n),
        centroids_alt=np.zeros(n),
        observation_counts=np.asarray(observation_counts, dtype=np.int32),
        confidence_scores=np.zeros(n),
        cluster_levels=np.zeros(n, dtype=np.int32),
        spread_m=np.asarray(spreads, dtype=np.float64),
    )


def test_geometry_filter_is_a_noop_when_both_knobs_are_off() -> None:
    best_sim = {0: 0.3, 1: 0.2}
    assert (
        filter_clusters_by_geometry(
            best_sim,
            _stats([1, 1], [9.0, 9.0]),
            min_observations=1,
            max_spread_m=None,
        )
        is best_sim
    ), "the default path must not even copy the dict"


def test_min_observations_drops_thin_clusters() -> None:
    kept = filter_clusters_by_geometry(
        {0: 0.3, 1: 0.25, 2: 0.2},
        _stats([5, 3, 1], [0.5, 0.5, 0.5]),
        min_observations=3,
        max_spread_m=None,
    )
    assert sorted(kept) == [0, 1]


def test_max_spread_drops_diffuse_clusters() -> None:
    kept = filter_clusters_by_geometry(
        {0: 0.3, 1: 0.25},
        _stats([5, 5], [0.4, 1.9]),
        min_observations=1,
        max_spread_m=1.0,
    )
    assert sorted(kept) == [0]


def test_geometry_filter_runs_before_the_ratio() -> None:
    """Dropping the best cluster must re-normalise the survivors, not keep its scale."""
    kept = filter_clusters_by_geometry(
        {0: 0.30, 1: 0.24},
        _stats([1, 5], [0.5, 0.5]),
        min_observations=5,
        max_spread_m=None,
    )
    (ranking,) = rank_localization_clusters(cluster_best_sim=kept, min_similarity=0.2)
    assert ranking.cluster_id == 1
    assert ranking.match_score == pytest.approx(1.0)


# ------------------------------------------------------- end-to-end response shape


def _geo_transform() -> GeoTransform:
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(Level(value=0.0, min_altitude=-2.0, max_altitude=4.0),),
    )


def _geo_transform_with_basement() -> GeoTransform:
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(
            Level(value=-1.0, min_altitude=-4.0, max_altitude=-2.0),
            Level(value=0.0, min_altitude=-1.0, max_altitude=4.0),
        ),
    )


def _candidate(
    candidate_id: int,
    eus_xyz: tuple[float, float, float],
    keyframe_id: int,
    similarity: float,
    level: int | None = 0,
    keyframe_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> EnrichedCandidate:
    pose = Pose.from_position_orientation(
        vector3.from_xyz(*keyframe_origin), quaternion.identity()
    )
    return EnrichedCandidate(
        id=candidate_id,
        similarity=similarity,
        eus_xyz=eus_xyz,
        lat=48.8566,
        lng=2.3522,
        alt=36.0,
        level=level,
        video_keyframe_id=keyframe_id,
        theta_center=0.1,
        phi_center=-0.2,
        geokeyframe_pose=pose,
        thumbnail="thumbs/1.jpg",
        angular_width=1.0,
        angular_height=0.8,
        vkf_lat=48.8565,
        vkf_lng=2.3521,
        vkf_alt=35.5,
        vkf_level=level,
        video_keyframe_filename="kf.jpg",
        video_keyframe_heading=12.0,
        video_keyframe_depth="kf.tif",
    )


def test_response_has_the_shape_the_benchmark_parses() -> None:
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.9),
        _candidate(2, (0.3, 0.5, 0.0), keyframe_id=11, similarity=0.85),
    ]
    response = build_localize_response(
        candidates,
        _geo_transform(),
        params=LocalizationParams(min_keyframes_per_cluster=2),
        time_embedding_ms=12,
        time_retrieval_ms=34,
    )

    assert set(response) == {"localizations", "time_embedding_ms", "time_retrieval_ms"}
    assert response["time_embedding_ms"] == 12
    assert response["time_retrieval_ms"] == 34

    (localization,) = response["localizations"]
    # The benchmark reads coordinates as [lat, lng, alt] and scores on match_score.
    lat, lng, alt = localization["coordinates"]
    assert 48.0 < lat < 49.0
    assert 2.0 < lng < 3.0
    assert isinstance(alt, float)
    assert 0.0 <= localization["match_score"] <= 1.0
    assert localization["observation_count"] == 2
    assert localization["keyframe_ids"] == ["10", "11"]

    observation = localization["observations"][0]
    # The placeholder bbox is part of the contract (livemap overlay is disabled).
    assert observation["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert len(observation["quaternion"]) == 4
    assert 0.0 <= observation["heading"] < 360.0


def test_cdog_and_ray_centroid_are_wired_from_candidate_poses() -> None:
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    candidates = [
        _candidate(
            1,
            (0.5, 0.0, 0.0),
            keyframe_id=10,
            similarity=0.9,
            keyframe_origin=(0.0, 0.0, 0.0),
        ),
        _candidate(
            2,
            (1.0, 1.0, 0.0),
            keyframe_id=11,
            similarity=0.85,
            keyframe_origin=(1.0, -1.0, 0.0),
        ),
    ]
    geo_transform = _geo_transform()
    (localization,) = localize_from_enriched_candidates(
        candidates,
        geo_transform,
        LocalizationParams(
            association="cdog",
            cdog_delta=0.0,
            centroid_from="rays",
            min_keyframes_per_cluster=2,
        ),
    )
    expected = geo_transform.local_position_to_wgs84(
        vector3.from_xyz(float(target[0]), float(target[1]), float(target[2]))
    )
    np.testing.assert_allclose(
        localization["coordinates"],
        [expected.lat, expected.lng, expected.alt],
        atol=1e-6,
    )


def test_explicit_leader_canopy_reproduces_default_fixture() -> None:
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.9),
        _candidate(2, (0.3, 0.5, 0.0), keyframe_id=11, similarity=0.85),
        _candidate(3, (5.0, 0.5, 0.0), keyframe_id=12, similarity=0.8),
    ]
    baseline = localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        LocalizationParams(min_keyframes_per_cluster=1),
    )
    explicit = localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        LocalizationParams(association="leader_canopy", min_keyframes_per_cluster=1),
    )

    assert explicit == baseline


def test_returned_cluster_labels_match_localization_membership() -> None:
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.9),
        _candidate(2, (0.3, 0.5, 0.0), keyframe_id=11, similarity=0.85),
        _candidate(3, (5.0, 0.5, 0.0), keyframe_id=12, similarity=0.8),
    ]
    params = LocalizationParams(
        min_keyframes_per_cluster=1,
        max_observations_per_cluster=len(candidates),
        num_results=len(candidates),
    )

    localizations, selected, cluster_labels = localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        params,
        return_cluster_labels=True,
    )
    assert localizations == localize_from_enriched_candidates(
        candidates, _geo_transform(), params
    )
    response_cluster_by_detection = {
        int(observation["object_idx"]): response_cluster
        for response_cluster, localization in enumerate(localizations)
        for observation in localization["observations"]
    }

    assert [candidate.id for candidate in selected] == [1, 2, 3]
    for left_index, left in enumerate(selected):
        for right_index, right in enumerate(selected):
            same_returned_cluster = (
                response_cluster_by_detection[left.id]
                == response_cluster_by_detection[right.id]
            )
            assert same_returned_cluster == (
                int(cluster_labels[left_index]) == int(cluster_labels[right_index])
            )


def test_no_candidates_gives_no_localizations() -> None:
    response = build_localize_response([], _geo_transform())
    assert response["localizations"] == []


def test_basement_and_ground_floor_detections_at_same_spot_do_not_merge() -> None:
    candidates = [
        _candidate(1, (0.0, -3.0, 0.0), keyframe_id=10, similarity=0.9, level=-1),
        _candidate(2, (0.0, -3.0, 0.0), keyframe_id=11, similarity=0.8, level=0),
    ]
    localizations = localize_from_enriched_candidates(
        candidates,
        _geo_transform_with_basement(),
        LocalizationParams(min_keyframes_per_cluster=1),
    )
    assert len(localizations) == 2
    assert {localization["level"] for localization in localizations} == {-1, 0}


def test_basement_cluster_reports_minus_one_level() -> None:
    candidates = [
        _candidate(1, (0.0, -3.0, 0.0), keyframe_id=10, similarity=0.9, level=-1),
        _candidate(2, (0.1, -3.0, 0.0), keyframe_id=11, similarity=0.8, level=-1),
    ]
    response = build_localize_response(candidates, _geo_transform_with_basement())
    (localization,) = response["localizations"]
    assert localization["level"] == -1


def test_genuinely_unresolved_cluster_reports_null_level() -> None:
    candidates = [
        _candidate(1, (0.0, 20.0, 0.0), keyframe_id=10, similarity=0.9, level=None),
        _candidate(2, (0.1, 20.0, 0.0), keyframe_id=11, similarity=0.8, level=None),
    ]
    response = build_localize_response(candidates, _geo_transform_with_basement())
    (localization,) = response["localizations"]
    assert localization["level"] is None


def test_observations_per_cluster_are_capped_and_sorted_by_similarity() -> None:
    candidates = [
        _candidate(i, (0.0, 0.5, 0.0), keyframe_id=10 + i, similarity=0.5 + i / 100)
        for i in range(8)
    ]
    localizations = localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        LocalizationParams(max_observations_per_cluster=3),
    )
    (localization,) = localizations
    assert localization["observation_count"] == 8, "count reflects the whole cluster"
    observations = localization["observations"]
    assert len(observations) == 3, "but only max_observations_per_cluster are returned"
    scores = [o["similarity_score"] for o in observations]
    assert scores == sorted(scores, reverse=True)


def test_num_results_caps_the_localization_count() -> None:
    # Three well-separated pairs → three clusters, capped to two.
    candidates = []
    for cluster_index, x in enumerate((0.0, 100.0, 200.0)):
        for k in range(2):
            candidates.append(
                _candidate(
                    cluster_index * 10 + k,
                    (x, 0.5, 0.0),
                    keyframe_id=100 + cluster_index * 10 + k,
                    similarity=0.9 - cluster_index / 10,
                )
            )
    localizations = localize_from_enriched_candidates(
        candidates, _geo_transform(), LocalizationParams(num_results=2)
    )
    assert len(localizations) == 2


def _geo_transform_without_a_band_at_the_origin() -> GeoTransform:
    """A georef whose only level band excludes the test clusters' altitude.

    `compute_cluster_statistics` re-resolves a cluster left at `UNRESOLVED_LEVEL` from
    its centroid's local-up coordinate. With the usual fixture that fallback rescues
    every unresolved cluster to level 0, which hides what the level *strategy* picked.
    Putting the band out of reach isolates the strategy, which is what these two tests
    are about.
    """
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(Level(value=0.0, min_altitude=10.0, max_altitude=20.0),),
    )


def test_median_level_strategy_outvotes_the_seed_detection() -> None:
    # Same spot, four detections. The highest-similarity one carries no resolved level;
    # the other three agree on 0. `"seed"` follows the outlier, `"median"` the majority.
    # The outlier has to be UNRESOLVED for the cluster to form at all:
    # `_levels_compatible` refuses to merge two *resolved* levels, so the median can
    # only ever differ from the seed where UNRESOLVED is involved.
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.95, level=None),
        _candidate(2, (0.1, 0.5, 0.0), keyframe_id=11, similarity=0.90, level=0),
        _candidate(3, (0.2, 0.5, 0.0), keyframe_id=12, similarity=0.85, level=0),
        _candidate(4, (0.3, 0.5, 0.0), keyframe_id=13, similarity=0.80, level=0),
    ]
    geo_transform = _geo_transform_without_a_band_at_the_origin()
    (seeded,) = localize_from_enriched_candidates(
        candidates, geo_transform, LocalizationParams(min_keyframes_per_cluster=1)
    )
    (median,) = localize_from_enriched_candidates(
        candidates,
        geo_transform,
        LocalizationParams(min_keyframes_per_cluster=1, level_strategy="median"),
    )
    # UNRESOLVED_LEVEL is serialized as None (see `build_localize_response`).
    assert seeded["level"] is None, "production follows the seed detection"
    assert median["level"] == 0, "the median follows the three agreeing detections"


def test_median_level_takes_the_lower_middle_value_on_an_even_count() -> None:
    # Levels are ordinal. Two detections on 2 and two unresolved must resolve to an
    # observed value, not to their average. Asserted on the statistics rather than the
    # response, which maps UNRESOLVED_LEVEL to None.
    positions = np.array(
        [[0.0, 0.5, 0.0], [0.1, 0.5, 0.0], [0.2, 0.5, 0.0], [0.3, 0.5, 0.0]],
        dtype=np.float64,
    )
    stats = compute_cluster_statistics(
        positions,
        np.zeros(4, dtype=np.int32),
        np.array([0.95, 0.90, 0.85, 0.80], dtype=np.float64),
        np.array([2, 2, UNRESOLVED_LEVEL, UNRESOLVED_LEVEL], dtype=np.int32),
        np.array([10, 11, 12, 13], dtype=np.int64),
        _geo_transform_without_a_band_at_the_origin(),
        level_strategy="median",
    )
    assert int(stats.cluster_levels[0]) == UNRESOLVED_LEVEL


def test_median_level_gives_one_vote_per_keyframe_not_per_detection() -> None:
    # One keyframe on an unresolved level contributes three detections; two keyframes
    # on level 0 contribute one each. Per detection the vote is 3-2 and the answer is
    # sentinel; per keyframe it is 1-2 and the answer is 0. A level is a property of the
    # camera pose, so the keyframe count is what may decide it.
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.95, level=None),
        _candidate(2, (0.1, 0.5, 0.0), keyframe_id=10, similarity=0.94, level=None),
        _candidate(3, (0.2, 0.5, 0.0), keyframe_id=10, similarity=0.93, level=None),
        _candidate(4, (0.3, 0.5, 0.0), keyframe_id=11, similarity=0.90, level=0),
        _candidate(5, (0.4, 0.5, 0.0), keyframe_id=12, similarity=0.85, level=0),
    ]
    (median,) = localize_from_enriched_candidates(
        candidates,
        _geo_transform_without_a_band_at_the_origin(),
        LocalizationParams(min_keyframes_per_cluster=1, level_strategy="median"),
    )
    assert median["level"] == 0


def test_median_level_never_falls_back_to_the_projected_position() -> None:
    # No keyframe resolves a level, and the cluster sits squarely inside level 0's
    # altitude band. `"seed"` resolves it from that band — i.e. from the depth-projected
    # centroid; `"median"` must leave it unresolved, the floor being a keyframe property
    # under that strategy.
    candidates = [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.95, level=None),
        _candidate(2, (0.1, 0.5, 0.0), keyframe_id=11, similarity=0.90, level=None),
    ]
    (seeded,) = localize_from_enriched_candidates(
        candidates, _geo_transform(), LocalizationParams(min_keyframes_per_cluster=1)
    )
    (median,) = localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        LocalizationParams(min_keyframes_per_cluster=1, level_strategy="median"),
    )
    assert seeded["level"] == 0, "production resolves it from the centroid altitude"
    assert median["level"] is None, "the median reads keyframe poses only"


def test_depth_cap_drops_only_detections_beyond_it() -> None:
    # Depth is the distance from the keyframe pose to the object position, so these
    # sit at 1 m and 12 m from their respective cameras.
    near = _candidate(1, (1.0, 0.0, 0.0), keyframe_id=10, similarity=0.9)
    far = _candidate(
        2,
        (12.0, 0.0, 0.0),
        keyframe_id=11,
        similarity=0.95,
        keyframe_origin=(0.0, 0.0, 0.0),
    )

    assert localize.detection_depths([near, far]).tolist() == [
        pytest.approx(1.0),
        pytest.approx(12.0),
    ]
    assert localize._filter_by_max_depth([near, far], None) == [near, far]
    assert localize._filter_by_max_depth([near, far], 15.0) == [near, far]
    assert localize._filter_by_max_depth([near, far], 5.0) == [near]


def test_depth_cap_measures_depth_from_each_own_keyframe() -> None:
    # Same object position, two cameras: the cap is per detection, not a distance to
    # the origin, so only the far observer is dropped.
    close_observer = _candidate(
        1,
        (10.0, 0.0, 0.0),
        keyframe_id=10,
        similarity=0.9,
        keyframe_origin=(9.0, 0.0, 0.0),
    )
    far_observer = _candidate(
        2,
        (10.0, 0.0, 0.0),
        keyframe_id=11,
        similarity=0.9,
        keyframe_origin=(0.0, 0.0, 0.0),
    )

    kept = localize._filter_by_max_depth([close_observer, far_observer], 5.0)

    assert kept == [close_observer]


def test_depth_cap_can_empty_the_selection_without_raising() -> None:
    candidates = [_candidate(1, (30.0, 0.0, 0.0), keyframe_id=10, similarity=0.9)]

    assert (
        localize_from_enriched_candidates(
            candidates, _geo_transform(), LocalizationParams(max_depth_m=5.0)
        )
        == []
    )


def test_depth_cap_rejects_a_non_positive_value() -> None:
    with pytest.raises(ValueError, match="max_depth_m"):
        localize._filter_by_max_depth(
            [_candidate(1, (1.0, 0.0, 0.0), keyframe_id=10, similarity=0.9)], 0.0
        )


def test_depth_cap_before_selection_backfills_from_deeper_in_the_ranking() -> None:
    """`before_select` keeps `candidate_count` usable detections;

    `after_select` does not."""
    near_first = _candidate(1, (1.0, 0.0, 0.0), keyframe_id=10, similarity=0.90)
    far = _candidate(2, (30.0, 0.0, 0.0), keyframe_id=11, similarity=0.85)
    near_last = _candidate(3, (2.0, 0.0, 0.0), keyframe_id=12, similarity=0.10)
    retrieved = [near_first, far, near_last]

    after = localize._filter_by_max_depth(
        localize.select_top_candidates(retrieved, 2), 15.0
    )
    before = localize.select_top_candidates(
        localize._filter_by_max_depth(retrieved, 15.0), 2
    )

    # Same cap, same retrieved set: capping after the truncation leaves one detection,
    # capping before it pulls the low-similarity near one up into the working set.
    assert [c.id for c in after] == [1]
    assert [c.id for c in before] == [1, 3]
