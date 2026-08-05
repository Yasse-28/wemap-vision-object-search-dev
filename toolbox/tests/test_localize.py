"""Tests for the ported clustering, ranking and response shape.

`localize.py` differs from `backend/object_search/v1_5_logic.py` only in its import
lines, so these tests are really pinning down *production* behaviour. They exist
because the backend has no tests for that module — the port is the first chance to
add them.

Two things are asserted deliberately rather than incidentally:

- the **ranking weights** (0.50 similarity / 0.15 confidence / 0.35 keyframes), so a
  well-meaning tweak here shows up as a failing test rather than as silently
  different result ordering than production;
- the **response shape**, because `toolbox/benchmark/object_search_http_benchmark.py`
  parses `localizations[].coordinates` and `match_score` directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.localize import (
    UNRESOLVED_LEVEL,
    LocalizationParams,
    build_localize_response,
    cluster_detections_leader_canopy,
    filter_clusters_by_min_keyframes,
    localize_from_enriched_candidates,
    rank_localization_clusters,
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


# ---------------------------------------------------------------------- ranking


def test_match_score_uses_the_production_weights() -> None:
    """match_score = 0.50·normalised_similarity + 0.15·confidence + 0.35·keyframes.

    keyframe_score saturates at 3 distinct keyframes.
    """
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.9},
        cluster_confidence={0: 1.0},
        cluster_keyframes={0: {"a", "b", "c"}},
        min_similarity=0.2,
    )
    (ranking,) = rankings
    # Single eligible cluster ⇒ normalised similarity is 1.0 by construction.
    assert ranking.normalized_similarity == pytest.approx(1.0)
    assert ranking.match_score == pytest.approx(0.50 * 1.0 + 0.15 * 1.0 + 0.35 * 1.0)


def test_keyframe_score_saturates_at_three() -> None:
    def score(n_keyframes: int) -> float:
        (ranking,) = rank_localization_clusters(
            cluster_best_sim={0: 0.9},
            cluster_confidence={0: 0.0},
            cluster_keyframes={0: {str(i) for i in range(n_keyframes)}},
            min_similarity=0.2,
        )
        return ranking.match_score

    assert score(1) == pytest.approx(0.50 + 0.35 / 3.0)
    assert score(3) == pytest.approx(0.50 + 0.35)
    assert score(10) == score(3), "keyframe_score must cap at 1.0"


def test_clusters_below_min_similarity_are_dropped() -> None:
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.9, 1: 0.05},
        cluster_confidence={},
        cluster_keyframes={},
        min_similarity=0.2,
    )
    assert [r.cluster_id for r in rankings] == [0]


def test_rankings_are_sorted_by_match_score_descending() -> None:
    rankings = rank_localization_clusters(
        cluster_best_sim={0: 0.5, 1: 0.9, 2: 0.7},
        cluster_confidence={0: 0.0, 1: 0.0, 2: 0.0},
        cluster_keyframes={0: {"a"}, 1: {"a"}, 2: {"a"}},
        min_similarity=0.2,
    )
    scores = [r.match_score for r in rankings]
    assert scores == sorted(scores, reverse=True)
    assert rankings[0].cluster_id == 1


# ------------------------------------------------------- end-to-end response shape


def _geo_transform() -> GeoTransform:
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(Level(value=0.0, min_altitude=-2.0, max_altitude=4.0),),
    )


def _candidate(
    candidate_id: int,
    eus_xyz: tuple[float, float, float],
    keyframe_id: int,
    similarity: float,
    level: int | None = 0,
) -> EnrichedCandidate:
    pose = Pose.from_position_orientation(
        vector3.from_xyz(0.0, 0.0, 0.0), quaternion.identity()
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


def test_no_candidates_gives_no_localizations() -> None:
    response = build_localize_response([], _geo_transform())
    assert response["localizations"] == []


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
