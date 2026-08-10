"""Tests for the ported clustering, ranking and response shape.

These tests pin the production-compatible default path plus the deliberate dev-only
divergences documented in `AI_CONTEXT/bricks.md`. They exist because the backend has
no tests for that module — the port is the first chance to add them.

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
    compute_cluster_statistics,
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
