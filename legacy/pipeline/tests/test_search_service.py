import numpy as np

from pipeline.core.types import (
    UNRESOLVED_LEVEL_SENTINEL,
    LoadedIndex,
    ObjectSearchIndexMetadata,
    default_created_utc,
)
from pipeline.online.localize_3d import (
    _top_candidate_indices,
    cluster_ocr_scores_from_summaries,
    rank_localization_clusters,
)
from pipeline.online.search_service import (
    _cluster_level,
    _heading_to_north_deg,
    _merge_cutout_results_by_keyframe,
)


def _build_index() -> LoadedIndex:
    metadata = ObjectSearchIndexMetadata(
        projection_dim=2,
        created_utc=default_created_utc(),
        cutout_count=3,
        object_count=2,
    )
    return LoadedIndex(
        metadata=metadata,
        cutout_embeddings=np.zeros((3, 2), dtype=np.float32),
        cutout_ids=np.array([1024, 1025, 2048], dtype=np.int64),
        cutout_keyframe_ids=np.array([1, 1, 2], dtype=np.int64),
        cutout_center_xy=np.zeros((3, 2), dtype=np.float32),
        cutout_rotation_cutout_to_equirect=np.zeros((3, 4, 4), dtype=np.float32),
        object_embeddings=np.zeros((2, 2), dtype=np.float32),
        object_keyframe_ids=np.array([1, 2], dtype=np.int64),
        object_cutout_ids=np.array([1025, 2048], dtype=np.int64),
        object_bboxes=np.zeros((2, 4), dtype=np.float32),
    )


def test_merge_cutout_results_by_keyframe_keeps_best_score_per_keyframe():
    index = _build_index()

    merged = _merge_cutout_results_by_keyframe(
        [
            ("1024", 0.70),
            ("1025", 0.92),
            ("2048", 0.81),
        ],
        index,
        limit=10,
    )

    assert merged == [("1", 0.92), ("2", 0.81)]


def test_merge_cutout_results_by_keyframe_ignores_unknown_cutout_ids():
    index = _build_index()

    merged = _merge_cutout_results_by_keyframe(
        [("999999", 0.99), ("2048", 0.81)],
        index,
        limit=10,
    )

    assert merged == [("2", 0.81)]


def test_top_candidate_indices_returns_sorted_best_matches():
    similarities = np.array([0.1, 0.8, 0.4, 0.9], dtype=np.float32)

    indices = _top_candidate_indices(similarities, candidate_count=3)

    assert indices.tolist() == [3, 1, 2]


def test_heading_to_north_deg_matches_expected_cardinal_directions():
    north = _heading_to_north_deg(
        origin_lat=45.0,
        origin_lon=6.0,
        target_lat=45.001,
        target_lon=6.0,
    )
    east = _heading_to_north_deg(
        origin_lat=45.0,
        origin_lon=6.0,
        target_lat=45.0,
        target_lon=6.001,
    )
    west = _heading_to_north_deg(
        origin_lat=45.0,
        origin_lon=6.0,
        target_lat=45.0,
        target_lon=5.999,
    )

    assert np.isclose(north, 0.0, atol=1e-3)
    assert np.isclose(east, 90.0, atol=1e-3)
    assert np.isclose(west, -90.0, atol=1e-3)


def test_rank_localization_clusters_filters_low_absolute_similarity():
    ranked = rank_localization_clusters(
        cluster_best_sim={0: 0.14, 1: 0.16},
        cluster_confidence={0: 1.0, 1: 0.2},
        cluster_keyframes={0: {"10", "11"}, 1: {"20", "21"}},
        min_similarity=0.15,
    )

    assert [r.cluster_id for r in ranked] == [1]


def test_rank_localization_clusters_uses_spatial_support_in_match_score():
    ranked = rank_localization_clusters(
        cluster_best_sim={0: 0.30, 1: 0.29},
        cluster_confidence={0: 0.1, 1: 1.0},
        cluster_keyframes={0: {"10"}, 1: {"20", "21", "22"}},
        min_similarity=0.15,
    )

    assert ranked[0].cluster_id == 1
    assert ranked[0].match_score > ranked[1].match_score


def test_rank_localization_clusters_boosts_matching_ocr_cluster():
    ranked = rank_localization_clusters(
        cluster_best_sim={0: 0.92, 1: 0.91, 2: 0.90},
        cluster_confidence={0: 0.8, 1: 0.8, 2: 0.8},
        cluster_keyframes={
            0: {"10", "11"},
            1: {"20", "21"},
            2: {"30", "31"},
        },
        cluster_ocr_scores={0: 0.0, 1: 1.0, 2: 0.0},
        min_similarity=0.15,
    )

    assert [r.cluster_id for r in ranked] == [1, 0, 2]
    assert ranked[0].ocr_score == 1.0


def test_rank_localization_clusters_keeps_no_ocr_cluster_eligible():
    ranked = rank_localization_clusters(
        cluster_best_sim={0: 0.92, 1: 0.91},
        cluster_confidence={0: 0.8, 1: 0.8},
        cluster_keyframes={0: {"10", "11"}, 1: {"20", "21"}},
        cluster_ocr_scores={0: 0.0, 1: 1.0},
        min_similarity=0.15,
    )

    assert {r.cluster_id for r in ranked} == {0, 1}
    assert ranked[0].cluster_id == 1


def test_rank_localization_clusters_without_ocr_scores_keeps_existing_formula():
    ranked = rank_localization_clusters(
        cluster_best_sim={0: 0.30, 1: 0.29},
        cluster_confidence={0: 0.1, 1: 1.0},
        cluster_keyframes={0: {"10"}, 1: {"20", "21", "22"}},
        min_similarity=0.15,
    )

    assert ranked[0].cluster_id == 1
    assert ranked[0].ocr_score == 0.0


def test_cluster_ocr_scores_from_summaries_returns_none_for_missing_arrays():
    scores = cluster_ocr_scores_from_summaries(
        query_text="repère de quai 16",
        cluster_ids={0, 1},
        cluster_ocr_keys=None,
        cluster_ocr_tokens=None,
        cluster_ocr_texts=None,
    )

    assert scores is None


def test_cluster_ocr_scores_from_summaries_ignores_non_identity_prompt():
    scores = cluster_ocr_scores_from_summaries(
        query_text="bench",
        cluster_ids={0, 1},
        cluster_ocr_keys=np.asarray(["numbers=16", "numbers=17"]),
        cluster_ocr_tokens=np.asarray(["16", "17"]),
        cluster_ocr_texts=None,
    )

    assert scores is None


def test_cluster_level_returns_negative_levels_when_valid():
    index = _build_index()
    index.cluster_levels = np.array([0, 2, -1], dtype=np.int32)

    assert _cluster_level(index, 0) == 0
    assert _cluster_level(index, 1) == 2
    assert _cluster_level(index, 2) == -1
    assert _cluster_level(index, 99) is None


def test_cluster_level_returns_none_for_unresolved_sentinel():
    index = _build_index()
    index.cluster_levels = np.array([UNRESOLVED_LEVEL_SENTINEL], dtype=np.int32)

    assert _cluster_level(index, 0) is None
