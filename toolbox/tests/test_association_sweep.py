from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from toolbox.benchmark import association_sweep
from toolbox.benchmark.association_sweep import (
    _params_from_grid_entry,
    fetch_prompt_candidates,
    nearest_annotation_labels,
    partition_metrics,
    shared_threshold_metrics,
)
from toolbox.benchmark.object_search_http_benchmark import Annotation, Prediction
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.localize import LocalizationParams
from toolbox.bricks.vendored.geo_transform import Pose
from toolbox.bricks.vendored.maths import quaternion, vector3


def _annotation(prompt: str) -> Annotation:
    return Annotation(
        id=f"target-{prompt}",
        class_name=prompt,
        prompt=prompt,
        lat=48.0,
        lng=2.0,
        accuracy_m=5.0,
    )


def _prediction(identifier: str, score: float, *, is_match: bool) -> Prediction:
    return Prediction(
        id=identifier,
        lat=48.0 if is_match else 49.0,
        lng=2.0,
        score=score,
    )


def test_shared_threshold_and_loo_are_fitted_across_prompts() -> None:
    annotations = {"a": [_annotation("a")], "b": [_annotation("b")]}
    predictions = {
        "a": [
            _prediction("a-tp", 0.9, is_match=True),
            _prediction("a-fp", 0.8, is_match=False),
        ],
        "b": [
            _prediction("b-fp", 0.9, is_match=False),
            _prediction("b-tp", 0.7, is_match=True),
        ],
    }

    metrics = shared_threshold_metrics(
        predictions, annotations, grouped=False, group_radius_m=2.0
    )

    # At 0.7 both prompts have F1=2/3. In LOO, b fits 0.7 for held-out a
    # (F1=2/3), while a fits 0.9 for held-out b (F1=0).
    assert metrics.threshold == pytest.approx(0.7)
    assert metrics.macro_f1 == pytest.approx(2.0 / 3.0)
    assert metrics.loo_macro_f1 == pytest.approx(1.0 / 3.0)


def test_candidate_cache_round_trip_skips_ann_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    map_path = tmp_path / "map"
    cache_dir = tmp_path / "cache"
    annotation = _annotation("lamp")
    calls = {"ann": 0, "enrich": 0, "connect": 0}
    cached_candidate = cast(EnrichedCandidate, SimpleNamespace(id=7, similarity=0.42))

    monkeypatch.setattr(
        association_sweep.map_manifest,
        "load_map_manifest",
        lambda _path: SimpleNamespace(
            path=map_path / "map_2_20260811_120000.json",
            map_id="map",
            geo_ref_id=42,
        ),
    )
    monkeypatch.setattr(
        association_sweep.georef_source,
        "load_pose_source",
        lambda _path: SimpleNamespace(geo_transform=object()),
    )
    monkeypatch.setattr(
        association_sweep,
        "load_annotations",
        lambda _path, _accuracy: [annotation],
    )

    def fake_connect() -> Any:
        calls["connect"] += 1
        return nullcontext(object())

    def fake_query(*_args: Any) -> list[dict[str, Any]]:
        calls["ann"] += 1
        return []

    def fake_enrich(*_args: Any, **_kwargs: Any) -> list[EnrichedCandidate]:
        calls["enrich"] += 1
        return [cached_candidate]

    monkeypatch.setattr(association_sweep.db, "connect", fake_connect)
    monkeypatch.setattr(association_sweep.service, "query_by_text", fake_query)
    monkeypatch.setattr(
        association_sweep.candidates, "load_enriched_candidates", fake_enrich
    )

    first = fetch_prompt_candidates(map_path, "http://ann", 1000, cache_dir)
    second = fetch_prompt_candidates(map_path, "http://ann", 1000, cache_dir)

    assert first == second == {"lamp": [cached_candidate]}
    assert calls == {"ann": 1, "enrich": 1, "connect": 1}
    assert len(list(cache_dir.glob("*.pickle"))) == 1


def test_unknown_grid_parameter_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown_parameter"):
        _params_from_grid_entry(
            {"label": "typo", "unknown_parameter": 3}, LocalizationParams()
        )


def test_partition_metrics_penalize_over_merging_and_over_splitting() -> None:
    annotation_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    over_merged = partition_metrics(
        np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int32), annotation_labels
    )
    over_split = partition_metrics(
        np.asarray([0, 0, 1, 2, 2, 3], dtype=np.int32), annotation_labels
    )

    assert over_merged.pair_precision == pytest.approx(2.0 / 5.0)
    assert over_merged.pair_recall == pytest.approx(1.0)
    assert over_split.pair_precision == pytest.approx(1.0)
    assert over_split.pair_recall == pytest.approx(1.0 / 3.0)
    assert over_merged.pair_f1 == pytest.approx(4.0 / 7.0)
    assert over_split.pair_f1 == pytest.approx(0.5)


def test_partition_metrics_exclude_unlabelled_detections() -> None:
    metrics = partition_metrics(
        np.asarray([0, 0, 0, 1], dtype=np.int32),
        np.asarray([0, 0, -1, -1], dtype=np.int64),
    )

    assert metrics.labelled_detections == 2
    assert metrics.pair_precision == pytest.approx(1.0)
    assert metrics.pair_recall == pytest.approx(1.0)
    assert metrics.rand_index == pytest.approx(1.0)


def test_no_nearby_annotations_yields_zero_pair_metrics() -> None:
    detections = [
        cast(EnrichedCandidate, SimpleNamespace(lat=48.0, lng=2.0)),
        cast(EnrichedCandidate, SimpleNamespace(lat=48.0, lng=2.0001)),
    ]
    distant = Annotation(
        id="far",
        class_name="lamp",
        prompt="lamp",
        lat=49.0,
        lng=2.0,
        accuracy_m=5.0,
    )

    annotation_labels = nearest_annotation_labels(detections, [distant], near_m=1.0)
    metrics = partition_metrics(np.asarray([0, 0], dtype=np.int32), annotation_labels)

    assert annotation_labels.tolist() == [-1, -1]
    assert metrics == association_sweep.PartitionMetrics(0.0, 0.0, 0.0, 0.0, 0)


def _unit(*values: float) -> np.ndarray:
    """A unit-norm embedding, so a cosine is just the dot product."""
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _feedback_candidate(
    candidate_id: int, similarity: float, pos_sim: float, neg_sim: float
) -> EnrichedCandidate:
    """A real candidate — `apply_feedback` rebuilds it with `dataclasses.replace`."""
    pose = Pose.from_position_orientation(
        vector3.from_xyz(0.0, 0.0, 0.0), quaternion.identity()
    )
    return EnrichedCandidate(
        id=candidate_id,
        similarity=similarity,
        eus_xyz=(1.0, 0.0, 0.0),
        lat=48.0,
        lng=2.0,
        alt=36.0,
        level=0,
        video_keyframe_id=10,
        theta_center=0.0,
        phi_center=0.0,
        geokeyframe_pose=pose,
        thumbnail=None,
        angular_width=1.0,
        angular_height=1.0,
        vkf_lat=48.0,
        vkf_lng=2.0,
        vkf_alt=35.0,
        vkf_level=0,
        video_keyframe_filename="kf.jpg",
        video_keyframe_heading=0.0,
        video_keyframe_depth="kf.tif",
        pos_sim=pos_sim,
        neg_sim=neg_sim,
    )


def test_feedback_off_returns_the_cached_candidates_untouched() -> None:
    cached = [_feedback_candidate(1, 0.30, 0.80, 0.10)]

    unchanged = association_sweep.apply_feedback(cached, LocalizationParams())

    # Identity, not equality: with both gains at zero the baseline must be the
    # cached objects themselves, so no recomputation can perturb it.
    assert unchanged[0] is cached[0]


def test_feedback_gains_are_swept_from_the_raw_prototype_columns() -> None:
    cached = [
        _feedback_candidate(1, 0.30, 0.90, 0.10),
        _feedback_candidate(2, 0.28, 0.70, 0.60),
    ]

    boosted = association_sweep.apply_feedback(
        cached, LocalizationParams(feedback_alpha=0.1, feedback_beta=0.2)
    )

    assert [c.similarity_boosted for c in boosted] == [
        pytest.approx(0.30 + 0.1 * 0.90 - 0.2 * 0.10),
        pytest.approx(0.28 + 0.1 * 0.70 - 0.2 * 0.60),
    ]
    # The raw columns stay raw under "none", which is what makes one cache enough.
    assert [c.pos_sim_applied for c in boosted] == [0.90, 0.70]


def test_feedback_normalization_is_applied_across_the_retrieved_set() -> None:
    cached = [
        _feedback_candidate(1, 0.30, 0.90, 0.0),
        _feedback_candidate(2, 0.28, 0.70, 0.0),
        _feedback_candidate(3, 0.26, 0.80, 0.0),
    ]

    boosted = association_sweep.apply_feedback(
        cached,
        LocalizationParams(feedback_alpha=1.0, feedback_normalization="center"),
    )

    # Median of the column (0.80) subtracted, so the middle candidate gains nothing.
    assert [c.pos_sim_applied for c in boosted] == [
        pytest.approx(0.10),
        pytest.approx(-0.10),
        pytest.approx(0.0),
    ]
    assert boosted[2].similarity_boosted == pytest.approx(0.26)


def test_feedback_cache_entries_are_separate_from_the_plain_ones(
    tmp_path: Path,
) -> None:
    plain = association_sweep._cache_path(
        tmp_path, "map", "lamp", 1000, with_feedback=False
    )
    with_feedback = association_sweep._cache_path(
        tmp_path, "map", "lamp", 1000, with_feedback=True
    )

    assert plain != with_feedback
    # And the plain digest is the one written before feedback existed, so the caches
    # already on disk (hours of ANN + enrichment) stay addressable.
    assert plain.name == (
        "map-d63a91c3680123c215d638f23f982428101b69633fe3b6b761f93780302909e8.pickle"
    )


def _prototypes(vectors: list[list[float]]) -> association_sweep.PromptPrototypes:
    array = np.asarray(vectors, dtype=np.float32)
    empty = np.empty((0, array.shape[1] if array.size else 2), dtype=np.float32)
    return association_sweep.PromptPrototypes(
        positive=array if array.size else empty,
        negative=empty,
        positive_requested=len(vectors),
        negative_requested=0,
    )


def test_identity_rescorer_reproduces_the_baseline_scores() -> None:
    cached = [
        replace(_feedback_candidate(1, 0.30, 0.0, 0.0), embedding=_unit(1.0, 0.0)),
        replace(_feedback_candidate(2, 0.28, 0.0, 0.0), embedding=_unit(0.0, 1.0)),
    ]

    rescored = association_sweep.apply_feedback(
        cached,
        LocalizationParams(rescorer="identity"),
        _prototypes([[1.0, 0.0]]),
    )

    # `rescorer` alone turns feedback on, so the boosted column is now what ranking
    # reads — identity must therefore carry the base similarity, not None. To float32
    # only: the seam's arrays are float32, which is why identity is a measured control
    # row in every comparison rather than an assumed no-op.
    assert [c.similarity_boosted for c in rescored] == [
        pytest.approx(0.30, abs=1e-7),
        pytest.approx(0.28, abs=1e-7),
    ]


def test_max_prototype_rescorer_matches_the_sql_boost() -> None:
    cached = [
        replace(_feedback_candidate(1, 0.30, 0.0, 0.0), embedding=_unit(1.0, 0.0)),
        replace(_feedback_candidate(2, 0.28, 0.0, 0.0), embedding=_unit(0.0, 1.0)),
    ]

    rescored = association_sweep.apply_feedback(
        cached,
        LocalizationParams(
            rescorer="max_prototype", rescorer_params={"alpha": 0.1, "beta": 0.0}
        ),
        _prototypes([[1.0, 0.0]]),
    )

    # Cosine to the single prototype is 1 for the first candidate and 0 for the
    # second, which is exactly what `pos_sim` would have held.
    assert rescored[0].similarity_boosted == pytest.approx(0.30 + 0.1, abs=1e-6)
    assert rescored[1].similarity_boosted == pytest.approx(0.28, abs=1e-6)


def test_rescorer_without_prototypes_fails_instead_of_doing_nothing() -> None:
    cached = [
        replace(_feedback_candidate(1, 0.30, 0.0, 0.0), embedding=_unit(1.0, 0.0))
    ]

    with pytest.raises(ValueError, match="--with-feedback"):
        association_sweep.apply_feedback(
            cached, LocalizationParams(rescorer="knn_cache"), None
        )


def test_rescorer_without_cached_embeddings_fails_loudly() -> None:
    with pytest.raises(ValueError, match="embeddings"):
        association_sweep.apply_feedback(
            [_feedback_candidate(1, 0.30, 0.0, 0.0)],
            LocalizationParams(rescorer="identity"),
            _prototypes([[1.0, 0.0]]),
        )
