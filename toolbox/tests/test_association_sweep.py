from __future__ import annotations

from contextlib import nullcontext
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
