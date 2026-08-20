"""Unit tests for the typed error decomposition."""

from __future__ import annotations

import pytest

from toolbox.benchmark.error_decomposition import (
    decompose,
    separability,
    type_prompt_predictions,
)
from toolbox.benchmark.object_search_http_benchmark import Annotation, Prediction

#: One ten-thousandth of a degree of latitude, near enough 1.11 m.
DEGREE_M = 1e-5


def _annotation(
    identifier: str, prompt: str, north_m: float, extent_m: float | None = 1.0
) -> Annotation:
    return Annotation(
        id=identifier,
        class_name=prompt,
        prompt=prompt,
        lat=48.0 + north_m * DEGREE_M / 1.11,
        lng=2.0,
        accuracy_m=5.0,
        extent_m=extent_m,
        object_id=f"object-{identifier}",
    )


def _prediction(identifier: str, north_m: float, score: float) -> Prediction:
    return Prediction(
        id=identifier,
        lat=48.0 + north_m * DEGREE_M / 1.11,
        lng=2.0,
        score=score,
    )


def _verdicts(predictions: list[Prediction]) -> list[str]:
    own = [_annotation("a1", "chaise", 0.0)]
    others = [_annotation("a2", "table", 3.0)]
    verdicts, _ = type_prompt_predictions(predictions, own, others)
    return [verdicts[prediction.id] for prediction in predictions]


def test_a_cluster_on_the_object_is_correct() -> None:
    assert _verdicts([_prediction("p1", 0.0, 0.9)]) == ["correct"]


def test_the_second_cluster_on_one_object_is_a_duplicate() -> None:
    # Both reach the object; the walk follows the ranking, so the better one takes it.
    assert _verdicts([_prediction("p1", 0.0, 0.9), _prediction("p2", 0.1, 0.8)]) == [
        "correct",
        "duplicate",
    ]


def test_the_duplicate_is_the_worse_ranked_one_whatever_the_order_given() -> None:
    verdicts = _verdicts([_prediction("p1", 0.1, 0.5), _prediction("p2", 0.0, 0.9)])

    # p1 is listed first but scores lower, so it is the duplicate.
    assert verdicts == ["duplicate", "correct"]


def test_a_cluster_beside_its_object_is_a_localisation_error() -> None:
    # Radius is extent/2 = 0.5 m; the localisation band reaches twice that.
    assert _verdicts([_prediction("p1", 0.7, 0.9)]) == ["localisation"]


def test_a_cluster_on_another_class_is_a_classification_error() -> None:
    assert _verdicts([_prediction("p1", 3.0, 0.9)]) == ["classification"]


def test_a_cluster_on_nothing_is_a_background_error() -> None:
    assert _verdicts([_prediction("p1", 25.0, 0.9)]) == ["background"]


def test_an_object_no_cluster_reached_is_missed() -> None:
    own = [_annotation("a1", "chaise", 0.0), _annotation("a2", "chaise", 10.0)]
    _, missed = type_prompt_predictions([_prediction("p1", 0.0, 0.9)], own, [])

    assert missed == ["a2"]


def test_junk_ranked_above_the_answer_costs_the_cutoff_and_nothing_else() -> None:
    # The signature of a pure ranking failure, and the reason both axes are reported:
    # the object *is* returned, just below two background clusters.
    chaise = _annotation("a1", "chaise", 0.0)
    decomposition = decompose(
        {
            "chaise": [
                _prediction("p1", 25.0, 0.99),
                _prediction("p2", 35.0, 0.98),
                _prediction("p3", 0.0, 0.50),
            ]
        },
        [chaise],
        k=2,
    )
    background = decomposition.by_type()["background"]

    assert background.count == 2
    assert background.delta_recall_at_k == pytest.approx(1.0)
    assert background.delta_recall_all == pytest.approx(0.0)


def test_a_lost_object_costs_both_axes() -> None:
    chaise = _annotation("a1", "chaise", 0.0)
    lost = _annotation("a2", "chaise", 40.0)
    decomposition = decompose(
        {"chaise": [_prediction("p1", 0.0, 0.9)]}, [chaise, lost], k=10
    )
    missed = decomposition.by_type()["missed"]

    assert missed.count == 1
    assert missed.delta_recall_at_k == pytest.approx(0.5)
    assert missed.delta_recall_all == pytest.approx(0.5)


def test_classes_further_apart_than_their_radius_are_separable() -> None:
    gate = separability(
        [_annotation("a1", "chaise", 0.0), _annotation("a2", "table", 3.0)]
    )

    assert gate.overlap_share == 0.0
    assert gate.classification_measurable
    assert gate.reason is None


def test_a_five_metre_radius_on_neighbouring_classes_is_refused() -> None:
    # What today's ground truth looks like: no extent, so the radius is `accuracy_m`.
    gate = separability(
        [
            _annotation("a1", "chaise", 0.0, extent_m=None),
            _annotation("a2", "table", 1.0, extent_m=None),
        ]
    )

    assert gate.overlap_share == 1.0
    assert not gate.classification_measurable
    assert gate.reason is not None


def test_a_refused_gate_withholds_only_the_two_classification_types() -> None:
    chaise = _annotation("a1", "chaise", 0.0, extent_m=None)
    table = _annotation("a2", "table", 1.0, extent_m=None)
    decomposition = decompose(
        {"chaise": [_prediction("p1", 25.0, 0.9)], "table": []}, [chaise, table]
    )
    types = decomposition.by_type()

    assert not types["classification"].measurable
    assert not types["classification_localisation"].measurable
    assert types["background"].measurable
    assert types["missed"].measurable


def test_a_single_class_ground_truth_cannot_confuse_anything() -> None:
    gate = separability([_annotation("a1", "chaise", 0.0)])

    assert gate.overlap_share == 0.0
    assert gate.classification_measurable


def test_different_levels_are_not_confusable_neighbours() -> None:
    ground = _annotation("a1", "chaise", 0.0, extent_m=None)
    upstairs = Annotation(
        id="a2",
        class_name="table",
        prompt="table",
        lat=ground.lat,
        lng=ground.lng,
        accuracy_m=5.0,
        level="1",
    )
    same_floor = Annotation(
        id="a3",
        class_name="table",
        prompt="table",
        lat=ground.lat,
        lng=ground.lng,
        accuracy_m=5.0,
        level="0",
    )

    on_top = Annotation(**{**vars(ground), "level": "0"})
    assert separability([on_top, upstairs]).overlap_share == 0.0
    assert separability([on_top, same_floor]).overlap_share == 1.0
