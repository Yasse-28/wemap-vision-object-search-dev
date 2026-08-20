"""Unit tests for the label-set metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from toolbox.benchmark.label_set_metrics import (
    category_of,
    evaluate_label_sets,
    has_label_sets,
    rank_labels,
    report_lines,
    set_ranking,
    top_n_frequency,
    vocabulary,
)
from toolbox.benchmark.object_search_http_benchmark import Annotation


def _chair(**overrides: object) -> Annotation:
    base: dict[str, object] = {
        "id": "a1",
        "class_name": "chaise",
        "prompt": "chaise",
        "lat": 48.0,
        "lng": 2.0,
        "accuracy_m": 5.0,
        "synonyms": ("chair", "seat"),
        "depictions": ("chair pictogram",),
        "visually_similar": ("stool", "armchair"),
        "clutter": ("table",),
    }
    base.update(overrides)
    return Annotation(**base)  # type: ignore[arg-type]


def test_the_class_name_counts_as_a_synonym() -> None:
    assert category_of("chaise", _chair()) == "synonyms"
    assert category_of("seat", _chair()) == "synonyms"


def test_labels_compare_without_case_or_stray_spaces() -> None:
    assert category_of("  SEAT ", _chair()) == "synonyms"


def test_each_category_is_recognised_and_a_stranger_is_not() -> None:
    chair = _chair()

    assert category_of("chair pictogram", chair) == "depictions"
    assert category_of("stool", chair) == "visually_similar"
    assert category_of("table", chair) == "clutter"
    assert category_of("fire extinguisher", chair) is None


def test_a_label_in_two_categories_is_credited_with_the_more_precise_one() -> None:
    # An annotator who repeats a word must not be able to lower a score by accident.
    chair = _chair(clutter=("chair", "table"))

    assert category_of("chair", chair) == "synonyms"


def test_an_annotation_with_only_a_class_name_carries_no_sets() -> None:
    bare = Annotation(
        id="a1", class_name="chaise", prompt="chaise", lat=48.0, lng=2.0, accuracy_m=5.0
    )

    assert not has_label_sets(bare)
    assert has_label_sets(_chair())


def test_top_n_reports_the_categories_the_proposals_reached() -> None:
    reached = top_n_frequency(["stool", "fire extinguisher", "seat"], _chair(), n=2)

    # `seat` sits third and the cutoff is two, so the synonym is not reached.
    assert reached["visually_similar"]
    assert not reached["synonyms"]


def test_the_ideal_order_scores_one() -> None:
    chair = _chair()
    ideal = ["chaise", "chair", "seat", "chair pictogram", "stool"]

    assert set_ranking(ideal, chair, n=5) == pytest.approx(1.0)


def test_proposing_nothing_useful_scores_zero() -> None:
    assert set_ranking(["fire extinguisher", "lift"], _chair(), n=5) == 0.0


def test_reversing_the_categories_costs_the_ranking_but_not_top_n() -> None:
    chair = _chair()
    good = ["chair", "chair pictogram", "stool", "table", "chaise"]
    bad = ["table", "stool", "chair pictogram", "chair", "chaise"]

    # Both reach every category inside the cutoff, so top-N cannot tell them apart.
    assert top_n_frequency(good, chair, n=5) == top_n_frequency(bad, chair, n=5)
    assert set_ranking(good, chair, n=5) > set_ranking(bad, chair, n=5)


def test_an_object_with_no_category_at_all_has_no_ideal_to_compare_to() -> None:
    bare = Annotation(
        id="a1", class_name="", prompt="chaise", lat=48.0, lng=2.0, accuracy_m=5.0
    )

    assert math.isnan(set_ranking(["chair"], bare, n=5))


def test_the_aggregate_skips_annotations_with_no_sets_and_says_so() -> None:
    bare = Annotation(
        id="a2", class_name="table", prompt="table", lat=48.0, lng=2.0, accuracy_m=5.0
    )
    report = evaluate_label_sets({"a1": ["chair"]}, [_chair(), bare], n=5)

    assert report.scored == 1
    assert report.annotations == 2
    assert report.coverage == pytest.approx(0.5)
    assert report.top_n["synonyms"] == pytest.approx(1.0)


def test_an_unannotated_map_reports_nothing_rather_than_zeros() -> None:
    bare = Annotation(
        id="a1", class_name="chaise", prompt="chaise", lat=48.0, lng=2.0, accuracy_m=5.0
    )
    report = evaluate_label_sets({}, [bare])

    assert report.scored == 0
    assert report.coverage == 0.0
    assert math.isnan(report.mean_set_ranking)
    assert "rien à mesurer" in "\n".join(report_lines(report))


def test_an_object_nobody_proposed_a_label_for_counts_as_a_failure() -> None:
    report = evaluate_label_sets({}, [_chair()], n=5)

    assert report.unproposed == 1
    assert report.top_n["synonyms"] == 0.0
    assert report.mean_set_ranking == 0.0


def test_the_vocabulary_is_every_label_any_object_mentions() -> None:
    assert vocabulary([_chair()]) == (
        "armchair",
        "chair",
        "chair pictogram",
        "chaise",
        "seat",
        "stool",
        "table",
    )


def test_ranking_a_vocabulary_follows_the_cosine() -> None:
    labels = ["chair", "table", "lift"]
    vectors = np.eye(3, dtype=np.float32)
    # Closest to the second axis, then the first.
    embedding = np.array([0.3, 0.9, 0.1], dtype=np.float32)

    assert rank_labels(embedding, vectors, labels) == ["table", "chair", "lift"]


def test_ranking_refuses_vectors_that_do_not_match_the_vocabulary() -> None:
    with pytest.raises(ValueError):
        rank_labels(np.ones(3), np.eye(2, 3, dtype=np.float32), ["chair"])
