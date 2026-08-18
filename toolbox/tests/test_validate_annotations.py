"""Unit tests for the annotation contract check."""

from __future__ import annotations

import pytest

from toolbox.benchmark.object_search_http_benchmark import Annotation
from toolbox.benchmark.validate_annotations import (
    duplicate_clicks,
    field_coverage,
    validate,
)


def _annotation(identifier: str, **overrides: object) -> Annotation:
    base: dict[str, object] = {
        "id": identifier,
        "class_name": "chaise",
        "prompt": "chaise",
        "lat": 48.0,
        "lng": 2.0,
        "accuracy_m": 5.0,
    }
    base.update(overrides)
    return Annotation(**base)  # type: ignore[arg-type]


def test_a_map_annotated_the_old_way_is_incomplete_but_not_contradictory() -> None:
    findings = validate([_annotation("a1"), _annotation("a2", lat=48.001)])

    assert not findings.blocking
    assert any("extent_m" in line and "absent" in line for line in findings.missing)


def test_a_half_filled_field_is_called_out_as_worse_than_an_absent_one() -> None:
    findings = validate(
        [
            _annotation("a1", extent_m=0.5),
            _annotation("a2", lat=48.001),
            _annotation("a3", lat=48.002),
        ]
    )

    line = next(item for item in findings.missing if "extent_m" in item)
    assert "PARTIEL" in line


def test_coverage_counts_each_field_separately() -> None:
    coverage = field_coverage(
        [_annotation("a1", extent_m=0.5, synonyms=("chair",)), _annotation("a2")]
    )

    assert coverage["extent_m"] == pytest.approx(0.5)
    assert coverage["labels.synonyms"] == pytest.approx(0.5)
    assert coverage["object_id"] == 0.0


def test_one_click_recorded_twice_is_found() -> None:
    groups = duplicate_clicks([_annotation("a1"), _annotation("a2")])

    assert len(groups) == 1
    assert {item.id for item in groups[0]} == {"a1", "a2"}


def test_two_objects_a_metre_apart_are_not_duplicates() -> None:
    assert duplicate_clicks([_annotation("a1"), _annotation("a2", lat=48.00001)]) == []


def test_two_classes_at_one_spot_are_not_duplicates_of_each_other() -> None:
    assert (
        duplicate_clicks([_annotation("a1"), _annotation("a2", class_name="table")])
        == []
    )


def test_duplicates_block_and_distinct_ids_at_one_spot_are_flagged() -> None:
    findings = validate(
        [
            _annotation("a1", object_id="chaise-1"),
            _annotation("a2", object_id="chaise-2"),
        ]
    )

    assert findings.blocking
    assert any("object_id différents" in line for line in findings.inconsistent)


def test_one_object_id_under_two_classes_is_contradictory() -> None:
    findings = validate(
        [
            _annotation("a1", object_id="thing-1"),
            _annotation("a2", object_id="thing-1", class_name="table", lat=48.001),
        ]
    )

    assert any("2 classes" in line for line in findings.inconsistent)


def test_one_object_id_under_two_extents_is_contradictory() -> None:
    findings = validate(
        [
            _annotation("a1", object_id="thing-1", extent_m=0.5),
            _annotation("a2", object_id="thing-1", extent_m=2.0, lat=48.001),
        ]
    )

    assert any("2 emprises" in line for line in findings.inconsistent)


def test_an_extent_in_centimetres_is_caught_as_a_typo() -> None:
    findings = validate([_annotation("a1", extent_m=50.0)])

    assert any("emprises hors de" in line for line in findings.inconsistent)


def test_one_class_with_two_synonym_sets_is_contradictory() -> None:
    findings = validate(
        [
            _annotation("a1", synonyms=("chair", "seat")),
            _annotation("a2", synonyms=("chair",), lat=48.001),
        ]
    )

    assert any("ensembles de" in line for line in findings.inconsistent)


def test_the_same_synonym_set_written_differently_is_not_a_disagreement() -> None:
    findings = validate(
        [
            _annotation("a1", synonyms=("Chair", "seat")),
            _annotation("a2", synonyms=("seat", "chair"), lat=48.001),
        ]
    )

    assert not any("ensembles de" in line for line in findings.inconsistent)


def test_separable_classes_earn_the_classification_columns() -> None:
    findings = validate(
        [
            _annotation("a1", extent_m=0.5),
            _annotation(
                "a2", class_name="table", prompt="table", lat=48.001, extent_m=1.0
            ),
        ]
    )

    assert any("DISPONIBLES" in line for line in findings.earned)


def test_the_flat_radius_is_named_as_the_thing_to_fix() -> None:
    findings = validate([_annotation("a1"), _annotation("a2", lat=48.001)])

    assert any("annoter extent_m" in line for line in findings.earned)


def test_a_partly_zoned_map_says_how_much_is_covered() -> None:
    findings = validate(
        [
            _annotation("a1", exhaustive_zone="lobby"),
            _annotation("a2", lat=48.001),
        ]
    )

    assert any("zone exhaustive" in line for line in findings.earned)


def test_an_empty_file_is_reported_as_contradictory() -> None:
    findings = validate([])

    assert findings.blocking
