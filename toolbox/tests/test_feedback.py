"""Tests for the review-feedback loader and the boost formula.

The loader is the half that touches the outside world (an SQLite file another
service writes), so its failure modes are all "returns nothing, quietly". The
formula half is pure arithmetic but carries the property the whole feature rests
on: `alpha = beta = 0` must be exact disablement, not approximate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from toolbox.bricks.candidates import (
    _ENRICH_SQL,
    _build_enrich_query,
    apply_feedback_boost,
)
from toolbox.bricks.feedback import (
    ReviewFeedback,
    load_review_feedback,
    normalize_query,
)

SLUG = "test-map"


def _write_annotation_db(
    data_dir: Path, rows: list[tuple[int, str, str]], slug: str = SLUG
) -> Path:
    """A minimal `detection_review` table; `rows` is `(target_id, query, status)`."""
    path = data_dir / slug / "object-search-annotations.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE detection_review (
            detection_review_id INTEGER PRIMARY KEY,
            target_type TEXT NOT NULL,
            target_id   INTEGER NOT NULL,
            query       TEXT NOT NULL,
            status      TEXT NOT NULL
        )
        """)
    conn.executemany(
        "INSERT INTO detection_review (target_type, target_id, query, status) "
        "VALUES ('object', ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return path


# ------------------------------------------------------------------------- loader


def test_missing_db_returns_none(tmp_path: Path, monkeypatch) -> None:
    """A map with no annotations must not be a special case for the caller."""
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    assert load_review_feedback(SLUG, "fire exit") is None


def test_unknown_query_returns_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(tmp_path, [(1, "fire exit", "true_positive")])
    assert load_review_feedback(SLUG, "vending machine") is None


def test_mixed_reviews_split_by_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(
        tmp_path,
        [
            (10, "fire exit", "true_positive"),
            (11, "fire exit", "false_positive"),
            (12, "fire exit", "true_positive"),
            (99, "other query", "true_positive"),  # must not leak in
        ],
    )

    feedback = load_review_feedback(SLUG, "fire exit")

    assert feedback is not None
    assert feedback.positive_ids == [10, 12]
    assert feedback.negative_ids == [11]


@pytest.mark.parametrize("stored, asked", [("FIDS", "fids"), ("fids", " FIDS ")])
def test_query_match_is_case_and_whitespace_insensitive(
    tmp_path: Path, monkeypatch, stored: str, asked: str
) -> None:
    """The stored query is raw user input; `FIDS` and `fids` are one search."""
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(tmp_path, [(7, stored, "false_positive")])

    feedback = load_review_feedback(SLUG, asked)

    assert feedback is not None
    assert feedback.negative_ids == [7]


def test_inner_whitespace_is_not_collapsed(tmp_path: Path, monkeypatch) -> None:
    """ "fire exit" and "fire  exit" are genuinely different searches."""
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(tmp_path, [(7, "fire  exit", "true_positive")])
    assert load_review_feedback(SLUG, "fire exit") is None


def test_another_maps_annotations_are_not_visible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(tmp_path, [(1, "fire exit", "true_positive")], slug="other")
    assert load_review_feedback(SLUG, "fire exit") is None


def test_duplicate_ids_are_collapsed_and_sorted(tmp_path: Path, monkeypatch) -> None:
    """Order must not depend on SQLite's row order — fixtures would be flaky."""
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    _write_annotation_db(
        tmp_path,
        [
            (5, "q", "true_positive"),
            (3, "q", "true_positive"),
            (5, "q", "true_positive"),
        ],
    )
    feedback = load_review_feedback(SLUG, "q")
    assert feedback is not None
    assert feedback.positive_ids == [3, 5]


def test_corrupt_db_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    """A broken annotation file must never take down a search."""
    monkeypatch.setenv("ANNOTATION_DATA_DIR", str(tmp_path))
    path = tmp_path / SLUG / "object-search-annotations.db"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a database")
    assert load_review_feedback(SLUG, "fire exit") is None


def test_normalize_query_folds_case_and_strips() -> None:
    assert normalize_query("  Fire Exit  ") == "fire exit"
    assert normalize_query("STRASSE") == "strasse"  # casefold, not lower


# -------------------------------------------------------------------------- boost


def test_boost_combines_both_terms() -> None:
    assert apply_feedback_boost(0.80, 0.90, 0.10, 0.2, 0.5) == pytest.approx(0.93)


def test_boost_clips_to_unit_range() -> None:
    assert apply_feedback_boost(0.95, 1.0, 0.0, 0.5, 0.0) == 1.0
    assert apply_feedback_boost(-0.9, 0.0, 1.0, 0.0, 0.5) == -1.0


@pytest.mark.parametrize("pos, neg", [(0.0, 0.0), (0.9, 0.1), (1.0, 1.0)])
def test_zero_alpha_beta_is_exact_disablement(pos: float, neg: float) -> None:
    """Not "close to" the raw similarity — bit-for-bit equal to it.

    This is the property that makes the baseline literally the current code path,
    so it is asserted with `==`, deliberately, and not `approx`.
    """
    similarity = 0.8137254901960784
    assert apply_feedback_boost(similarity, pos, neg, 0.0, 0.0) == similarity


# ---------------------------------------------------------------------- SQL shape


def test_no_feedback_emits_todays_sql_unchanged() -> None:
    """Identity, not equality: the untouched path must reuse the same constant."""
    sql, params, has_pos, has_neg = _build_enrich_query(3, [1, 2], None)
    assert sql is _ENRICH_SQL
    assert params == [3, [1, 2]]
    assert (has_pos, has_neg) == (False, False)


def test_empty_feedback_also_emits_todays_sql() -> None:
    empty = ReviewFeedback(positive_ids=[], negative_ids=[])
    sql, params, has_pos, has_neg = _build_enrich_query(3, [1, 2], empty)
    assert sql is _ENRICH_SQL
    assert params == [3, [1, 2]]
    assert (has_pos, has_neg) == (False, False)


def test_prototype_params_precede_the_where_clause_params() -> None:
    """SELECT-list subqueries bind first. Wrong order = a map id where ids belong."""
    feedback = ReviewFeedback(positive_ids=[10, 12], negative_ids=[11])
    sql, params, has_pos, has_neg = _build_enrich_query(3, [1, 2], feedback)

    assert (has_pos, has_neg) == (True, True)
    assert params == [[10, 12], 3, [11], 3, 3, [1, 2]]
    assert sql.count("%s") == len(params)
    assert "AS pos_sim" in sql and "AS neg_sim" in sql
    # The prototype subqueries must sit in the SELECT list, before FROM.
    assert sql.index("AS neg_sim") < sql.index("FROM object_search_candidate AS c")


def test_only_the_populated_side_is_queried() -> None:
    feedback = ReviewFeedback(positive_ids=[], negative_ids=[11])
    sql, params, has_pos, has_neg = _build_enrich_query(3, [1, 2], feedback)

    assert (has_pos, has_neg) == (False, True)
    assert "AS pos_sim" not in sql
    assert params == [[11], 3, 3, [1, 2]]


def test_prototypes_are_restricted_to_the_same_georef() -> None:
    """Without this, one map borrows another's annotations via a stale id."""
    feedback = ReviewFeedback(positive_ids=[10], negative_ids=[])
    sql, _, _, _ = _build_enrich_query(3, [1, 2], feedback)
    assert "p.geo_ref_id = %s" in sql
