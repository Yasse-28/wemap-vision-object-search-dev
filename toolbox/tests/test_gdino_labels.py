"""Unit tests for the G-DINO label recovery."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolbox.benchmark.gdino_labels import (
    PLACEHOLDER_LABEL,
    ArgmaxLabels,
    assign_labels,
    validate,
    venue_classes,
)
from toolbox.bricks.ingest_cli import EMBEDDING_DIM


def _map(tmp_path, labels: list[str], sources: list[str], vectors: np.ndarray):
    directory = tmp_path / "object-search"
    directory.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "row_index": pa.array(range(len(labels)), type=pa.int64()),
                "detector_source": pa.array(sources, type=pa.string()),
                "label": pa.array(labels, type=pa.string()),
            }
        ),
        directory / "metadata.parquet",
    )
    vectors.astype(np.float16).tofile(directory / "embeddings.npy")
    return tmp_path


def _unit(rows: list[list[float]]) -> np.ndarray:
    dense = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        dense[index, : len(row)] = row
    dense /= np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-6)
    return dense


def test_the_nearest_phrase_wins_and_the_margin_says_by_how_much(tmp_path) -> None:
    classes = _unit([[1.0, 0.0], [0.0, 1.0]])
    path = _map(
        tmp_path,
        [PLACEHOLDER_LABEL, PLACEHOLDER_LABEL],
        ["gdino", "gdino"],
        _unit([[1.0, 0.05], [0.05, 1.0]]),
    )

    result = assign_labels(path, ["left", "right"], classes)

    assert result.label.tolist() == ["left", "right"]
    assert (result.margin > 0.5).all()


def test_an_ambiguous_cutout_gets_a_margin_near_zero(tmp_path) -> None:
    classes = _unit([[1.0, 0.0], [0.0, 1.0]])
    path = _map(tmp_path, [PLACEHOLDER_LABEL], ["gdino"], _unit([[1.0, 1.0]]))

    result = assign_labels(path, ["left", "right"], classes)

    assert float(result.margin[0]) < 1e-3


def test_only_the_requested_detector_is_labelled(tmp_path) -> None:
    classes = _unit([[1.0, 0.0]])
    path = _map(
        tmp_path,
        ["yolo_thing", PLACEHOLDER_LABEL],
        ["yolo", "gdino"],
        _unit([[1.0], [1.0]]),
    )

    result = assign_labels(path, ["only"], classes)

    assert result.row_index.tolist() == [1]


def test_every_row_can_be_labelled_when_the_source_filter_is_dropped(tmp_path) -> None:
    classes = _unit([[1.0, 0.0]])
    path = _map(
        tmp_path,
        ["yolo_thing", PLACEHOLDER_LABEL],
        ["yolo", "gdino"],
        _unit([[1.0], [1.0]]),
    )

    result = assign_labels(path, ["only"], classes, source=None)

    assert result.row_index.tolist() == [0, 1]


def test_a_length_mismatch_refuses_rather_than_guesses(tmp_path) -> None:
    classes = _unit([[1.0, 0.0]])
    path = _map(tmp_path, [PLACEHOLDER_LABEL] * 2, ["gdino"] * 2, _unit([[1.0]]))

    with pytest.raises(SystemExit, match="refusing to guess"):
        assign_labels(path, ["only"], classes)


def test_validation_says_so_when_there_is_only_the_placeholder(tmp_path) -> None:
    path = _map(tmp_path, [PLACEHOLDER_LABEL], ["gdino"], _unit([[1.0]]))
    result = ArgmaxLabels(
        np.array([0]), np.array(["guess"]), np.array([0.5]), np.array([0.1])
    )

    lines = validate(path, result)

    assert "aucun label réel" in lines[0]


def test_validation_scores_the_agreement_with_the_stored_labels(tmp_path) -> None:
    path = _map(tmp_path, ["chair", "lamp"], ["gdino", "gdino"], _unit([[1.0], [1.0]]))
    result = ArgmaxLabels(
        np.array([0, 1]),
        np.array(["chair", "sofa"]),
        np.array([0.5, 0.5]),
        np.array([0.1, 0.1]),
    )

    lines = validate(path, result)

    assert "accord top-1 : 50.0%" in lines[0]


def test_an_unknown_venue_refuses_instead_of_labelling_with_nothing(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no GroundingDINO vocabulary"):
        venue_classes(tmp_path, venue="submarine")


def test_the_venue_vocabularies_are_the_ones_prepare_used() -> None:
    _, hotel = venue_classes(_missing_path(), venue="hotel")
    _, airport = venue_classes(_missing_path(), venue="airport")

    assert "smoke detector" in hotel
    assert "x ray machine" in airport


def _missing_path():
    from pathlib import Path

    return Path("/nonexistent")
