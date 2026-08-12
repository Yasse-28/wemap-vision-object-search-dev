"""Tests for graph label-propagation review rescoring."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from toolbox.bricks.rescoring import RescoreInput, build_rescorer
from toolbox.bricks.rescoring_graph_propagation import GraphPropagationRescorer


def _input(
    embeddings: list[list[float]],
    positive: list[list[float]],
    negative: list[list[float]],
) -> RescoreInput:
    """Build a small two-dimensional rescorer input."""
    candidate_embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, 2)
    return RescoreInput(
        candidate_ids=np.arange(len(embeddings), dtype=np.int64),
        embeddings=candidate_embeddings,
        base_similarity=np.linspace(
            0.2, 0.2 + 0.01 * max(len(embeddings) - 1, 0), len(embeddings)
        ).astype(np.float32),
        positive_embeddings=np.asarray(positive, dtype=np.float32).reshape(-1, 2),
        negative_embeddings=np.asarray(negative, dtype=np.float32).reshape(-1, 2),
    )


def test_hand_computed_one_iteration_case() -> None:
    inp = _input(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        positive=[[1.0, 0.0]],
        negative=[[0.0, 1.0]],
    )

    result = GraphPropagationRescorer(
        k=1, alpha_prop=0.5, n_iterations=1, gamma=0.1
    ).score(inp)

    np.testing.assert_allclose(result.scores, [0.25, 0.16], atol=1e-7)
    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    np.testing.assert_allclose(
        result.positive_evidence, np.asarray([0.5, 0.0]), atol=1e-7
    )
    np.testing.assert_allclose(
        result.negative_evidence, np.asarray([0.0, 0.5]), atol=1e-7
    )
    np.testing.assert_array_equal(
        result.positive_evidence_applied, result.positive_evidence
    )
    np.testing.assert_array_equal(
        result.negative_evidence_applied, result.negative_evidence
    )


def test_is_deterministic_with_tied_similarities() -> None:
    inp = _input(
        embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        positive=[[1.0, 0.0]],
        negative=[[0.0, 1.0]],
    )
    rescorer = GraphPropagationRescorer(k=2)

    first = rescorer.score(inp)
    second = rescorer.score(inp)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.positive_evidence, second.positive_evidence)
    np.testing.assert_array_equal(first.negative_evidence, second.negative_evidence)


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        ([], [[0.0, 1.0]]),
        ([[1.0, 0.0]], []),
        ([[1.0, 0.0]], [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_degenerate_review_counts_do_not_raise(
    positive: list[list[float]], negative: list[list[float]]
) -> None:
    inp = _input([[1.0, 0.0]], positive, negative)

    result = GraphPropagationRescorer(k=50).score(inp)

    assert result.scores.shape == (1,)
    assert np.all(np.isfinite(result.scores))


def test_no_reviews_falls_back_to_finite_base_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inp = _input([[1.0, 0.0], [0.0, 1.0]], [], [])
    inp.base_similarity[0] = np.nan

    with caplog.at_level(logging.INFO):
        result = GraphPropagationRescorer().score(inp)

    np.testing.assert_allclose(result.scores, [0.0, 0.21], atol=1e-7)
    np.testing.assert_array_equal(result.positive_evidence, [0.0, 0.0])
    np.testing.assert_array_equal(result.negative_evidence, [0.0, 0.0])
    assert len(caplog.records) == 1
    assert "cannot fit without reviews" in caplog.text


def test_empty_candidates_do_not_raise() -> None:
    inp = _input([], [[1.0, 0.0]], [[0.0, 1.0]])

    result = GraphPropagationRescorer().score(inp)

    assert result.scores.shape == (0,)
    assert np.all(np.isfinite(result.scores))


def test_registered_with_documented_defaults() -> None:
    rescorer = build_rescorer("graph_propagation", {})

    assert isinstance(rescorer, GraphPropagationRescorer)
    assert rescorer.k == 10
    assert rescorer.alpha_prop == 0.8
    assert rescorer.n_iterations == 20
    assert rescorer.gamma == 0.1
