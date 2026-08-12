"""Tests for deterministic multiple-prototype rescoring."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from toolbox.bricks.rescoring import RescoreInput, build_rescorer
from toolbox.bricks.rescoring_multi_prototype import MultiPrototypeRescorer


def _input(
    embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    base_similarity: np.ndarray | None = None,
) -> RescoreInput:
    """Build a compact rescorer input for synthetic tests."""
    if base_similarity is None:
        base_similarity = np.zeros(embeddings.shape[0], dtype=np.float32)
    return RescoreInput(
        candidate_ids=np.arange(embeddings.shape[0], dtype=np.int64),
        embeddings=embeddings,
        base_similarity=base_similarity,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
    )


@pytest.mark.parametrize("missing_label", ["positive", "negative"])
def test_missing_label_falls_back_to_base_and_logs_once(
    missing_label: str, caplog: pytest.LogCaptureFixture
) -> None:
    embeddings = np.eye(2, dtype=np.float32)
    empty = np.empty((0, 2), dtype=np.float32)
    positive = empty if missing_label == "positive" else embeddings[:1]
    negative = empty if missing_label == "negative" else embeddings[1:]
    base = np.asarray([0.2, 0.4], dtype=np.float32)

    with caplog.at_level(logging.INFO, logger="pipeline"):
        result = MultiPrototypeRescorer(alpha=0.5, beta=0.5).score(
            _input(embeddings, positive, negative, base)
        )

    np.testing.assert_array_equal(result.scores, base)
    assert result.positive_evidence is None
    assert result.negative_evidence is None
    fallback_messages = [
        message
        for message in caplog.messages
        if "could not fit positive and negative prototypes" in message
    ]
    assert len(fallback_messages) == 1


def test_one_positive_and_k_larger_than_samples_do_not_raise() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positive = np.asarray([[1.0, 0.0]], dtype=np.float32)
    negative = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = MultiPrototypeRescorer(
        k_positive=10,
        k_negative=10,
        alpha=0.2,
        beta=0.1,
    ).score(_input(embeddings, positive, negative))

    np.testing.assert_allclose(result.scores, [0.2, -0.1], rtol=0.0, atol=1e-6)


def test_more_prototypes_than_candidates_do_not_raise() -> None:
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    positive = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    negative = np.asarray([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    result = MultiPrototypeRescorer(alpha=0.2, beta=0.1).score(
        _input(embeddings, positive, negative)
    )

    assert result.scores.shape == (1,)
    assert np.all(np.isfinite(result.scores))


def test_scoring_is_deterministic() -> None:
    rng = np.random.default_rng(91)
    embeddings = rng.normal(size=(8, 3)).astype(np.float32)
    positive = rng.normal(size=(6, 3)).astype(np.float32)
    negative = rng.normal(size=(5, 3)).astype(np.float32)
    inp = _input(embeddings, positive, negative)
    rescorer = MultiPrototypeRescorer(alpha=0.2, beta=0.1, n_init=5, seed=17)

    first = rescorer.score(inp)
    second = rescorer.score(inp)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.positive_evidence, second.positive_evidence)
    np.testing.assert_array_equal(first.negative_evidence, second.negative_evidence)


def test_hand_computed_cluster_evidence_and_scores() -> None:
    root_half = np.float32(1.0 / np.sqrt(2.0))
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [root_half, root_half]], dtype=np.float32
    )
    positive = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    negative = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    base = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    result = MultiPrototypeRescorer(
        k_positive=2,
        k_negative=2,
        alpha=0.2,
        beta=0.1,
        normalization="none",
    ).score(_input(embeddings, positive, negative, base))

    expected_positive = np.asarray([1.0, 1.0, root_half], dtype=np.float32)
    expected_negative = np.asarray([0.0, 0.0, -root_half], dtype=np.float32)
    expected_scores = base + 0.2 * expected_positive - 0.1 * expected_negative
    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    np.testing.assert_allclose(
        result.positive_evidence, expected_positive, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        result.negative_evidence, expected_negative, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(result.scores, expected_scores, rtol=0.0, atol=1e-6)


def test_empty_candidates_return_finite_empty_arrays() -> None:
    embeddings = np.empty((0, 2), dtype=np.float32)
    positive = np.asarray([[1.0, 0.0]], dtype=np.float32)
    negative = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = MultiPrototypeRescorer().score(
        _input(embeddings, positive, negative, np.empty(0, dtype=np.float32))
    )

    assert result.scores.shape == (0,)
    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    assert result.positive_evidence.shape == (0,)
    assert result.negative_evidence.shape == (0,)


def test_registry_builds_multi_prototype_with_defaults() -> None:
    rescorer = build_rescorer("multi_prototype", {})

    assert isinstance(rescorer, MultiPrototypeRescorer)
    assert rescorer.k_positive == 4
    assert rescorer.k_negative == 4
    assert rescorer.alpha == 0.0
    assert rescorer.beta == 0.0
    assert rescorer.normalization == "none"
    assert rescorer.n_init == 3
    assert rescorer.seed == 0
