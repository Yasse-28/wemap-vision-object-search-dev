"""Tests for annotated-neighbour cache candidate rescoring."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from toolbox.bricks.rescoring import RescoreInput, build_rescorer
from toolbox.bricks.rescoring_knn_cache import KnnCacheRescorer


def _input(
    embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    base_similarity: np.ndarray | None = None,
) -> RescoreInput:
    """Build a compact rescorer input for two-dimensional test embeddings."""
    n_candidates = embeddings.shape[0]
    return RescoreInput(
        candidate_ids=np.arange(n_candidates, dtype=np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        base_similarity=(
            np.zeros(n_candidates, dtype=np.float32)
            if base_similarity is None
            else np.asarray(base_similarity, dtype=np.float32)
        ),
        positive_embeddings=np.asarray(positive_embeddings, dtype=np.float32).reshape(
            -1, 2
        ),
        negative_embeddings=np.asarray(negative_embeddings, dtype=np.float32).reshape(
            -1, 2
        ),
    )


def test_knn_cache_hand_computed_weighted_vote() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        positive_embeddings=np.asarray([[1.0, 0.0]]),
        negative_embeddings=np.asarray([[0.0, 1.0]]),
        base_similarity=np.asarray([0.3, 0.3, 0.3]),
    )

    result = KnnCacheRescorer(k=2, p=2.0, gamma=0.2).score(inp)

    np.testing.assert_allclose(result.scores, [0.5, 0.356, 0.1], atol=1e-6)
    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    np.testing.assert_allclose(result.positive_evidence, np.asarray([1.0, 0.64, 0.0]))
    np.testing.assert_allclose(result.negative_evidence, np.asarray([0.0, 0.36, 1.0]))


def test_knn_cache_zero_reviews_returns_base_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inp = _input(
        embeddings=np.eye(2, dtype=np.float32),
        positive_embeddings=np.empty((0, 2), dtype=np.float32),
        negative_embeddings=np.empty((0, 2), dtype=np.float32),
        base_similarity=np.asarray([0.2, 0.4]),
    )

    with caplog.at_level(logging.INFO):
        result = KnnCacheRescorer().score(inp)

    np.testing.assert_array_equal(result.scores, inp.base_similarity)
    np.testing.assert_array_equal(result.positive_evidence, [0.0, 0.0])
    np.testing.assert_array_equal(result.negative_evidence, [0.0, 0.0])
    assert caplog.messages == [
        "knn_cache has no reviewed embeddings; returning base similarity."
    ]


def test_knn_cache_zero_positive_reviews_uses_negative_votes() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0]]),
        positive_embeddings=np.empty((0, 2), dtype=np.float32),
        negative_embeddings=np.asarray([[1.0, 0.0]]),
    )

    result = KnnCacheRescorer().score(inp)

    np.testing.assert_allclose(result.scores, [-0.1])
    np.testing.assert_array_equal(result.positive_evidence, [0.0])
    np.testing.assert_array_equal(result.negative_evidence, [1.0])


def test_knn_cache_zero_negative_reviews_and_one_positive() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        positive_embeddings=np.asarray([[1.0, 0.0]]),
        negative_embeddings=np.empty((0, 2), dtype=np.float32),
    )

    result = KnnCacheRescorer(k=5).score(inp)

    np.testing.assert_allclose(result.scores, [0.1, 0.0])
    np.testing.assert_array_equal(result.positive_evidence, [1.0, 0.0])
    np.testing.assert_array_equal(result.negative_evidence, [0.0, 0.0])


def test_knn_cache_handles_more_reviews_than_candidates_and_large_k() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0]]),
        positive_embeddings=np.asarray([[1.0, 0.0], [0.5, 0.5]]),
        negative_embeddings=np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
    )

    result = KnnCacheRescorer(k=100, p=1.0).score(inp)

    np.testing.assert_allclose(result.scores, [0.1])
    assert np.all(np.isfinite(result.scores))


def test_knn_cache_no_neighbour_above_similarity_floor_has_zero_vote() -> None:
    base = np.asarray([0.25], dtype=np.float32)
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0]]),
        positive_embeddings=np.asarray([[0.5, 0.5]]),
        negative_embeddings=np.asarray([[0.4, 0.6]]),
        base_similarity=base,
    )

    result = KnnCacheRescorer(min_similarity=0.5).score(inp)

    np.testing.assert_array_equal(result.scores, base)
    np.testing.assert_array_equal(result.positive_evidence, [0.0])
    np.testing.assert_array_equal(result.negative_evidence, [0.0])


def test_knn_cache_is_deterministic_with_tied_neighbours() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0]]),
        positive_embeddings=np.asarray([[0.5, 0.5]]),
        negative_embeddings=np.asarray([[0.5, -0.5]]),
    )
    rescorer = KnnCacheRescorer(k=1)

    first = rescorer.score(inp)
    second = rescorer.score(inp)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.positive_evidence, second.positive_evidence)
    np.testing.assert_array_equal(first.negative_evidence, second.negative_evidence)
    np.testing.assert_allclose(first.scores, [0.1])


def test_knn_cache_is_registered_with_default_parameters() -> None:
    rescorer = build_rescorer("knn_cache", {})

    assert isinstance(rescorer, KnnCacheRescorer)
    assert rescorer.k == 5
    assert rescorer.p == 3.0
    assert rescorer.gamma == 0.1
    assert rescorer.min_similarity == 0.0
