"""Tests for the dual-space linear-probe rescorer."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from toolbox.bricks.rescoring import RescoreInput, build_rescorer
from toolbox.bricks.rescoring_linear_probe import LinearProbeRescorer


def _input(
    *,
    embeddings: np.ndarray | None = None,
    positive: np.ndarray | None = None,
    negative: np.ndarray | None = None,
) -> RescoreInput:
    """Build a small two-dimensional rescore input."""
    candidate_embeddings = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        if embeddings is None
        else embeddings
    )
    return RescoreInput(
        candidate_ids=np.arange(candidate_embeddings.shape[0], dtype=np.int64),
        embeddings=candidate_embeddings,
        base_similarity=np.linspace(
            0.2, 0.3, candidate_embeddings.shape[0], dtype=np.float32
        ),
        positive_embeddings=(
            np.asarray([[1.0, 0.0]], dtype=np.float32) if positive is None else positive
        ),
        negative_embeddings=(
            np.asarray([[0.0, 1.0]], dtype=np.float32) if negative is None else negative
        ),
    )


def test_linear_probe_is_registered_with_default_hyperparameters() -> None:
    rescorer = build_rescorer("linear_probe", {})

    assert isinstance(rescorer, LinearProbeRescorer)
    assert rescorer.lambda_ == 1.0
    assert rescorer.w_probe == 0.5
    assert rescorer.n_iterations == 50
    assert rescorer.n_weak_negatives == 0
    assert rescorer.seed == 0
    assert rescorer.mix == "linear"


@pytest.mark.parametrize(
    ("positive", "negative", "reason"),
    [
        (
            np.empty((0, 2), dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            "no positive reviews",
        ),
        (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            "no negative reviews",
        ),
    ],
)
def test_linear_probe_missing_class_falls_back_once(
    positive: np.ndarray,
    negative: np.ndarray,
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    inp = _input(positive=positive, negative=negative)

    with caplog.at_level(logging.INFO):
        result = LinearProbeRescorer().score(inp)

    np.testing.assert_array_equal(result.scores, inp.base_similarity)
    assert result.positive_evidence is None
    assert result.negative_evidence is None
    matching = [record for record in caplog.records if reason in record.message]
    assert len(matching) == 1


def test_linear_probe_one_positive_fits() -> None:
    result = LinearProbeRescorer().score(_input())

    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    assert np.all(np.isfinite(result.scores))
    assert result.scores[0] > result.scores[1]


def test_linear_probe_accepts_more_reviews_than_candidates() -> None:
    inp = _input(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        positive=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        negative=np.asarray([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
    )

    result = LinearProbeRescorer().score(inp)

    assert result.scores.shape == (1,)
    assert np.all(np.isfinite(result.scores))


def test_linear_probe_caps_weak_negatives_at_candidate_count() -> None:
    inp = _input(negative=np.empty((0, 2), dtype=np.float32))

    result = LinearProbeRescorer(n_weak_negatives=10, seed=4).score(inp)

    assert result.scores.shape == inp.base_similarity.shape
    assert result.positive_evidence is not None
    assert np.all(np.isfinite(result.scores))


def test_linear_probe_is_deterministic() -> None:
    inp = _input(
        embeddings=np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
            dtype=np.float32,
        )
    )
    rescorer = LinearProbeRescorer(n_weak_negatives=1, seed=17)

    first = rescorer.score(inp)
    second = rescorer.score(inp)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.positive_evidence, second.positive_evidence)
    np.testing.assert_array_equal(first.negative_evidence, second.negative_evidence)


def test_linear_probe_one_iteration_matches_hand_computation() -> None:
    inp = RescoreInput(
        candidate_ids=np.asarray([10, 20], dtype=np.int64),
        embeddings=np.eye(2, dtype=np.float32),
        base_similarity=np.asarray([0.2, 0.4], dtype=np.float32),
        positive_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        negative_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    probe_weight = 0.25

    result = LinearProbeRescorer(
        lambda_=1.0, w_probe=probe_weight, n_iterations=1
    ).score(inp)

    # At a=0, R=0.25 I and mean-loss regularization contributes 2 I.
    # The first Newton coefficients are therefore [+0.5/2.25, -0.5/2.25].
    decision = np.asarray([2.0 / 9.0, -2.0 / 9.0])
    expected_positive = 1.0 / (1.0 + np.exp(-decision))
    expected_negative = 1.0 - expected_positive
    expected_scores = (
        1.0 - probe_weight
    ) * inp.base_similarity + probe_weight * expected_positive
    assert result.positive_evidence is not None
    assert result.negative_evidence is not None
    np.testing.assert_allclose(
        result.positive_evidence, expected_positive, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        result.negative_evidence, expected_negative, rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(result.scores, expected_scores, rtol=0.0, atol=1e-7)


def test_linear_probe_explicit_linear_mix_matches_default_bit_for_bit() -> None:
    inp = _input(
        embeddings=np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
            dtype=np.float32,
        )
    )

    default = LinearProbeRescorer().score(inp)
    explicit = LinearProbeRescorer(mix="linear").score(inp)

    np.testing.assert_array_equal(explicit.scores, default.scores)
    np.testing.assert_array_equal(explicit.positive_evidence, default.positive_evidence)
    np.testing.assert_array_equal(explicit.negative_evidence, default.negative_evidence)


def test_linear_probe_standardized_mix_is_invariant_to_affine_base_scale() -> None:
    inp = _input(
        embeddings=np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
            dtype=np.float32,
        )
    )
    scaled_inp = RescoreInput(
        candidate_ids=inp.candidate_ids,
        embeddings=inp.embeddings,
        base_similarity=3.5 * inp.base_similarity + 7.0,
        positive_embeddings=inp.positive_embeddings,
        negative_embeddings=inp.negative_embeddings,
    )
    rescorer = LinearProbeRescorer(mix="standardized")

    original = rescorer.score(inp)
    scaled = rescorer.score(scaled_inp)

    np.testing.assert_allclose(scaled.scores, original.scores, rtol=0.0, atol=2e-6)


def test_linear_probe_standardized_mix_without_spread_falls_back_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inp = _input()
    inp = RescoreInput(
        candidate_ids=inp.candidate_ids,
        embeddings=inp.embeddings,
        base_similarity=np.full_like(inp.base_similarity, 0.25),
        positive_embeddings=inp.positive_embeddings,
        negative_embeddings=inp.negative_embeddings,
    )

    with caplog.at_level(logging.INFO):
        result = LinearProbeRescorer(mix="standardized").score(inp)

    np.testing.assert_array_equal(result.scores, inp.base_similarity)
    matching = [
        record for record in caplog.records if "column with no spread" in record.message
    ]
    assert len(matching) == 1
