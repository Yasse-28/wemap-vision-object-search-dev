"""Tests for the shared review-rescoring seam and baseline implementation.

Ported from the `feat/rescoring-*` worktrees, minus their ranking-path test: that
branch predates the ratio score and made `rank_localization_clusters` switch to a
min-max normalisation when a rescorer was set. Here the seam is offline-only — a
rescorer writes `similarity_boosted` and the ratio consumes it like any other
boost — so there is no such branch to pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from toolbox.bricks.candidates import (
    FeedbackNormalization,
    apply_feedback_boost,
    normalize_prototype_similarities,
)
from toolbox.bricks.rescoring import (
    IdentityRescorer,
    MaxPrototypeRescorer,
    RescoreInput,
    build_rescorer,
)


def test_identity_returns_base_similarity_without_evidence() -> None:
    base = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    inp = RescoreInput(
        candidate_ids=np.arange(3, dtype=np.int64),
        embeddings=np.eye(3, dtype=np.float32),
        base_similarity=base,
        positive_embeddings=np.eye(3, dtype=np.float32),
        negative_embeddings=np.eye(3, dtype=np.float32),
    )

    result = IdentityRescorer().score(inp)

    assert result.scores is base
    assert result.positive_evidence is None
    assert result.negative_evidence is None
    assert result.positive_evidence_applied is None
    assert result.negative_evidence_applied is None


def test_build_identity_rescorer_takes_no_parameters() -> None:
    assert isinstance(build_rescorer("identity", {}), IdentityRescorer)
    with pytest.raises(TypeError):
        build_rescorer("identity", {"alpha": 0.0})


@pytest.mark.parametrize("normalization", ["none", "center", "standardize"])
def test_max_prototype_matches_sql_boost(
    normalization: FeedbackNormalization,
) -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, 0.8]],
        dtype=np.float32,
    )
    positive = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    negative = np.asarray([[-1.0, 0.0]], dtype=np.float32)
    base = np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    alpha, beta = 0.2, 0.5

    pos_sim = normalize_prototype_similarities(
        np.max(embeddings @ positive.T, axis=1).tolist(), normalization
    )
    neg_sim = normalize_prototype_similarities(
        np.max(embeddings @ negative.T, axis=1).tolist(), normalization
    )
    sql_scores = np.asarray(
        [
            apply_feedback_boost(raw, pos, neg, alpha, beta)
            for raw, pos, neg in zip(base, pos_sim, neg_sim)
        ],
        dtype=np.float32,
    )
    inp = RescoreInput(
        candidate_ids=np.arange(4, dtype=np.int64),
        embeddings=embeddings,
        base_similarity=base,
        positive_embeddings=positive,
        negative_embeddings=negative,
    )

    result = MaxPrototypeRescorer(alpha, beta, normalization).score(inp)

    np.testing.assert_allclose(result.scores, sql_scores, rtol=0.0, atol=1e-6)
    assert result.positive_evidence_applied is not None
    assert result.negative_evidence_applied is not None
    np.testing.assert_allclose(
        result.positive_evidence_applied,
        np.asarray(pos_sim, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.negative_evidence_applied,
        np.asarray(neg_sim, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_max_prototype_defaults_to_base_similarity() -> None:
    base = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    inp = RescoreInput(
        candidate_ids=np.arange(3, dtype=np.int64),
        embeddings=np.eye(3, dtype=np.float32),
        base_similarity=base,
        positive_embeddings=np.eye(3, dtype=np.float32),
        negative_embeddings=np.eye(3, dtype=np.float32),
    )

    result = MaxPrototypeRescorer().score(inp)

    np.testing.assert_array_equal(result.scores, base)


def test_build_rescorer_none_is_the_off_branch() -> None:
    assert build_rescorer(None, {}) is None


def test_build_rescorer_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown rescorer"):
        build_rescorer("missing", {})
