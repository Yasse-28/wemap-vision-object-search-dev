"""Annotated-neighbour cache rescorer for reviewed candidate embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toolbox.bricks.rescoring import RescoreInput, RescoreResult

logger = logging.getLogger(__name__)


class KnnCacheRescorer:
    """Rescore candidates with similarity-weighted reviewed-neighbour votes.

    Each candidate's ``k`` nearest reviewed embeddings vote with label ``+1``
    for a positive review and ``-1`` for a negative review. A neighbour's weight
    is ``max(0, cosine_similarity) ** p`` when its similarity is strictly above
    ``min_similarity``. The final score explicitly combines that vote with the
    MetaCLIP score as ``base_similarity + gamma * vote``.

    ``positive_evidence`` and ``negative_evidence`` are the normalized positive
    and negative vote masses per candidate. They sum to one when any neighbour
    has weight and their difference is the vote applied to the base score.

    Args:
        k: Maximum number of reviewed neighbours allowed to vote.
        p: Positive exponent controlling how strongly the nearest votes dominate.
        gamma: Gain applied to the signed neighbour vote.
        min_similarity: Strict cosine-similarity floor for voting neighbours.
    """

    name = "knn_cache"

    def __init__(
        self,
        k: int = 5,
        p: float = 3.0,
        gamma: float = 0.1,
        min_similarity: float = 0.0,
    ) -> None:
        self.k = int(k)
        self.p = float(p)
        self.gamma = float(gamma)
        self.min_similarity = float(min_similarity)
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if not np.isfinite(self.p) or self.p <= 0.0:
            raise ValueError("p must be finite and greater than 0.")
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite.")
        if not np.isfinite(self.min_similarity):
            raise ValueError("min_similarity must be finite.")

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Combine base similarities with reviewed-neighbour vote evidence."""
        from toolbox.bricks.rescoring import RescoreResult

        n_candidates = inp.embeddings.shape[0]
        n_positive = inp.positive_embeddings.shape[0]
        n_negative = inp.negative_embeddings.shape[0]
        n_reviewed = n_positive + n_negative
        if n_reviewed == 0:
            logger.info(
                "knn_cache has no reviewed embeddings; returning base similarity."
            )
            no_evidence = np.zeros(n_candidates, dtype=np.float32)
            return RescoreResult(
                scores=inp.base_similarity,
                positive_evidence=no_evidence,
                negative_evidence=no_evidence.copy(),
            )

        reviewed = np.concatenate(
            (inp.positive_embeddings, inp.negative_embeddings), axis=0
        )
        labels = np.concatenate(
            (
                np.ones(n_positive, dtype=np.float64),
                -np.ones(n_negative, dtype=np.float64),
            )
        )
        similarities = np.asarray(inp.embeddings @ reviewed.T, dtype=np.float64)
        neighbour_count = min(self.k, n_reviewed)
        neighbour_indices = np.argsort(-similarities, axis=1, kind="stable")[
            :, :neighbour_count
        ]
        neighbour_similarities = np.take_along_axis(
            similarities, neighbour_indices, axis=1
        )
        neighbour_labels = labels[neighbour_indices]

        clipped_similarities = np.maximum(neighbour_similarities, 0.0)
        is_eligible = (
            np.isfinite(neighbour_similarities)
            & (neighbour_similarities > self.min_similarity)
            & (clipped_similarities > 0.0)
        )
        max_similarity = np.max(
            np.where(is_eligible, clipped_similarities, 0.0), axis=1
        )
        scaled_similarities = np.divide(
            clipped_similarities,
            max_similarity[:, np.newaxis],
            out=np.zeros_like(clipped_similarities),
            where=is_eligible & (max_similarity[:, np.newaxis] > 0.0),
        )
        weights = np.where(is_eligible, scaled_similarities**self.p, 0.0)
        total_weight = np.sum(weights, axis=1)
        positive_weight = np.sum(weights * (neighbour_labels > 0.0), axis=1)
        negative_weight = np.sum(weights * (neighbour_labels < 0.0), axis=1)
        positive_evidence = np.divide(
            positive_weight,
            total_weight,
            out=np.zeros(n_candidates, dtype=np.float64),
            where=total_weight > 0.0,
        )
        negative_evidence = np.divide(
            negative_weight,
            total_weight,
            out=np.zeros(n_candidates, dtype=np.float64),
            where=total_weight > 0.0,
        )
        vote = positive_evidence - negative_evidence
        scores = np.asarray(inp.base_similarity + self.gamma * vote, dtype=np.float32)
        return RescoreResult(
            scores=scores,
            positive_evidence=np.asarray(positive_evidence, dtype=np.float32),
            negative_evidence=np.asarray(negative_evidence, dtype=np.float32),
        )
