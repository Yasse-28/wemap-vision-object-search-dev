"""Multiple-prototype rescoring with deterministic spherical k-means."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toolbox.bricks.candidates import (
    FEEDBACK_NORMALIZATIONS,
    FeedbackNormalization,
    apply_feedback_boost,
    normalize_prototype_similarities,
)
from toolbox.logging import logger

if TYPE_CHECKING:
    from toolbox.bricks.rescoring import RescoreInput, RescoreResult

_MAX_LLOYD_ITERATIONS = 100


def _normalized_rows(rows: np.ndarray) -> np.ndarray | None:
    """Return unit-length centroid rows, or ``None`` for unusable rows."""
    norms = np.linalg.norm(rows, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    return np.asarray(rows / norms[:, np.newaxis], dtype=np.float32)


def _fit_spherical_kmeans(
    samples: np.ndarray, k: int, n_init: int, seed: int
) -> np.ndarray | None:
    """Fit spherical k-means and return the best restart's unit centroids."""
    n_samples = samples.shape[0]
    if n_samples == 0:
        return None

    n_clusters = min(k, n_samples)
    rng = np.random.default_rng(seed)
    best_centroids: np.ndarray | None = None
    best_objective = -np.inf

    for _ in range(n_init):
        indices = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = _normalized_rows(samples[indices])
        if centroids is None:
            continue

        previous_labels: np.ndarray | None = None
        for _ in range(_MAX_LLOYD_ITERATIONS):
            similarities = samples @ centroids.T
            labels = np.argmax(similarities, axis=1)
            updated = centroids.copy()
            for cluster_index in range(n_clusters):
                members = samples[labels == cluster_index]
                if members.shape[0] == 0:
                    continue
                mean = np.mean(members, axis=0, dtype=np.float64)
                norm = float(np.linalg.norm(mean))
                if np.isfinite(norm) and norm > 0.0:
                    updated[cluster_index] = np.asarray(mean / norm, dtype=np.float32)
            centroids = updated
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels

        objective = float(np.sum(np.max(samples @ centroids.T, axis=1)))
        if np.isfinite(objective) and objective > best_objective:
            best_objective = objective
            best_centroids = centroids.copy()

    return best_centroids


def _finite_base_similarity(base_similarity: np.ndarray) -> np.ndarray:
    """Preserve finite base scores and replace non-finite inputs safely."""
    return np.nan_to_num(
        np.asarray(base_similarity, dtype=np.float32),
        copy=True,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )


class MultiPrototypeRescorer:
    """Rescore candidates against clustered positive and negative appearances.

    Positive and negative evidence are each the candidate's best cosine
    similarity to the corresponding spherical-k-means centroids. The applied
    evidence contains those columns after the same per-query normalization used
    by ``MaxPrototypeRescorer``. Final scores explicitly combine the original
    MetaCLIP similarity as ``base + alpha * positive - beta * negative`` and use
    the shared feedback clipping behavior.

    Args:
        k_positive: Maximum number of positive appearance clusters.
        k_negative: Maximum number of negative appearance clusters.
        alpha: Gain applied to normalized positive evidence.
        beta: Gain applied to normalized negative evidence.
        normalization: Evidence normalization mode shared with prototype scoring.
        n_init: Number of deterministic random initializations per label set.
        seed: Seed used for centroid initialization.
    """

    name = "multi_prototype"

    def __init__(
        self,
        k_positive: int = 4,
        k_negative: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        normalization: FeedbackNormalization = "none",
        n_init: int = 3,
        seed: int = 0,
    ) -> None:
        if k_positive <= 0 or k_negative <= 0:
            raise ValueError("Prototype cluster counts must be positive.")
        if n_init <= 0:
            raise ValueError("n_init must be positive.")
        if normalization not in FEEDBACK_NORMALIZATIONS:
            choices = ", ".join(FEEDBACK_NORMALIZATIONS)
            raise ValueError(
                f"Unknown prototype normalization {normalization!r}; "
                f"available: {choices}."
            )

        self.k_positive = int(k_positive)
        self.k_negative = int(k_negative)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalization = normalization
        self.n_init = int(n_init)
        self.seed = int(seed)
        # Validate even when a query later cannot fit prototypes.
        normalize_prototype_similarities([], normalization)

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Return final scores and best-centroid evidence in candidate order."""
        # Deferred to avoid a module cycle: the registry imports this class.
        from toolbox.bricks.rescoring import RescoreResult

        positive_centroids = _fit_spherical_kmeans(
            inp.positive_embeddings, self.k_positive, self.n_init, self.seed
        )
        negative_centroids = _fit_spherical_kmeans(
            inp.negative_embeddings, self.k_negative, self.n_init, self.seed
        )
        if positive_centroids is None or negative_centroids is None:
            logger.info(
                "Rescorer '%s' could not fit positive and negative prototypes; "
                "using base similarity.",
                self.name,
            )
            return RescoreResult(scores=_finite_base_similarity(inp.base_similarity))

        positive = np.asarray(
            np.max(inp.embeddings @ positive_centroids.T, axis=1), dtype=np.float32
        )
        negative = np.asarray(
            np.max(inp.embeddings @ negative_centroids.T, axis=1), dtype=np.float32
        )
        positive_applied = np.asarray(
            normalize_prototype_similarities(positive.tolist(), self.normalization),
            dtype=np.float32,
        )
        negative_applied = np.asarray(
            normalize_prototype_similarities(negative.tolist(), self.normalization),
            dtype=np.float32,
        )
        scores = np.asarray(
            [
                apply_feedback_boost(base, pos, neg, self.alpha, self.beta)
                for base, pos, neg in zip(
                    _finite_base_similarity(inp.base_similarity),
                    positive_applied,
                    negative_applied,
                )
            ],
            dtype=np.float32,
        )
        return RescoreResult(
            scores=scores,
            positive_evidence=positive,
            negative_evidence=negative,
            positive_evidence_applied=positive_applied,
            negative_evidence_applied=negative_applied,
        )
