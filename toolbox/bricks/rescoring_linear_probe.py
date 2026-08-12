"""Dual-space logistic-regression rescoring from per-query reviews."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from toolbox.bricks.candidates import normalize_prototype_similarities

if TYPE_CHECKING:
    from toolbox.bricks.rescoring import RescoreInput, RescoreResult

logger = logging.getLogger(__name__)

_MIN_CURVATURE = 1e-8

LinearProbeMix = Literal["linear", "standardized"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate the logistic sigmoid without overflowing."""
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _select_weak_negatives(
    embeddings: np.ndarray,
    base_similarity: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample weak negatives from the lower-scoring half of the retrieval."""
    sample_count = min(count, embeddings.shape[0])
    if sample_count == 0:
        return np.empty((0, embeddings.shape[1]), dtype=np.float64)

    ranked_indices = np.argsort(base_similarity, kind="stable")
    tail_size = min(
        embeddings.shape[0], max(sample_count, (embeddings.shape[0] + 1) // 2)
    )
    tail_indices = ranked_indices[:tail_size]
    generator = np.random.default_rng(seed)
    selected = np.asarray(
        np.sort(generator.choice(tail_indices, size=sample_count, replace=False)),
        dtype=np.int64,
    )
    return np.asarray(embeddings[selected], dtype=np.float64)


def _fit_dual_coefficients(
    train_embeddings: np.ndarray,
    labels: np.ndarray,
    lambda_: float,
    n_iterations: int,
) -> np.ndarray | None:
    """Fit ridge-logistic coefficients using Newton steps on the Gram matrix."""
    gram = train_embeddings @ train_embeddings.T
    sample_count = train_embeddings.shape[0]
    regularization = lambda_ * sample_count
    coefficients = np.zeros(sample_count, dtype=np.float64)

    for _ in range(n_iterations):
        logits = gram @ coefficients
        probabilities = _sigmoid(logits)
        curvature = np.maximum(probabilities * (1.0 - probabilities), _MIN_CURVATURE)
        system = curvature[:, np.newaxis] * gram
        system.flat[:: sample_count + 1] += regularization
        right_hand_side = labels - probabilities - regularization * coefficients
        try:
            step = np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError:
            return None
        coefficients += step
        if not np.all(np.isfinite(coefficients)):
            return None

    return coefficients


class LinearProbeRescorer:
    """Mix base similarity with a per-query logistic probe probability.

    The probe minimizes mean logistic loss plus
    ``lambda_ / 2 * ||w||²`` without an intercept. It is represented in the
    reviewed-sample space as ``w = X.T @ a`` and fitted by a fixed number of
    deterministic Newton/IRLS steps on the Gram matrix. Candidate embeddings are
    not renormalized.

    By default, the final score is
    ``(1 - w_probe) * base + w_probe * sigmoid(f(candidate))``. With a
    standardized mix, the base similarity and raw decision function are each
    robustly z-scored before applying the same weighted mix.

    ``positive_evidence`` is the candidate's fitted positive-class probability;
    ``negative_evidence`` is its complementary negative-class probability. If
    fitting is impossible, the base similarity is returned and evidence is absent.

    Args:
        lambda_: Strength of L2 regularization on the probe weights.
        w_probe: Weight assigned to the probe probability in the final score.
        n_iterations: Fixed number of Newton/IRLS fitting iterations.
        n_weak_negatives: Candidate embeddings to sample from the retrieval tail
            as additional weak negatives. Zero disables weak negatives.
        seed: Seed used to sample weak negatives deterministically.
        mix: ``"linear"`` for the original probability mix, or
            ``"standardized"`` for a scale-fair mix of robust z-scores.
    """

    name = "linear_probe"

    def __init__(
        self,
        lambda_: float = 1.0,
        w_probe: float = 0.5,
        n_iterations: int = 50,
        n_weak_negatives: int = 0,
        seed: int = 0,
        mix: LinearProbeMix = "linear",
    ) -> None:
        if not np.isfinite(lambda_) or lambda_ <= 0.0:
            raise ValueError("lambda_ must be finite and greater than zero.")
        if not np.isfinite(w_probe) or not 0.0 <= w_probe <= 1.0:
            raise ValueError("w_probe must be finite and between zero and one.")
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if n_weak_negatives < 0:
            raise ValueError("n_weak_negatives must be non-negative.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if mix not in ("linear", "standardized"):
            raise ValueError(
                f"Unknown linear probe mix {mix!r}; available: linear, standardized."
            )

        self.lambda_ = float(lambda_)
        self.w_probe = float(w_probe)
        self.n_iterations = int(n_iterations)
        self.n_weak_negatives = int(n_weak_negatives)
        self.seed = int(seed)
        self.mix = mix

    def _fallback(self, inp: RescoreInput, reason: str) -> RescoreResult:
        """Return finite base scores and log one no-fit reason."""
        from toolbox.bricks.rescoring import RescoreResult

        logger.info("linear_probe could not fit; using base similarity: %s", reason)
        scores = np.nan_to_num(
            np.asarray(inp.base_similarity, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return RescoreResult(scores=scores)

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Fit the probe and return mixed scores in candidate input order."""
        from toolbox.bricks.rescoring import RescoreResult

        embeddings = np.asarray(inp.embeddings, dtype=np.float64)
        base_similarity = np.asarray(inp.base_similarity, dtype=np.float64)
        positive = np.asarray(inp.positive_embeddings, dtype=np.float64)
        negative = np.asarray(inp.negative_embeddings, dtype=np.float64)

        if embeddings.ndim != 2 or base_similarity.shape != (embeddings.shape[0],):
            return self._fallback(inp, "invalid candidate shapes")
        dimension = embeddings.shape[1]
        if positive.ndim != 2 or positive.shape[1] != dimension:
            return self._fallback(inp, "invalid positive embedding shape")
        if negative.ndim != 2 or negative.shape[1] != dimension:
            return self._fallback(inp, "invalid negative embedding shape")
        if positive.shape[0] == 0:
            return self._fallback(inp, "no positive reviews")
        if negative.shape[0] == 0 and self.n_weak_negatives == 0:
            return self._fallback(inp, "no negative reviews")
        if self.n_iterations == 0:
            return self._fallback(inp, "n_iterations is zero")
        if not all(
            np.all(np.isfinite(values))
            for values in (embeddings, base_similarity, positive, negative)
        ):
            return self._fallback(inp, "non-finite input")

        weak_negatives = _select_weak_negatives(
            embeddings,
            base_similarity,
            self.n_weak_negatives,
            self.seed,
        )
        if negative.shape[0] == 0 and weak_negatives.shape[0] == 0:
            return self._fallback(inp, "no usable negative examples")

        train_embeddings = np.concatenate((positive, negative, weak_negatives), axis=0)
        labels = np.concatenate(
            (
                np.ones(positive.shape[0], dtype=np.float64),
                np.zeros(negative.shape[0] + weak_negatives.shape[0], dtype=np.float64),
            )
        )
        coefficients = _fit_dual_coefficients(
            train_embeddings,
            labels,
            self.lambda_,
            self.n_iterations,
        )
        if coefficients is None:
            return self._fallback(inp, "Newton solve failed")

        decision = (embeddings @ train_embeddings.T) @ coefficients
        positive_evidence = _sigmoid(decision)
        negative_evidence = _sigmoid(-decision)
        if self.mix == "linear":
            scores = (
                1.0 - self.w_probe
            ) * base_similarity + self.w_probe * positive_evidence
        else:
            standardized_base = np.asarray(
                normalize_prototype_similarities(
                    base_similarity.tolist(), "standardize"
                ),
                dtype=np.float64,
            )
            standardized_decision = np.asarray(
                normalize_prototype_similarities(decision.tolist(), "standardize"),
                dtype=np.float64,
            )
            if not np.any(standardized_base) or not np.any(standardized_decision):
                logger.info(
                    "linear_probe standardized mix has a column with no spread; "
                    "using base similarity"
                )
                scores = base_similarity
            else:
                scores = (
                    1.0 - self.w_probe
                ) * standardized_base + self.w_probe * standardized_decision
        if not all(
            np.all(np.isfinite(values))
            for values in (positive_evidence, negative_evidence, scores)
        ):
            return self._fallback(inp, "fit produced non-finite scores")

        return RescoreResult(
            scores=np.asarray(scores, dtype=np.float32),
            positive_evidence=np.asarray(positive_evidence, dtype=np.float32),
            negative_evidence=np.asarray(negative_evidence, dtype=np.float32),
        )
