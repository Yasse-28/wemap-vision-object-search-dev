"""Shared client-side seam for review-based candidate rescoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from toolbox.bricks.candidates import (
    FEEDBACK_NORMALIZATIONS,
    FeedbackNormalization,
    apply_feedback_boost,
    normalize_prototype_similarities,
)
from toolbox.bricks.rescoring_graph_propagation import GraphPropagationRescorer
from toolbox.bricks.rescoring_knn_cache import KnnCacheRescorer
from toolbox.bricks.rescoring_linear_probe import LinearProbeRescorer
from toolbox.bricks.rescoring_multi_prototype import MultiPrototypeRescorer


@dataclass(frozen=True)
class RescoreInput:
    """Candidate and per-query review embeddings supplied to a rescorer."""

    candidate_ids: np.ndarray
    embeddings: np.ndarray
    base_similarity: np.ndarray
    positive_embeddings: np.ndarray
    negative_embeddings: np.ndarray


@dataclass(frozen=True)
class RescoreResult:
    """Final scores plus optional per-candidate diagnostic evidence."""

    scores: np.ndarray
    positive_evidence: np.ndarray | None = None
    negative_evidence: np.ndarray | None = None
    positive_evidence_applied: np.ndarray | None = None
    negative_evidence_applied: np.ndarray | None = None


class Rescorer(Protocol):
    """A per-query model that returns one final score per candidate."""

    name: str

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Return final scores and any diagnostic evidence in input order."""


class IdentityRescorer:
    """Preserve base similarities while exercising the rescorer ranking path."""

    name = "identity"

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Return base similarities unchanged and without diagnostic evidence."""
        return RescoreResult(scores=inp.base_similarity)


def _max_cosine(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Maximum cosine to a unit-normalized prototype set, or neutral zeros."""
    if prototypes.shape[0] == 0:
        return np.zeros(embeddings.shape[0], dtype=np.float32)
    return np.asarray(np.max(embeddings @ prototypes.T, axis=1), dtype=np.float32)


class MaxPrototypeRescorer:
    """Client-side equivalent of the existing SQL max-prototype boost."""

    name = "max_prototype"

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        normalization: FeedbackNormalization = "none",
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        if normalization not in FEEDBACK_NORMALIZATIONS:
            choices = ", ".join(FEEDBACK_NORMALIZATIONS)
            raise ValueError(
                f"Unknown prototype normalization {normalization!r}; "
                f"available: {choices}."
            )
        self.normalization = normalization
        # Validate even when a query later has no prototypes.
        normalize_prototype_similarities([], normalization)

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Apply normalized positive and negative max-cosine evidence."""
        positive = _max_cosine(inp.embeddings, inp.positive_embeddings)
        negative = _max_cosine(inp.embeddings, inp.negative_embeddings)
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
                    inp.base_similarity, positive_applied, negative_applied
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


RESCORERS: dict[str, type[Rescorer]] = {
    IdentityRescorer.name: IdentityRescorer,
    MaxPrototypeRescorer.name: MaxPrototypeRescorer,
    MultiPrototypeRescorer.name: MultiPrototypeRescorer,
    LinearProbeRescorer.name: LinearProbeRescorer,
    KnnCacheRescorer.name: KnnCacheRescorer,
    GraphPropagationRescorer.name: GraphPropagationRescorer,
}


def build_rescorer(name: str | None, params: dict[str, object]) -> Rescorer | None:
    """Build a registered rescorer, with ``None`` as the explicit off branch."""
    if name is None:
        return None
    try:
        rescorer_type = RESCORERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RESCORERS))
        raise ValueError(f"Unknown rescorer {name!r}; available: {choices}.") from exc
    return cast(Rescorer, rescorer_type(**params))
