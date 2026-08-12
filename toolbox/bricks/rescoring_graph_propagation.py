"""Graph label-propagation rescorer for reviewed object-search candidates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toolbox.bricks.rescoring import RescoreInput, RescoreResult

logger = logging.getLogger(__name__)


class GraphPropagationRescorer:
    """Propagate review labels over a candidate-and-prototype cosine graph.

    ``positive_evidence`` and ``negative_evidence`` are the independently
    propagated positive and negative label masses for each candidate. The final
    score explicitly combines them with the MetaCLIP base similarity as
    ``base + gamma * (positive_evidence - negative_evidence)``.

    Non-positive cosine edges have zero weight so that the symmetric degree
    normalization stays real and finite. This method has a failure mode that the
    other rescorers do not: two visually similar but distinct object classes
    joined by one strong edge trade labels, and tuning ``alpha_prop`` cannot
    separate them.

    Args:
        k: Maximum number of outgoing nearest-neighbour edges per node.
        alpha_prop: Weight assigned to propagated evidence at each iteration.
        n_iterations: Fixed number of propagation iterations.
        gamma: Gain applied to signed propagated evidence.
    """

    name = "graph_propagation"

    def __init__(
        self,
        k: int = 10,
        alpha_prop: float = 0.8,
        n_iterations: int = 20,
        gamma: float = 0.1,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        if not 0.0 <= alpha_prop <= 1.0:
            raise ValueError("alpha_prop must be between 0 and 1.")
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if not np.isfinite(gamma):
            raise ValueError("gamma must be finite.")
        self.k = int(k)
        self.alpha_prop = float(alpha_prop)
        self.n_iterations = int(n_iterations)
        self.gamma = float(gamma)

    def score(self, inp: RescoreInput) -> RescoreResult:
        """Return base similarity adjusted by propagated review evidence."""
        # Imported here to avoid a module cycle: rescoring.py owns the shared result
        # type and imports this implementation to register it.
        from toolbox.bricks.rescoring import RescoreResult

        candidate_count = inp.embeddings.shape[0]
        base = _finite_base_similarity(inp.base_similarity)
        no_evidence = np.zeros(candidate_count, dtype=np.float32)
        if candidate_count == 0:
            logger.info("Graph propagation cannot fit without candidates; using base.")
            return RescoreResult(
                scores=base,
                positive_evidence=no_evidence,
                negative_evidence=no_evidence,
                positive_evidence_applied=no_evidence,
                negative_evidence_applied=no_evidence,
            )

        positive_count = inp.positive_embeddings.shape[0]
        negative_count = inp.negative_embeddings.shape[0]
        if positive_count + negative_count == 0:
            logger.info("Graph propagation cannot fit without reviews; using base.")
            return RescoreResult(
                scores=base,
                positive_evidence=no_evidence,
                negative_evidence=no_evidence,
                positive_evidence_applied=no_evidence,
                negative_evidence_applied=no_evidence,
            )

        nodes = np.concatenate(
            (
                inp.embeddings,
                inp.positive_embeddings,
                inp.negative_embeddings,
            ),
            axis=0,
        )
        normalized_graph = _build_normalized_graph(nodes, self.k)
        seeds = np.zeros((nodes.shape[0], 2), dtype=np.float32)
        positive_start = candidate_count
        negative_start = positive_start + positive_count
        seeds[positive_start:negative_start, 0] = 1.0
        seeds[negative_start:, 1] = 1.0

        evidence = seeds.copy()
        retained_seed_weight = 1.0 - self.alpha_prop
        for _ in range(self.n_iterations):
            evidence = (
                self.alpha_prop * (normalized_graph @ evidence)
                + retained_seed_weight * seeds
            )

        positive = np.asarray(evidence[:candidate_count, 0], dtype=np.float32)
        negative = np.asarray(evidence[:candidate_count, 1], dtype=np.float32)
        scores = np.asarray(base + self.gamma * (positive - negative), dtype=np.float32)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        return RescoreResult(
            scores=scores,
            positive_evidence=positive,
            negative_evidence=negative,
            positive_evidence_applied=positive,
            negative_evidence_applied=negative,
        )


def _finite_base_similarity(base_similarity: np.ndarray) -> np.ndarray:
    """Return float32 base scores with non-finite values replaced safely."""
    base = np.asarray(base_similarity, dtype=np.float32)
    return np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=-1.0)


def _build_normalized_graph(nodes: np.ndarray, k: int) -> np.ndarray:
    """Build the symmetric, degree-normalized non-negative cosine kNN graph."""
    node_count = nodes.shape[0]
    if node_count <= 1:
        return np.zeros((node_count, node_count), dtype=np.float32)

    similarities = np.asarray(nodes @ nodes.T, dtype=np.float32)
    similarities = np.nan_to_num(similarities, nan=-np.inf, posinf=1.0, neginf=-np.inf)
    np.fill_diagonal(similarities, -np.inf)

    neighbour_count = min(k, node_count - 1)
    # Stable sorting makes equal-similarity neighbour selection deterministic by
    # retaining ascending node-index order for ties.
    neighbours = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbour_count]
    rows = np.arange(node_count, dtype=np.int64)[:, np.newaxis]
    weights = np.zeros_like(similarities, dtype=np.float32)
    selected_weights = np.maximum(similarities[rows, neighbours], 0.0)
    weights[rows, neighbours] = selected_weights
    weights = np.maximum(weights, weights.T)
    np.fill_diagonal(weights, 0.0)

    degree = np.asarray(np.sum(weights, axis=1, dtype=np.float32), dtype=np.float32)
    inverse_sqrt_degree = np.zeros(node_count, dtype=np.float32)
    connected = degree > 0.0
    inverse_sqrt_degree[connected] = 1.0 / np.sqrt(degree[connected])
    normalized = (
        inverse_sqrt_degree[:, np.newaxis]
        * weights
        * inverse_sqrt_degree[np.newaxis, :]
    )
    return np.asarray(
        np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float32,
    )
