"""Inverted-softmax rescoring of retrieved candidates — dev-only, opt-in.

State-of-the-art joint embeddings suffer from hubness: a few gallery rows are the
nearest neighbour of almost every query, so every prompt inherits the same false
positives. Inverted softmax (QB-Norm, Bogolin et al., CVPR 2022) divides each
query-candidate similarity by how strongly that candidate answers a fixed bank of
*probe* queries, which demotes rows that answer everything.

Everything expensive is precomputed: the only thing needed at query time is one scalar
per candidate,

    score(q, d) = exp( beta * sim(q, d) - log sum_b exp(beta * sim(b, d)) )

where the log-sum term depends on the bank and the candidate, never on the query. So
this needs **no query embedding, no mirror change and no re-ingest** — unlike centring,
which needs both sides of the similarity and therefore the online service. See
`AI_CONTEXT/bricks.md`.

Measured at retrieval level before wiring: R@1 0.135 -> 0.293 on bbhotel and
0.049 -> 0.086 on vinci, at the cost of tail recall (R@100 0.544 -> 0.470 on vinci).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

#: Where the builder writes the per-candidate denominators, under the map.
DENOMINATOR_FILENAME = Path("benchmark") / "is-denominators.npz"
#: Inverse temperature for cosine similarity, the value the QB-Norm paper reports.
DEFAULT_BETA = 20.0


@dataclass(frozen=True)
class Denominators:
    """`log sum_b exp(beta * sim(b, d))` per candidate id, and the beta it was built at.

    `beta` is recorded because the denominator is not rescalable: changing beta means
    rebuilding, and silently reusing a denominator from another beta would produce a
    score that is neither.
    """

    beta: float
    by_id: dict[int, float]

    def score(self, candidate_id: int, similarity: float, beta: float) -> float | None:
        """The rescored similarity in (0, 1], or None when there is no denominator.

        Returns the **normalised** form `exp(beta * sim - log_denom)` rather than the
        logit `beta * sim - log_denom`. The two rank identically — `exp` is monotone —
        but the logit is negative, and `match_score` downstream is `sim / best` clipped
        to [0, 1], which a negative best turns into a constant 1.0 for every cluster.
        Ranking is not the only consumer of the number; its scale has to stay usable.

        A candidate the bank never saw — a row ingested after the build — keeps its raw
        similarity rather than being scored on a missing term.
        """
        denominator = self.by_id.get(int(candidate_id))
        if denominator is None:
            return None
        return float(np.exp(beta * float(similarity) - denominator))


def denominator_path(map_path: Path) -> Path:
    """Where this map's denominators live."""
    return map_path / DENOMINATOR_FILENAME


def load_denominators(map_path: Path) -> Denominators | None:
    """Read the map's denominators, or None when they were never built."""
    path = denominator_path(map_path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        ids = data["id"].astype(np.int64)
        values = data["log_denom"].astype(np.float64)
        beta = float(data["beta"]) if "beta" in data else DEFAULT_BETA
    return Denominators(beta=beta, by_id=dict(zip(ids.tolist(), values.tolist())))


def apply_inverted_softmax(
    candidates: Iterable[Any], denominators: Denominators, beta: float
) -> tuple[list[Any], int]:
    """Rescored candidates, and how many actually carried a denominator.

    Writes `similarity_boosted`, because `EnrichedCandidate.effective_similarity` is the
    one hook ranking reads. The review-feedback boost writes the same field, so the two
    must not run together — the caller is what enforces that.

    Returns new candidates rather than mutating: `EnrichedCandidate` is frozen, which is
    what stops a rescoring from leaking into a caller that asked for raw similarities.
    """
    rescored: list[Any] = []
    count = 0
    for candidate in candidates:
        value = denominators.score(candidate.id, candidate.similarity, beta)
        if value is None:
            rescored.append(candidate)
            continue
        rescored.append(replace(candidate, similarity_boosted=value))
        count += 1
    return rescored, count
