"""
Post-processing hook after retrieval.

Future work: plug in depth/geo/geometry localization here without changing the
retrieval API. Default implementation is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from pipeline.core.types import LoadedIndex

ResultPair = Tuple[str, float]


def postprocess_results(
    results: List[ResultPair],
    *,
    index: "LoadedIndex",
    route_type: str,
    query_text: str,
    extra: Dict[str, Any] | None = None,
) -> List[ResultPair]:
    """Shape or enrich ranked (id, score) pairs before returning to the client."""
    return results
