from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteResult:
    text: str
    type: str
    canonical_object: str


class Router:
    """Lightweight standalone router.

    The production router depends on separate translation and LLM model wrappers.
    This sandbox default keeps `auto` deterministic and dependency-free; callers can
    still force `search_type="object"` or `search_type="cutout"`.
    """

    def route(self, text: str) -> RouteResult:
        return RouteResult(text=text, type="cutout", canonical_object=text)
