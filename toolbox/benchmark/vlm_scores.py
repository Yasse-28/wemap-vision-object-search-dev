"""Cache the VLM's `p(yes)` per (prompt, candidate), for both gate levels.

One scoring pass serves every configuration in a sweep. `p(yes)` depends on the query
and the cutout and on nothing else — not on the association, the granularity, the depth
cap or the ranking — so it is computed once per prompt and reused by every grid row,
the same way the enriched candidates themselves are. That is what makes gating
measurable at sweep speed rather than at VLM speed.

It is also what lets the two gate levels share one table: the cluster-level gate
aggregates the very scores the detection-level gate applies individually, so a
difference between them is a difference of *where the score is applied*, not of what
was measured.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.render_cutouts import resolve_cutout_path
from toolbox.bricks.vlm_gate import GateConfig, VlmYesNoScorer
from toolbox.logging import logger


def cache_path(cache_dir: Path, map_id: str, prompt: str, config: GateConfig) -> Path:
    """Path of the score table for one prompt and one gate configuration."""
    key = json.dumps(
        {
            "map_id": map_id,
            "prompt": prompt,
            "model_id": config.model_id,
            "question": config.question,
            "quantization": config.quantization,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return cache_dir / f"vlm-{map_id}-{digest}.npz"


def load_or_score(
    prompt: str,
    prompt_candidates: Sequence[EnrichedCandidate],
    *,
    map_path: Path,
    map_id: str,
    cache_dir: Path,
    scorer: VlmYesNoScorer,
    cutout_root: Path | None = None,
    refresh: bool = False,
) -> dict[int, float]:
    """Return `p(yes)` per candidate id, scoring only what is not cached.

    Candidates whose cutout cannot be read are absent from the result rather than
    scored zero: the converted v1 indexes carry *virtual* thumbnail paths that no file
    backs, and a missing image must not read as the model rejecting the candidate.

    Args:
        prompt: The query, as the benchmark issues it.
        prompt_candidates: Candidates to score, carrying their thumbnail keys.
        map_path: Map directory the thumbnail keys are relative to.
        map_id: Toolbox map identifier, part of the cache key.
        cache_dir: Directory holding the score tables.
        scorer: Yes/no scorer, loaded lazily on the first miss.
        cutout_root: Where `render_benchmark_cutouts` wrote the cutouts of a converted
            v1 index, whose stored thumbnail keys are virtual.
        refresh: Rescore and overwrite an existing table.

    Returns:
        Candidate id mapped to `p(yes)`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_path(cache_dir, map_id, prompt, scorer.config)
    cached: dict[int, float] = {}
    if table_path.is_file() and not refresh:
        with np.load(table_path) as data:
            cached = {
                int(key): float(value)
                for key, value in zip(data["ids"], data["scores"])
            }
    resolved: list[tuple[EnrichedCandidate, Path]] = []
    unresolved = 0
    for candidate in prompt_candidates:
        if candidate.id in cached or not candidate.thumbnail:
            continue
        cutout_path = resolve_cutout_path(map_path, candidate.thumbnail, cutout_root)
        if cutout_path is None:
            unresolved += 1
            continue
        resolved.append((candidate, cutout_path))
    if unresolved:
        logger.warning(
            "Prompt %r: %d candidate(s) have a virtual thumbnail and no rendered "
            "cutout — run `python -m toolbox.benchmark.render_benchmark_cutouts`",
            prompt,
            unresolved,
        )
    missing = [candidate for candidate, _ in resolved]
    if missing:
        logger.info(
            "Prompt %r: scoring %d cutout(s) with the VLM (%d already cached)",
            prompt,
            len(missing),
            len(cached),
        )
        scores = scorer.score_paths([path for _, path in resolved], prompt)
        unreadable = 0
        for candidate, score in zip(missing, scores.tolist()):
            if np.isfinite(score):
                cached[candidate.id] = float(score)
            else:
                unreadable += 1
        if unreadable:
            logger.warning(
                "Prompt %r: %d/%d cutout(s) could not be read and stay unscored",
                prompt,
                unreadable,
                len(missing),
            )
        ids = np.asarray(sorted(cached), dtype=np.int64)
        np.savez(
            table_path,
            ids=ids,
            scores=np.asarray([cached[int(i)] for i in ids], dtype=np.float64),
        )
    return cached


def coverage(
    prompt_candidates: Sequence[EnrichedCandidate], scores: dict[int, float]
) -> float:
    """Fraction of candidates that actually carry a VLM score."""
    if not prompt_candidates:
        return 0.0
    scored = sum(1 for candidate in prompt_candidates if candidate.id in scores)
    return scored / len(prompt_candidates)
