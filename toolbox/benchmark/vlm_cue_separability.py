"""Measure how well a VLM's yes/no probability separates reviewed cutouts.

The same diagnostic shape as `pair_cue_separability`, and for the same reason: before
implementing a gate, measure whether the cue it would gate on carries any signal. Here
the labels are free — every `detection_review` row is a human saying "this cutout is /
is not what I searched for", which is exactly the question the gate asks.

Read the AUC against the cues already measured on this data:

    depth-point distance between two detections   0.879
    cutout <-> cutout cosine (MetaCLIP)           0.529   <- semantics, today
    VLM p(yes)                                    this script

The middle row is why this is worth running: at the candidate level our own embedding
similarity is barely better than a coin flip, so a semantic gate can only help if it
brings something the embedding does not have.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from toolbox.benchmark.pair_cue_separability import _percentiles, _rank_auc
from toolbox.bricks import candidates, db, map_manifest
from toolbox.bricks.feedback import annotation_db_path, normalize_query
from toolbox.bricks.vlm_gate import DEFAULT_QUESTION, GateConfig, VlmYesNoScorer
from toolbox.logging import logger

_SELECT_REVIEWS = """
SELECT query, target_id, status
FROM detection_review
WHERE target_type = 'object' AND status IN ('true_positive', 'false_positive')
"""


class PromptReport(TypedDict):
    """Separability of the VLM score on one prompt's reviewed cutouts."""

    positive_count: int
    negative_count: int
    unresolved_reviews: int
    positive_percentiles: dict[str, float]
    negative_percentiles: dict[str, float]
    auc: float | None


def load_reviews(map_path: Path) -> dict[str, dict[int, bool]]:
    """Read reviewed candidate ids per normalised query.

    Args:
        map_path: Map directory holding `object-search-annotations.db`.

    Returns:
        Normalised query mapped to candidate id mapped to "is a true positive".
    """
    path = annotation_db_path(map_path.name, map_path)
    if not path.is_file():
        raise ValueError(f"No annotation database at {path}")
    by_query: dict[str, dict[int, bool]] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for query, target_id, status in conn.execute(_SELECT_REVIEWS):
            normalised = normalize_query(query or "")
            if not normalised:
                continue
            by_query.setdefault(normalised, {})[int(target_id)] = (
                status == "true_positive"
            )
    return by_query


def evaluate_prompt(
    scorer: VlmYesNoScorer,
    query: str,
    labels: dict[int, bool],
    thumbnails: dict[int, str],
    map_path: Path,
    *,
    max_per_class: int | None,
) -> PromptReport:
    """Score one prompt's reviewed cutouts and summarise the separation.

    Args:
        scorer: Loaded (or lazily loadable) yes/no scorer.
        query: The prompt, as the user typed it.
        labels: Candidate id mapped to "is a true positive".
        thumbnails: Candidate id mapped to thumbnail key.
        map_path: Map directory the thumbnail keys are relative to.
        max_per_class: Cap on cutouts scored per class, or None for all of them.

    Returns:
        Percentiles and AUC for the two classes.
    """
    positives = [i for i, is_positive in labels.items() if is_positive]
    negatives = [i for i, is_positive in labels.items() if not is_positive]
    unresolved = sum(1 for i in labels if i not in thumbnails)
    positives = [i for i in positives if i in thumbnails][:max_per_class]
    negatives = [i for i in negatives if i in thumbnails][:max_per_class]
    scored: dict[str, np.ndarray] = {}
    for name, ids in (("positive", positives), ("negative", negatives)):
        paths = [map_path / thumbnails[i] for i in ids]
        values = scorer.score_paths(paths, query)
        scored[name] = values[np.isfinite(values)]
        logger.info("Prompt %r: scored %d %s cutout(s)", query, scored[name].size, name)
    return {
        "positive_count": int(scored["positive"].size),
        "negative_count": int(scored["negative"].size),
        "unresolved_reviews": unresolved,
        "positive_percentiles": _percentiles(scored["positive"]),
        "negative_percentiles": _percentiles(scored["negative"]),
        "auc": _rank_auc(scored["positive"], scored["negative"], higher_is_same=True),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse VLM cue separability command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--model", default=GateConfig().model_id)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--quantization", default="4bit", choices=("4bit", "none"))
    parser.add_argument("--batch-size", type=int, default=GateConfig().batch_size)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap the cutouts scored per class and prompt, for a quick look.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score every reviewed cutout and emit a JSON separability report."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    manifest = map_manifest.load_map_manifest(map_path)
    if manifest.geo_ref_id is None:
        raise ValueError(f"{manifest.path}: manifest records no geo_ref_id")
    reviews = load_reviews(map_path)
    scorer = VlmYesNoScorer(
        GateConfig(
            model_id=args.model,
            question=args.question,
            quantization=args.quantization,
            batch_size=args.batch_size,
        )
    )
    report: dict[str, PromptReport] = {}
    with db.connect() as conn:
        for query, labels in reviews.items():
            thumbnails = candidates.load_thumbnail_keys(
                conn, int(manifest.geo_ref_id), list(labels)
            )
            report[query] = evaluate_prompt(
                scorer,
                query,
                labels,
                thumbnails,
                map_path,
                max_per_class=args.max_per_class,
            )
    pooled_auc = _pooled_auc(report)
    payload = {"per_prompt": report, "pooled_auc": pooled_auc, "model": args.model}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


def _pooled_auc(report: dict[str, PromptReport]) -> float | None:
    """Mean of the per-prompt AUCs, over the prompts where one exists.

    Deliberately not an AUC over the pooled scores: the gate always runs inside one
    query, so separating cutouts of *different* queries is not a skill it needs, and
    pooling would credit it for exactly that.
    """
    values = [
        item["auc"]
        for item in report.values()
        if item["auc"] is not None and item["positive_count"] and item["negative_count"]
    ]
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
