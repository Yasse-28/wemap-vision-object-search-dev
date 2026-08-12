"""Does contrasting a prompt against negative prompts separate reviewed cutouts?

Same protocol as `vlm_cue_separability`, on the same free labels: every
`detection_review` row is a human saying whether a cutout matches the query. Here the
cue is not a VLM but a **relative** text score — the query against a set of negative
prompts, on the text↔image axis retrieval already uses.

Three scores per candidate, all from one text encoder pass:

    raw       cos(query, cutout)                     — the retrieval score, the baseline
    margin    cos(query, c) - max_j cos(neg_j, c)    — keeps the query's own scale
    softmax   log p(query | {query} u negatives)     — scale-free per query

The last one is the interesting one, and the reason to try this at all. Absolute
text↔image similarities are not comparable across prompts (each query has its own
offset), which is why the ranking is a ratio and why one shared acceptance threshold
transfers badly. A softmax over a query and its negatives removes that offset by
construction, so it is a candidate for improving *threshold transfer* rather than
within-prompt ranking.

Three sources of negatives, which do not measure the same thing:

- `benchmark`: the map's other benchmark prompts. Free, and exactly the hard negatives
  (a check-in counter against a check-in kiosk), but **closed-set**: production is not
  handed the list of classes, so this is an upper bound, not a result;
- `venue`: the venue vocabulary `prepare` already uses for detection, minus the query
  itself. The honest open-set version;
- `manual`: hand-written negatives for the confusions the reviews actually contain.
  A product would get these from an "exclude" field.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from toolbox.benchmark.pair_cue_separability import _percentiles, _rank_auc
from toolbox.bricks import candidates as candidates_module
from toolbox.bricks import db, map_manifest
from toolbox.bricks.feedback import annotation_db_path, normalize_query
from toolbox.logging import logger

_SELECT_REVIEWS = """
SELECT query, target_id, status
FROM detection_review
WHERE target_type = 'object' AND status IN ('true_positive', 'false_positive')
"""

# Hand-written negatives for the confusions the vinci reviews actually contain — read
# off the cutouts, not invented: the false positives of "check in counter" are
# self-service kiosks with "Check-in" printed on them, and those of "emergency power
# plant" are the building that houses the generator.
# Keyed by `_manual_key`, i.e. letters only, so one entry serves "check in counter" and
# "checkin counter".
MANUAL_NEGATIVES: dict[str, tuple[str, ...]] = {
    "checkincounter": (
        "self-service check-in kiosk",
        "a standing touchscreen terminal",
        "an airport sign",
    ),
    "checkinkiosk": (
        "a staffed check-in counter desk",
        "a baggage drop belt",
    ),
    "emergencypowerplant": (
        "a plain building exterior",
        "a wall with a door",
        "an air conditioning unit",
    ),
    "egates": (
        "a check-in counter desk",
        "a security scanner",
        "a glass wall",
    ),
    "xraymachine": (
        "a conveyor belt without a scanner",
        "a metal detector arch",
        "a luggage trolley",
    ),
    "flightinformationdisplaysystem": (
        "an advertising screen",
        "a television showing a movie",
        "a wall-mounted sign",
    ),
    "poubelle": ("un bac à plantes", "une borne", "une valise"),
    "lampe": ("un détecteur de fumée", "un haut-parleur au plafond", "une fenêtre"),
    "plante": ("un vase vide", "un motif végétal imprimé", "un arbre dehors"),
    "tv": ("un écran d'information", "un tableau accroché au mur", "un miroir"),
}


class CueReport(TypedDict):
    """Separability of one score on one prompt's reviewed cutouts."""

    positive_count: int
    negative_count: int
    auc: float | None
    positive_percentiles: dict[str, float]
    negative_percentiles: dict[str, float]


def winning_negatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    negatives: np.ndarray,
    names: Sequence[str],
    *,
    top: int = 5,
) -> list[tuple[str, int]]:
    """Which negative prompt wins on the *true positives*, and how often.

    The diagnostic that explains a negative set rather than merely scoring it: when a
    negative is a synonym of the query, it takes the maximum on the cutouts the query
    is supposed to match, and the contrast subtracts the very signal being measured.
    """
    if negatives.size == 0 or not labels.any():
        return []
    winners = np.asarray(names)[(embeddings[labels] @ negatives.T).argmax(axis=1)]
    counts = Counter(winners.tolist())
    return counts.most_common(top)


def load_reviews(map_path: Path) -> dict[str, dict[int, bool]]:
    """Reviewed candidate ids per normalised query, mapped to "is a true positive"."""
    path = annotation_db_path(map_path.name, map_path)
    if not path.is_file():
        raise ValueError(f"No annotation database at {path}")
    by_query: dict[str, dict[int, bool]] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for query, target_id, status in conn.execute(_SELECT_REVIEWS):
            normalised = normalize_query(query or "")
            if normalised:
                by_query.setdefault(normalised, {})[int(target_id)] = (
                    status == "true_positive"
                )
    return by_query


def _manual_key(query: str) -> str:
    """Lookup key for `MANUAL_NEGATIVES`, insensitive to spacing.

    Users type what they type: the same prompt appears as "check in counter" on one
    map and "checkin counter" on another, and `normalize_query` deliberately leaves
    inner whitespace alone because "fire exit" and "fire  exit" are different searches.
    For a hand-written table keyed by concept, though, that distinction is noise.
    """
    return "".join(
        character for character in normalize_query(query) if character.isalnum()
    )


def venue_vocabulary(venue_type: str | None) -> tuple[str, ...]:
    """The detection vocabulary `prepare` uses for this venue, as negative prompts."""
    from prepare import prompts as prepare_prompts
    from prepare.yolo_proposals import BROAD_VOCAB

    vocab: list[str] = list(BROAD_VOCAB)
    vocab.extend(prepare_prompts.yolo_specific_vocab(venue_type) or ())
    vocab.extend(prepare_prompts.gdino_classes(venue_type) or ())
    seen: dict[str, None] = {}
    for entry in vocab:
        seen.setdefault(normalize_query(entry), None)
    return tuple(seen)


def negatives_for(
    query: str, source: str, *, all_queries: Sequence[str], venue_type: str | None
) -> tuple[str, ...]:
    """Negative prompts for one query, from the requested source.

    The query is always removed, **compared without spacing**: the reviews store
    "checkin kiosk" and the venue vocabulary says "check in kiosk", and a plain
    normalised comparison leaves the query in its own negative set — where it wins on
    180 of that prompt's true positives and inverts the score.

    Near synonyms are *not* removed. "self check in kiosk" against "check in kiosk", or
    "decorative plant" against "plante", is the open-set reality, and pretending
    otherwise would measure a vocabulary nobody can build in advance.
    """
    squashed = _manual_key(query)
    if source == "benchmark":
        pool: tuple[str, ...] = tuple(all_queries)
    elif source == "venue":
        pool = venue_vocabulary(venue_type)
    elif source == "manual":
        pool = MANUAL_NEGATIVES.get(squashed, ())
    else:
        raise ValueError(f"Unknown negative source {source!r}")
    return tuple(entry for entry in pool if _manual_key(entry) != squashed)


def score_variants(
    embeddings: np.ndarray,
    positive: np.ndarray,
    negatives: np.ndarray,
    *,
    logit_scale: float,
) -> dict[str, np.ndarray]:
    """The three scores, for one query over one candidate matrix.

    Args:
        embeddings: `(n, d)` unit-norm cutout embeddings.
        positive: `(d,)` unit-norm query embedding.
        negatives: `(m, d)` unit-norm negative embeddings; may be empty.
        logit_scale: The CLIP temperature, i.e. `exp(logit_scale)` from the model.

    Returns:
        Score name mapped to `(n,)` values, higher meaning "more like the query".
    """
    raw = embeddings @ positive
    scores = {"raw": raw}
    if negatives.size == 0:
        return scores
    against = embeddings @ negatives.T
    scores["margin"] = raw - against.max(axis=1)
    # log p(query) under the zero-shot classifier over {query} u negatives. Written as
    # -logsumexp of the *differences* so nothing overflows at scale 100.
    logits = np.concatenate([raw[:, None], against], axis=1) * logit_scale
    shifted = logits - logits[:, :1]
    scores["softmax"] = -np.log(np.exp(shifted).sum(axis=1))
    return scores


def _report(values: np.ndarray, labels: np.ndarray) -> CueReport:
    positive = values[labels]
    negative = values[~labels]
    return {
        "positive_count": int(positive.size),
        "negative_count": int(negative.size),
        "auc": _rank_auc(positive, negative, higher_is_same=True),
        "positive_percentiles": _percentiles(positive),
        "negative_percentiles": _percentiles(negative),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse negative-prompt cue arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sources", default="benchmark,venue,manual", help="Comma-separated."
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score every reviewed cutout with each negative source and report AUCs."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    manifest = map_manifest.load_map_manifest(map_path)
    if manifest.geo_ref_id is None:
        raise ValueError(f"{manifest.path}: manifest records no geo_ref_id")
    reviews = load_reviews(map_path)
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]

    from inference.embedder import METACLIP_DTYPE, METACLIP_MODEL_ID, MetaClipEmbedder

    embedder = MetaClipEmbedder(METACLIP_MODEL_ID, args.device, METACLIP_DTYPE)
    logit_scale = float(embedder.model.logit_scale.exp().item())
    logger.info("MetaCLIP2 logit scale %.1f", logit_scale)

    def embed(text: str) -> np.ndarray:
        return np.asarray(embedder.embed_text(text), dtype=np.float64)

    report: dict[str, dict[str, CueReport]] = {}
    winners: dict[str, dict[str, list[tuple[str, int]]]] = {}
    with db.connect() as conn:
        for query, labels in reviews.items():
            by_id = candidates_module.load_prototype_embeddings(
                conn, int(manifest.geo_ref_id), list(labels)
            )
            resolved = [
                candidate_id for candidate_id in labels if candidate_id in by_id
            ]
            if not resolved:
                logger.warning(
                    "Prompt %r: no reviewed id resolves to an embedding", query
                )
                continue
            matrix = np.asarray(
                [by_id[candidate_id] for candidate_id in resolved], dtype=np.float64
            )
            truth = np.asarray([labels[i] for i in resolved], dtype=bool)
            if truth.all() or not truth.any():
                logger.warning("Prompt %r: reviews are all one class", query)
                continue
            positive = embed(query)
            prompt_report: dict[str, CueReport] = {}
            explained: dict[str, list[tuple[str, int]]] = {}
            for source in sources:
                negative_prompts = negatives_for(
                    query,
                    source,
                    all_queries=list(reviews),
                    venue_type=manifest.venue_type,
                )
                negatives = (
                    np.asarray([embed(text) for text in negative_prompts])
                    if negative_prompts
                    else np.empty((0, matrix.shape[1]))
                )
                for name, values in score_variants(
                    matrix, positive, negatives, logit_scale=logit_scale
                ).items():
                    key = "raw" if name == "raw" else f"{source}/{name}"
                    prompt_report[key] = _report(values, truth)
                explained[source] = winning_negatives(
                    matrix, truth, negatives, negative_prompts
                )
                logger.info(
                    "Prompt %r, %s: %d negative(s); wins on true positives: %s",
                    query,
                    source,
                    len(negative_prompts),
                    explained[source][:3],
                )
            report[query] = prompt_report
            winners[query] = explained

    pooled = _pooled(report)
    payload = {
        "per_prompt": report,
        "pooled_auc": pooled,
        "winning_negatives": winners,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
    for name, value in sorted(pooled.items()):
        print(f"{name:24} {value:.3f}")
    return 0


def _pooled(report: dict[str, dict[str, CueReport]]) -> dict[str, float]:
    """Mean per-prompt AUC per score, over the prompts where one exists."""
    names = {name for prompt in report.values() for name in prompt}
    pooled: dict[str, float] = {}
    for name in names:
        values: list[float] = []
        for prompt in report.values():
            cue = prompt.get(name)
            if cue is not None and cue["auc"] is not None:
                values.append(cue["auc"])
        if values:
            pooled[name] = float(np.mean(values))
    return pooled


if __name__ == "__main__":
    raise SystemExit(main())
