"""Precompute the inverted-softmax denominator for every candidate of one map.

Reads the *live index* rather than the parquet: the denominator is keyed by
`object_search_candidate.id`, which is what the online service returns and therefore the
only key the rescoring can look up.

The probe bank is the query distribution, not a generic vocabulary. It is built from the
Vinci asset classes (the client's own inventory, see `AI_CONTEXT/bricks.md`) plus every
class this map is actually annotated with, each under a few phrasings — hub detection is
a property of the gallery *relative to a query distribution*, so probing with the wrong
distribution characterises the wrong hubs.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

from toolbox.benchmark.annotation_store import ANNOTATION_DB_FILENAME
from toolbox.bricks import db
from toolbox.bricks.georef_source import load_pose_source
from toolbox.bricks.inverted_softmax import DEFAULT_BETA, denominator_path
from toolbox.logging import logger

#: The client's asset inventory for the Vinci PoC test zones, deduplicated.
ASSET_CLASSES = (
    "door", "escalator", "smoke detector", "emergency exit signage",
    "emergency lighting", "direction signage", "security signage",
    "non smoking signage", "fire extinguisher sign", "fire hose reel",
    "fire alarm system", "fire extinguisher", "sprinkler head", "light",
    "motion sensor", "air diffuser", "hole air diffuser", "air curtain",
    "badge reader", "maglock", "CCTV camera", "360 CCTV camera", "FIDS screen",
    "IT screen", "sono speaker", "wifi antenna", "passenger seat",
    "bench of passenger seats", "boarding desk", "boarding pass scanner",
    "garbage bin", "security guard desk", "security curtain",
    "front manual tray conveyor", "rear manual tray conveyor",
    "walk through metal detector", "EDS machine", "EDS working station",
)
#: One term is one probe under each phrasing. Ensembling probes the query distribution
#: better than adding unrelated terms would.
TEMPLATES = (
    "{}", "a photo of a {}", "a {} in an airport terminal",
    "a close-up of a {}",
)
#: Rows pulled from postgres per batch. 1024 halfvec floats per row.
CHUNK = 20000


def annotated_classes(map_path: Path) -> list[str]:
    """Every class this map carries ground truth for, or none if it has no store."""
    store = map_path / ANNOTATION_DB_FILENAME
    if not store.is_file():
        return []
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT DISTINCT class FROM ground_truth_point WHERE class IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()
    return sorted({str(row[0]).strip() for row in rows if str(row[0]).strip()})


def build_probes(map_path: Path) -> list[str]:
    """The bank: asset classes plus this map's own classes, under every phrasing."""
    terms = sorted(set(ASSET_CLASSES) | set(annotated_classes(map_path)))
    return [template.format(term) for term in terms for template in TEMPLATES]


def embed_probes(probes: list[str], device: str) -> np.ndarray:
    """L2-normalised text embeddings, from the mirror's own embedder.

    The mirror is the only embedder in this repo on purpose: a second one would drift
    from what the index was built with, and every similarity here would be off.
    """
    from inference.embedder import METACLIP_MODEL_ID, MetaClipEmbedder

    embedder = MetaClipEmbedder(METACLIP_MODEL_ID, device, "float32")
    bank = np.stack([embedder.embed_text(p) for p in probes]).astype(np.float64)
    norms = np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-6)
    return np.asarray(bank / norms, dtype=np.float64)


def log_sum_exp_over_bank(
    conn: Any, geo_ref_id: int, bank: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the index and return `(ids, log sum_b exp(beta * sim(b, d)))`.

    Streamed in chunks because a georef holds hundreds of thousands of 1024-d vectors;
    the reduction over the bank is what makes the per-candidate result one scalar.
    """
    ids: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    with conn.cursor(name="is_denominators") as cursor:
        cursor.itersize = CHUNK
        cursor.execute(
            "SELECT id, embedding FROM object_search_candidate WHERE geo_ref_id = %s"
            " ORDER BY id",
            [geo_ref_id],
        )
        seen = 0
        while True:
            rows = cursor.fetchmany(CHUNK)
            if not rows:
                break
            chunk_ids = np.fromiter(
                (r[0] for r in rows), dtype=np.int64, count=len(rows)
            )
            vectors = np.stack([_as_vector(r[1]) for r in rows])
            vectors /= np.maximum(
                np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6
            )
            scaled = beta * (bank @ vectors.T)
            # Subtract the column max before exponentiating: beta = 20 on a cosine
            # already reaches e^20, and a bigger bank would overflow float64.
            peak = scaled.max(axis=0)
            chunk_denominator = peak + np.log(np.exp(scaled - peak).sum(axis=0))
            ids.append(chunk_ids)
            denominators.append(chunk_denominator)
            seen += len(rows)
            logger.info("  %d lignes traitees", seen)
    if not ids:
        raise SystemExit(f"georef {geo_ref_id}: aucune ligne dans l'index.")
    return np.concatenate(ids), np.concatenate(denominators)


def _as_vector(value: Any) -> np.ndarray:
    """One halfvec column as float64, whichever way the driver hands it over."""
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float64)
    if isinstance(value, str):
        return np.fromstring(value.strip().strip("[]"), sep=",", dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_path", type=Path)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    map_path = args.map_path.expanduser().resolve()
    geo_ref_id = load_pose_source(map_path).geo_ref_id
    if geo_ref_id is None:
        raise SystemExit(f"{map_path.name}: le manifeste ne porte pas de geo_ref.")

    probes = build_probes(map_path)
    logger.info(
        "banque : %d sondes (%d termes x %d gabarits), georef %s, beta %g",
        len(probes),
        len(probes) // len(TEMPLATES),
        len(TEMPLATES),
        geo_ref_id,
        args.beta,
    )
    bank = embed_probes(probes, args.device)

    with db.connect() as conn:
        ids, denominators = log_sum_exp_over_bank(
            conn, int(geo_ref_id), bank, args.beta
        )

    target = denominator_path(map_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, id=ids, log_denom=denominators, beta=np.float64(args.beta))
    logger.info(
        "%d denominateurs ecrits dans %s (min %.3f, med %.3f, max %.3f)",
        ids.size,
        target,
        denominators.min(),
        np.median(denominators),
        denominators.max(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
