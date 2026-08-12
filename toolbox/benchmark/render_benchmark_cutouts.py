"""Render the cutouts a map's benchmark actually needs, for the VLM gate.

A converted v1 index stores no cutout JPEGs, so the gate has nothing to show the model.
Rendering the whole index is possible and pointless: the gate only ever sees the
candidates a prompt retrieves, plus the ones a human reviewed. That is a few thousand
images per map instead of a million, which fits on the local disk in seconds of I/O.

Reads the sweep's own candidate cache, so it renders exactly the set the sweep will
score — no ANN query, no second source of truth about what "the candidates" are.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from toolbox.benchmark.association_sweep import (
    DEFAULT_TIMEOUT_S,
    fetch_prompt_candidates,
)
from toolbox.benchmark.vlm_cue_separability import load_reviews
from toolbox.bricks import candidates as candidates_module
from toolbox.bricks import db, map_manifest, render_cutouts
from toolbox.logging import logger

DEFAULT_CUTOUT_ROOT = Path.home() / ".cache" / "wemap-object-search" / "cutouts"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse cutout-rendering command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://127.0.0.1:45677")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Defaults to {DEFAULT_CUTOUT_ROOT}/<map_id>, on the local disk.",
    )
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutout-batch", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--with-reviews",
        action="store_true",
        help="Also render the reviewed cutouts, which the cue diagnostic scores.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render every cutout the benchmark of one map will ask the VLM about."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    manifest = map_manifest.load_map_manifest(map_path)
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else DEFAULT_CUTOUT_ROOT / manifest.map_id
    )

    prompt_candidates = fetch_prompt_candidates(
        map_path,
        args.ann_base_url,
        args.candidate_count,
        args.cache_dir,
        timeout_s=args.timeout,
    )
    keys = [
        candidate.thumbnail
        for prompt_list in prompt_candidates.values()
        for candidate in prompt_list
    ]
    if args.with_reviews:
        if manifest.geo_ref_id is None:
            raise ValueError(f"{manifest.path}: manifest records no geo_ref_id")
        reviewed_ids = sorted(
            {
                candidate_id
                for labels in load_reviews(map_path).values()
                for candidate_id in labels
            }
        )
        with db.connect() as conn:
            thumbnails = candidates_module.load_thumbnail_keys(
                conn, int(manifest.geo_ref_id), reviewed_ids
            )
        keys.extend(thumbnails.values())

    rows = render_cutouts.rows_for_thumbnail_keys(keys)
    if not rows:
        logger.info(
            "Nothing to render: %s stores real cutouts already", manifest.map_id
        )
        return 0
    rendered = render_cutouts.render_rows(
        map_path,
        rows,
        out_dir,
        device=args.device,
        batch=args.cutout_batch,
    )
    logger.info("%d/%d cutout(s) available under %s", len(rendered), len(rows), out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
