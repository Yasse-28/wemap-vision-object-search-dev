"""Run the mirrored `prepare` job over a map directory.

## Why this exists — do not replace it with `python -m prepare`

`prepare` has a CLI, but it is a research convenience and **cannot be chained into
the local ingest**. Two of its choices are wrong for us, one of them dangerously:

1. **`video_keyframe_id` is a positional index.** `prepare/cli.py` does
   `image_entries = list(enumerate(image_paths))`, so the id it writes into
   `metadata.parquet` is 0, 1, 2… in *sorted-path* order — not the manifest's
   keyframe id. Both are small integers, so many of them collide by accident and
   candidates get silently attached to **the wrong keyframes**, which puts every
   object in the wrong place with no error anywhere.
2. **It never passes `crops_output_dir`**, so `thumbnail_file` is `""` for every row
   and no cutout thumbnails are written at all.

Production has the same split: the Django `object_search_prepare` command builds
real `(keyframe.id, path)` pairs and calls `run_prepare` as a module, exactly as this
does. So this module is the local counterpart of that command, not a wrapper around
the CLI.

## Resolving keyframe ids

Ids come from the map manifest (`georef_source.load_pose_source`): the index in
`geo_keyframes`, keyed on the `image_url` basename. Lookup is by full filename and
never by stem — since ids are array indices, a stray `2.jpg` would otherwise pair an
unrelated image with a real pose.

Images that resolve to no keyframe are skipped and counted, because indexing an image
whose pose we cannot find produces candidates that can never be positioned.

The venue comes from the manifest too (`map.venue_type`), so there is no `--venue`
flag: an override would silently change what gets indexed relative to production.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prepare.config import PrepareConfig
from prepare.pipeline import run_prepare

from toolbox.bricks.georef_source import PoseSource, load_pose_source
from toolbox.bricks.vendored import proposal_cutouts
from toolbox.logging import logger

# The map directory mirrors the S3 layout: images/ beside depths/.
DEFAULT_IMAGES_DIRNAME = "images"
DEFAULT_OUTPUTS_DIRNAME = "object-search"
# Must agree with prepare_postprocess.DEFAULT_THUMBNAIL_PREFIX, which is resolved
# relative to the map directory when the toolbox serves a preview.
THUMBNAILS_DIRNAME = "thumbnails"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def resolve_images_dir(map_path: Path, images_dirname: str | None = None) -> Path:
    """The ERP images directory: the caller's, else `images/`."""
    map_path = Path(map_path)
    images_dir = map_path / (images_dirname or DEFAULT_IMAGES_DIRNAME)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No ERP images directory at '{images_dir}'.")
    return images_dir


def collect_image_entries(
    map_path: Path,
    images_dirname: str | None = None,
    pose_source: PoseSource | None = None,
) -> list[tuple[int, Path]]:
    """`(keyframe_id, image_path)` pairs for every resolvable ERP image.

    Sorted by keyframe id for reproducible output ordering. Raises if the directory
    is missing or nothing resolves — an empty run would otherwise write an empty
    index and look like a success.
    """
    images_dir = resolve_images_dir(Path(map_path), images_dirname)
    source = pose_source or load_pose_source(map_path)

    paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            f"No {'/'.join(IMAGE_SUFFIXES)} images in '{images_dir}'."
        )

    # Lookup is by full filename, never by stem: manifest keyframe ids are
    # `geo_keyframes` indices, so a file named `2.jpg` must not resolve to
    # keyframe 2 — it would silently pair an unrelated image with a real pose.
    filename_to_id = source.image_filename_to_keyframe_id

    entries: list[tuple[int, Path]] = []
    skipped: list[str] = []
    for path in paths:
        keyframe_id = filename_to_id.get(path.name)
        if keyframe_id is None:
            skipped.append(path.name)
            continue
        entries.append((int(keyframe_id), path))

    if skipped:
        logger.warning(
            "Skipping %d/%d image(s) the manifest does not list (e.g. %s) — without a "
            "pose their candidates could never be positioned.",
            len(skipped),
            len(paths),
            skipped[:5],
        )
    if not entries:
        raise ValueError(
            f"None of the {len(paths)} image(s) in '{images_dir}' resolved to a "
            f"keyframe id via {source.path.name}. The filenames on disk do not match "
            "the ones it records — are these the right map's images?"
        )

    duplicates = len(entries) - len({kf_id for kf_id, _ in entries})
    if duplicates:
        raise ValueError(
            f"{duplicates} image(s) resolved to a keyframe id already used. Ingest "
            "would attach their candidates to the same pose."
        )

    unknown = len(source.poses) - len(entries)
    if unknown > 0:
        logger.info(
            "%d of %d keyframe(s) in %s have no image on disk; only the %d present "
            "will be indexed.",
            unknown,
            len(source.poses),
            source.path.name,
            len(entries),
        )

    entries.sort(key=lambda entry: entry[0])
    logger.info("Resolved %d image(s) to keyframe ids.", len(entries))
    return entries


def run(
    map_path: Path,
    *,
    output_dir: Path | None = None,
    images_dirname: str | None = None,
    limit: int | None = None,
    device: str = "auto",
    batch_size: int = 16,
    yolo_weights: str | None = None,
    cutout_batch: int = proposal_cutouts.DEFAULT_CUTOUT_BATCH,
) -> Path:
    """Detect, cut out and embed. Returns the directory holding the outputs.

    The venue comes from the manifest's `map.venue_type` — it selects the detection
    prompts, so overriding it would index a different candidate set than production.

    `batch_size` and `cutout_batch` are unrelated despite the names: the first is the
    MetaCLIP embedding batch, the second bounds the ERP-replication peak during cutout
    extraction. Only the second affects the OOM the mirrored cutout code hits.
    """
    map_path = Path(map_path)
    output_dir = output_dir or map_path / DEFAULT_OUTPUTS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / THUMBNAILS_DIRNAME
    crops_dir.mkdir(parents=True, exist_ok=True)

    source = load_pose_source(map_path)
    venue = source.venue_type
    if venue is not None:
        logger.info("Using venue '%s' from %s.", venue, source.path.name)
    else:
        logger.warning(
            "No venue: %s records none. Detection falls back to BROAD-only YOLO with "
            "GroundingDINO skipped, which indexes a different candidate set than "
            "production would for this map.",
            source.path.name,
        )

    entries = collect_image_entries(map_path, images_dirname, pose_source=source)
    if limit is not None:
        entries = entries[: max(0, int(limit))]

    config = PrepareConfig(batch_size=int(batch_size))
    config.yolo.device = device
    if yolo_weights:
        config.yolo.weights = yolo_weights

    # The mirrored cutout code keeps two replicated ERPs alive at once, doubling its
    # peak and OOMing an 8 GB card. Swap in the bounded copy; see its module docstring.
    proposal_cutouts.install(int(cutout_batch))
    if int(cutout_batch) != proposal_cutouts.DEFAULT_CUTOUT_BATCH:
        logger.info(
            "Cutout batch set to %d (default %d): lower means less peak GPU memory "
            "and more grid_sample calls. Output is unaffected.",
            int(cutout_batch),
            proposal_cutouts.DEFAULT_CUTOUT_BATCH,
        )

    result = run_prepare(
        image_entries=entries,
        output_dir=output_dir,
        config=config,
        crops_output_dir=crops_dir,
        venue=venue,
    )
    logger.info(
        "prepare wrote %d row(s), dim %d → %s",
        result.n_rows,
        result.embedding_dim,
        result.metadata_path,
    )
    if result.n_rows == 0:
        logger.error(
            "prepare produced no rows: the detectors found nothing in %d image(s). "
            "Check --venue and that the images are equirectangular.",
            len(entries),
        )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the mirrored prepare job over a map directory, resolving real "
            "keyframe ids from the map's v2 manifest — which `python -m prepare` "
            "does not."
        )
    )
    parser.add_argument("map_path", type=Path, help="Map directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--images-dirname",
        default=None,
        help=f"ERP images directory, relative to map_path (default: "
        f"{DEFAULT_IMAGES_DIRNAME}).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="MetaCLIP embedding batch. Does NOT affect cutout memory — see "
        "--cutout-batch.",
    )
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument(
        "--cutout-batch",
        type=int,
        default=proposal_cutouts.DEFAULT_CUTOUT_BATCH,
        help="Proposals per grid_sample call during cutout extraction (default: "
        f"{proposal_cutouts.DEFAULT_CUTOUT_BATCH}, production's value). Peak memory "
        "is this many float32 copies of the ERP: ~199 MB each at 5760x2880. On an 8 GB "
        "card the default needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; 4 "
        "works without it. Output is unaffected either way.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.map_path.is_dir():
        parser.error(f"No such map directory: '{args.map_path}'.")

    run(
        args.map_path,
        output_dir=args.output_dir,
        images_dirname=args.images_dirname,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        yolo_weights=args.yolo_weights,
        cutout_batch=args.cutout_batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
