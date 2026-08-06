"""Bridge `prepare`'s output to what `ingest` expects.

**This step is not optional, and its absence is silent.**

`prepare/writer.py` emits `metadata.parquet` with a `thumbnail_file` basename and
**no `depth` column**. `ingest` reads `thumbnail_key` and `depth`. In production
the gap is closed by the Django command `object_search_prepare` (it rewrites
`thumbnail_file` → the S3 `thumbnail_key`, then adds `depth` via `_sample_depths`).
That command is Django-coupled, so this module reproduces the two steps locally.

Skip it and nothing errors: `bulk_copy` writes NULL for the missing columns,
`_compute_object_positions` returns all-NaN, every `object_position` is NULL,
candidate enrichment filters those rows out, and `localize` returns an empty list.
That failure looks exactly like "the model found nothing", which is why it is
called out here and in the ADR.

## On-disk conventions assumed

    {map}/depths/{depth_url basename}   the exact filename the manifest records

That filename is the only thing tried. Keyframe ids are `geo_keyframes` indices, so
a `{kf_id}.tif` fallback would happily serve an unrelated depth map for a valid
request — silently, and with a plausible result.

Zarr depth is not supported: the mirrored pipeline's depth format is the frozen
sqrt-quantised uint16 TIFF. The standalone's zarr reader went to `legacy/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prepare.convention import theta_phi_to_uv

from toolbox.bricks.georef_source import PoseSource, load_pose_source
from toolbox.bricks.vendored.depth_decode import (
    DepthMapNotFound,
    decode_uint16_meters,
    load_depth_map_from_path,
)
from toolbox.logging import logger

DEFAULT_THUMBNAIL_PREFIX = "object-search/thumbnails/"
DEFAULT_DEPTH_DIRNAME = "depths"


def _resolve_depth_path(depth_dir: Path, depth_filename: str | None) -> Path | None:
    """The depth map the manifest names for this keyframe, if it is on disk."""
    if not depth_filename:
        return None
    candidate = depth_dir / depth_filename
    return candidate if candidate.is_file() else None


def sample_depths(
    metadata: pd.DataFrame, map_path: Path, pose_source: PoseSource | None = None
) -> np.ndarray:
    """Per-row depth in metres at each candidate's `(theta_center, phi_center)`.

    Ported from `object_search_prepare._sample_depths`. Returns a float64 array
    aligned with `metadata`; NaN means "no depth available" (no TIFF on disk, or
    the pixel held the invalid sentinel 0). `ingest` persists NaN as NULL.

    Each TIFF is loaded, sampled and freed immediately — full-resolution ERP depth
    maps run ~30 MB each, and holding 400+ at once exhausts memory.
    """
    if len(metadata) == 0:
        return np.empty((0,), dtype=np.float64)

    depth_dir = Path(map_path) / DEFAULT_DEPTH_DIRNAME
    source = pose_source or load_pose_source(map_path)
    depth_by_id = source.depth_filename_by_keyframe_id

    vk_ids = metadata["video_keyframe_id"].to_numpy()
    theta_all = metadata["theta_center"].to_numpy().astype(np.float64)
    phi_all = metadata["phi_center"].to_numpy().astype(np.float64)

    depths = np.full(len(metadata), np.nan, dtype=np.float64)
    missing: list[int] = []
    for vk_id in (int(x) for x in np.unique(vk_ids)):
        idxs = np.flatnonzero(vk_ids == vk_id)
        if idxs.size == 0:
            continue
        depth_path = _resolve_depth_path(depth_dir, depth_by_id.get(vk_id))
        if depth_path is None:
            missing.append(vk_id)
            continue
        try:
            depth_uint16, (h, w) = load_depth_map_from_path(depth_path)
        except (DepthMapNotFound, ValueError) as exc:
            logger.warning("Keyframe %s: unreadable depth map (%s)", vk_id, exc)
            missing.append(vk_id)
            continue

        u, v = theta_phi_to_uv(theta_all[idxs], phi_all[idxs], w, h)
        xi = np.mod(np.round(u).astype(np.int64), w)
        yi = np.clip(np.round(v).astype(np.int64), 0, h - 1)
        raw = depth_uint16[yi, xi]
        decoded = decode_uint16_meters(raw)
        # raw==0 sentinel → decode returns 0.0 → mark as NaN
        depths[idxs] = np.where(raw == 0, np.nan, decoded.astype(np.float64))
        del depth_uint16  # free immediately — do not accumulate all maps in RAM

    if missing:
        logger.warning(
            "No depth map for %d/%d keyframe(s) (e.g. %s) — their candidates get no "
            "3D position and will be invisible to localize.",
            len(missing),
            len(np.unique(vk_ids)),
            missing[:5],
        )
    return depths


def postprocess_metadata(
    metadata_path: Path,
    map_path: Path,
    *,
    thumbnail_prefix: str = DEFAULT_THUMBNAIL_PREFIX,
    pose_source: PoseSource | None = None,
) -> pd.DataFrame:
    """Add `thumbnail_key` + `depth` to a prepare `metadata.parquet`, in place.

    Idempotent: re-running on an already-postprocessed file re-samples depth and
    leaves an existing `thumbnail_key` alone.
    """
    metadata_path = Path(metadata_path)
    metadata = pq.read_table(metadata_path).to_pandas()

    # thumbnail_file (basename) → thumbnail_key. Production makes this an S3 key;
    # locally it is a path relative to the map directory, which is what the
    # toolbox serves cutout images from.
    if "thumbnail_file" in metadata.columns:
        metadata["thumbnail_key"] = metadata["thumbnail_file"].apply(
            lambda f: (thumbnail_prefix + str(f)) if f else ""
        )
        metadata = metadata.drop(columns=["thumbnail_file"])
    elif "thumbnail_key" not in metadata.columns:
        logger.warning(
            "%s has neither thumbnail_file nor thumbnail_key — cutout images will "
            "be unavailable.",
            metadata_path,
        )

    metadata["depth"] = sample_depths(metadata, Path(map_path), pose_source)
    pq.write_table(pa.Table.from_pandas(metadata), metadata_path)

    n_with_depth = int(metadata["depth"].notna().sum())
    logger.info(
        "Sampled depth for %d/%d candidates in %s.",
        n_with_depth,
        len(metadata),
        metadata_path,
    )
    if n_with_depth == 0 and len(metadata) > 0:
        logger.error(
            "No candidate got a depth value. Every object_position will be NULL and "
            "localize will return nothing. Check that %s contains uint16 .tif depth "
            "maps named exactly as the manifest's depth_url basenames.",
            Path(map_path) / DEFAULT_DEPTH_DIRNAME,
        )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add thumbnail_key + depth columns to a prepare metadata.parquet, so "
            "toolbox.bricks.ingest_cli can lift candidates into 3D."
        )
    )
    parser.add_argument("map_path", type=Path, help="Map directory (holds depths/).")
    parser.add_argument(
        "metadata_path",
        type=Path,
        nargs="?",
        help="metadata.parquet to rewrite. Defaults to "
        "<map_path>/object-search/metadata.parquet.",
    )
    parser.add_argument(
        "--thumbnail-prefix",
        default=DEFAULT_THUMBNAIL_PREFIX,
        help="Prefix prepended to thumbnail_file "
        f"(default: {DEFAULT_THUMBNAIL_PREFIX}).",
    )
    args = parser.parse_args(argv)

    metadata_path = (
        args.metadata_path or args.map_path / "object-search" / "metadata.parquet"
    )
    if not metadata_path.is_file():
        parser.error(f"No metadata.parquet at '{metadata_path}'.")

    postprocess_metadata(
        metadata_path, args.map_path, thumbnail_prefix=args.thumbnail_prefix
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
