"""Render the cutouts a converted v1 index never wrote (dev-only).

`v1_index_convert` gives every row a **virtual** `thumbnail_key`,
`{outputs}/rows/{row_index}.png`, because v1 rendered its previews from the ERP on
request and stored none. The toolbox re-renders those on demand for the UI; anything
that needs the pixels *offline* — the VLM gate, above all — has nothing to open.

This module renders them once, to a directory of the caller's choosing, so the map's
own directory (often a slow external disk, and 23 GB of JPEGs for a full index) stays
untouched.

**The geometry is the mirror's, not a lookalike.** The stored angles are inverted back
to an ERP pixel box and handed to the same `create_proposal_cutouts` the indexer uses,
so the cutout the model sees is the cutout the embedder saw. The round trip goes
through integer pixels, so the angles the function recomputes differ from the stored
ones by at most half a pixel — visible in a diff, invisible in an image.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from toolbox.logging import logger

# The virtual key `v1_index_convert` writes, and the row index inside it. Kept in sync
# with `VIRTUAL_ROW_PREVIEW` in `toolbox/backend/src/workbench-index.ts`.
VIRTUAL_ROW_KEY = re.compile(r"^.+/rows/(?P<row>\d+)\.(?:png|jpg)$")
DEFAULT_OUT_SIZE = 224
DEFAULT_CUTOUT_BATCH = 4


def virtual_row_index(thumbnail_key: str | None) -> int | None:
    """Row index behind a virtual thumbnail key, or None for a real file."""
    if not thumbnail_key:
        return None
    match = VIRTUAL_ROW_KEY.match(thumbnail_key)
    return int(match.group("row")) if match else None


def rendered_path(out_dir: Path, row_index: int) -> Path:
    """Where this module writes (and later finds) one rendered cutout.

    Sharded by thousands: a converted index has a million rows, and a single
    directory with a million entries is slow to stat on every filesystem here.
    """
    return out_dir / f"{row_index // 1000:04d}" / f"{row_index}.jpg"


def resolve_cutout_path(
    map_path: Path, thumbnail_key: str | None, cutout_root: Path | None
) -> Path | None:
    """Resolve a thumbnail key to a file, preferring a real one.

    Args:
        map_path: Map directory real keys are relative to.
        thumbnail_key: The key as stored on the candidate.
        cutout_root: Directory holding rendered cutouts, if any.

    Returns:
        A path to open, or None when the key is virtual and nothing was rendered.
    """
    if not thumbnail_key:
        return None
    row_index = virtual_row_index(thumbnail_key)
    if row_index is None:
        return map_path / thumbnail_key
    if cutout_root is None:
        return None
    return rendered_path(cutout_root, row_index)


def _erp_boxes(
    theta: np.ndarray,
    phi: np.ndarray,
    angular_width: np.ndarray,
    angular_height: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    """Invert `prepare.convention.erp_pixel_centers_to_spherical` back to boxes."""
    cx = (theta + math.pi) / (2.0 * math.pi) * width
    cy = (math.pi / 2.0 - phi) / math.pi * height
    half_w = angular_width / (2.0 * math.pi) * width / 2.0
    half_h = angular_height / math.pi * height / 2.0
    return [
        (
            int(round(x - dx)),
            int(round(y - dy)),
            int(round(x + dx)),
            int(round(y + dy)),
        )
        for x, y, dx, dy in zip(cx, cy, half_w, half_h)
    ]


def render_rows(
    map_path: Path,
    row_indices: Iterable[int],
    out_dir: Path,
    *,
    device: str = "cuda",
    out_size: int = DEFAULT_OUT_SIZE,
    batch: int = DEFAULT_CUTOUT_BATCH,
    metadata_dirname: str = "object-search",
) -> dict[int, Path]:
    """Render the requested parquet rows to `out_dir`, skipping what exists.

    Rows are grouped by keyframe so each ERP is decoded once, which is the whole cost:
    a 5760x2880 JPEG dwarfs the projection itself.

    Args:
        map_path: Map directory holding `images/` and the metadata parquet.
        row_indices: Parquet row indices to render.
        out_dir: Destination for the JPEGs.
        device: Torch device for the gnomonic sampling.
        out_size: Square cutout side; 224 is what the embedder used.
        batch: Proposals sampled per `grid_sample` call — see the vendored module.
        metadata_dirname: Prepare-output directory inside the map.

    Returns:
        Row index mapped to the rendered file, including rows already present.

    Raises:
        FileNotFoundError: If the metadata parquet or the images directory is absent.
    """
    # Imported lazily: torch and pyarrow are heavy, and this is an occasional job.
    import pyarrow.parquet as pq
    from PIL import Image

    from toolbox.bricks.vendored.proposal_cutouts import create_proposal_cutouts

    wanted = sorted({int(row) for row in row_indices})
    result: dict[int, Path] = {}
    pending: list[int] = []
    for row_index in wanted:
        path = rendered_path(out_dir, row_index)
        if path.is_file():
            result[row_index] = path
        else:
            pending.append(row_index)
    if not pending:
        logger.info("All %d cutout(s) already rendered in %s", len(wanted), out_dir)
        return result

    metadata_path = map_path / metadata_dirname / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata parquet at {metadata_path}")
    images_dir = map_path / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No images directory at {images_dir}")

    table = pq.read_table(
        metadata_path,
        columns=[
            "row_index",
            "vk_image_path",
            "theta_center",
            "phi_center",
            "angular_width",
            "angular_height",
        ],
    ).to_pydict()
    by_row = {
        int(row): index for index, row in enumerate(table["row_index"])
    }  # parquet row order is not guaranteed to equal row_index

    grouped: dict[str, list[int]] = {}
    for row_index in pending:
        position = by_row.get(row_index)
        if position is None:
            logger.warning("Row %d is not in %s", row_index, metadata_path)
            continue
        grouped.setdefault(str(table["vk_image_path"][position]), []).append(row_index)

    logger.info(
        "Rendering %d cutout(s) from %d keyframe(s) into %s",
        sum(len(rows) for rows in grouped.values()),
        len(grouped),
        out_dir,
    )
    for done, (image_name, rows) in enumerate(sorted(grouped.items()), start=1):
        image_path = images_dir / image_name
        if not image_path.is_file():
            logger.warning(
                "Missing ERP %s; %d row(s) stay unrendered", image_path, len(rows)
            )
            continue
        with Image.open(image_path) as handle:
            erp = np.asarray(handle.convert("RGB"))
        height, width = erp.shape[:2]
        positions = [by_row[row_index] for row_index in rows]
        boxes = _erp_boxes(
            np.asarray([table["theta_center"][p] for p in positions], dtype=np.float64),
            np.asarray([table["phi_center"][p] for p in positions], dtype=np.float64),
            np.asarray(
                [table["angular_width"][p] for p in positions], dtype=np.float64
            ),
            np.asarray(
                [table["angular_height"][p] for p in positions], dtype=np.float64
            ),
            width,
            height,
        )
        cutouts = create_proposal_cutouts(erp, boxes, device, out_size, batch=batch)
        for row_index, cutout in zip(rows, cutouts):
            path = rendered_path(out_dir, row_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cutout.image).save(path, quality=92)
            result[row_index] = path
        if done % 100 == 0:
            logger.info("Rendered %d/%d keyframe(s)", done, len(grouped))
    return result


def rows_for_thumbnail_keys(thumbnail_keys: Sequence[str | None]) -> list[int]:
    """The virtual row indices among a set of thumbnail keys."""
    rows = [virtual_row_index(key) for key in thumbnail_keys]
    return sorted({row for row in rows if row is not None})
