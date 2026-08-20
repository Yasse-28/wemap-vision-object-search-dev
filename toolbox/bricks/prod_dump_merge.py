"""Fold a per-capture prod dump into the single-file layout everything else expects.

Prod runs `prepare` **per VideoCapture** (`object_search_prepare` writes
`<map_id>/<vc_id>/object_search/`), so a dump arrives as one
`metadata.parquet` + `embeddings.npy` pair per capture. `build-index.sh` produces
the other shape: one pair for the whole map. Both are valid — `ingest_cli`'s
`discover_capture_dirs` reads either — but the toolbox's explorer only reads the
single-file one: it keys rows by `row_index` and assumes a dense 0-based counter,
which restarts at 0 in every capture file.

This module converts the first shape into the second, so a prod dump becomes
browsable exactly like a locally prepared map.

## Relation to `benchmarks/merge_prepare_outputs`

The mirror already does this, for the benchmarks: `benchmarks/lib.py`'s
`merge_prepare_outputs` + `save_index`. The row selection here is deliberately the
same (per-capture `video_keyframe_id` membership, `row_index` as the position into
that capture's embeddings, `geokeyframe_id` + `vk_image_path` attached). Two
deviations, both forced:

- **Embeddings are streamed to disk, not concatenated in RAM.** The mirrored helper
  holds every embedding at once — ~2 GB for a 1M-row dump, ~4 GB at the
  `np.concatenate` peak, plus another copy in `save_index`'s `astype`. That does not
  fit on a 16 GB dev box with a browser open.
- **`gk_rows` comes from the manifest + `trajectory.json`, not a Django fixture.**
  `load_fixture_geokeyframes` needs `manage.py dumpdata` output. Locally the poses
  are in the manifest and the `vk_id → video_capture_id` grouping — which the
  manifest does *not* carry — is in the 360-viewer `trajectory.json`, the same file
  `prod_dump_remap` already needs.

## Order of operations

**Remap first.** `prod_dump_remap` must have run: the row selection matches
`video_keyframe_id` against manifest indices, so an unremapped capture (still
carrying prod `VideoKeyframe.id`s) would contribute nothing, or worse, contribute
the handful of ids that collide with a valid index. This refuses to run on one.

No thinning is applied. The merged file is the whole dump; `ingest_cli` thins at
ingest time as it always did, so the explorer can still see the proposals that
thinning drops — which is the interesting part when a query returns nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from toolbox.bricks.ingest_cli import (
    DEFAULT_OUTPUTS_DIRNAME,
    EMBEDDING_DIM,
    discover_capture_dirs,
)
from toolbox.bricks.map_manifest import MapManifest, load_map_manifest
from toolbox.bricks.prod_dump_remap import (
    TRAJECTORY_FILENAME,
    build_keyframe_id_map,
    find_trajectory,
    is_remapped,
)
from toolbox.logging import logger

METADATA_FILENAME = "metadata.parquet"
EMBEDDINGS_FILENAME = "embeddings.npy"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class MergeResult:
    rows: int
    captures: int
    skipped_captures: tuple[str, ...]


def capture_id_by_keyframe(trajectory_path: Path) -> dict[int, int]:
    """`video_keyframe_id → video_capture_id`, from a 360-viewer `trajectory.json`.

    The ids here are prod's; callers hold local ones, so this is only usable
    together with the same mapping `prod_dump_remap` builds. It is used the other
    way round: to know which capture directory a local keyframe's rows live in.
    """
    data = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    out: dict[int, int] = {}
    for capture in data.get("captures") or []:
        vc_id = int(capture["video_capture_id"])
        for keyframe in capture.get("keyframes") or []:
            out[int(keyframe["id"])] = vc_id
    return out


def _keyframes_by_capture(
    manifest: MapManifest, trajectory_path: Path
) -> dict[int, dict[int, str]]:
    """`video_capture_id → {local keyframe id: image filename}`.

    Mirrors `merge_prepare_outputs`' `by_capture`, minus the poses: the merge itself
    needs only the id membership and the image path, and `ingest_cli` is what reads
    poses later.
    """
    id_map = build_keyframe_id_map(manifest, trajectory_path)
    vc_by_prod_id = capture_id_by_keyframe(trajectory_path)
    image_by_local: dict[int, str] = {
        kf.keyframe_id: kf.image_filename for kf in manifest.keyframes
    }

    out: dict[int, dict[int, str]] = defaultdict(dict)
    for prod_id, local_id in zip(id_map.video_keyframe_ids, id_map.local_keyframe_ids):
        vc_id = vc_by_prod_id.get(int(prod_id))
        if vc_id is not None:
            out[vc_id][int(local_id)] = image_by_local.get(int(local_id), "")
    return dict(out)


def _read_capture_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    """A capture's raw float16 dump as `(rows, EMBEDDING_DIM)`.

    Despite the `.npy` name these are `tofile` output — see `ingest_cli`.
    """
    flat = np.fromfile(path, dtype=np.float16)
    embeddings = flat.reshape(-1, EMBEDDING_DIM)
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{path}: {embeddings.shape[0]} embedding rows for "
            f"{expected_rows} metadata rows."
        )
    return embeddings


def merge_dump(
    map_path: Path,
    *,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    trajectory_path: Path | None = None,
) -> MergeResult:
    """Write `{map}/{outputs_dirname}/{metadata.parquet,embeddings.npy}` from the
    per-capture directories beside them.

    Captures are visited in sorted directory order and `row_index` is renumbered
    `0..N-1` over the result, so it stays the dense counter the explorer needs and
    keeps indexing the merged embeddings positionally. `thumbnail_key` is carried
    over untouched — it holds the original per-capture index in its filename and is
    never recomputed from `row_index`.

    Raises:
        FileNotFoundError: no capture directory, or no `trajectory.json`.
        RuntimeError: a capture has not been remapped yet, or nothing matched.
    """
    map_path = Path(map_path)
    root = map_path / outputs_dirname
    trajectory = Path(trajectory_path) if trajectory_path else None
    if trajectory is None:
        trajectory = find_trajectory(map_path, outputs_dirname)
    if trajectory is None or not trajectory.is_file():
        raise FileNotFoundError(
            f"No {TRAJECTORY_FILENAME} for '{map_path}': it carries the capture "
            "grouping this merge needs."
        )

    capture_dirs = [
        d for d in discover_capture_dirs(map_path, outputs_dirname) if d != root
    ]
    if not capture_dirs:
        raise FileNotFoundError(
            f"No per-capture directory under '{root}'. Nothing to merge — a merged "
            f"{METADATA_FILENAME} may already be there."
        )

    manifest = load_map_manifest(map_path)
    by_capture = _keyframes_by_capture(manifest, trajectory)

    metadata_tmp = root / f"{METADATA_FILENAME}.tmp"
    embeddings_tmp = root / f"{EMBEDDINGS_FILENAME}.tmp"
    chunks: list[pd.DataFrame] = []
    skipped: list[str] = []
    total = 0

    try:
        with open(embeddings_tmp, "wb") as sink:
            for capture_dir in capture_dirs:
                table = pq.read_table(capture_dir / METADATA_FILENAME)
                if not is_remapped(table):
                    raise RuntimeError(
                        f"{capture_dir} still carries prod keyframe ids. Run "
                        "`python -m toolbox.bricks.prod_dump_remap` first."
                    )
                wanted = by_capture.get(_capture_id(capture_dir), {})
                frame = table.to_pandas()
                keep = frame["video_keyframe_id"].isin(wanted.keys()).to_numpy()
                if not keep.any():
                    skipped.append(capture_dir.name)
                    logger.warning(
                        "%s: no row belongs to a manifest keyframe — skipped.",
                        capture_dir.name,
                    )
                    del frame
                    continue

                embeddings = _read_capture_embeddings(
                    capture_dir / EMBEDDINGS_FILENAME, len(frame)
                )
                selected = frame.loc[keep].copy()
                # row_index is the position into this capture's embeddings, exactly as
                # `merge_prepare_outputs` uses it.
                embeddings[selected["row_index"].to_numpy()].tofile(sink)
                del embeddings

                selected["geokeyframe_id"] = selected["video_keyframe_id"].astype(
                    np.int64
                )
                selected["vk_image_path"] = (
                    selected["video_keyframe_id"].map(wanted).astype(str)
                )
                chunks.append(selected)
                total += len(selected)
                logger.info(
                    "%s: %d/%d rows merged (running total %d).",
                    capture_dir.name,
                    len(selected),
                    len(frame),
                    total,
                )
                del frame
            sink.flush()
            os.fsync(sink.fileno())

        if not chunks:
            raise RuntimeError("No capture contributed a row; nothing written.")

        merged = pd.concat(chunks, ignore_index=True)
        del chunks[:]
        merged["row_index"] = np.arange(len(merged), dtype=np.int64)
        pq.write_table(pa.Table.from_pandas(merged), metadata_tmp)
        with open(metadata_tmp, "rb") as handle:
            os.fsync(handle.fileno())
        # Both or neither: a metadata.parquet whose embeddings are still the old
        # file is the one inconsistency no reader can detect.
        os.replace(metadata_tmp, root / METADATA_FILENAME)
        os.replace(embeddings_tmp, root / EMBEDDINGS_FILENAME)
    except BaseException:
        metadata_tmp.unlink(missing_ok=True)
        embeddings_tmp.unlink(missing_ok=True)
        raise

    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "georef_id": manifest.geo_ref_id,
                "map_id": manifest.map_id,
                "map_version": manifest.map_version,
                "images_dir": str(map_path / "images"),
                "source": "prod dump, merged by toolbox.bricks.prod_dump_merge",
                "thin_distance": None,
                "captures": [d.name for d in capture_dirs if d.name not in skipped],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Merged %d rows from %d capture(s) into %s.",
        total,
        len(capture_dirs) - len(skipped),
        root,
    )
    return MergeResult(
        rows=total,
        captures=len(capture_dirs) - len(skipped),
        skipped_captures=tuple(skipped),
    )


def _capture_id(capture_dir: Path) -> int:
    """The `video_capture_id` a dump directory is named after."""
    try:
        return int(capture_dir.name)
    except ValueError as exc:
        raise RuntimeError(
            f"{capture_dir}: directory name is not a video_capture_id, so its rows "
            "cannot be attributed to a capture."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a per-capture prod dump into one metadata.parquet + "
            "embeddings.npy, the layout build-index.sh produces and the toolbox "
            "explorer reads."
        )
    )
    parser.add_argument(
        "map_path", type=Path, help="Map directory (holds the v2 manifest)."
    )
    parser.add_argument(
        "--outputs-dirname",
        default=DEFAULT_OUTPUTS_DIRNAME,
        help=f"Dump directory under the map (default: {DEFAULT_OUTPUTS_DIRNAME}).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help=f"Path to {TRAJECTORY_FILENAME}. Defaults to the dump dir, then the "
        "map root.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.map_path.is_dir():
        parser.error(f"No such map directory: '{args.map_path}'.")

    merge_dump(
        args.map_path,
        outputs_dirname=args.outputs_dirname,
        trajectory_path=args.trajectory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
