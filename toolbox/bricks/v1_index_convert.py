"""Convert a v1 SQLite object-search index into the v2 prepare-output layout.

The retired standalone pipeline (see `legacy/`) stored one row per detection in
`object-search.db`, with the embedding inline as a float32 BLOB. The v2 pipeline
reads `metadata.parquet` + `embeddings.npy` (see `prepare/writer.py`). Both halves
describe the same thing, and — this is what makes the conversion possible at all —
they agree on the two contracts that matter:

* **the embedding space**: both embed proposal cutouts with
  `facebook/metaclip-2-worldwide-huge-quickgelu`, L2-normalised (1024-d). v1 stores
  float32, v2 float16; the cast is the only change.
* **the angle convention**: v1's `bbox_spherical_coordinates` is
  `[theta, phi, fov_x, fov_y]` in radians
  (`legacy/pipeline/offline/detect/common.py::compute_bbox_spherical`), the same
  quantities as v2's `theta_center` / `phi_center` / `angular_{width,height}`.

so the conversion is a re-shaping, not a re-computation. Nothing is re-detected and
nothing is re-embedded.

## `phi` is negated, and that is not optional

v1 builds its ray from face pixels as `[(cx - c)/f, (cy - c)/f, 1]` — OpenCV, so
its `y` points **down** — and then stores `phi = asin(y)`. v2's `phi` points **up**
(`prepare/convention.py`: `v = (pi/2 - phi)/pi * H`). So v1's `phi` is the negative
of v2's, and the conversion flips it.

Nothing in v1 noticed, because v1 sampled depth from the cubemap pixel rather than
from the stored angle — the column was display-only there. In v2 it is load-bearing
twice: `ingest_cli._compute_object_positions` builds the lifting ray from it, and
`prepare_postprocess.sample_depths` reads the depth TIFF with it. Keep the sign and
every object lands mirrored about the horizon, with no error anywhere.

Measured on 1714 rows over 20 keyframes, v1's own stored depth against the map's
depth TIFF re-sampled at the stored angle: median |Δ| 0.09 m and correlation 0.95
with the flip, against 1.31 m and 0.15 without it.

## What cannot be carried over

- `cutout`, `cluster`, `cluster_cutout` and the OCR columns have no v2
  counterpart: v2 clusters per query, at query time.
- `position_world` / `position_local` / `level` are re-derived from `depth` and
  the manifest pose at ingest (`ingest_cli._compute_object_positions`).
- `detection_score`: v1 never stored the detector confidence. Written as NaN →
  NULL, which is debug/stats only — `localize`'s `confidence` comes from cluster
  geometry, not from this column.
- thumbnails: v1 rendered cutout previews on request and stored none, so there is no
  JPEG to point at. `thumbnail_key` therefore gets a **virtual** path,
  `{outputs_dirname}/rows/{row_index}.png`, which the toolbox's preview route
  re-renders from the ERP on demand (`workbench-index.ts`, `VIRTUAL_ROW_PREVIEW`).
  Writing the files instead would cost 12.6 GB of JPEGs for a million rows — and
  ~139 GB once exFAT's 128 KB clusters are counted. The rendered patch is the stored
  angles re-projected, *not* the pixels v1 embedded; the two differ because v1 cut its
  crops from cubemap faces.

`keyframe_id` is remapped: v1 ids are `georef.db` rows, v2 ids are indices into the
manifest's `geo_keyframes`. The two are joined on the image filename — the only
identifier both sides share. Rows whose keyframe is absent from the manifest, or
which carry no spherical bbox, cannot be converted and are counted and reported.

    python -m toolbox.bricks.v1_index_convert <v1_dir> <map_path> [--v1-db …]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toolbox.bricks.map_manifest import MapManifest, load_map_manifest
from toolbox.logging import logger

EMBEDDING_DIM = 1024
DEFAULT_V1_DB = "object-search.db"
DEFAULT_GEOREF_DB = "georef.db"
DEFAULT_OUTPUTS_DIRNAME = "object-search"
DEFAULT_BATCH_ROWS = 20_000

# `thumbnail_key` points here instead of at a JPEG: no file exists at
# `{outputs_dirname}/rows/{row_index}.png`, and the toolbox's preview route recognises
# the shape and re-renders the cutout from the ERP (see the module docstring). The
# directory name is shared with `workbench-index.ts`'s VIRTUAL_ROW_PREVIEW pattern.
VIRTUAL_THUMBNAIL_DIRNAME = "rows"

# Column order and types are the v2 contract: `prepare/writer.py` for the first ten,
# `prepare_postprocess` for `thumbnail_key`/`depth`, and the prod dump for the last two.
SCHEMA = pa.schema(
    [
        ("row_index", pa.int64()),
        ("video_keyframe_id", pa.int64()),
        ("theta_center", pa.float16()),
        ("phi_center", pa.float16()),
        ("angular_width", pa.float16()),
        ("angular_height", pa.float16()),
        ("detector_source", pa.string()),
        ("label", pa.string()),
        ("detection_score", pa.float32()),
        ("thumbnail_key", pa.string()),
        ("depth", pa.float64()),
        ("geokeyframe_id", pa.int64()),
        ("vk_image_path", pa.string()),
    ]
)


@dataclass
class ConversionStats:
    """Row accounting, so a partial conversion is never silent."""

    read: int = 0
    written: int = 0
    unknown_keyframe: int = 0
    no_spherical_bbox: int = 0
    no_depth: int = 0
    keyframes: set[int] = field(default_factory=set)


def load_keyframe_map(
    georef_db: Path, manifest: MapManifest
) -> tuple[dict[int, int], dict[int, str]]:
    """Map v1 `GeoRefKeyframe.id` → v2 manifest keyframe index, by image filename.

    Returns the id map and, alongside it, the image filename per v2 index (written
    to `vk_image_path` so a mismatch stays visible in the parquet).
    """
    by_filename = manifest.image_filename_to_keyframe_id()
    image_by_index = {kf.keyframe_id: kf.image_filename for kf in manifest.keyframes}

    with sqlite3.connect(f"file:{georef_db}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT id, image_filename FROM GeoRefKeyframe").fetchall()

    id_map = {
        int(v1_id): by_filename[filename]
        for v1_id, filename in rows
        if filename in by_filename
    }
    missing = len(rows) - len(id_map)
    logger.info(
        "Keyframe map: %d/%d v1 keyframes matched a manifest keyframe (%d unmatched).",
        len(id_map),
        len(rows),
        missing,
    )
    if not id_map:
        raise ValueError(
            f"No v1 keyframe in {georef_db.name} matches an image filename in the "
            "manifest. The two describe different maps, or the manifest was "
            "re-exported from different imagery."
        )
    return id_map, image_by_index


def _iter_object_batches(v1_db: Path, batch_rows: int) -> Iterator[list[sqlite3.Row]]:
    """Stream `object` rows in `object_idx` order, in batches.

    Streaming is not an optimisation here: the embeddings alone are ~4 KB per row,
    so a full index does not fit in memory twice.
    """
    conn = sqlite3.connect(f"file:{v1_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT keyframe_id, bbox_spherical_coordinates, embedding, depth, "
            "label, detection_source FROM object ORDER BY object_idx"
        )
        while True:
            batch = cursor.fetchmany(batch_rows)
            if not batch:
                return
            yield batch
    finally:
        conn.close()


def _convert_batch(
    batch: list[sqlite3.Row],
    id_map: dict[int, int],
    image_by_index: dict[int, str],
    stats: ConversionStats,
    thumbnail_prefix: str,
) -> tuple[dict[str, list], np.ndarray]:
    """Turn one batch of v1 rows into v2 columns plus their float16 embeddings."""
    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    embeddings: list[np.ndarray] = []

    for row in batch:
        stats.read += 1
        vk_id = id_map.get(int(row["keyframe_id"]))
        if vk_id is None:
            stats.unknown_keyframe += 1
            continue
        sph_blob = row["bbox_spherical_coordinates"]
        if sph_blob is None or len(sph_blob) != 16:
            stats.no_spherical_bbox += 1
            continue
        theta, phi, fov_x, fov_y = np.frombuffer(sph_blob, dtype=np.float32)
        depth = row["depth"]
        if depth is None:
            stats.no_depth += 1

        row_index = stats.written
        columns["row_index"].append(row_index)
        columns["video_keyframe_id"].append(vk_id)
        columns["theta_center"].append(float(theta))
        # v1's phi is positive downwards; v2's is positive upwards. See the module
        # docstring — this single sign is the difference between correct positions
        # and every object mirrored about the horizon.
        columns["phi_center"].append(-float(phi))
        columns["angular_width"].append(float(fov_x))
        columns["angular_height"].append(float(fov_y))
        columns["detector_source"].append(row["detection_source"] or "")
        columns["label"].append(row["label"])
        columns["detection_score"].append(float("nan"))
        # No JPEG exists — v1 rendered previews on request. This is the *virtual* key
        # the toolbox re-renders from the ERP on demand (see the module docstring).
        columns["thumbnail_key"].append(f"{thumbnail_prefix}{row_index}.png")
        columns["depth"].append(float("nan") if depth is None else float(depth))
        columns["geokeyframe_id"].append(vk_id)
        columns["vk_image_path"].append(image_by_index.get(vk_id, ""))

        embeddings.append(np.frombuffer(row["embedding"], dtype=np.float32))
        stats.keyframes.add(vk_id)
        stats.written += 1

    if not embeddings:
        return columns, np.empty((0, EMBEDDING_DIM), dtype=np.float16)
    stacked = np.stack(embeddings).astype(np.float16)
    if stacked.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"v1 embeddings are {stacked.shape[1]}-d, not {EMBEDDING_DIM}-d — this "
            "index was not built with MetaCLIP2 and cannot be queried by the v2 "
            "online service."
        )
    return columns, stacked


def _table_from_columns(columns: dict[str, list]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(columns[name], type=SCHEMA.field(name).type)
            for name in SCHEMA.names
        ],
        schema=SCHEMA,
    )


def convert(
    *,
    v1_dir: Path,
    map_path: Path,
    v1_db_name: str = DEFAULT_V1_DB,
    georef_db_name: str = DEFAULT_GEOREF_DB,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> ConversionStats:
    """Write `{map_path}/{outputs_dirname}/` from the v1 index in `v1_dir`.

    Refuses to overwrite an existing output directory: the caller is expected to
    move the previous prepare output aside first, so it stays available for a
    re-ingest.
    """
    v1_db = Path(v1_dir) / v1_db_name
    georef_db = Path(v1_dir) / georef_db_name
    for path in (v1_db, georef_db):
        if not path.is_file():
            raise FileNotFoundError(f"No such v1 database: '{path}'.")

    output_dir = Path(map_path) / outputs_dirname
    if output_dir.exists():
        raise FileExistsError(
            f"'{output_dir}' already exists. Move it aside first — overwriting it "
            "would destroy the prepare output the current index was built from."
        )

    manifest = load_map_manifest(Path(map_path))
    id_map, image_by_index = load_keyframe_map(georef_db, manifest)

    output_dir.mkdir(parents=True)
    stats = ConversionStats()
    writer: pq.ParquetWriter | None = None
    try:
        with (output_dir / "embeddings.npy").open("wb") as embeddings_file:
            writer = pq.ParquetWriter(output_dir / "metadata.parquet", SCHEMA)
            for batch_number, batch in enumerate(
                _iter_object_batches(v1_db, batch_rows), start=1
            ):
                columns, embeddings = _convert_batch(
                    batch,
                    id_map,
                    image_by_index,
                    stats,
                    f"{outputs_dirname}/{VIRTUAL_THUMBNAIL_DIRNAME}/",
                )
                if columns["row_index"]:
                    writer.write_table(_table_from_columns(columns))
                    embeddings.tofile(embeddings_file)
                if batch_number % 10 == 0:
                    logger.info(
                        "… %d rows read, %d written, %d keyframes",
                        stats.read,
                        stats.written,
                        len(stats.keyframes),
                    )
    finally:
        if writer is not None:
            writer.close()

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "georef_id": manifest.geo_ref_id,
                "map_id": manifest.map_id,
                "map_version": manifest.map_version,
                "images_dir": str(Path(map_path) / "images"),
                "source": (
                    f"v1 index {v1_db.name}, converted by "
                    "toolbox.bricks.v1_index_convert"
                ),
                "v1_dir": str(Path(v1_dir)),
                "rows": stats.written,
                "keyframes": len(stats.keyframes),
                "thin_distance": None,
                "captures": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Converted %d/%d v1 rows over %d keyframes into %s.",
        stats.written,
        stats.read,
        len(stats.keyframes),
        output_dir,
    )
    if stats.unknown_keyframe or stats.no_spherical_bbox:
        logger.warning(
            "Skipped %d row(s) with an unknown keyframe and %d with no spherical bbox.",
            stats.unknown_keyframe,
            stats.no_spherical_bbox,
        )
    if stats.no_depth:
        logger.warning(
            "%d/%d converted row(s) carry no depth: v1 could not localise them, so "
            "their object_position stays NULL and localize will not see them.",
            stats.no_depth,
            stats.written,
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a v1 SQLite object-search index into the v2 "
            "metadata.parquet + embeddings.npy layout, ready for "
            "toolbox.bricks.ingest_cli."
        )
    )
    parser.add_argument("v1_dir", type=Path, help="v1 map directory (holds the .db).")
    parser.add_argument(
        "map_path", type=Path, help="v2 map directory (holds the manifest)."
    )
    parser.add_argument("--v1-db", default=DEFAULT_V1_DB, help=f"({DEFAULT_V1_DB})")
    parser.add_argument(
        "--georef-db", default=DEFAULT_GEOREF_DB, help=f"({DEFAULT_GEOREF_DB})"
    )
    parser.add_argument(
        "--outputs-dirname",
        default=DEFAULT_OUTPUTS_DIRNAME,
        help=f"Output directory under the map ({DEFAULT_OUTPUTS_DIRNAME}).",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"SQLite fetch size ({DEFAULT_BATCH_ROWS}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    convert(
        v1_dir=args.v1_dir,
        map_path=args.map_path,
        v1_db_name=args.v1_db,
        georef_db_name=args.georef_db,
        outputs_dirname=args.outputs_dirname,
        batch_rows=args.batch_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
