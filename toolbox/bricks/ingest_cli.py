"""Local entrypoint for the object-search ingest step.

Ported from `backend/object_search/management/commands/object_search_ingest.py`.
For a given map directory this:

  1. Loads keyframe poses and the georef id from the map's v2 manifest, and
     upserts the poses into `geokeyframe`.
  2. Spatially thins keyframes to keep only those >= min_distance apart.
  3. Reads `metadata.parquet` + `embeddings.npy` from disk for every prepare
     output directory found under the map.
  4. Bulk-COPYs candidates into `object_search_candidate` (see `ingest.py`).
  5. Creates a per-georef partial HNSW index on the embedding column.

Idempotent: re-running for the same georef wipes prior rows before re-inserting,
so the partial HNSW index is always rebuilt cleanly.

## What changed from production

| Production | Here |
|---|---|
| `BaseCommand` | `argparse` |
| `GeoKeyframe` / `GeoRef` / `VideoCapture` ORM | the v2 manifest + a file scan |
| S3 `get_object` | reads files under the map directory |
| `select_keyframes_by_distance` | `indexing.grid.filter_by_distance` [1] |
| Slack `ProcessTracker` | `_step()`, a logging context manager |
| `GeoRef.object_search_status` writes | dropped (no such table) |

[1] The mirror's original, of which the backend helper is a vendored copy — so
    this port removes a vendoring rather than adding one.

`_compute_object_positions`, `EMBEDDING_DIM` and `DEFAULT_MIN_DISTANCE` are
verbatim — that formula is the whole point of the brick.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from indexing.grid import filter_by_distance

from toolbox.bricks import Connection, db_schema
from toolbox.bricks.georef_source import KeyframePose, load_pose_source
from toolbox.bricks.ingest import (
    bulk_copy,
    create_partial_hnsw_index,
    drop_partial_hnsw_index,
)
from toolbox.bricks.vendored import erp
from toolbox.bricks.vendored.maths import quaternion
from toolbox.logging import logger

EMBEDDING_DIM = 1024
DEFAULT_MIN_DISTANCE = 1.5

DEFAULT_OUTPUTS_DIRNAME = "object-search"


@contextlib.contextmanager
def _step(label: str) -> Iterator[None]:
    """Log-and-time one pipeline step.

    Stands in for Slack's `ProcessTracker.execute`. Production's tracker fails via
    `sys.exit(1)`, which is why the command there catches `SystemExit`; here an
    exception is just an exception.
    """
    logger.info("→ %s", label)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        logger.error("✗ %s (after %.1fs)", label, time.perf_counter() - started)
        raise
    logger.info("✓ %s (%.1fs)", label, time.perf_counter() - started)


def _compute_object_positions(
    metadata_all: pd.DataFrame,
    geokeyframe_ids_all: np.ndarray,
    gk_by_vk: dict,
    depths: np.ndarray | None,
) -> np.ndarray:
    """Compute per-candidate object positions in EUS (LocalFrame).

    Returns an (N, 3) float64 array. Rows where depth is NaN/missing are set to
    NaN and will be serialized as NULL by bulk_copy.

    Formula: object_pos = camera_pos + depth * rotate(orientation, ray(theta, phi))

    The ray uses the ERP → OpenGL camera convention from erp.py:
        ray = [cos_phi * sin(theta), sin(phi), -cos_phi * cos(theta)]
    """
    n = len(metadata_all)
    result = np.full((n, 3), np.nan, dtype=np.float64)

    if depths is None:
        return result

    # gk_by_vk: video_keyframe_id → (gk_id, x, y, z, orientation[w,x,y,z]).
    # Build reverse map: gk_id → (x, y, z, orientation).
    gk_id_to_cam: dict[int, tuple] = {
        gk_id: (x, y, z, orientation)
        for gk_id, x, y, z, orientation in gk_by_vk.values()
    }

    cam_xyz = np.empty((n, 3), dtype=np.float64)
    orientations = np.empty((n, 4), dtype=np.float64)
    for i, gk_id in enumerate(geokeyframe_ids_all):
        cam_data = gk_id_to_cam[int(gk_id)]
        cam_xyz[i, 0] = cam_data[0]
        cam_xyz[i, 1] = cam_data[1]
        cam_xyz[i, 2] = cam_data[2]
        orientations[i] = cam_data[3]  # [w, x, y, z]

    theta = metadata_all["theta_center"].to_numpy(dtype=np.float64)
    phi = metadata_all["phi_center"].to_numpy(dtype=np.float64)
    rays_cam = erp.theta_phi_to_opengl_ray_batch(theta, phi)

    # Rotate each ray by its corresponding keyframe orientation.
    # cast_batch is a no-op that only satisfies the NewType; the formula is verbatim.
    rays_world = quaternion.rotate_vectors_batch(
        quaternion.cast_batch(orientations), rays_cam
    )  # (N, 3)

    # object_pos = camera_pos + depth * ray_world (only where depth is valid)
    valid = np.isfinite(depths)
    result[valid] = cam_xyz[valid] + depths[valid, np.newaxis] * rays_world[valid]
    return result


def _ingest_capture(
    conn: Connection,
    geo_ref_id: int,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    geokeyframe_ids: np.ndarray,
    gk_by_vk: dict,
) -> int:
    """Extract candidate columns for one capture, compute object positions, and
    stream them into ``object_search_candidate`` via binary COPY.

    Called once per prepare output directory so no cross-capture arrays are ever
    concatenated — peak memory is bounded by the largest single capture.
    """

    def column(name: str) -> np.ndarray | None:
        return metadata[name].to_numpy() if name in metadata.columns else None

    depths = column("depth")
    if depths is None:
        logger.warning(
            "metadata has no 'depth' column — every object_position will be NULL and "
            "localize will return nothing. Run toolbox.bricks.prepare_postprocess "
            "first."
        )
    object_positions = _compute_object_positions(
        metadata, geokeyframe_ids, gk_by_vk, depths
    )
    return bulk_copy(
        conn,
        geo_ref_id=geo_ref_id,
        geokeyframe_ids=geokeyframe_ids,
        theta_center=metadata["theta_center"].to_numpy(),
        phi_center=metadata["phi_center"].to_numpy(),
        angular_width=metadata["angular_width"].to_numpy(),
        angular_height=metadata["angular_height"].to_numpy(),
        embeddings=embeddings,
        thumbnail_keys=column("thumbnail_key"),
        depths=depths,
        object_positions=object_positions,
        detector_sources=column("detector_source"),
        labels=column("label"),
        detection_scores=column("detection_score"),
    )


def discover_capture_dirs(map_path: Path, outputs_dirname: str) -> list[Path]:
    """Prepare output directories under a map: those holding a `metadata.parquet`.

    Supports both a single `{map}/object-search/` and a per-capture
    `{map}/object-search/{capture}/` layout, which is how the CLI's `--output-dir`
    tends to be used locally.
    """
    root = map_path / outputs_dirname
    if not root.is_dir():
        return []
    if (root / "metadata.parquet").is_file():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "metadata.parquet").is_file())


def _read_capture_outputs(
    capture_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Read `metadata.parquet` + `embeddings.npy` for one capture directory.

    Returns ``None`` if the prepare outputs are missing. Mirrors production's
    length-mismatch guard, which catches a half-written prepare run.
    """
    metadata_path = capture_dir / "metadata.parquet"
    embeddings_path = capture_dir / "embeddings.npy"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        logger.warning("Prepare outputs missing in %s; skipping.", capture_dir)
        return None

    metadata = pq.read_table(metadata_path).to_pandas()
    # Despite the .npy name, `prepare.writer.StreamingWriter` streams the rows with
    # `tofile` — a headerless raw float16 dump, so `np.load` rejects it. Production
    # reads it the same way, with `np.frombuffer` on the S3 body.
    embeddings = np.fromfile(embeddings_path, dtype=np.float16)
    embeddings = embeddings.reshape(-1, EMBEDDING_DIM)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"{capture_dir}: metadata/embeddings length mismatch "
            f"({len(metadata)} vs {embeddings.shape[0]})."
        )
    return metadata, embeddings


def _upsert_geokeyframes(
    conn: Connection, geo_ref_id: int, poses: dict[int, KeyframePose]
) -> None:
    """Write manifest poses into the local `geokeyframe` stand-in table.

    The `geo_keyframes` index doubles as both the geokeyframe and video-keyframe
    id — see `georef_source`.
    """
    rows = [
        (
            pose.keyframe_id,
            geo_ref_id,
            pose.keyframe_id,
            [float(v) for v in pose.orientation_wxyz],
            float(pose.position_eus[0]),
            float(pose.position_eus[1]),
            float(pose.position_eus[2]),
        )
        for pose in poses.values()
    ]
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM geokeyframe WHERE geo_ref_id = %s", [geo_ref_id])
        cursor.executemany(
            """
            INSERT INTO geokeyframe
                (id, geo_ref_id, video_keyframe_id, orientation, position)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s, %s), 0))
            ON CONFLICT (geo_ref_id, id) DO UPDATE SET
                video_keyframe_id = EXCLUDED.video_keyframe_id,
                orientation = EXCLUDED.orientation,
                position = EXCLUDED.position
            """,
            rows,
        )
    logger.info("Upserted %d geokeyframe rows for georef %s.", len(rows), geo_ref_id)


def _unit_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalised float64 copy. The centroid is only meaningful on unit vectors."""
    unit = np.asarray(embeddings, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-6)
    return np.asarray(unit / norms, dtype=np.float64)


def _embedding_centroid(
    capture_dirs: Sequence[Path], kept_vk_ids: set[int]
) -> np.ndarray | None:
    """Mean of every kept unit embedding across all captures, None when there are none.

    A first pass over the parquets, because the centroid has to be the same vector for
    every row written: computing it per capture would store several different centres in
    one index and the online service could only subtract one of them.
    """
    total = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    count = 0
    for capture_dir in capture_dirs:
        outputs = _read_capture_outputs(capture_dir)
        if outputs is None:
            continue
        metadata, embeddings = outputs
        keep = metadata["video_keyframe_id"].isin(kept_vk_ids).to_numpy()
        if not keep.any():
            continue
        unit = _unit_rows(embeddings[keep])
        total += unit.sum(axis=0)
        count += unit.shape[0]
    if count == 0:
        return None
    return total / count


def _center_embeddings(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Unit embeddings with the centroid removed, renormalised.

    The same three lines s6 measures its `centré` column with, so what the benchmark
    scores is what the analysis predicted. Renormalising matters: the HNSW index is
    `halfvec_l2_ops`, and L2 ranks like cosine only on unit vectors.
    """
    centred = _unit_rows(embeddings) - centroid
    centred /= np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-6)
    return np.asarray(centred, dtype=np.float32)


def _store_centroid(
    conn: Connection, geo_ref_id: int, centroid: np.ndarray | None
) -> None:
    """Record the centroid this index was built with, or clear a stale one.

    Clearing on an uncentred ingest is the half that keeps the two sides honest: a
    leftover row would make the online service centre queries against raw vectors,
    which scores worse than either choice made consistently.
    """
    with conn.cursor() as cursor:
        if centroid is None:
            cursor.execute(
                "DELETE FROM object_search_embedding_centroid WHERE geo_ref_id = %s",
                [geo_ref_id],
            )
            return
        # A text literal cast to halfvec: this module packs the COPY stream by hand
        # rather than depending on pgvector's python types, and one row does not
        # justify pulling one in.
        literal = "[" + ",".join(repr(float(value)) for value in centroid) + "]"
        cursor.execute(
            """
            INSERT INTO object_search_embedding_centroid (geo_ref_id, centroid)
            VALUES (%s, %s::halfvec)
            ON CONFLICT (geo_ref_id) DO UPDATE
              SET centroid = EXCLUDED.centroid, created_at = now()
            """,
            [geo_ref_id, literal],
        )


def run_ingest(
    conn: Connection,
    *,
    map_path: Path,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
    center_embeddings: bool = False,
) -> int:
    """Ingest every prepare output under `map_path`. Returns the row count.

    The georef id comes from the manifest — it is the partition key of the
    candidate table and of the partial HNSW index, so taking it from anywhere else
    risks a mismatch with what the online service queries, which returns zero hits
    with no error.
    """
    with _step("Loading keyframe poses"):
        source = load_pose_source(map_path)
        poses = source.poses
        if source.geo_ref_id is None:
            raise ValueError(
                f"{source.path.name} records no geo_ref_id: its geo_levels carry no "
                "'geo_ref'. The manifest is the only source for it — re-export it."
            )
        geo_ref_id = int(source.geo_ref_id)
        logger.info("Using geo_ref_id %s from %s.", geo_ref_id, source.path.name)
        db_schema.ensure_schema(conn)
        _upsert_geokeyframes(conn, geo_ref_id, poses)
        conn.commit()

        # gk_by_vk: video_keyframe_id → (gk_id, x, y, z, orientation[w,x,y,z])
        gk_by_vk = {
            pose.keyframe_id: (
                pose.keyframe_id,
                float(pose.position_eus[0]),
                float(pose.position_eus[1]),
                float(pose.position_eus[2]),
                pose.orientation_wxyz,
            )
            for pose in poses.values()
        }

    with _step(f"Spatial thinning at {min_distance} m"):
        # filter_by_distance wants (gk_id, vk_id, vc_id, x, y, z); vc_id is unused
        # here (one logical capture per map dir), so pass 0.
        rows = [
            (gk_id, vk_id, 0, x, y, z)
            for vk_id, (gk_id, x, y, z, _orientation) in sorted(gk_by_vk.items())
        ]
        kept_vk_ids = {r[1] for r in filter_by_distance(rows, distance=min_distance)}
        logger.info(
            "Spatial thinning: %d / %d keyframes kept.", len(kept_vk_ids), len(rows)
        )

    capture_dirs = discover_capture_dirs(Path(map_path), outputs_dirname)
    if not capture_dirs:
        raise RuntimeError(
            f"No prepare outputs under '{Path(map_path) / outputs_dirname}'. "
            "Run `python -m prepare --output-dir ...` first."
        )

    centroid: np.ndarray | None = None
    if center_embeddings:
        with _step("Computing the embedding centroid"):
            centroid = _embedding_centroid(capture_dirs, kept_vk_ids)
            if centroid is None:
                raise RuntimeError(
                    "Centring was asked for but no kept row carries an embedding."
                )
            logger.info(
                "Centroid over the kept rows: norm %.4f — subtracted from every "
                "stored vector, and from every query for this georef.",
                float(np.linalg.norm(centroid)),
            )

    with _step(f"Reading prepare outputs + bulk COPY ({len(capture_dirs)} capture(s))"):
        total_inserted = 0
        captures_with_data = 0
        conn.autocommit = False
        try:
            # Index-free COPY, then one bulk build below. Leave the partial HNSW
            # index in place and every row is an incremental insert into it — hours
            # instead of minutes, and a worse graph. See `drop_partial_hnsw_index`.
            drop_partial_hnsw_index(conn, geo_ref_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM object_search_candidate WHERE geo_ref_id = %s",
                    [geo_ref_id],
                )
            for capture_dir in capture_dirs:
                outputs = _read_capture_outputs(capture_dir)
                if outputs is None:
                    continue
                metadata, embeddings = outputs

                keep = metadata["video_keyframe_id"].isin(kept_vk_ids).to_numpy()
                if not keep.any():
                    continue
                metadata = metadata.loc[keep].reset_index(drop=True)
                embeddings = embeddings[keep]
                if centroid is not None:
                    embeddings = _center_embeddings(embeddings, centroid)

                geokeyframe_ids = np.fromiter(
                    (
                        gk_by_vk[vk][0]
                        for vk in metadata["video_keyframe_id"].to_numpy()
                    ),
                    dtype=np.int64,
                    count=len(metadata),
                )
                n = _ingest_capture(
                    conn, geo_ref_id, metadata, embeddings, geokeyframe_ids, gk_by_vk
                )
                total_inserted += n
                captures_with_data += 1
                logger.info(
                    "%s: %d candidates copied (running total %d).",
                    capture_dir.name,
                    n,
                    total_inserted,
                )

            if captures_with_data == 0:
                raise RuntimeError(
                    "No prepare outputs matched a kept keyframe; nothing indexed."
                )
            # Same transaction as the rows: an index and its centroid must never be
            # committed apart, or queries centre against vectors that are not.
            _store_centroid(conn, geo_ref_id, centroid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    with _step("Creating partial HNSW index"):
        # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
        conn.autocommit = True
        create_partial_hnsw_index(conn, geo_ref_id)

    logger.info(
        "Object-search index built for georef %s (%d candidates).",
        geo_ref_id,
        total_inserted,
    )
    return total_inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lift prepare outputs into 3D and bulk-COPY them into pgvector, then "
            "build the per-georef partial HNSW index."
        )
    )
    parser.add_argument(
        "map_path", type=Path, help="Map directory (holds the v2 manifest)."
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_MIN_DISTANCE,
        help="Minimum distance between keyframes, in metres "
        f"(default: {DEFAULT_MIN_DISTANCE}).",
    )
    parser.add_argument(
        "--outputs-dirname",
        default=DEFAULT_OUTPUTS_DIRNAME,
        help="Prepare-outputs directory under the map "
        f"(default: {DEFAULT_OUTPUTS_DIRNAME}).",
    )
    parser.add_argument(
        "--center-embeddings",
        action="store_true",
        help=(
            "Subtract the index's own centroid from every stored vector, and record it"
            " so the online service subtracts it from queries too. Reduces hubness"
            " (s6 of map_analysis measures by how much). Off by default: it changes"
            " what the index contains, so it is an experiment, not a default."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.map_path.is_dir():
        parser.error(f"No such map directory: '{args.map_path}'.")

    from toolbox.bricks.db import connect

    with connect() as conn:
        run_ingest(
            conn,
            map_path=args.map_path,
            min_distance=args.min_distance,
            outputs_dirname=args.outputs_dirname,
            center_embeddings=args.center_embeddings,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
