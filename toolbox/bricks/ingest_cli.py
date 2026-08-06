"""Local entrypoint for the object-search ingest step.

Ported from `backend/object_search/management/commands/object_search_ingest.py`.
For a given map directory and georef id this:

  1. Loads keyframe poses from the map's v2 manifest (or legacy `georef.db`)
     and upserts them into `geokeyframe`.
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
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from indexing.grid import filter_by_distance

from toolbox.bricks import Connection, db_schema
from toolbox.bricks.georef_source import KeyframePose, load_pose_source
from toolbox.bricks.ingest import bulk_copy, create_partial_hnsw_index
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
    """Write `georef.db` poses into the local `geokeyframe` stand-in table.

    `GeoRefKeyframe.id` doubles as both the geokeyframe and video-keyframe id —
    see `georef_source`.
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
            ON CONFLICT (id) DO UPDATE SET
                geo_ref_id = EXCLUDED.geo_ref_id,
                video_keyframe_id = EXCLUDED.video_keyframe_id,
                orientation = EXCLUDED.orientation,
                position = EXCLUDED.position
            """,
            rows,
        )
    logger.info("Upserted %d geokeyframe rows for georef %s.", len(rows), geo_ref_id)


def run_ingest(
    conn: Connection,
    *,
    map_path: Path,
    geo_ref_id: int | None = None,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    outputs_dirname: str = DEFAULT_OUTPUTS_DIRNAME,
) -> int:
    """Ingest every prepare output under `map_path`. Returns the row count.

    `geo_ref_id` defaults to the one the v2 manifest records, which is the id
    production uses. v1 maps record none, so there it must be passed.
    """
    with _step("Loading keyframe poses"):
        source = load_pose_source(map_path)
        poses = source.poses
        if geo_ref_id is None:
            if source.geo_ref_id is None:
                raise ValueError(
                    f"{source.path.name} records no geo_ref_id (legacy format), so "
                    "--geo-ref-id is required. It must match the id the online service "
                    "is queried with."
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

    with _step(f"Reading prepare outputs + bulk COPY ({len(capture_dirs)} capture(s))"):
        total_inserted = 0
        captures_with_data = 0
        conn.autocommit = False
        try:
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
    parser.add_argument("map_path", type=Path, help="Map directory (holds georef.db).")
    parser.add_argument(
        "--geo-ref-id",
        type=int,
        default=None,
        help="Georef id to index under. Defaults to the one the v2 manifest records; "
        "required for legacy georef.db maps. Must match what the online service is "
        "queried with.",
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
            geo_ref_id=args.geo_ref_id,
            map_path=args.map_path,
            min_distance=args.min_distance,
            outputs_dirname=args.outputs_dirname,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
