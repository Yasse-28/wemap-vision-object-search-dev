#!/usr/bin/env python3
"""Hybrid offline pipeline targeting /localize.

Combines YOLO-World + GroundingDINO proposals with MetaCLIP2 embeddings and
optional 3D localisation (without offline clustering, which is performed at
request time by the online service).

Stages per mini-batch (single-pass, each ERP loaded exactly once)
------------------------------------------------------------------
GPU main thread:
  1. YOLO-World detect_batch (2 super-vocab passes)
  2. GDINO detect_batch (1 venue-specific prompt)
  3. MetaCLIP2 get_image_features for cutout faces + proposal crops combined

CPU post-process worker (overlaps with GPU of next batch):
  4. Class-agnostic spherical NMS + ERP→face projection + per-face cap
  5. Depth zarr load + 3D projection (positions_local/world) in parallel sub-threads
  6. Streaming write to DB (complete object rows + cutout rows)

Resume / checkpoint
-------------------
The DB is the source of truth at all times.  On restart the pipeline:
- Checks which cutout_ids already have real embeddings → skips those keyframes
  for embedding
- Checks the ``processed_keyframe`` table → skips those keyframes for detection
  (legacy ``processed_cutout_ids`` blob in params is folded in for old DBs)
- Each mini-batch (cutouts + objects + processed-keyframe rows) commits as a
  single transaction, so a crash leaves the batch fully applied or not at all
- Objects with NULL position_local (ENU metric) can be re-localised if depths
  become available later

Usage
-----
    python -m pipeline.offline.build_index \\
        --map_path /path/to/maps/<map-id> \\
        --config pipeline/config/config_hybrid_airport.yaml

    # limit for testing
    python -m pipeline.offline.build_index \\
        --map_path ... --config ... --limit 50
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.core.io_streaming import (  # noqa: E402
    CutoutRow,
    ObjectRow,
    load_embedded_cutout_ids,
    load_processed_keyframe_ids,
    mark_keyframes_processed,
    open_build_db,
    write_cutout_batch,
    write_index_metadata,
    write_object_batch,
)
from pipeline.core.logging import logger  # noqa: E402
from pipeline.core.models.base_model import load_model  # noqa: E402
from pipeline.core.models.metaclip import MetaCLIP  # noqa: E402
from pipeline.core.types import (  # noqa: E402
    DEFAULT_ID_STRIDE,
    OBJECT_SEARCH_INDEX_DB_FILENAME,
    UNRESOLVED_LEVEL_SENTINEL,
    ObjectSearchIndexMetadata,
    default_created_utc,
)
from pipeline.offline.detect.common import (  # noqa: E402
    FACE_NAME_TO_LOCAL_INDEX,
    HybridDetection,
    _area_filter,
    _aspect_ratio_filter,
    _clip_bboxes_to_image,
    compute_bbox_spherical,
    project_erp_bbox_to_face,
    spherical_nms,
)
from pipeline.offline.detect.hybrid_detector import (  # noqa: E402
    GdinoVenueDetector,
    YoloWorldDetector,
)
from pipeline.offline.detect.prompts import (  # noqa: E402
    DEFAULT_GDINO_LABEL,
    DEFAULT_VENUE,
    VENUE_PROMPTS,
)
from pipeline.offline.ingest.equirect_extract import (  # noqa: E402
    extract_cutouts_cubemap_for_keyframe,
    load_equirect_rgb,
)
from pipeline.offline.ingest.image_io import load_image_paths  # noqa: E402
from pipeline.offline.ingest.keyframe_id import (  # noqa: E402
    keyframe_id_from_image_path,
)
from pipeline.offline.localize.depth_io import (  # noqa: E402
    load_depth_from_tif,
    load_depth_from_zarr,
)
from pipeline.offline.localize.georef import (  # noqa: E402
    load_georef_from_db,
    load_image_filename_to_keyframe_id,
    wds_to_enu,
)
from pipeline.offline.localize.localize_3d import (  # noqa: E402
    cubemap_pixel_to_equirect_uv,
    cubemap_pixel_to_ray,
    keyframe_levels_from_poses,
    load_keyframe_poses_from_georef,
    sample_depth_at_pixel,
    transform_to_world,
)
from pipeline.offline.shared.geometry import CUBEMAP_FACE_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# Prefetch helper: load ERP + extract cubemap faces in a worker thread
# ---------------------------------------------------------------------------

PrepResult = Optional[
    Tuple[int, np.ndarray, list]
]  # (keyframe_id, equirect_rgb, cutouts)


def _sample_image_paths_by_min_dist(
    image_paths: List[Path],
    *,
    image_filename_to_keyframe_id: Optional[Dict[str, int]],
    keyframe_poses: Dict[int, np.ndarray],
    min_dist_m: float,
) -> List[Path]:
    """Select keyframes at least ``min_dist_m`` apart, matching prepare_reloc order.

    The production prepare_reloc step walks keyframes by descending id, suppresses
    later candidates whose camera centers are too close, then returns selected
    keyframes sorted by ascending id. This applies the same rule to image paths.
    """
    if min_dist_m <= 0.0:
        return image_paths

    candidates: List[Tuple[int, Path, np.ndarray]] = []
    missing_pose = 0
    for image_path in image_paths:
        keyframe_id = keyframe_id_from_image_path(
            image_path,
            image_filename_to_keyframe_id=image_filename_to_keyframe_id,
        )
        if keyframe_id is None:
            continue
        pose = keyframe_poses.get(int(keyframe_id))
        if pose is None:
            missing_pose += 1
            continue
        camera_position = np.linalg.inv(pose)[:3, 3]
        candidates.append((int(keyframe_id), image_path, camera_position))

    if missing_pose:
        logger.warning(
            "min_dist sampling dropped %d images with no keyframe pose",
            missing_pose,
        )
    if not candidates:
        logger.warning(
            "min_dist sampling found no keyframes with poses; "
            "keeping original image set",
        )
        return image_paths

    candidates.sort(key=lambda item: item[0], reverse=True)
    positions = np.asarray([item[2] for item in candidates], dtype=np.float64)
    keep_mask = np.ones(len(candidates), dtype=bool)

    for i in range(len(positions)):
        if not keep_mask[i]:
            continue
        dists = np.linalg.norm(positions[i + 1 :] - positions[i], axis=1)
        keep_mask[i + 1 :][dists < min_dist_m] = False

    selected = [candidates[i] for i in range(len(candidates)) if keep_mask[i]]
    selected.sort(key=lambda item: item[0])
    sampled_paths = [path for _keyframe_id, path, _position in selected]
    logger.info(
        "min_dist sampling: selected %d / %d keyframes (min_dist=%.2f m)",
        len(sampled_paths),
        len(candidates),
        min_dist_m,
    )
    return sampled_paths


def _prep_erp_data(
    image_path: Path,
    *,
    image_filename_to_keyframe_id: Optional[Dict[str, int]],
    cubemap_face_size: int,
    cubemap_fov_deg: float,
) -> PrepResult:
    keyframe_id = keyframe_id_from_image_path(
        image_path,
        image_filename_to_keyframe_id=image_filename_to_keyframe_id,
    )
    if keyframe_id is None:
        return None
    equirect_rgb = load_equirect_rgb(image_path)
    if equirect_rgb is None:
        return None
    cutouts = extract_cutouts_cubemap_for_keyframe(
        equirect_rgb,
        keyframe_id,
        face_size=cubemap_face_size,
        fov_deg=cubemap_fov_deg,
        id_stride=DEFAULT_ID_STRIDE,
    )
    return (keyframe_id, equirect_rgb, cutouts)


def _iter_prefetched_erps(
    image_paths: List[Path],
    *,
    image_filename_to_keyframe_id: Optional[Dict[str, int]],
    cubemap_face_size: int,
    cubemap_fov_deg: float,
    workers: int,
    prefetch: int,
) -> Iterator[Tuple[Path, PrepResult]]:
    def _prep(p: Path) -> PrepResult:
        return _prep_erp_data(
            p,
            image_filename_to_keyframe_id=image_filename_to_keyframe_id,
            cubemap_face_size=cubemap_face_size,
            cubemap_fov_deg=cubemap_fov_deg,
        )

    if workers <= 0:
        for p in image_paths:
            yield p, _prep(p)
        return

    max_pending = max(1, int(prefetch), int(workers))
    pending: deque[Tuple[Path, Future[PrepResult]]] = deque()
    iter_paths = iter(image_paths)

    with ThreadPoolExecutor(
        max_workers=int(workers), thread_name_prefix="erp-loader"
    ) as ex:
        for _ in range(max_pending):
            try:
                path = next(iter_paths)
            except StopIteration:
                break
            pending.append((path, ex.submit(_prep, path)))

        while pending:
            path, future = pending.popleft()
            yield path, future.result()
            try:
                next_path = next(iter_paths)
                pending.append((next_path, ex.submit(_prep, next_path)))
            except StopIteration:
                pass


# ---------------------------------------------------------------------------
# Localization-only recovery pass
# ---------------------------------------------------------------------------


def _run_localization_pass(
    conn: sqlite3.Connection,
    *,
    keyframe_poses: dict,
    depth_dir: Path,
    kf_id_to_depth_stem: Dict[int, str],
    face_size: int,
    fov_deg: float,
    min_depth_m: float,
    max_depth_m: float,
    georef: Any = None,
    chunk_size: int = 50_000,
) -> int:
    """Re-localize objects that have NULL position_local (set when depth lookup failed).

    Streams unlocalized object rows in keyset chunks ordered by ``object_idx``
    (objects are inserted in keyframe order, so a keyframe's rows stay contiguous
    and its depth zarr is loaded once per chunk).  Each chunk projects every bbox
    to a 3D world position and writes the results back via bulk UPDATE, keeping
    the RAM footprint bounded regardless of table size.

    Returns the number of objects successfully localized.
    """
    chunk_size = max(1, int(chunk_size))
    n_total = int(
        conn.execute(
            "SELECT count(*) FROM object WHERE position_local IS NULL"
        ).fetchone()[0]
    )
    if n_total == 0:
        return 0

    logger.info(
        "Localization recovery: processing %d objects with missing positions", n_total
    )

    n_localized = 0
    n_seen = 0
    last_object_idx = -1

    while True:
        rows = conn.execute(
            "SELECT object_idx, keyframe_id, cutout_id, bbox_coordinates"
            " FROM object WHERE position_local IS NULL AND object_idx > ?"
            " ORDER BY object_idx LIMIT ?",
            (last_object_idx, chunk_size),
        ).fetchall()
        if not rows:
            break
        last_object_idx = int(rows[-1][0])
        n_seen += len(rows)

        updates: list = []
        depth_cache: Dict[int, Optional[np.ndarray]] = {}  # kf_id → depth (per chunk)

        for row in rows:
            object_idx = int(row[0])
            kf_id = int(row[1])
            cutout_id = int(row[2])

            pose = keyframe_poses.get(kf_id)
            if pose is None:
                updates.append((0, None, None, None, None, object_idx))
                continue
            if kf_id not in depth_cache:
                depth_cache[kf_id] = _load_depth_one(
                    kf_id, depth_dir, kf_id_to_depth_stem
                )[1]
            depth_map = depth_cache[kf_id]
            if depth_map is None:
                updates.append((0, None, None, None, None, object_idx))
                continue
            face_idx = cutout_id % DEFAULT_ID_STRIDE
            if face_idx >= len(CUBEMAP_FACE_ORDER):
                updates.append((0, None, None, None, None, object_idx))
                continue
            face_name = CUBEMAP_FACE_ORDER[face_idx]

            bx1, by1, bx2, by2 = np.frombuffer(row[3], dtype=np.float32)
            loc = _localize_detection_3d(
                bbox_face=(float(bx1), float(by1), float(bx2), float(by2)),
                face_name=face_name,
                face_size=face_size,
                fov_deg=fov_deg,
                depth_map=depth_map,
                pose=pose,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                georef=georef,
            )
            loc_valid = (
                loc.localization_valid if loc.localization_valid is not None else 0
            )
            updates.append(
                (
                    loc_valid,
                    loc.position_keyframe,
                    loc.position_local,
                    loc.position_world,
                    loc.depth,
                    object_idx,
                )
            )
            if loc_valid == 1:
                n_localized += 1

        conn.executemany(
            "UPDATE object SET localization_valid=?, position_keyframe=?,"
            " position_local=?, position_world=?, depth=?"
            " WHERE object_idx=?",
            updates,
        )
        conn.commit()
        logger.info(
            "Localization recovery: %d / %d processed (%d localized)",
            n_seen,
            n_total,
            n_localized,
        )

    logger.info(
        "Localization recovery: %d / %d objects successfully localized",
        n_localized,
        n_total,
    )
    return n_localized


# ---------------------------------------------------------------------------
# Post-process + depth localise + write  (runs in a single worker thread)
# ---------------------------------------------------------------------------


def _embedding_dim_from_db(conn: sqlite3.Connection, default: int = 1024) -> int:
    """Infer the stored embedding dimension from an existing embedding blob.

    Embeddings are persisted as raw float32 bytes, so the dimension is the
    blob length divided by 4. Used on the resume path, where the model may not
    be loaded but the index already contains embeddings.

    Args:
        conn: Open SQLite connection to the index database.
        default: Dimension to assume when no embedding is present yet.

    Returns:
        The embedding dimension in elements.
    """
    for table in ("object", "cutout"):
        row = conn.execute(
            f"SELECT embedding FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row and row[0] is not None:
            return len(row[0]) // np.dtype(np.float32).itemsize
    return default


def _load_depth_one(
    kf_id: int, depth_dir: Path, kf_id_to_stem: Dict[int, str]
) -> Tuple[int, Optional[np.ndarray]]:
    """Resolve and load one keyframe's depth map.

    Depth files may be named after the keyframe ID or the original image filename
    stem (UUID maps), in zarr or tif. Resolution order, first existing wins:
    {stem}.zarr → {kf_id}.zarr → {stem}.tif → {kf_id}.tif.
    """
    stem = kf_id_to_stem.get(kf_id)
    zarr_candidates: List[Path] = []
    tif_candidates: List[Path] = []
    if stem:
        zarr_candidates.append(depth_dir / f"{stem}.zarr")
        tif_candidates.append(depth_dir / f"{stem}.tif")
    zarr_candidates.append(depth_dir / f"{kf_id}.zarr")
    tif_candidates.append(depth_dir / f"{kf_id}.tif")
    for path in zarr_candidates:
        if path.exists():
            try:
                dm, _ = load_depth_from_zarr(str(path))
                return kf_id, dm
            except Exception:
                pass
    for path in tif_candidates:
        if path.exists():
            try:
                dm, _ = load_depth_from_tif(str(path))
                return kf_id, dm
            except Exception:
                pass
    return kf_id, None


def _load_depth_maps(
    keyframe_ids: List[int], depth_dir: Path, kf_id_to_stem: Dict[int, str]
) -> Dict[int, Optional[np.ndarray]]:
    """Load depth maps for the batch's unique keyframes in parallel."""
    depth_maps: Dict[int, Optional[np.ndarray]] = {}
    kf_ids_needed = list({int(kf) for kf in keyframe_ids})
    with ThreadPoolExecutor(
        max_workers=min(4, max(1, len(kf_ids_needed))),
        thread_name_prefix="depth-load",
    ) as depth_pool:
        for kf_id, dm in depth_pool.map(
            lambda k: _load_depth_one(k, depth_dir, kf_id_to_stem), kf_ids_needed
        ):
            depth_maps[kf_id] = dm
    return depth_maps


class _Localization3D(NamedTuple):
    position_keyframe: Optional[bytes]
    position_local: Optional[bytes]
    position_world: Optional[bytes]
    depth: Optional[float]
    localization_valid: Optional[int]


def _localize_detection_3d(
    *,
    bbox_face: Tuple[float, float, float, float],
    face_name: str,
    face_size: int,
    fov_deg: float,
    depth_map: Optional[np.ndarray],
    pose: Any,
    min_depth_m: float,
    max_depth_m: float,
    georef: Any,
) -> _Localization3D:
    """Project a face-local bbox center to a 3D world position via depth.

    ``localization_valid`` is None when not attempted (no depth map / pose),
    0 when the sampled depth was invalid/out-of-range or projection raised, and
    1 on success.
    """
    if depth_map is None or pose is None:
        return _Localization3D(None, None, None, None, None)
    fx1, fy1, fx2, fy2 = bbox_face
    cx_face = 0.5 * (fx1 + fx2)
    cy_face = 0.5 * (fy1 + fy2)
    try:
        u_eq, v_eq = cubemap_pixel_to_equirect_uv(
            cx_face, cy_face, face_name, face_size, fov_deg
        )
        depth_sampled = sample_depth_at_pixel(depth_map, u_eq, v_eq)
        if not (
            np.isfinite(depth_sampled) and min_depth_m <= depth_sampled <= max_depth_m
        ):
            return _Localization3D(None, None, None, None, 0)
        ray_pano = cubemap_pixel_to_ray(cx_face, cy_face, face_name, face_size, fov_deg)
        pos_keyframe = depth_sampled * ray_pano
        pos_world = transform_to_world(pos_keyframe, pose)
        pos_enu = wds_to_enu(pos_world)

        pos_geo_bytes: Optional[bytes] = None
        if georef is not None:
            try:
                geopose = georef.local_position_to_world(pos_world)
                lat = geopose.position.get_latitude_deg()
                lon = geopose.position.get_longitude_deg()
                alt = geopose.position.get_altitude()
                pos_geo_bytes = np.array([lat, lon, alt], np.float32).tobytes()
            except Exception:
                pass
        return _Localization3D(
            pos_keyframe.astype(np.float32).tobytes(),
            pos_enu.astype(np.float32).tobytes(),
            pos_geo_bytes,
            float(depth_sampled),
            1,
        )
    except Exception:
        return _Localization3D(None, None, None, None, 0)


def _post_process_and_write(
    *,
    conn: sqlite3.Connection,
    batch_payload: dict,
    keyframe_poses: dict,
    keyframe_levels: dict,
    depth_dir: Path,
    face_size: int,
    fov_deg: float,
    min_depth_m: float,
    max_depth_m: float,
    do_localize: bool,
    next_object_idx: List[int],  # one-element list used as mutable counter
    georef: Any = None,
) -> None:
    """CPU-side work submitted to the post-process worker pool.

    The main GPU loop has already applied NMS + projection + per-face cap.
    This worker only loads depth zarrs, projects accepted bboxes to 3D world
    coordinates, and writes complete rows to the DB.
    """
    # Payload produced by the main GPU loop (NMS + projection + cap already done).
    cutout_rows: List[CutoutRow] = batch_payload["cutout_rows"]
    # capped_dets: List[List[(det, face_name, bbox_face)]] — one list per ERP
    capped_dets_per_erp = batch_payload["capped_dets"]
    keyframe_ids: List[int] = batch_payload["keyframe_ids"]
    crop_embs_per_erp: List[List[np.ndarray]] = batch_payload["crop_embs"]

    # Write cutout rows (embeddings already computed by GPU).
    # All writes for this batch share one transaction (committed at the end)
    # so a crash leaves the batch fully applied or not at all — no half-written
    # keyframe that would re-detect and duplicate objects on resume.
    write_cutout_batch(conn, cutout_rows, commit=False)

    kf_id_to_stem: Dict[int, str] = batch_payload.get("kf_id_to_stem", {})
    depth_maps: Dict[int, Optional[np.ndarray]] = {}
    if do_localize and depth_dir.is_dir():
        depth_maps = _load_depth_maps(keyframe_ids, depth_dir, kf_id_to_stem)

    object_rows: List[ObjectRow] = []

    textness_per_erp = batch_payload.get("textness_per_erp") or []

    for i, keyframe_id in enumerate(keyframe_ids):
        capped = capped_dets_per_erp[i]  # [(det, face_name, bbox_face), ...]
        emb_list = crop_embs_per_erp[i]  # [emb, ...] aligned with capped
        textness_list = textness_per_erp[i] if i < len(textness_per_erp) else None
        if len(emb_list) != len(capped):
            # Fallback: zero-embeddings if alignment is off (should not happen)
            dim = (
                int(emb_list[0].shape[0])
                if emb_list
                else int(batch_payload.get("projection_dim", 512))
            )
            emb_list = [np.zeros(dim, dtype=np.float32)] * len(capped)

        pose = keyframe_poses.get(int(keyframe_id))
        depth_map = depth_maps.get(int(keyframe_id)) if do_localize else None
        # Level is a keyframe-level property; resolve once per keyframe.
        raw_level = keyframe_levels.get(int(keyframe_id), UNRESOLVED_LEVEL_SENTINEL)
        level = int(raw_level) if raw_level != UNRESOLVED_LEVEL_SENTINEL else None

        for det_i, ((det, face_name, bbox_face), emb) in enumerate(
            zip(capped, emb_list)
        ):
            local_idx = FACE_NAME_TO_LOCAL_INDEX[face_name]
            cutout_id = int(keyframe_id) * DEFAULT_ID_STRIDE + local_idx
            fx1, fy1, fx2, fy2 = (float(v) for v in bbox_face)

            # 3D localisation: use the pre-projected face-local bbox center
            loc = _localize_detection_3d(
                bbox_face=(fx1, fy1, fx2, fy2),
                face_name=face_name,
                face_size=face_size,
                fov_deg=fov_deg,
                depth_map=depth_map,
                pose=pose,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                georef=georef,
            )

            # Compute spherical bbox coordinates
            bbox_sph = compute_bbox_spherical(
                fx1,
                fy1,
                fx2,
                fy2,
                face_name,
                face_size,
                fov_deg,
            )

            # textness_list is aligned with capped (same order), so det_i is the index
            textness = (
                float(textness_list[det_i])
                if textness_list is not None and det_i < len(textness_list)
                else None
            )

            object_rows.append(
                ObjectRow(
                    object_idx=next_object_idx[0],
                    keyframe_id=int(keyframe_id),
                    cutout_id=cutout_id,
                    bbox_coordinates=np.array(
                        [fx1, fy1, fx2, fy2], dtype=np.float32
                    ).tobytes(),
                    bbox_spherical_coordinates=bbox_sph.tobytes(),
                    embedding=emb.astype(np.float32).tobytes(),
                    position_keyframe=loc.position_keyframe,
                    position_local=loc.position_local,
                    position_world=loc.position_world,
                    depth=loc.depth,
                    localization_valid=loc.localization_valid,
                    label=det.label,
                    detection_source=det.source,
                    level=level,
                    textness_score=textness,
                )
            )
            next_object_idx[0] += 1

    write_object_batch(conn, object_rows, commit=False)
    # Mark every keyframe in this batch as fully processed, then commit once so
    # cutouts + objects + progress are atomic.
    mark_keyframes_processed(
        conn,
        (int(kf) for kf in keyframe_ids),
        commit=False,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Lightweight OCR pass  (Stage B — post-build, CPU)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OcrCandidateFilter:
    """Filters selecting unprocessed OCR candidate rows from the object table."""

    textness_threshold: float
    include_labels: Tuple[str, ...]
    exclude_labels: Tuple[str, ...]
    include_detection_sources: Tuple[str, ...]


def _ocr_candidate_where(
    flt: _OcrCandidateFilter, after_object_idx: Optional[int] = None
) -> Tuple[str, List[object]]:
    """Build the SQL WHERE clause + params selecting unprocessed OCR candidates.

    ``after_object_idx`` enables keyset pagination over object_idx.
    """
    clauses = [
        "textness_score > ?",
        "ocr_key IS NULL",
        "COALESCE(ocr_candidate, 0) = 0",
    ]
    params: List[object] = [float(flt.textness_threshold)]
    if after_object_idx is not None:
        clauses.append("object_idx > ?")
        params.append(int(after_object_idx))
    if flt.include_labels:
        placeholders = ",".join("?" for _ in flt.include_labels)
        clauses.append(f"COALESCE(label, '') IN ({placeholders})")
        params.extend(flt.include_labels)
    if flt.exclude_labels:
        placeholders = ",".join("?" for _ in flt.exclude_labels)
        clauses.append(f"COALESCE(label, '') NOT IN ({placeholders})")
        params.extend(flt.exclude_labels)
    if flt.include_detection_sources:
        placeholders = ",".join("?" for _ in flt.include_detection_sources)
        clauses.append(f"COALESCE(detection_source, '') IN ({placeholders})")
        params.extend(flt.include_detection_sources)
    return " AND ".join(clauses), params


class _OcrPrecheck(NamedTuple):
    n_eligible: int
    n_existing_candidates: int
    per_keyframe_attempts: Dict[int, int]


def _ocr_precheck(
    conn: sqlite3.Connection,
    flt: _OcrCandidateFilter,
    *,
    max_candidates: int,
    max_candidates_per_keyframe: int,
) -> Optional[_OcrPrecheck]:
    """Count eligible candidates and seed per-keyframe attempt counts.

    Returns None (caller should return 0) when there is nothing to process or
    the global candidate cap is already met.
    """
    where_sql, where_params = _ocr_candidate_where(flt)
    n_eligible = int(
        conn.execute(
            f"SELECT count(*) FROM object WHERE {where_sql}", where_params
        ).fetchone()[0]
    )
    if n_eligible == 0:
        n_null_textness = conn.execute(
            "SELECT count(*) FROM object WHERE textness_score IS NULL",
        ).fetchone()[0]
        if n_null_textness:
            logger.warning(
                "OCR pass: no candidates, and %d objects have NULL textness_score. "
                "This DB was likely built before OCR textness scoring was enabled; "
                "rebuild the index from scratch to populate OCR candidates.",
                n_null_textness,
            )
        else:
            logger.info(
                "OCR pass: no unprocessed candidates "
                "(textness > %.3f AND ocr_key IS NULL AND ocr_candidate != 1)",
                flt.textness_threshold,
            )
        return None

    n_existing_candidates = 0
    if max_candidates:
        n_existing_candidates = int(
            conn.execute(
                "SELECT count(*) FROM object WHERE COALESCE(ocr_candidate, 0) = 1",
            ).fetchone()[0]
        )
        if n_existing_candidates >= max_candidates:
            logger.info(
                "OCR pass: max_candidates=%d already reached by existing OCR attempts",
                max_candidates,
            )
            return None

    per_keyframe_attempts: Dict[int, int] = {}
    if max_candidates_per_keyframe:
        per_keyframe_attempts = {
            int(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT keyframe_id, count(*) FROM object "
                "WHERE COALESCE(ocr_candidate, 0) = 1 GROUP BY keyframe_id",
            ).fetchall()
        }
    return _OcrPrecheck(n_eligible, n_existing_candidates, per_keyframe_attempts)


def _read_ocr_crop(
    *,
    ocr: Any,
    face_img: Optional[Image.Image],
    bbox_blob: Optional[bytes],
    object_idx: int,
    label: str,
    textness: float,
    lightweight_preprocess: str,
    log_error: bool,
) -> Tuple[Optional[tuple], bool]:
    """Crop, preprocess, and OCR-read one object.

    Returns (accepted UPDATE tuple or None, read_failed). On a backend exception
    read_failed is True and (when ``log_error``) a warning is emitted; the caller
    owns the running error count and abort decision.
    """
    from pipeline.core.models.ocr.identity import ocr_key_string
    from pipeline.offline.refine.ocr_refine import preprocess_for_ocr

    if face_img is None or bbox_blob is None:
        return None, False
    bbox = np.frombuffer(bbox_blob, dtype=np.float32)
    if bbox.size < 4:
        return None, False
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop = face_img.crop((x1, y1, x2, y2))
    if crop.width < 4 or crop.height < 4:
        return None, False
    try:
        if lightweight_preprocess != "raw":
            crop = preprocess_for_ocr(crop, lightweight_preprocess)
        result = ocr.read(crop)
    except Exception as exc:
        if log_error:
            logger.warning(
                "OCR read failed for object_idx=%d label=%s textness=%.4f "
                "crop=%sx%s: %s",
                object_idx,
                label,
                textness,
                crop.width,
                crop.height,
                exc,
            )
        return None, True
    if result.accepted:
        tokens_str = " ".join(result.identity.tokens)
        key_str = ocr_key_string(result.identity)
        return (result.text, tokens_str, key_str, 1, object_idx), False
    return None, False


def _flush_ocr_updates(
    conn: sqlite3.Connection, accepted_updates: list, rejected_updates: list
) -> None:
    """Commit accepted/rejected OCR row updates and clear the buffers."""
    if accepted_updates:
        conn.executemany(
            "UPDATE object SET ocr_text=?, ocr_tokens=?, ocr_key=?, "
            "ocr_candidate=1, ocr_assigned=1, ocr_source=? WHERE object_idx=?",
            accepted_updates,
        )
        accepted_updates.clear()
    if rejected_updates:
        conn.executemany(
            "UPDATE object SET ocr_candidate=1, ocr_assigned=0, ocr_source=0 "
            "WHERE object_idx=?",
            rejected_updates,
        )
        rejected_updates.clear()
    conn.commit()


def _run_ocr_pass(
    conn: sqlite3.Connection,
    *,
    image_paths: List[Path],
    image_filename_to_keyframe_id: Optional[Dict[str, int]],
    cubemap_face_size: int,
    cubemap_fov_deg: float,
    textness_threshold: float,
    lightweight_lang: str,
    lightweight_device: str,
    preferred_device: str,
    lightweight_preprocess: str,
    lightweight_min_score: float,
    lightweight_single_letter_min_score: float,
    candidate_chunk_size: int = 1000,
    max_candidates: int = 0,
    max_candidates_per_keyframe: int = 0,
    progress_interval: int = 1000,
    max_read_errors: int = 20,
    include_labels: Tuple[str, ...] = (),
    exclude_labels: Tuple[str, ...] = (),
    include_detection_sources: Tuple[str, ...] = (),
) -> int:
    """Run lightweight PP-OCR on high-textness objects without an OCR result yet.

    Candidate rows are streamed from SQLite in keyset chunks.  Every attempted
    candidate is marked with ``ocr_candidate=1`` so failed reads are not paid for
    again on later resumes; accepted reads also get ``ocr_assigned=1`` and
    ``ocr_source=1``.

    Returns the number of objects that received an OCR key.
    """
    candidate_chunk_size = max(1, int(candidate_chunk_size))
    max_candidates = max(0, int(max_candidates))
    max_candidates_per_keyframe = max(0, int(max_candidates_per_keyframe))
    progress_interval = max(0, int(progress_interval))
    max_read_errors = max(1, int(max_read_errors))
    resolved_lightweight_device = (
        lightweight_device
        if lightweight_device
        else _default_lightweight_paddle_device(preferred_device=preferred_device)
    )
    flt = _OcrCandidateFilter(
        textness_threshold=float(textness_threshold),
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        include_detection_sources=include_detection_sources,
    )

    precheck = _ocr_precheck(
        conn,
        flt,
        max_candidates=max_candidates,
        max_candidates_per_keyframe=max_candidates_per_keyframe,
    )
    if precheck is None:
        return 0
    n_eligible = precheck.n_eligible
    n_existing_candidates = precheck.n_existing_candidates
    per_keyframe_attempts = precheck.per_keyframe_attempts

    logger.info(
        "OCR pass: %d eligible candidates; chunk_size=%d max_candidates=%s "
        "max_candidates_per_keyframe=%s device=%s",
        n_eligible,
        candidate_chunk_size,
        max_candidates or "unlimited",
        max_candidates_per_keyframe or "unlimited",
        resolved_lightweight_device,
    )

    try:
        from pipeline.core.models.ocr.paddle_lightweight import (
            LightweightPaddleOcrRecognizer,
        )
        from pipeline.offline.ingest.equirect_extract import LazyCutoutProvider
    except ImportError as exc:
        logger.warning("OCR pass skipped — missing dependency: %s", exc)
        return 0

    try:
        ocr = LightweightPaddleOcrRecognizer(
            lang=lightweight_lang,
            device=resolved_lightweight_device,
            min_score=lightweight_min_score,
            single_letter_min_score=lightweight_single_letter_min_score,
        )
    except Exception as exc:
        logger.warning("OCR pass skipped — failed to load PP-OCR model: %s", exc)
        return 0

    provider = LazyCutoutProvider(
        image_paths,
        "cubemap",
        id_stride=DEFAULT_ID_STRIDE,
        cubemap_face_size=cubemap_face_size,
        cubemap_fov_deg=cubemap_fov_deg,
        image_filename_to_keyframe_id=image_filename_to_keyframe_id,
    )

    accepted_updates: list = []
    rejected_updates: list = []
    last_object_idx = -1
    n_attempted = 0
    n_accepted = 0
    n_read_errors = 0
    n_skipped_per_keyframe = 0
    cap_reached = False
    aborted_for_errors = False

    while True:
        if max_candidates and (n_existing_candidates + n_attempted) >= max_candidates:
            cap_reached = True
            break

        where_sql, where_params = _ocr_candidate_where(flt, last_object_idx)
        rows = conn.execute(
            "SELECT object_idx, keyframe_id, cutout_id, bbox_coordinates, "
            "label, textness_score "
            f"FROM object WHERE {where_sql} ORDER BY object_idx LIMIT ?",
            (*where_params, candidate_chunk_size),
        ).fetchall()
        if not rows:
            break
        last_object_idx = int(rows[-1]["object_idx"])

        for row in rows:
            if (
                max_candidates
                and (n_existing_candidates + n_attempted) >= max_candidates
            ):
                cap_reached = True
                break

            object_idx = int(row["object_idx"])
            keyframe_id = int(row["keyframe_id"])
            if (
                max_candidates_per_keyframe
                and per_keyframe_attempts.get(keyframe_id, 0)
                >= max_candidates_per_keyframe
            ):
                n_skipped_per_keyframe += 1
                continue

            per_keyframe_attempts[keyframe_id] = (
                per_keyframe_attempts.get(keyframe_id, 0) + 1
            )
            n_attempted += 1

            accepted_update, read_failed = _read_ocr_crop(
                ocr=ocr,
                face_img=provider.get_image(int(row["cutout_id"])),
                bbox_blob=row["bbox_coordinates"],
                object_idx=object_idx,
                label=row["label"] or "",
                textness=float(row["textness_score"]),
                lightweight_preprocess=lightweight_preprocess,
                log_error=n_read_errors < 5,
            )
            if accepted_update is not None:
                accepted_updates.append(accepted_update)
                n_accepted += 1
            elif read_failed:
                n_read_errors += 1
                if n_read_errors >= max_read_errors:
                    logger.warning(
                        "OCR pass aborted after %d read errors; "
                        "leaving remaining candidates unmarked for retry",
                        n_read_errors,
                    )
                    aborted_for_errors = True
                    break
            else:
                # Normal non-accepted result: mark attempted so resumes skip it.
                # (Backend exceptions leave rows unmarked for later retry.)
                rejected_updates.append((object_idx,))

            if progress_interval and n_attempted % progress_interval == 0:
                logger.info(
                    "OCR pass: attempted=%d accepted=%d read_errors=%d "
                    "skipped_by_keyframe_cap=%d",
                    n_attempted,
                    n_accepted,
                    n_read_errors,
                    n_skipped_per_keyframe,
                )

        _flush_ocr_updates(conn, accepted_updates, rejected_updates)

        if cap_reached or aborted_for_errors:
            break

    logger.info(
        "OCR pass: %d / %d attempted candidates received OCR key "
        "(eligible=%d read_errors=%d skipped_by_keyframe_cap=%d "
        "cap_reached=%s aborted_for_errors=%s)",
        n_accepted,
        n_attempted,
        n_eligible,
        n_read_errors,
        n_skipped_per_keyframe,
        cap_reached,
        aborted_for_errors,
    )
    try:
        ocr.close()
    except Exception:
        pass
    return n_accepted


def _encode_textness_text_features(preferred_device: str) -> np.ndarray:
    """Encode the textness positive/negative prompts into a feature matrix.

    Retries on CPU if CUDA prompt encoding fails.
    """
    from pipeline.offline.refine.ocr_refine import (
        TEXTNESS_NEGATIVE_PROMPTS,
        TEXTNESS_POSITIVE_PROMPTS,
    )

    def _encode(model_device: str) -> np.ndarray:
        model = load_model(
            {"name": "metaclip2", "device": model_device}, device=model_device
        )
        prompt_features = [
            model.get_text_features([prompt]).detach().float().cpu().numpy()[0]
            for prompt in (*TEXTNESS_POSITIVE_PROMPTS, *TEXTNESS_NEGATIVE_PROMPTS)
        ]
        return np.vstack(prompt_features).astype(np.float32)

    try:
        return _encode(preferred_device)
    except Exception as exc:
        if not str(preferred_device).startswith("cuda"):
            raise
        logger.warning(
            "OCR recovery: MetaCLIP textness prompt encoding failed on CUDA; "
            "retrying on CPU. Error: %s",
            exc,
        )
        MetaCLIP.destroy_instance()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _encode("cpu")


def _backfill_textness_updates(
    conn: sqlite3.Connection,
    *,
    text_features: np.ndarray,
    n_positive_prompts: int,
    batch_size: int,
    n_missing: int,
) -> int:
    """Stream object embeddings in chunks, scoring textness and writing it back.

    Each chunk's rows are removed from the candidate set by the UPDATE (their
    textness_score becomes non-NULL), so the unkeyed query makes progress.
    """
    n_updated = 0
    while True:
        chunk = conn.execute(
            "SELECT object_idx, embedding FROM object "
            "WHERE textness_score IS NULL AND embedding IS NOT NULL "
            "ORDER BY object_idx LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not chunk:
            break

        object_ids: List[int] = []
        embeddings: List[np.ndarray] = []
        for row in chunk:
            emb = np.frombuffer(row[1], dtype=np.float32)
            if emb.size == 0:
                continue
            object_ids.append(int(row[0]))
            embeddings.append(emb)
        if not embeddings:
            continue

        object_features = np.vstack(embeddings).astype(np.float32)
        scores = object_features @ text_features.T
        pos = scores[:, :n_positive_prompts].max(axis=1)
        neg = scores[:, n_positive_prompts:].max(axis=1)
        textness_scores = (pos - neg).astype(np.float32)
        updates = [
            (float(score), object_idx)
            for object_idx, score in zip(object_ids, textness_scores)
        ]
        conn.executemany(
            "UPDATE object SET textness_score=? WHERE object_idx=?",
            updates,
        )
        conn.commit()
        n_updated += len(updates)
        logger.info(
            "OCR recovery: textness backfill %d / %d",
            n_updated,
            n_missing,
        )
    return n_updated


def _backfill_textness_scores(
    conn: sqlite3.Connection,
    *,
    preferred_device: str,
    batch_size: int = 2048,
) -> int:
    """Populate missing textness scores from stored object embeddings."""
    n_missing = int(
        conn.execute(
            "SELECT count(*) FROM object "
            "WHERE textness_score IS NULL AND embedding IS NOT NULL",
        ).fetchone()[0]
    )
    if not n_missing:
        return 0

    from pipeline.offline.refine.ocr_refine import TEXTNESS_POSITIVE_PROMPTS

    text_features = _encode_textness_text_features(preferred_device)
    logger.info("OCR recovery: backfilling textness_score for %d objects", n_missing)
    n_updated = _backfill_textness_updates(
        conn,
        text_features=text_features,
        n_positive_prompts=len(TEXTNESS_POSITIVE_PROMPTS),
        batch_size=batch_size,
        n_missing=n_missing,
    )
    logger.info("OCR recovery: populated textness_score for %d objects", n_updated)
    return n_updated


def _default_lightweight_paddle_device(*, preferred_device: str) -> str:
    """Return the best available device string for PaddleOCR."""
    try:
        from pipeline.core.models.ocr.paddle_utils import (
            default_lightweight_paddle_device,
        )

        return default_lightweight_paddle_device(preferred_device)
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Main — config, image-set helpers, detection loop, summary
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict:
    if config_path.is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_ocr_config(cfg: dict, offline_cfg: dict) -> dict:
    """Resolve hybrid OCR config.

    Hybrid config files keep OCR at the top level, while an earlier draft of
    this builder read ``offline.ocr``.  Keep the nested form as a compatibility
    fallback, but prefer the documented top-level section.
    """
    top_level = cfg.get("ocr")
    nested = offline_cfg.get("ocr")
    if top_level is not None:
        if nested is not None:
            logger.warning(
                "Both top-level 'ocr' and 'offline.ocr' are set; using top-level 'ocr'",
            )
        return top_level or {}
    return nested or {}


def _as_normalized_tuple(value: object) -> Tuple[str, ...]:
    """Parse optional YAML string/list filters into normalized terms."""
    if value is None:
        return ()
    raw_values: Sequence[object]
    if isinstance(value, str):
        raw_values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    normalized = []
    for item in raw_values:
        term = " ".join(str(item).lower().strip().split())
        if term:
            normalized.append(term)
    return tuple(dict.fromkeys(normalized))


@dataclass
class BuildConfig:
    """All resolved configuration for one build run."""

    map_path: Path
    output_db_path: Path
    device: str
    venue: str
    # Batch / prefetch
    mini_batch_size: int
    prefetch_workers: int
    prefetch_queue: int
    post_process_pending: int
    # Cubemap geometry
    cubemap_face_size: int
    cubemap_fov_deg: float
    # NMS + area / aspect filters
    nms_iou: float
    min_area_ratio: float
    max_area_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    max_proposals_per_face: int
    collect_area_stats: bool
    skip_clustering: bool
    # 3-D localization
    depth_dir: Path
    min_depth_m: float
    max_depth_m: float
    do_localize: bool
    # Detectors
    gdino_prompt: str
    gdino_label: str
    gdino_score: float
    gdino_min_conf: float
    gdino_model: str
    yolo_weights: str
    yolo_imgsz: int
    yolo_conf_broad: float
    yolo_conf_specific: float
    # OCR
    ocr_enabled: bool
    ocr_textness_threshold: float
    ocr_lightweight_lang: str
    ocr_lightweight_device: str
    ocr_lightweight_preprocess: str
    ocr_lightweight_min_score: float
    ocr_lightweight_single_letter_min_score: float
    ocr_candidate_chunk_size: int
    ocr_max_candidates: int
    ocr_max_candidates_per_keyframe: int
    ocr_progress_interval: int
    ocr_include_labels: Tuple[str, ...]
    ocr_exclude_labels: Tuple[str, ...]
    ocr_include_detection_sources: Tuple[str, ...]
    # Image-set sampling
    min_dist: float
    limit: int


class _ImageSet(NamedTuple):
    paths: List[Path]
    filename_to_kf_id: Optional[Dict[str, int]]
    kf_id_to_depth_stem: Dict[int, str]


class _GeorefState(NamedTuple):
    keyframe_poses: dict
    keyframe_levels: dict
    georef: Any


class _DetectionStats(NamedTuple):
    n_erps_timed: int
    t_yolo: float
    t_gdino: float
    t_mc_faces: float
    t_mc_crops: float
    skipped: int
    yolo_raw: List[int]
    gdino_raw: List[int]
    after_cap: List[int]
    nms_removed: List[int]
    crops_per_erp: List[int]
    area_ratios: List[float]
    union_area_ratios: List[float]
    nms_iou: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid offline build pipeline for /localize",
    )
    parser.add_argument("--map_path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: <map_path>/object_search_prompts.yaml)",
    )
    parser.add_argument(
        "--output_db",
        type=Path,
        default=None,
        help=f"Output DB (default: <map_path>/{OBJECT_SEARCH_INDEX_DB_FILENAME})",
    )
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--venue",
        choices=tuple(VENUE_PROMPTS.keys()),
        default=None,
        help="Venue preset for GDINO prompt (overridden by config venue key)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--min_dist",
        type=float,
        default=1.5,
        help="Minimum spacing between sampled keyframes in meters (<= 0 disables)",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace, map_path: Path) -> BuildConfig:
    """Build a BuildConfig from CLI args and YAML config."""
    config_path = args.config or (map_path / "object_search_prompts.yaml")
    raw = _load_config(config_path)
    offline_cfg = raw.get("offline", {})
    inf_cfg = offline_cfg.get("inference", {})
    loc_cfg = offline_cfg.get("localization", {})
    yolo_cfg = raw.get("yolo_params", {})
    gdino_cfg = raw.get("gdino_params", {})
    ocr_cfg = _resolve_ocr_config(raw, offline_cfg)

    depth_dir = map_path / str(loc_cfg.get("depth_dir", "depths"))
    localization_requested = bool(loc_cfg.get("enabled", True))
    do_localize = localization_requested and depth_dir.is_dir()
    if localization_requested and not depth_dir.is_dir():
        logger.warning(
            "Localization is enabled in config but depth directory does not exist: %s"
            " — objects will be written without 3D positions. "
            "Download depth maps or set localization.enabled: false to suppress "
            "this warning.",
            depth_dir,
        )

    venue = args.venue or raw.get("venue", DEFAULT_VENUE)
    # conf_specific falls back to legacy "conf_airport_specific" key.
    yolo_conf_specific = float(
        yolo_cfg.get("conf_specific", yolo_cfg.get("conf_airport_specific", 0.03))
    )

    return BuildConfig(
        map_path=map_path,
        output_db_path=args.output_db or (map_path / OBJECT_SEARCH_INDEX_DB_FILENAME),
        device=args.device,
        venue=venue,
        mini_batch_size=max(1, int(offline_cfg.get("mini_batch_size", 4))),
        prefetch_workers=int(offline_cfg.get("prefetch_workers", 2)),
        prefetch_queue=int(offline_cfg.get("prefetch_queue", 4)),
        post_process_pending=max(0, int(offline_cfg.get("post_process_pending", 2))),
        cubemap_face_size=int(inf_cfg.get("cubemap_face_size", 512)),
        cubemap_fov_deg=float(inf_cfg.get("cubemap_fov_deg", 90.0)),
        # Single class-agnostic spherical (FoV-IoU) NMS over pooled YOLO+GDINO.
        # Removes same- and cross-label duplicates in one pass; 0 disables.
        nms_iou=float(offline_cfg.get("nms_iou", 0.55)),
        min_area_ratio=float(offline_cfg.get("min_area_ratio", 0.0)),
        max_area_ratio=float(offline_cfg.get("max_area_ratio", 0.95)),
        min_aspect_ratio=float(offline_cfg.get("min_aspect_ratio", 0.05)),
        max_aspect_ratio=float(offline_cfg.get("max_aspect_ratio", 20.0)),
        max_proposals_per_face=int(offline_cfg.get("max_proposals_per_face", 30)),
        # Per-ERP bbox-coverage diagnostics (off by default; pure logging overhead).
        collect_area_stats=bool(offline_cfg.get("collect_area_stats", False)),
        skip_clustering=bool(loc_cfg.get("skip_clustering", True)),
        depth_dir=depth_dir,
        min_depth_m=float(loc_cfg.get("min_depth_m", 0.5)),
        max_depth_m=float(loc_cfg.get("max_depth_m", 25.0)),
        do_localize=do_localize,
        gdino_prompt=gdino_cfg.get("prompt") or VENUE_PROMPTS.get(
            venue, VENUE_PROMPTS[DEFAULT_VENUE]
        ),
        gdino_label=gdino_cfg.get("label", DEFAULT_GDINO_LABEL),
        gdino_score=float(gdino_cfg.get("score", 0.06)),
        gdino_min_conf=float(gdino_cfg.get("min_confidence", 0.10)),
        gdino_model=str(gdino_cfg.get("model", "IDEA-Research/grounding-dino-tiny")),
        yolo_weights=str(yolo_cfg.get("weights", "yolov8s-worldv2.pt")),
        yolo_imgsz=int(yolo_cfg.get("imgsz", 1600)),
        yolo_conf_broad=float(yolo_cfg.get("conf_broad", 0.05)),
        yolo_conf_specific=yolo_conf_specific,
        ocr_enabled=bool(ocr_cfg.get("enabled", False)),
        ocr_textness_threshold=float(ocr_cfg.get("textness_threshold", 0.02)),
        ocr_lightweight_lang=str(ocr_cfg.get("lightweight_lang", "fr")),
        ocr_lightweight_device=str(ocr_cfg.get("lightweight_device", "") or ""),
        ocr_lightweight_preprocess=str(ocr_cfg.get("lightweight_preprocess", "raw")),
        ocr_lightweight_min_score=float(ocr_cfg.get("lightweight_min_score", 0.7)),
        ocr_lightweight_single_letter_min_score=float(
            ocr_cfg.get("lightweight_single_letter_min_score", 0.85)
        ),
        ocr_candidate_chunk_size=int(ocr_cfg.get("candidate_chunk_size", 1000)),
        ocr_max_candidates=int(ocr_cfg.get("max_candidates", 0)),
        ocr_max_candidates_per_keyframe=int(
            ocr_cfg.get("max_candidates_per_keyframe", 0)
        ),
        ocr_progress_interval=int(ocr_cfg.get("progress_interval", 1000)),
        ocr_include_labels=_as_normalized_tuple(ocr_cfg.get("include_labels")),
        ocr_exclude_labels=_as_normalized_tuple(ocr_cfg.get("exclude_labels")),
        ocr_include_detection_sources=_as_normalized_tuple(
            ocr_cfg.get("include_detection_sources")
        ),
        min_dist=args.min_dist,
        limit=args.limit,
    )


def _load_image_set(map_path: Path, cfg: BuildConfig) -> _ImageSet:
    """Resolve the ordered set of ERP images to process."""
    images_dir = map_path / "images_360"
    if not images_dir.is_dir():
        raise SystemExit(f"No images_360 directory at {images_dir}")
    image_paths = load_image_paths(images_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    georef_db = map_path / "georef.db"
    filename_to_kf_id = load_image_filename_to_keyframe_id(georef_db)
    # Reverse mapping: kf_id → image filename stem (without extension).
    # Used to resolve depth zarr paths: {depth_dir}/{stem}.zarr falls back to
    # {depth_dir}/{kf_id}.zarr so both UUID-named and integer-named maps work.
    kf_id_to_depth_stem: Dict[int, str] = (
        {kf_id: Path(fn).stem for fn, kf_id in filename_to_kf_id.items()}
        if filename_to_kf_id is not None
        else {}
    )
    if filename_to_kf_id is not None:
        logger.info("UUID filename mapping: %d entries", len(filename_to_kf_id))
        before = len(image_paths)
        image_paths = [
            p
            for p in image_paths
            if keyframe_id_from_image_path(
                p, image_filename_to_keyframe_id=filename_to_kf_id
            )
            is not None
        ]
        if before - len(image_paths):
            logger.warning(
                "Dropped %d images with no keyframe match", before - len(image_paths)
            )
    else:
        logger.info("No filename column in georef.db; falling back to int(path.stem)")

    return _ImageSet(
        paths=image_paths,
        filename_to_kf_id=filename_to_kf_id,
        kf_id_to_depth_stem=kf_id_to_depth_stem,
    )


def _setup_georef_and_poses(
    map_path: Path,
    cfg: BuildConfig,
    image_set: _ImageSet,
) -> Tuple[_GeorefState, List[Path]]:
    """Load keyframe poses + georef; apply min_dist sampling; return todo list."""
    georef_db = map_path / "georef.db"
    keyframe_poses: dict = {}
    if georef_db.is_file():
        try:
            keyframe_poses = load_keyframe_poses_from_georef(georef_db)
        except Exception as exc:
            logger.warning("Failed to load keyframe poses from %s: %s", georef_db, exc)
        if keyframe_poses:
            logger.info("Loaded %d keyframe poses", len(keyframe_poses))
        elif cfg.min_dist > 0.0 or cfg.do_localize:
            logger.warning(
                "No keyframe poses found in %s%s",
                georef_db,
                (
                    " — localization will produce 0 localized objects even"
                    " though depths are available"
                    if cfg.do_localize
                    else ""
                ),
            )

    image_paths = list(image_set.paths)
    if cfg.min_dist > 0.0:
        if keyframe_poses:
            image_paths = _sample_image_paths_by_min_dist(
                image_paths,
                image_filename_to_keyframe_id=image_set.filename_to_kf_id,
                keyframe_poses=keyframe_poses,
                min_dist_m=cfg.min_dist,
            )
        else:
            logger.warning(
                "min_dist sampling requested but no georef poses are available;"
                " keeping %d images",
                len(image_paths),
            )
    if cfg.limit > 0:
        image_paths = image_paths[: cfg.limit]

    georef = None
    if cfg.do_localize and georef_db.is_file():
        if not keyframe_poses:
            logger.warning(
                "No keyframe poses found in %s — localization will produce "
                "0 localized objects even though depths are available",
                georef_db,
            )
        georef = load_georef_from_db(georef_db)
        if georef is None:
            logger.warning(
                "GeoRef origin not found in %s — position_world will be in "
                "the map's local metric frame (not converted to lat/lng)",
                georef_db,
            )

    # Level is a detection-phase property resolved from the keyframe camera
    # position — available even for objects that fail depth localization.
    keyframe_levels: dict = {}
    if keyframe_poses and georef is not None:
        keyframe_levels = keyframe_levels_from_poses(keyframe_poses, georef)
        logger.info(
            "Resolved level for %d / %d keyframes",
            sum(1 for v in keyframe_levels.values() if v != UNRESOLVED_LEVEL_SENTINEL),
            len(keyframe_poses),
        )

    return (
        _GeorefState(
            keyframe_poses=keyframe_poses,
            keyframe_levels=keyframe_levels,
            georef=georef,
        ),
        image_paths,
    )


def _run_ocr_if_enabled(
    conn: sqlite3.Connection,
    cfg: BuildConfig,
    image_paths: List[Path],
    image_set: _ImageSet,
) -> None:
    """Backfill textness scores and run the OCR pass when OCR is enabled."""
    if not cfg.ocr_enabled:
        return
    n_missing = conn.execute(
        "SELECT count(*) FROM object WHERE textness_score IS NULL",
    ).fetchone()[0]
    if n_missing:
        _backfill_textness_scores(conn, preferred_device=cfg.device)
    _run_ocr_pass(
        conn,
        image_paths=image_paths,
        image_filename_to_keyframe_id=image_set.filename_to_kf_id,
        cubemap_face_size=cfg.cubemap_face_size,
        cubemap_fov_deg=cfg.cubemap_fov_deg,
        textness_threshold=cfg.ocr_textness_threshold,
        lightweight_lang=cfg.ocr_lightweight_lang,
        lightweight_device=cfg.ocr_lightweight_device,
        preferred_device=cfg.device,
        lightweight_preprocess=cfg.ocr_lightweight_preprocess,
        lightweight_min_score=cfg.ocr_lightweight_min_score,
        lightweight_single_letter_min_score=cfg.ocr_lightweight_single_letter_min_score,
        candidate_chunk_size=cfg.ocr_candidate_chunk_size,
        max_candidates=cfg.ocr_max_candidates,
        max_candidates_per_keyframe=cfg.ocr_max_candidates_per_keyframe,
        progress_interval=cfg.ocr_progress_interval,
        include_labels=cfg.ocr_include_labels,
        exclude_labels=cfg.ocr_exclude_labels,
        include_detection_sources=cfg.ocr_include_detection_sources,
    )


def _handle_resume_complete(
    conn: sqlite3.Connection,
    cfg: BuildConfig,
    image_paths: List[Path],
    image_set: _ImageSet,
    geo: _GeorefState,
    images_dir: Path,
) -> bool:
    """Handle the case where all images are already processed.

    Runs any pending localization recovery or OCR, updates metadata, and
    returns True so main() can exit early.
    """
    n_missing_positions = conn.execute(
        "SELECT count(*) FROM object WHERE position_local IS NULL"
    ).fetchone()[0]

    if cfg.do_localize and n_missing_positions > 0:
        logger.info(
            "Detection/embedding complete but %d objects have no position. "
            "Running localization-only recovery pass.",
            n_missing_positions,
        )
        _run_localization_pass(
            conn,
            keyframe_poses=geo.keyframe_poses,
            depth_dir=cfg.depth_dir,
            kf_id_to_depth_stem=image_set.kf_id_to_depth_stem,
            face_size=cfg.cubemap_face_size,
            fov_deg=cfg.cubemap_fov_deg,
            min_depth_m=cfg.min_depth_m,
            max_depth_m=cfg.max_depth_m,
            georef=geo.georef,
        )
    else:
        logger.info("All images already processed.")

    _run_ocr_if_enabled(conn, cfg, image_paths, image_set)

    n_cutouts = conn.execute("SELECT count(*) FROM cutout").fetchone()[0]
    n_objects = conn.execute("SELECT count(*) FROM object").fetchone()[0]
    n_localized = conn.execute(
        "SELECT count(*) FROM object WHERE localization_valid = 1"
    ).fetchone()[0]
    logger.info(
        "Resume complete. cutouts=%d objects=%d localized=%d (%.1f%%)",
        n_cutouts,
        n_objects,
        n_localized,
        100.0 * n_localized / max(1, n_objects),
    )
    meta = ObjectSearchIndexMetadata(
        schema_version=3,
        projection_dim=_embedding_dim_from_db(conn),
        created_utc=default_created_utc(),
        object_detector_prompt=f"Hybrid YOLO-World + GDINO ({cfg.venue})",
        cutout_count=n_cutouts,
        object_count=n_objects,
        source_images_dir=str(images_dir.resolve()),
        source="equirect360",
        geometry="cubemap",
        id_stride=DEFAULT_ID_STRIDE,
        cubemap_face_size=cfg.cubemap_face_size,
        cubemap_fov_deg=cfg.cubemap_fov_deg,
        gdino_params_json="{}",
        build_params_json="{}",
        notes=(
            f"Hybrid pipeline. venue={cfg.venue} skip_clustering={cfg.skip_clustering}"
        ),
    )
    write_index_metadata(conn, meta)
    conn.close()
    return True


# Each capped detection: (detection, cubemap face name, bbox in face coords).
CappedDet = Tuple[HybridDetection, str, Tuple[float, float, float, float]]
_UNION_GRID_SCALE = 0.1  # downsample factor for union-area rasterisation


@dataclass
class _ErpCounters:
    """Per-ERP detection-count accumulators for the build summary."""

    yolo_raw: List[int] = field(default_factory=list)
    gdino_raw: List[int] = field(default_factory=list)
    after_cap: List[int] = field(default_factory=list)
    nms_removed: List[int] = field(default_factory=list)
    crops_per_erp: List[int] = field(default_factory=list)
    area_ratios: List[float] = field(default_factory=list)
    union_area_ratios: List[float] = field(default_factory=list)


def _pull_batch(
    prefetched_iter: Iterator[Any], mini_batch_size: int
) -> Tuple[list, int]:
    """Pull up to mini_batch_size prepped ERPs from the prefetch iterator."""
    batch_preps: list = []
    n_pulled = 0
    for _ in range(mini_batch_size):
        try:
            _path, prep = next(prefetched_iter)
        except StopIteration:
            break
        batch_preps.append(prep)
        n_pulled += 1
    return batch_preps, n_pulled


def _project_and_cap(
    accepted: list, erp_w: int, erp_h: int, cfg: BuildConfig
) -> List[CappedDet]:
    """Project accepted ERP detections to cubemap faces and cap per face."""
    face_buckets: Dict[
        str, List[Tuple[HybridDetection, Tuple[float, float, float, float]]]
    ] = {}
    for det in accepted:
        proj = project_erp_bbox_to_face(
            det.bbox,
            erp_w,
            erp_h,
            face_size=cfg.cubemap_face_size,
            fov_deg=cfg.cubemap_fov_deg,
        )
        if proj is None:
            continue
        fname, bbox_face = proj
        face_buckets.setdefault(fname, []).append((det, bbox_face))

    erp_capped: List[CappedDet] = []
    for fname, items in face_buckets.items():
        if cfg.max_proposals_per_face > 0:
            items.sort(key=lambda x: -x[0].score)
            items = items[: cfg.max_proposals_per_face]
        for det, bbox_face in items:
            erp_capped.append((det, fname, bbox_face))
    return erp_capped


def _accumulate_area_stats(
    erp_capped: List[CappedDet], erp_h: int, erp_w: int, counters: _ErpCounters
) -> None:
    """Append per-ERP sum/union bbox-coverage ratios (opt-in diagnostics)."""
    erp_area = float(erp_h * erp_w)
    total_bbox_area = sum(
        max(0.0, det.bbox[2] - det.bbox[0]) * max(0.0, det.bbox[3] - det.bbox[1])
        for det, _fn, _bf in erp_capped
    )
    counters.area_ratios.append(total_bbox_area / max(1.0, erp_area))
    gh = max(1, int(erp_h * _UNION_GRID_SCALE))
    gw = max(1, int(erp_w * _UNION_GRID_SCALE))
    mask = np.zeros((gh, gw), dtype=bool)
    for det, _fn, _bf in erp_capped:
        x1, y1, x2, y2 = det.bbox
        mx1 = max(0, int(x1 * gw / erp_w))
        my1 = max(0, int(y1 * gh / erp_h))
        mx2 = min(gw, int(x2 * gw / erp_w) + 1)
        my2 = min(gh, int(y2 * gh / erp_h) + 1)
        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = True
    counters.union_area_ratios.append(float(mask.sum()) / (gh * gw))


def _detect_and_cap_batch(
    *,
    erps: list,
    erp_shapes: List[Tuple[int, int]],
    yolo_detector: Any,
    gdino_detector: Any,
    cfg: BuildConfig,
    counters: _ErpCounters,
    sync: Callable[[], None],
) -> Tuple[List[List[CappedDet]], float, float]:
    """Detect (YOLO+GDINO), filter, spherical-NMS, project, and per-face cap.

    Applying the cap here (before MetaCLIP) is the key performance fix: it
    reduces MetaCLIP input from ~100 crops/ERP to at most
    max_proposals_per_face × 6 per ERP (3-5× fewer crops). Returns the capped
    detections per ERP plus (yolo_seconds, gdino_seconds); updates counters.
    """
    sync()
    t0 = time.perf_counter()
    yolo_dets_per_erp = yolo_detector.detect_batch(erps)
    sync()
    dt_yolo = time.perf_counter() - t0

    t0 = time.perf_counter()
    gdino_dets_per_erp = gdino_detector.detect_batch(erps)
    sync()
    dt_gdino = time.perf_counter() - t0

    capped_dets_per_erp: List[List[CappedDet]] = []
    for i, (erp_h, erp_w) in enumerate(erp_shapes):
        combined = yolo_dets_per_erp[i] + gdino_dets_per_erp[i]
        counters.yolo_raw.append(len(yolo_dets_per_erp[i]))
        counters.gdino_raw.append(len(gdino_dets_per_erp[i]))
        combined = _clip_bboxes_to_image(combined, erp_w, erp_h)
        combined = _area_filter(
            combined, erp_w, erp_h, cfg.min_area_ratio, cfg.max_area_ratio
        )
        combined = _aspect_ratio_filter(
            combined, cfg.min_aspect_ratio, cfg.max_aspect_ratio
        )
        n_before_nms = len(combined)
        accepted = spherical_nms(combined, erp_w, erp_h, cfg.nms_iou)
        counters.nms_removed.append(n_before_nms - len(accepted))

        erp_capped = _project_and_cap(accepted, erp_w, erp_h, cfg)
        capped_dets_per_erp.append(erp_capped)
        counters.after_cap.append(len(erp_capped))

        # Coverage diagnostics are opt-in (pure logging overhead at scale).
        if cfg.collect_area_stats:
            _accumulate_area_stats(erp_capped, erp_h, erp_w, counters)
    return capped_dets_per_erp, dt_yolo, dt_gdino


def _collect_face_and_crop_pils(
    *,
    cutouts_list: list,
    capped_dets_per_erp: List[List[CappedDet]],
) -> Tuple[
    List[Image.Image], List[Tuple[int, int]], List[Image.Image], List[Tuple[int, int]]
]:
    """Flatten cubemap faces and per-detection crops into MetaCLIP input lists.

    Faces are a fixed-size batch (mini_batch_size × 6) → compile-friendly; crops
    are bounded by the per-face cap. Returns the face/crop PIL lists alongside
    the (start, end) slice each ERP occupies in them.
    """
    all_face_pils: List[Image.Image] = []
    face_slice_per_erp: List[Tuple[int, int]] = []
    all_crop_pils: List[Image.Image] = []
    crop_slice_per_erp: List[Tuple[int, int]] = []

    for i, cutouts in enumerate(cutouts_list):
        fs = len(all_face_pils)
        all_face_pils.extend(c.image for c in cutouts)
        face_slice_per_erp.append((fs, len(all_face_pils)))

        cs = len(all_crop_pils)
        for det, fname, (fx1, fy1, fx2, fy2) in capped_dets_per_erp[i]:
            face_idx = FACE_NAME_TO_LOCAL_INDEX[fname]
            face_img = cutouts[face_idx].image
            crop = face_img.crop((int(fx1), int(fy1), int(fx2), int(fy2)))
            # HF processor is ambiguous when H=W=3; enforce a minimum side.
            min_side = 16
            if crop.width < min_side or crop.height < min_side:
                new_w = max(min_side, crop.width)
                new_h = max(min_side, crop.height)
                crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            all_crop_pils.append(crop)
        crop_slice_per_erp.append((cs, len(all_crop_pils)))
    return all_face_pils, face_slice_per_erp, all_crop_pils, crop_slice_per_erp


def _embed_faces_and_crops(
    *,
    metaclip: MetaCLIP,
    all_face_pils: List[Image.Image],
    all_crop_pils: List[Image.Image],
    sync: Callable[[], None],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    """Run MetaCLIP on face and crop PILs.

    Returns (face_embs, crop_embs_flat, faces_seconds, crops_seconds); arrays are
    None when their PIL list is empty.
    """
    face_embs: Optional[np.ndarray] = None
    dt_mc_faces = 0.0
    if all_face_pils:
        sync()
        t0 = time.perf_counter()
        face_embs_t = metaclip.get_image_features(all_face_pils)
        sync()
        dt_mc_faces = time.perf_counter() - t0
        face_embs = face_embs_t.detach().float().cpu().numpy()

    crop_embs_flat: Optional[np.ndarray] = None
    dt_mc_crops = 0.0
    if all_crop_pils:
        sync()
        t0 = time.perf_counter()
        crop_embs_t = metaclip.get_image_features(all_crop_pils)
        sync()
        dt_mc_crops = time.perf_counter() - t0
        crop_embs_flat = crop_embs_t.detach().float().cpu().numpy()
    return face_embs, crop_embs_flat, dt_mc_faces, dt_mc_crops


def _compute_batch_textness(
    cfg: BuildConfig, metaclip: MetaCLIP, crop_embs_flat: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Textness scores for crop embeddings when OCR is enabled (else None)."""
    if not (
        cfg.ocr_enabled
        and crop_embs_flat is not None
        and crop_embs_flat.shape[0] > 0
    ):
        return None
    from pipeline.offline.refine.ocr_refine import compute_textness_scores

    return compute_textness_scores(
        clip_model=metaclip, object_embeddings=crop_embs_flat
    )


def _build_cutout_rows(
    *,
    cutouts_list: list,
    face_embs: Optional[np.ndarray],
    face_slice_per_erp: List[Tuple[int, int]],
    crop_embs_flat: Optional[np.ndarray],
    crop_slice_per_erp: List[Tuple[int, int]],
    projection_dim: int,
) -> Tuple[List[CutoutRow], List[List[np.ndarray]]]:
    """Build cutout DB rows and per-ERP crop-embedding lists."""
    zero_emb = np.zeros(projection_dim, dtype=np.float32)
    cutout_rows: List[CutoutRow] = []
    crop_embs_per_erp: List[List[np.ndarray]] = []

    for i, cutouts in enumerate(cutouts_list):
        for cutout in cutouts:
            rotation = (
                np.asarray(cutout.rotation_cutout_to_equirect, dtype=np.float32)
                if cutout.rotation_cutout_to_equirect is not None
                else np.zeros((4, 4), dtype=np.float32)
            )
            face_global_idx = face_slice_per_erp[i][0] + FACE_NAME_TO_LOCAL_INDEX.get(
                cutout.face_label, 0
            )
            emb = face_embs[face_global_idx] if face_embs is not None else zero_emb
            cutout_rows.append(
                CutoutRow(
                    cutout_id=cutout.cutout_id,
                    keyframe_id=cutout.keyframe_id,
                    center_x=(
                        float(cutout.center_xy[0]) if cutout.center_xy else None
                    ),
                    center_y=(
                        float(cutout.center_xy[1]) if cutout.center_xy else None
                    ),
                    rotation=rotation.tobytes(),
                    embedding=emb.astype(np.float32).tobytes(),
                )
            )

        cs, ce = crop_slice_per_erp[i]
        crop_embs_per_erp.append(
            [crop_embs_flat[j] for j in range(cs, ce)]
            if crop_embs_flat is not None
            else []
        )
    return cutout_rows, crop_embs_per_erp


def _split_textness_per_erp(
    textness_flat: Optional[np.ndarray],
    crop_slice_per_erp: List[Tuple[int, int]],
    n_valid: int,
) -> List[Optional[List[float]]]:
    """Slice the flat textness vector back into per-ERP score lists."""
    textness_per_erp: List[Optional[List[float]]] = []
    for i in range(n_valid):
        cs, ce = crop_slice_per_erp[i]
        if textness_flat is not None and ce > cs:
            textness_per_erp.append([float(textness_flat[j]) for j in range(cs, ce)])
        else:
            textness_per_erp.append(None)
    return textness_per_erp


def _run_detection_loop(
    conn: sqlite3.Connection,
    todo_paths: List[Path],
    image_set: _ImageSet,
    yolo_detector: Any,
    gdino_detector: Any,
    metaclip: MetaCLIP,
    cfg: BuildConfig,
    geo: _GeorefState,
    next_object_idx: List[int],
) -> _DetectionStats:
    """GPU detection + embedding + write loop over todo_paths.

    Returns per-ERP detection stats for the build summary.
    """
    is_cuda = str(cfg.device).startswith("cuda") and torch.cuda.is_available()

    def _sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    post_pool: Optional[ThreadPoolExecutor] = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="post-process")
        if cfg.post_process_pending > 0
        else None
    )
    pending_futures: deque = deque()

    # Timing accumulators (seconds, excluding first-batch warmup)
    _t_yolo = _t_gdino = _t_mc_faces = _t_mc_crops = 0.0
    _n_batches = _n_erps_timed = 0
    counters = _ErpCounters()
    skipped = 0

    pbar = tqdm(total=len(todo_paths), desc="Hybrid build (ERP)")
    prefetched = _iter_prefetched_erps(
        todo_paths,
        image_filename_to_keyframe_id=image_set.filename_to_kf_id,
        cubemap_face_size=cfg.cubemap_face_size,
        cubemap_fov_deg=cfg.cubemap_fov_deg,
        workers=cfg.prefetch_workers,
        prefetch=cfg.prefetch_queue,
    )
    prefetched_iter = iter(prefetched)

    try:
        while True:
            batch_preps, n_pulled = _pull_batch(prefetched_iter, cfg.mini_batch_size)
            if n_pulled == 0:
                break

            valid_preps = [p for p in batch_preps if p is not None]
            skipped += n_pulled - len(valid_preps)
            if not valid_preps:
                pbar.update(n_pulled)
                continue

            keyframe_ids = [p[0] for p in valid_preps]
            erps = [p[1] for p in valid_preps]
            cutouts_list = [p[2] for p in valid_preps]
            erp_shapes = [(erp.shape[0], erp.shape[1]) for erp in erps]

            capped_dets_per_erp, _dt_yolo, _dt_gdino = _detect_and_cap_batch(
                erps=erps,
                erp_shapes=erp_shapes,
                yolo_detector=yolo_detector,
                gdino_detector=gdino_detector,
                cfg=cfg,
                counters=counters,
                sync=_sync,
            )

            (
                all_face_pils,
                face_slice_per_erp,
                all_crop_pils,
                crop_slice_per_erp,
            ) = _collect_face_and_crop_pils(
                cutouts_list=cutouts_list,
                capped_dets_per_erp=capped_dets_per_erp,
            )

            counters.crops_per_erp.extend(
                len(capped_dets_per_erp[i]) for i in range(len(valid_preps))
            )
            face_embs, crop_embs_flat, _dt_mc_faces, _dt_mc_crops = (
                _embed_faces_and_crops(
                    metaclip=metaclip,
                    all_face_pils=all_face_pils,
                    all_crop_pils=all_crop_pils,
                    sync=_sync,
                )
            )

            # Textness uses MetaCLIP (already loaded); negligible extra cost.
            textness_flat = _compute_batch_textness(cfg, metaclip, crop_embs_flat)

            # Accumulate timing (skip first batch — includes warmup / compile)
            if _n_batches > 0:
                _t_yolo += _dt_yolo
                _t_gdino += _dt_gdino
                _t_mc_faces += _dt_mc_faces
                _t_mc_crops += _dt_mc_crops
                _n_erps_timed += len(valid_preps)
            _n_batches += 1

            cutout_rows, crop_embs_per_erp = _build_cutout_rows(
                cutouts_list=cutouts_list,
                face_embs=face_embs,
                face_slice_per_erp=face_slice_per_erp,
                crop_embs_flat=crop_embs_flat,
                crop_slice_per_erp=crop_slice_per_erp,
                projection_dim=int(metaclip.projection_dim),
            )
            textness_per_erp = _split_textness_per_erp(
                textness_flat, crop_slice_per_erp, len(valid_preps)
            )

            batch_payload = {
                "cutout_rows": cutout_rows,
                "capped_dets": capped_dets_per_erp,
                "keyframe_ids": keyframe_ids,
                "erp_shapes": erp_shapes,
                "crop_embs": crop_embs_per_erp,
                "kf_id_to_stem": image_set.kf_id_to_depth_stem,
                "textness_per_erp": textness_per_erp,
                "projection_dim": int(metaclip.projection_dim),
            }
            post_kwargs = dict(
                conn=conn,
                batch_payload=batch_payload,
                keyframe_poses=geo.keyframe_poses,
                keyframe_levels=geo.keyframe_levels,
                depth_dir=cfg.depth_dir,
                face_size=cfg.cubemap_face_size,
                fov_deg=cfg.cubemap_fov_deg,
                min_depth_m=cfg.min_depth_m,
                max_depth_m=cfg.max_depth_m,
                do_localize=cfg.do_localize,
                next_object_idx=next_object_idx,
                georef=geo.georef,
            )

            if post_pool is None:
                _post_process_and_write(**post_kwargs)
            else:
                while len(pending_futures) >= cfg.post_process_pending:
                    pending_futures.popleft().result()
                pending_futures.append(
                    post_pool.submit(_post_process_and_write, **post_kwargs)
                )

            pbar.update(n_pulled)

    finally:
        while pending_futures:
            pending_futures.popleft().result()
        if post_pool is not None:
            post_pool.shutdown(wait=True)
        pbar.close()

    return _DetectionStats(
        n_erps_timed=_n_erps_timed,
        t_yolo=_t_yolo,
        t_gdino=_t_gdino,
        t_mc_faces=_t_mc_faces,
        t_mc_crops=_t_mc_crops,
        skipped=skipped,
        yolo_raw=counters.yolo_raw,
        gdino_raw=counters.gdino_raw,
        after_cap=counters.after_cap,
        nms_removed=counters.nms_removed,
        crops_per_erp=counters.crops_per_erp,
        area_ratios=counters.area_ratios,
        union_area_ratios=counters.union_area_ratios,
        nms_iou=cfg.nms_iou,
    )


def _log_build_summary(
    *,
    n_cutouts: int,
    n_objects: int,
    n_localized: int,
    n_erps_done: int,
    stats: _DetectionStats,
) -> None:
    """Log output counts, per-ERP detection stats, and GPU stage timing."""

    def _arr_stats(values: List[int], label: str) -> None:
        if not values:
            logger.info("  %-35s (no samples)", label)
            return
        a = np.asarray(values, dtype=np.float32)
        logger.info(
            "  %-35s mean=%.1f  median=%.1f  p95=%.1f  max=%d",
            label,
            float(a.mean()),
            float(np.median(a)),
            float(np.percentile(a, 95)),
            int(a.max()),
        )

    logger.info("=" * 60)
    logger.info("Build complete")
    logger.info("=" * 60)
    logger.info("Output:")
    logger.info(
        "  ERPs processed:   %d  (skipped: %d)", n_erps_done, stats.skipped
    )
    logger.info(
        "  Cutouts written:  %d  (%.1f per ERP)",
        n_cutouts,
        n_cutouts / max(1, n_erps_done),
    )
    logger.info(
        "  Objects written:  %d  (%.1f per ERP)",
        n_objects,
        n_objects / max(1, n_erps_done),
    )
    logger.info(
        "  Localized:        %d / %d  (%.1f%%)",
        n_localized,
        n_objects,
        100.0 * n_localized / max(1, n_objects),
    )
    logger.info("  Objects/cutout:   %.2f", n_objects / max(1, n_cutouts))

    if stats.yolo_raw:
        logger.info("Detection counts (per ERP):")
        _arr_stats(stats.yolo_raw, "YOLO raw detections/ERP:")
        _arr_stats(stats.gdino_raw, "GDINO raw detections/ERP:")
        if stats.nms_iou > 0:
            logger.info(
                "  %-35s total=%d  (spherical IoU > %.2f)",
                "Duplicates removed by spherical NMS:",
                int(np.sum(stats.nms_removed)) if stats.nms_removed else 0,
                stats.nms_iou,
            )
        _arr_stats(stats.after_cap, "After NMS + cap/ERP (→ objects):")
        _arr_stats(stats.crops_per_erp, "Crops sent to MetaCLIP/ERP:")
        logger.info("  Objects/cutout (face): %.2f", n_objects / max(1, n_cutouts))
        if stats.area_ratios and stats.union_area_ratios:
            s = np.asarray(stats.area_ratios, dtype=np.float32) * 100
            u = np.asarray(stats.union_area_ratios, dtype=np.float32) * 100
            overlap = s / np.maximum(u, 0.01)
            logger.info(
                "  Sum area / ERP (%%):"
                "  mean=%.1f  median=%.1f  p95=%.1f  max=%.1f"
                "  (>100%% = multi-label overlap)",
                float(s.mean()),
                float(np.median(s)),
                float(np.percentile(s, 95)),
                float(s.max()),
            )
            logger.info(
                "  Union area / ERP (%%):"
                "  mean=%.1f  median=%.1f  p95=%.1f  max=%.1f"
                "  (true coverage, capped at 100%%)",
                float(u.mean()),
                float(np.median(u)),
                float(np.percentile(u, 95)),
                float(u.max()),
            )
            logger.info(
                "  Avg label-layers per covered px:"
                "  mean=%.2f  median=%.2f  p95=%.2f"
                "  (1.0 = no overlap, >2 = dense multi-label stacking)",
                float(overlap.mean()),
                float(np.median(overlap)),
                float(np.percentile(overlap, 95)),
            )

    if stats.n_erps_timed > 0:
        n = stats.n_erps_timed
        logger.info("GPU stage timing (avg per ERP, warmup excluded, n=%d ERPs):", n)
        logger.info(
            "  YOLO detect:          %6.3f s/ERP  (%5.0f ms)",
            stats.t_yolo / n,
            1000 * stats.t_yolo / n,
        )
        logger.info(
            "  GDINO detect:         %6.3f s/ERP  (%5.0f ms)",
            stats.t_gdino / n,
            1000 * stats.t_gdino / n,
        )
        logger.info(
            "  MetaCLIP faces:       %6.3f s/ERP  (%5.0f ms)",
            stats.t_mc_faces / n,
            1000 * stats.t_mc_faces / n,
        )
        logger.info(
            "  MetaCLIP crops:       %6.3f s/ERP  (%5.0f ms)",
            stats.t_mc_crops / n,
            1000 * stats.t_mc_crops / n,
        )
        gpu_total = (
            stats.t_yolo + stats.t_gdino + stats.t_mc_faces + stats.t_mc_crops
        ) / n
        logger.info(
            "  GPU total (measured): %6.3f s/ERP  (%5.0f ms)",
            gpu_total,
            1000 * gpu_total,
        )
        logger.info(
            "  Post-process+DB:      in worker thread (overlapped, not"
            " separately timed)"
        )
    else:
        logger.info(
            "GPU stage timing: only 1 batch processed (warmup excluded —"
            " no steady-state data)"
        )

    logger.info("=" * 60)


def main() -> None:
    args = _parse_args()
    map_path = args.map_path.resolve()
    cfg = _resolve_config(args, map_path)
    images_dir = map_path / "images_360"

    image_set = _load_image_set(map_path, cfg)
    geo, image_paths = _setup_georef_and_poses(map_path, cfg, image_set)

    logger.info("Hybrid build pipeline (ERP-direct, /localize target)")
    logger.info("  Map:        %s", map_path)
    logger.info("  Images:     %d", len(image_paths))
    logger.info("  Device:     %s", cfg.device)
    logger.info("  Venue:      %s  GDINO: %s", cfg.venue, cfg.gdino_model)
    logger.info(
        "  Localize:   %s  skip_clustering=%s", cfg.do_localize, cfg.skip_clustering
    )
    logger.info(
        "  Mini-batch: %d  prefetch: workers=%d queue=%d",
        cfg.mini_batch_size,
        cfg.prefetch_workers,
        cfg.prefetch_queue,
    )
    logger.info(
        "  OCR:        %s  textness_threshold=%.3f lang=%s preprocess=%s "
        "chunk=%d max_candidates=%s max/keyframe=%s",
        cfg.ocr_enabled,
        cfg.ocr_textness_threshold,
        cfg.ocr_lightweight_lang,
        cfg.ocr_lightweight_preprocess,
        cfg.ocr_candidate_chunk_size,
        cfg.ocr_max_candidates or "unlimited",
        cfg.ocr_max_candidates_per_keyframe or "unlimited",
    )
    logger.info("  Output DB:  %s", cfg.output_db_path)

    conn = open_build_db(cfg.output_db_path)

    embedded_cutout_ids = set(load_embedded_cutout_ids(conn).tolist())
    embedded_keyframe_ids = {cid // DEFAULT_ID_STRIDE for cid in embedded_cutout_ids}
    processed_keyframe_ids = set(
        load_processed_keyframe_ids(conn, id_stride=DEFAULT_ID_STRIDE).tolist()
    )
    logger.info(
        "Resume: %d keyframes with cutout embeddings, %d with detections",
        len(embedded_keyframe_ids),
        len(processed_keyframe_ids),
    )

    todo_paths = [
        p
        for p in image_paths
        if keyframe_id_from_image_path(
            p, image_filename_to_keyframe_id=image_set.filename_to_kf_id
        )
        not in processed_keyframe_ids
    ]
    logger.info("Images remaining: %d", len(todo_paths))

    if not todo_paths:
        _handle_resume_complete(conn, cfg, image_paths, image_set, geo, images_dir)
        return

    # Write initial metadata (projection_dim corrected after MetaCLIP loads)
    meta = ObjectSearchIndexMetadata(
        schema_version=3,
        projection_dim=0,
        created_utc=default_created_utc(),
        object_detector_prompt=f"Hybrid YOLO-World + GDINO ({cfg.venue})",
        cutout_count=0,
        object_count=0,
        source_images_dir=str(images_dir.resolve()),
        source="equirect360",
        geometry="cubemap",
        id_stride=DEFAULT_ID_STRIDE,
        cubemap_face_size=cfg.cubemap_face_size,
        cubemap_fov_deg=cfg.cubemap_fov_deg,
        gdino_params_json="{}",
        build_params_json="{}",
        notes=(
            f"Hybrid pipeline. venue={cfg.venue} skip_clustering={cfg.skip_clustering}"
        ),
    )
    write_index_metadata(conn, meta)

    yolo_detector = YoloWorldDetector(
        weights=cfg.yolo_weights,
        device=cfg.device,
        imgsz=cfg.yolo_imgsz,
        yolo_iou=0.5,
        max_det=400,
        enable_broad=True,
        venue=cfg.venue,
        conf_broad=cfg.yolo_conf_broad,
        conf_specific=cfg.yolo_conf_specific,
    )
    gdino_detector = GdinoVenueDetector(
        prompt=cfg.gdino_prompt,
        device=cfg.device,
        score=cfg.gdino_score,
        min_confidence=cfg.gdino_min_conf,
        synthetic_label=cfg.gdino_label,
        batch_size=cfg.mini_batch_size,
    )
    metaclip: MetaCLIP = load_model({"name": "metaclip2"}, device=cfg.device)
    meta.projection_dim = int(metaclip.projection_dim)
    write_index_metadata(conn, meta)

    next_object_idx = [conn.execute("SELECT count(*) FROM object").fetchone()[0]]

    det_stats = _run_detection_loop(
        conn,
        todo_paths,
        image_set,
        yolo_detector,
        gdino_detector,
        metaclip,
        cfg,
        geo,
        next_object_idx,
    )

    if det_stats.skipped:
        logger.warning(
            "Skipped %d images (no keyframe id match or load error)", det_stats.skipped
        )

    n_cutouts = conn.execute("SELECT count(*) FROM cutout").fetchone()[0]
    n_objects = conn.execute("SELECT count(*) FROM object").fetchone()[0]
    n_localized = conn.execute(
        "SELECT count(*) FROM object WHERE localization_valid = 1"
    ).fetchone()[0]
    n_erps_done = len(todo_paths) - det_stats.skipped

    _log_build_summary(
        n_cutouts=n_cutouts,
        n_objects=n_objects,
        n_localized=n_localized,
        n_erps_done=n_erps_done,
        stats=det_stats,
    )

    _run_ocr_if_enabled(conn, cfg, image_paths, image_set)

    meta.cutout_count = n_cutouts
    meta.object_count = n_objects
    write_index_metadata(conn, meta)
    conn.close()


if __name__ == "__main__":
    main()
