"""UI helper endpoints backed by object-search-gui enrichment utilities."""

from __future__ import annotations

import functools
import io
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

_GUI_ROOT = Path(__file__).resolve().parents[3] / "object-search-gui"
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))


def _import_gui() -> dict[str, Any]:
    from src.db import lookup_cutout_objects, lookup_keyframe
    from src.models import EnrichedResult, NormalizedResult
    from src.object_search_inspector import draw_detection_overlay
    from src.previews import load_preview_from_path
    from src.resolver import enrich_results

    return {
        "lookup_cutout_objects": lookup_cutout_objects,
        "lookup_keyframe": lookup_keyframe,
        "EnrichedResult": EnrichedResult,
        "NormalizedResult": NormalizedResult,
        "draw_detection_overlay": draw_detection_overlay,
        "load_preview_from_path": load_preview_from_path,
        "enrich_results": enrich_results,
    }


def enrich_text_search_results(
    *,
    map_path: Path,
    pairs: list[list[Any]],
    router_object_type: str | None,
    index_path_override: str | None,
) -> list[dict[str, Any]]:
    gui = _import_gui()
    normalized = [
        gui["NormalizedResult"](
            rank=idx,
            id=str(item[0]),
            score=float(item[1]),
            router_object_type=router_object_type,
        )
        for idx, item in enumerate(pairs, start=1)
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    enriched = gui["enrich_results"](
        str(map_path),
        normalized,
        router_object_type,
        index_path_override,
    )
    return [item.model_dump() for item in enriched]


def _connect_readonly(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _lookup_keyframe_from_flat_table(
    db_path: Path,
    keyframe_id: str,
) -> dict[str, Any] | None:
    conn = _connect_readonly(db_path)
    if conn is None:
        return None
    try:
        row = conn.execute(
            """
            SELECT id, lat, lon, alt, level
            FROM keyframes
            WHERE id = ?
            """,
            (keyframe_id,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        conn.close()

    if row is None:
        return None
    return {
        "id": str(row[0]),
        "lat": float(row[1]) if row[1] is not None else None,
        "lon": float(row[2]) if row[2] is not None else None,
        "alt": float(row[3]) if row[3] is not None else None,
        "level": str(row[4]) if row[4] is not None else None,
    }


def _lookup_keyframe_from_georef_pose(
    map_path: Path,
    keyframe_id: str,
) -> dict[str, Any] | None:
    georef_db_path = map_path / "georef.db"
    if not georef_db_path.exists():
        return None

    try:
        from pipeline.offline.localize.georef import load_georef_from_db
        from pipeline.offline.localize.localize_3d import (
            load_keyframe_poses_from_georef,
        )
    except ModuleNotFoundError:
        return None

    georef = load_georef_from_db(georef_db_path)
    if georef is None:
        return None

    try:
        keyframe_id_int = int(keyframe_id)
        pose = load_keyframe_poses_from_georef(
            georef_db_path,
            [keyframe_id_int],
        ).get(keyframe_id_int)
    except Exception:
        return None
    if pose is None:
        return None

    try:
        position_local = np.linalg.inv(np.asarray(pose, dtype=np.float64))[:3, 3]
        geopose = georef.local_position_to_world(position_local)
        coords = geopose.position
        level = coords.get_level()
        return {
            "id": keyframe_id,
            "lat": coords.get_latitude_deg(),
            "lon": coords.get_longitude_deg(),
            "alt": float(coords.get_altitude()),
            "level": str(level) if level is not None else None,
        }
    except Exception:
        return None


def _lookup_keyframe_standalone(
    map_path: Path, keyframe_id: str
) -> dict[str, Any] | None:
    for db_path in (map_path / "georef.db", map_path / "reloc.db"):
        metadata = _lookup_keyframe_from_flat_table(db_path, keyframe_id)
        if metadata is not None:
            return metadata
    return _lookup_keyframe_from_georef_pose(map_path, keyframe_id)


@functools.lru_cache(maxsize=100_000)
def keyframe_metadata(map_path: Path, keyframe_id: str) -> dict[str, Any] | None:
    standalone = _lookup_keyframe_standalone(map_path, keyframe_id)
    if standalone is not None:
        return standalone

    try:
        gui = _import_gui()
    except ModuleNotFoundError:
        return None
    meta = gui["lookup_keyframe"](map_path, keyframe_id)
    if meta is None:
        return None
    return {
        "id": keyframe_id,
        "lat": meta.lat,
        "lon": meta.lon,
        "alt": meta.alt,
        "level": meta.level,
    }


def cutout_detections(
    map_path: Path,
    cutout_id: str,
    *,
    index_path_override: str | None,
) -> list[dict[str, Any]]:
    gui = _import_gui()
    records = gui["lookup_cutout_objects"](
        map_path,
        cutout_id,
        index_path_override=index_path_override,
    )
    items: list[dict[str, Any]] = []
    for record in records:
        items.append(
            {
                "id": record.id,
                "label": record.label,
                "confidence": record.confidence,
                "source": record.source,
                "bbox": list(record.bbox),
            }
        )
    return items


def resolve_cutout_preview_path(
    map_path: Path,
    cutout_id: str,
    *,
    index_path_override: str | None,
    preview_path: str | None = None,
) -> str:
    if preview_path:
        return preview_path
    from src.db import lookup_cutout

    meta = lookup_cutout(map_path, cutout_id, index_path_override=index_path_override)
    if meta is None or not meta.preview_path:
        raise HTTPException(status_code=404, detail="Preview not available")
    return str(meta.preview_path)


def render_cutout_preview_png(
    *,
    map_path: Path,
    preview_path: str | None,
    cutout_id: str,
    index_path_override: str | None,
    selected_object_id: str | None,
) -> bytes:
    gui = _import_gui()
    preview_path = resolve_cutout_preview_path(
        map_path,
        cutout_id,
        index_path_override=index_path_override,
        preview_path=preview_path,
    )

    image = gui["load_preview_from_path"](preview_path, map_path=map_path)
    if image is None:
        raise HTTPException(status_code=404, detail="Preview file not found")

    detections = gui["lookup_cutout_objects"](
        map_path,
        cutout_id,
        index_path_override=index_path_override,
    )
    if detections:
        image = gui["draw_detection_overlay"](
            image,
            detections,
            selected_detection_id=selected_object_id,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preview_png_from_path(map_path: Path, preview_path: str) -> bytes:
    gui = _import_gui()
    image = gui["load_preview_from_path"](preview_path, map_path=map_path)
    if image is None:
        raise HTTPException(status_code=404, detail="Preview file not found")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
