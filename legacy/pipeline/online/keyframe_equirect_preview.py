"""Render equirectangular keyframe previews with projected object bounding boxes."""

from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw

_GUI_ROOT = Path(__file__).resolve().parents[3] / "object-search-gui"
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.bbox_postprocess import postprocess_detections  # noqa: E402
from src.standalone_index import (  # noqa: E402
    _cutout_params_from_manifest,
    _load_bundle,
    lookup_objects_for_keyframe_in_bundle,
)

from pipeline.offline.localize.georef import (  # noqa: E402
    resolve_keyframe_equirect_image_path,
)
from pipeline.offline.localize.localize_3d import (  # noqa: E402
    cubemap_pixel_to_equirect_uv,
)
from pipeline.offline.shared.geometry import CUBEMAP_FACE_ORDER  # noqa: E402

_BOX_COLOR = "#11b5ae"
_BOX_COLOR_SELECTED = "#ff6b6b"
_MAX_PREVIEW_WIDTH = 1600


def _import_lonlat_helpers() -> "tuple[Any, Any]":
    from wemap_vision.geometry.equirectangular_projector import lonlat2XY, xyz2lonlat

    return lonlat2XY, xyz2lonlat


def _bbox_corner_pixels(
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _horizon_cutout_pixel_to_equirect_px(
    u: float,
    v: float,
    relative_pose: np.ndarray,
    params: Any,
    shape: tuple[int, int, int],
) -> tuple[float, float]:
    lonlat2XY, xyz2lonlat = _import_lonlat_helpers()
    f = 0.5 * params.width * 1 / np.tan(0.5 * params.fov / 180.0 * np.pi)
    cx = (params.width - 1) / 2.0
    cy = (params.height - 1) / 2.0
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    k_inv = np.linalg.inv(k)
    xyz = np.array([u, v, 1.0], dtype=np.float64) @ k_inv.T
    xyz = xyz / np.linalg.norm(xyz)
    rot = np.asarray(relative_pose[:3, :3], dtype=np.float64).T
    xyz = xyz @ rot.T
    lonlat = xyz2lonlat(xyz.reshape(1, 1, 3))
    xy = lonlat2XY(lonlat, shape=shape[:2]).astype(np.float64)
    return float(xy[0, 0, 0]), float(xy[0, 0, 1])


def _project_bbox_to_equirect_polygon(
    bbox: tuple[float, float, float, float],
    *,
    geometry: str,
    equirect_shape: tuple[int, int, int],
    cutout_id: str,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
) -> list[tuple[float, float]] | None:
    idx = bundle["index_by_cutout_id"].get(str(cutout_id))
    if idx is None:
        return None

    erp_h, erp_w = equirect_shape[0], equirect_shape[1]
    points: list[tuple[float, float]] = []

    if geometry == "cubemap":
        id_stride = int(manifest.get("id_stride") or 0)
        face_size = int(manifest.get("cubemap_face_size") or 0)
        fov_deg = float(manifest.get("cubemap_fov_deg") or 90.0)
        if id_stride <= 0 or face_size <= 0:
            return None
        local_index = int(cutout_id) % id_stride
        if local_index >= len(CUBEMAP_FACE_ORDER):
            return None
        face_name = CUBEMAP_FACE_ORDER[local_index]
        for u, v in _bbox_corner_pixels(bbox):
            u_eq, v_eq = cubemap_pixel_to_equirect_uv(
                u,
                v,
                face_name,
                face_size,
                fov_deg,
            )
            points.append((u_eq * (erp_w - 1), v_eq * (erp_h - 1)))
        return points

    if geometry == "horizon":
        params = _cutout_params_from_manifest(manifest)
        if params is None:
            return None
        pose = bundle["cutout_rotation_cutout_to_equirect"][idx]
        for u, v in _bbox_corner_pixels(bbox):
            px, py = _horizon_cutout_pixel_to_equirect_px(
                u,
                v,
                pose,
                params,
                equirect_shape,
            )
            points.append((px, py))
        return points

    return None


def _postprocess_detections_by_cutout(
    detections: list[Any],
    *,
    nms_iou: float,
    min_bbox_area: float,
    max_bbox_area: float,
) -> list[Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in detections:
        grouped[str(item.cutout_id)].append(item)

    filtered: list[Any] = []
    for group in grouped.values():
        filtered.extend(
            postprocess_detections(
                group,
                nms_iou=nms_iou,
                min_bbox_area=min_bbox_area,
                max_bbox_area=max_bbox_area,
            ),
        )
    return filtered


def render_index_keyframe_equirect_preview_png(
    index_path: Path,
    keyframe_id: str,
    *,
    selected_object_id: str | None = None,
    nms_iou: float = 1.0,
    min_bbox_area: float = 0.0,
    max_bbox_area: float = 0.0,
) -> bytes:
    bundle = _load_bundle(str(index_path))
    manifest = bundle["manifest"]
    geometry = str(manifest.get("geometry") or "")
    if geometry not in {"cubemap", "horizon"}:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported index geometry for ERP preview: {geometry!r}",
        )

    map_path = index_path.parent
    manifest = bundle["manifest"]
    source_image_path, tried_paths = resolve_keyframe_equirect_image_path(
        map_path,
        keyframe_id,
    )
    if source_image_path is None or not source_image_path.is_file():
        detail = (
            "Equirectangular source image not found for keyframe {keyframe_id}.".format(
                keyframe_id=keyframe_id,
            )
        )
        if tried_paths:
            detail += " Tried: " + "; ".join(tried_paths[:8])
            if len(tried_paths) > 8:
                detail += f" (+{len(tried_paths) - 8} more)"
        raise HTTPException(status_code=404, detail=detail)

    try:
        image = Image.open(source_image_path).convert("RGB")
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail="Could not read source image"
        ) from exc

    equirect_shape = (image.height, image.width, 3)
    if image.width > _MAX_PREVIEW_WIDTH:
        scale = _MAX_PREVIEW_WIDTH / float(image.width)
        image = image.resize(
            (int(round(image.width * scale)), int(round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        scale_x = image.width / equirect_shape[1]
        scale_y = image.height / equirect_shape[0]
    else:
        scale_x = 1.0
        scale_y = 1.0

    draw = ImageDraw.Draw(image)
    objects = _postprocess_detections_by_cutout(
        lookup_objects_for_keyframe_in_bundle(index_path, keyframe_id),
        nms_iou=nms_iou,
        min_bbox_area=min_bbox_area,
        max_bbox_area=max_bbox_area,
    )
    for item in objects:
        polygon = _project_bbox_to_equirect_polygon(
            item.bbox,
            geometry=geometry,
            equirect_shape=equirect_shape,
            cutout_id=item.cutout_id,
            bundle=bundle,
            manifest=manifest,
        )
        if not polygon:
            continue
        scaled = [(px * scale_x, py * scale_y) for px, py in polygon]
        is_selected = item.object_id == selected_object_id
        color = _BOX_COLOR_SELECTED if is_selected else _BOX_COLOR
        width = 4 if is_selected else 2
        if len(scaled) >= 2:
            closed = scaled + [scaled[0]]
            draw.line(closed, fill=color, width=width)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
