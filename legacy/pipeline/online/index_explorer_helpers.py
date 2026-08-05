"""Object-search index (OS Data Explorer) helpers for the React UI."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

_GUI_ROOT = Path(__file__).resolve().parents[3] / "object-search-gui"
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))


def _import_index_modules() -> dict[str, Any]:
    from src.bbox_postprocess import postprocess_detections
    from src.db import lookup_keyframe
    from src.object_search_inspector import draw_detection_overlay
    from src.previews import load_preview_from_path
    from src.prompt_latent_space import (
        build_topk_records,
        compute_global_color_range,
        compute_object_pca_projection,
        compute_object_umap_projection,
        compute_similarity_matrix,
        embed_prompts,
        extract_default_prompts,
        get_cached_similarity_matrix,
        parse_prompt_lines,
    )
    from src.standalone_index import (
        ClusterOcrMetadata,
        StandaloneObjectMetadata,
        build_preview_ref,
        format_ocr_source,
        inspect_index_preview_ref,
        load_bundle_explorer_summary,
        load_cluster_ocr_records,
        load_object_embedding_latent_data,
        lookup_all_objects_in_bundle,
        lookup_objects_in_bundle,
        resolve_index_path,
    )

    from pipeline.core.types import OBJECT_SEARCH_INDEX_DB_FILENAME

    return {
        "postprocess_detections": postprocess_detections,
        "OBJECT_SEARCH_INDEX_DB_FILENAME": OBJECT_SEARCH_INDEX_DB_FILENAME,
        "lookup_keyframe": lookup_keyframe,
        "draw_detection_overlay": draw_detection_overlay,
        "load_preview_from_path": load_preview_from_path,
        "build_preview_ref": build_preview_ref,
        "inspect_index_preview_ref": inspect_index_preview_ref,
        "load_bundle_explorer_summary": load_bundle_explorer_summary,
        "load_cluster_ocr_records": load_cluster_ocr_records,
        "load_object_embedding_latent_data": load_object_embedding_latent_data,
        "lookup_all_objects_in_bundle": lookup_all_objects_in_bundle,
        "lookup_objects_in_bundle": lookup_objects_in_bundle,
        "resolve_index_path": resolve_index_path,
        "format_ocr_source": format_ocr_source,
        "parse_prompt_lines": parse_prompt_lines,
        "extract_default_prompts": extract_default_prompts,
        "get_cached_similarity_matrix": get_cached_similarity_matrix,
        "compute_similarity_matrix": compute_similarity_matrix,
        "compute_global_color_range": compute_global_color_range,
        "build_topk_records": build_topk_records,
        "compute_object_umap_projection": compute_object_umap_projection,
        "compute_object_pca_projection": compute_object_pca_projection,
        "embed_prompts": embed_prompts,
        "ClusterOcrMetadata": ClusterOcrMetadata,
        "StandaloneObjectMetadata": StandaloneObjectMetadata,
    }


def index_path_candidates(
    map_path: Path,
    index_path_override: str | None,
) -> list[Path]:
    mods = _import_index_modules()
    filename = mods["OBJECT_SEARCH_INDEX_DB_FILENAME"]
    candidates: list[Path] = []
    if index_path_override:
        candidates.append(Path(index_path_override).expanduser())
    candidates.extend(
        [
            map_path / "object-search" / filename,
            map_path / filename,
        ]
    )
    return candidates


def resolve_map_index_path(
    map_path: Path,
    index_path_override: str | None,
) -> Path | None:
    mods = _import_index_modules()
    result: Path | None = mods["resolve_index_path"](map_path, index_path_override)
    return result


def require_map_index_path(
    map_path: Path,
    index_path_override: str | None,
) -> Path:
    index_path = resolve_map_index_path(map_path, index_path_override)
    if index_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No object-search index database (object-search.db) found"
                " for this map."
            ),
        )
    return index_path


def index_object_record(item: Any) -> dict[str, Any]:
    mods = _import_index_modules()
    format_ocr_source = mods["format_ocr_source"]
    return {
        "id": item.object_id,
        "keyframe_id": item.keyframe_id,
        "cutout_id": item.cutout_id,
        "bbox": [round(float(v), 1) for v in item.bbox],
        "label": item.label,
        "detection_source": item.detection_source,
        "ocr_text": item.ocr_text,
        "ocr_tokens": item.ocr_tokens,
        "ocr_key": item.ocr_key,
        "ocr_source": format_ocr_source(item.ocr_source),
        "textness": (
            round(float(item.textness_score), 4)
            if item.textness_score is not None
            else None
        ),
        "ocr_candidate": item.ocr_candidate,
        "ocr_assigned": item.ocr_assigned,
        "cluster_id": item.cluster_id,
    }


def cluster_ocr_record(item: Any) -> dict[str, Any]:
    mods = _import_index_modules()
    format_ocr_source = mods["format_ocr_source"]
    return {
        "cluster_id": item.cluster_id,
        "ocr_text": item.ocr_text,
        "ocr_tokens": item.ocr_tokens,
        "ocr_key": item.ocr_key,
        "ocr_observation_count": item.ocr_observation_count,
        "ocr_source": format_ocr_source(item.ocr_source),
    }


def bundle_summary_payload(index_path: Path) -> dict[str, Any]:
    mods = _import_index_modules()
    summary = mods["load_bundle_explorer_summary"](index_path)
    format_ocr_source = mods["format_ocr_source"]
    source_counts = summary.ocr_source_counts or {}
    return {
        "index_path": str(summary.index_path),
        "manifest": summary.manifest,
        "keyframe_ids": list(summary.keyframe_ids),
        "cutout_ids_by_keyframe": {
            key: list(values) for key, values in summary.cutout_ids_by_keyframe.items()
        },
        "object_count_by_cutout": dict(summary.object_count_by_cutout),
        "ocr_text_count": summary.ocr_text_count,
        "ocr_key_count": summary.ocr_key_count,
        "ocr_candidate_count": summary.ocr_candidate_count,
        "ocr_assigned_count": summary.ocr_assigned_count,
        "ocr_source_counts": {
            format_ocr_source(int(source)): count
            for source, count in sorted(source_counts.items())
        },
        "cluster_ocr_count": summary.cluster_ocr_count,
        "default_prompts": list(
            mods["extract_default_prompts"](summary.manifest),
        ),
    }


def build_index_keyframe_markers(
    map_path: Path,
    keyframe_ids: list[str],
    selected_keyframe_id: str | None,
) -> list[dict[str, Any]]:
    mods = _import_index_modules()
    lookup_keyframe = mods["lookup_keyframe"]
    markers: list[dict[str, Any]] = []
    for keyframe_id in keyframe_ids:
        metadata = lookup_keyframe(map_path, keyframe_id)
        if metadata is None or metadata.lat is None or metadata.lon is None:
            continue
        is_selected = keyframe_id == selected_keyframe_id
        markers.append(
            {
                "id": keyframe_id,
                "latitude": metadata.lat,
                "longitude": metadata.lon,
                "level": metadata.level,
                "color": "#16a34a" if is_selected else "#9ca3af",
                "radius": 8 if is_selected else 6,
            }
        )
    return markers


def index_status_payload(
    map_path: Path,
    index_path_override: str | None,
    *,
    selected_keyframe_id: str | None = None,
) -> dict[str, Any]:
    checked_paths = [
        str(path) for path in index_path_candidates(map_path, index_path_override)
    ]
    index_path = resolve_map_index_path(map_path, index_path_override)
    if index_path is None:
        return {
            "available": False,
            "index_path": None,
            "map_path": str(map_path),
            "object_search_index_path": index_path_override,
            "checked_paths": checked_paths,
        }
    summary = bundle_summary_payload(index_path)
    markers = build_index_keyframe_markers(
        map_path,
        summary["keyframe_ids"],
        selected_keyframe_id,
    )
    return {
        "available": True,
        "index_path": str(index_path),
        "summary": summary,
        "markers": markers,
        "resolved_marker_count": len(markers),
    }


def index_objects_payload(index_path: Path) -> dict[str, Any]:
    mods = _import_index_modules()
    objects = mods["lookup_all_objects_in_bundle"](index_path)
    return {"objects": [index_object_record(item) for item in objects]}


def index_cluster_ocr_payload(index_path: Path) -> dict[str, Any]:
    mods = _import_index_modules()
    records = mods["load_cluster_ocr_records"](index_path)
    return {"records": [cluster_ocr_record(item) for item in records]}


def cutout_objects_payload(index_path: Path, cutout_id: str) -> dict[str, Any]:
    mods = _import_index_modules()
    detections = mods["lookup_objects_in_bundle"](index_path, cutout_id)
    preview_ref = mods["build_preview_ref"](index_path, cutout_id)
    preview_debug = mods["inspect_index_preview_ref"](preview_ref)
    return {
        "detections": [index_object_record(item) for item in detections],
        "preview_ref": preview_ref,
        "preview_debug": {
            "index_path": preview_debug.index_path,
            "cutout_id": preview_debug.cutout_id,
            "keyframe_id": preview_debug.keyframe_id,
            "source": preview_debug.source,
            "geometry": preview_debug.geometry,
            "source_images_dir": preview_debug.source_images_dir,
            "source_image_path": preview_debug.source_image_path,
            "params": preview_debug.params,
            "error": preview_debug.error,
        },
    }


def render_index_cutout_preview_png(
    index_path: Path,
    cutout_id: str,
    *,
    selected_object_id: str | None,
    draw_boxes: bool = True,
    nms_iou: float = 1.0,
    min_bbox_area: float = 0.0,
    max_bbox_area: float = 0.0,
) -> bytes:
    mods = _import_index_modules()
    preview_ref = mods["build_preview_ref"](index_path, cutout_id)
    image = mods["load_preview_from_path"](preview_ref)
    if image is None:
        raise HTTPException(status_code=404, detail="Preview file not found")
    detections = mods["lookup_objects_in_bundle"](index_path, cutout_id)
    detections = mods["postprocess_detections"](
        detections,
        nms_iou=nms_iou,
        min_bbox_area=min_bbox_area,
        max_bbox_area=max_bbox_area,
    )
    if draw_boxes and detections:
        image = mods["draw_detection_overlay"](
            image,
            detections,
            selected_detection_id=selected_object_id,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_index_object_crop_png(
    index_path: Path,
    cutout_id: str,
    bbox: tuple[float, float, float, float],
) -> bytes:
    mods = _import_index_modules()
    preview_ref = mods["build_preview_ref"](index_path, cutout_id)
    preview_img = mods["load_preview_from_path"](preview_ref)
    if preview_img is None:
        raise HTTPException(status_code=404, detail="Preview file not found")
    x0, y0, x1, y1 = bbox
    width, height = preview_img.size
    left = max(0, min(width, int(np.floor(x0))))
    top = max(0, min(height, int(np.floor(y0))))
    right = max(left + 1, min(width, int(np.ceil(x1))))
    bottom = max(top + 1, min(height, int(np.ceil(y1))))
    if left >= width or top >= height:
        raise HTTPException(status_code=404, detail="Invalid bbox for preview")
    crop = preview_img.crop((left, top, right, bottom)).copy()
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def _similarity_color(value: float, color_min: float, color_max: float) -> str:
    if color_max <= color_min:
        t = 0.5
    else:
        t = (value - color_min) / (color_max - color_min)
    t = max(0.0, min(1.0, t))
    # green (#16a34a) -> gray (#9ca3af)
    r = int(22 + (156 - 22) * (1 - t))
    g = int(163 + (163 - 163) * (1 - t))
    b = int(74 + (175 - 74) * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_prompt_latent_payload(
    index_path: Path,
    *,
    prompts_text: str,
    threshold: float,
    top_k: int,
    projection: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    selected_query: str | None,
    scale_points_by_bbox_area: bool,
) -> dict[str, Any]:
    mods = _import_index_modules()
    latent_data = mods["load_object_embedding_latent_data"](index_path)
    if latent_data.object_embeddings.shape[0] == 0:
        raise HTTPException(
            status_code=404,
            detail="No object embeddings are stored in this object-search index.",
        )

    summary = mods["load_bundle_explorer_summary"](index_path)
    default_prompts = list(mods["extract_default_prompts"](summary.manifest))
    prompts = mods["parse_prompt_lines"](prompts_text)
    if not prompts:
        return {
            "available": False,
            "default_prompts": default_prompts,
            "message": (
                "Enter at least one reference prompt to analyze the latent space."
            ),
        }

    warnings: list[str] = []
    try:
        _, similarity_matrix, text_source, fallback = mods[
            "get_cached_similarity_matrix"
        ](
            index_path,
            tuple(prompts),
            latent_data.object_embeddings,
            batch_size=4,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if fallback:
        warnings.append(str(fallback))
    color_min, color_max = mods["compute_global_color_range"](similarity_matrix)
    query_labels = list(prompts)
    selected = selected_query if selected_query in query_labels else query_labels[0]
    query_idx = query_labels.index(selected)
    query_similarities = similarity_matrix[query_idx]

    projection_key = projection.upper()
    if projection_key == "PCA":
        coords, variance = mods["compute_object_pca_projection"](str(index_path))
        projection_label = "PCA"
        projection_meta = {
            "variance_x": variance[0],
            "variance_y": variance[1],
        }
    else:
        try:
            coords = mods["compute_object_umap_projection"](
                str(index_path),
                int(n_neighbors),
                float(min_dist),
                int(random_state),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        projection_label = "UMAP"
        projection_meta = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        }

    areas = np.asarray(latent_data.bbox_areas, dtype=np.float32)
    if scale_points_by_bbox_area and areas.size:
        min_area = float(np.min(areas))
        max_area = float(np.max(areas))
        if np.isclose(min_area, max_area):
            sizes = [8.0] * areas.shape[0]
        else:
            sizes = [
                6.0 + 10.0 * (float(area) - min_area) / (max_area - min_area)
                for area in areas
            ]
    else:
        sizes = [8.0] * int(latent_data.object_embeddings.shape[0])

    points: list[dict[str, Any]] = []
    for idx in range(int(latent_data.object_embeddings.shape[0])):
        sim = float(query_similarities[idx])
        points.append(
            {
                "object_id": latent_data.object_ids[idx],
                "keyframe_id": latent_data.object_keyframe_ids[idx],
                "cutout_id": latent_data.object_cutout_ids[idx],
                "x": round(float(coords[idx, 0]), 4),
                "y": round(float(coords[idx, 1]), 4),
                "size": round(float(sizes[idx]), 2),
                "similarity": round(sim, 4),
                "active": sim >= threshold,
                "color": _similarity_color(sim, color_min, color_max),
            }
        )

    topk_records = mods["build_topk_records"](
        latent_data,
        query_similarities,
        k=max(1, int(top_k)),
    )

    return {
        "available": True,
        "default_prompts": default_prompts,
        "query_labels": query_labels,
        "selected_query": selected,
        "embedding_backend": text_source,
        "warnings": warnings,
        "color_min": color_min,
        "color_max": color_max,
        "threshold": threshold,
        "projection": projection_label,
        "projection_meta": projection_meta,
        "points": points,
        "topk": topk_records,
    }
