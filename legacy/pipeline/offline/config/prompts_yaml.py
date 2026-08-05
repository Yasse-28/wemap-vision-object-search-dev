"""Load GroundingDINO + post-process settings from YAML (`gdino_params` or
eval `experiments`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pipeline.core.logging import logger
from pipeline.core.types import (
    DEFAULT_ID_STRIDE,
    CutoutGeometryConfig,
    InferenceConfig,
    KeyframeFilterConfig,
    LocalizationConfig,
    OcrRefinementConfig,
    OfflineBuildConfig,
    VisualRefinementConfig,
)

# Align with the standalone GroundingDINOModel.default_conf["threshold"]
DEFAULT_GDINO_MODEL_THRESHOLD = 0.15
# Align with _postprocess_detections defaults in object_search/detectors/postprocess.py
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MIN_AREA_RATIO = 0.01
DEFAULT_MAX_AREA_RATIO = 0.50
DEFAULT_MIN_CONFIDENCE = 0.10
DEFAULT_CROP_PADDING_PX = 20


@dataclass
class GdinoParams:
    prompt: str
    score: Optional[float] = None
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    crop_padding_px: int = DEFAULT_CROP_PADDING_PX
    exclude_labels: List[str] | None = None

    def model_threshold_used(self) -> float:
        return (
            float(self.score)
            if self.score is not None
            else DEFAULT_GDINO_MODEL_THRESHOLD
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score": self.score,
            "model_threshold_used": self.model_threshold_used(),
            "iou_threshold": self.iou_threshold,
            "min_area_ratio": self.min_area_ratio,
            "max_area_ratio": self.max_area_ratio,
            "min_confidence": self.min_confidence,
            "crop_padding_px": self.crop_padding_px,
            "exclude_labels": list(self.exclude_labels or []),
        }

    def to_manifest_json(self) -> str:
        return json.dumps(self.to_manifest_dict(), indent=2)


def _resolve_gdino_prompt_string(data: Dict[str, Any], *, context: str) -> str:
    """Read non-empty `prompt` (str), or legacy single-item `prompts` list
    with a deprecation log."""
    raw_prompt = data.get("prompt")
    if raw_prompt is not None and str(raw_prompt).strip():
        return str(raw_prompt).strip()
    legacy = data.get("prompts")
    if isinstance(legacy, list) and len(legacy) == 1 and str(legacy[0]).strip():
        logger.warning(
            "Deprecated: %s uses `prompts` with a single entry; use"
            ' `prompt: "..."` instead',
            context,
        )
        return str(legacy[0]).strip()
    raise ValueError(
        f"{context} must set non-empty 'prompt' (string), "
        f"or legacy 'prompts' with exactly one non-empty string"
    )


def _load_yaml(yaml_path: Path) -> Dict[str, Any]:
    yaml_path = yaml_path.resolve()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty YAML: {yaml_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path} must contain a YAML mapping")
    return raw


def load_keyframe_filter_config(yaml_path: Path) -> KeyframeFilterConfig:
    yaml_path = yaml_path.resolve()
    raw = _load_yaml(yaml_path)
    filters = raw.get("keyframe_filters")
    if filters is None:
        return KeyframeFilterConfig()
    if not isinstance(filters, dict):
        raise ValueError("keyframe_filters must be a mapping")

    level_filter = (
        int(filters["level_filter"])
        if filters.get("level_filter") is not None
        else None
    )
    polygon_raw = filters.get("polygon_geojson_path")
    if polygon_raw is None or str(polygon_raw).strip() == "":
        return KeyframeFilterConfig(level_filter=level_filter)

    polygon_path = Path(str(polygon_raw)).expanduser()
    if not polygon_path.is_absolute():
        polygon_path = yaml_path.parent / polygon_path
    return KeyframeFilterConfig(
        level_filter=level_filter,
        polygon_geojson_path=polygon_path.resolve(),
    )


def _section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"offline.{key} must be a mapping")
    return value


def _optional_bool(value: Any, key: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _localization_enabled(localization: Dict[str, Any]) -> bool:
    if bool(localization.get("skip_localize_3d", False)):
        return False
    return bool(localization.get("enabled", True))


def _reject_misplaced_keyframe_filters(localization: Dict[str, Any]) -> None:
    misplaced = [
        key for key in ("level_filter", "polygon_geojson_path") if key in localization
    ]
    if misplaced:
        joined = ", ".join(misplaced)
        raise ValueError(
            f"Move {joined} from offline.localization to top-level keyframe_filters"
        )


def _reject_removed_flat_source(geometry: Dict[str, Any]) -> None:
    source = geometry.get("source")
    if source is not None and str(source) != "equirect360":
        raise ValueError(
            "offline.geometry.source no longer supports flat cutouts; "
            "use equirectangular images in {map_path}/images_360"
        )


def load_offline_build_config(yaml_path: Path) -> OfflineBuildConfig:
    raw = _load_yaml(yaml_path)
    offline = raw.get("offline")
    if offline is None:
        return OfflineBuildConfig()
    if not isinstance(offline, dict):
        raise ValueError("offline must be a mapping")

    inference = _section(offline, "inference")
    geometry = _section(offline, "geometry")
    localization = _section(offline, "localization")
    visual = _section(offline, "visual_refinement")
    ocr = _section(offline, "ocr_refinement")
    _reject_misplaced_keyframe_filters(localization)
    _reject_removed_flat_source(geometry)

    return OfflineBuildConfig(
        device=str(offline["device"]) if offline.get("device") is not None else None,
        skip_objects=_optional_bool(
            offline.get("skip_objects"), "offline.skip_objects"
        ),
        inference=(
            InferenceConfig(
                batch_size=int(inference.get("batch_size", 8)),
                cutout_workers=int(inference.get("cutout_workers", 2)),
                cutout_prefetch=int(inference.get("cutout_prefetch", 8)),
                max_detections_per_image=(
                    int(inference["max_detections_per_image"])
                    if inference.get("max_detections_per_image") is not None
                    else None
                ),
                object_crop_batch_size=(
                    int(inference["object_crop_batch_size"])
                    if inference.get("object_crop_batch_size") is not None
                    else None
                ),
            )
            if inference
            else None
        ),
        geometry=(
            CutoutGeometryConfig(
                geometry=geometry.get("geometry", "cubemap"),
                id_stride=int(geometry.get("id_stride", DEFAULT_ID_STRIDE)),
                cubemap_face_size=int(geometry.get("cubemap_face_size", 512)),
                cubemap_fov_deg=float(geometry.get("cubemap_fov_deg", 90.0)),
                horizon_nb_of_cuts=int(geometry.get("horizon_nb_of_cuts", 12)),
                horizon_fov=float(geometry.get("horizon_fov", 60.0)),
                horizon_inclination=float(geometry.get("horizon_inclination", 0.0)),
                horizon_width=int(geometry.get("horizon_width", 720)),
                horizon_height=int(geometry.get("horizon_height", 720)),
            )
            if geometry
            else None
        ),
        localization=(
            LocalizationConfig(
                enabled=_localization_enabled(localization),
                depth_dir=str(localization.get("depth_dir", "depths")),
                clustering_eps_m=float(localization.get("clustering_eps_m", 3.0)),
                min_depth_m=float(localization.get("min_depth_m", 0.5)),
                max_depth_m=float(localization.get("max_depth_m", 25.0)),
                embedding_similarity_threshold=float(
                    localization.get("embedding_similarity_threshold", 0.85)
                ),
            )
            if localization
            else None
        ),
        visual_refinement=(
            VisualRefinementConfig(
                enabled=bool(visual.get("enabled", False)),
                repo=str(visual.get("repo", "facebookresearch/dinov2")),
                model=str(visual.get("model", "dinov2_vitb14")),
                image_size=int(visual.get("image_size", 256)),
                batch_size=int(visual.get("batch_size", 16)),
                similarity_threshold=float(visual.get("similarity_threshold", 0.6)),
                min_cluster_observations=int(visual.get("min_cluster_observations", 3)),
                min_component_observations=int(
                    visual.get("min_component_observations", 2)
                ),
            )
            if visual
            else None
        ),
        ocr_refinement=(
            OcrRefinementConfig(
                enabled=bool(ocr.get("enabled", False)),
                lightweight_first=bool(ocr.get("lightweight_first", False)),
                lightweight_lang=str(ocr.get("lightweight_lang", "fr")),
                lightweight_preprocess=str(ocr.get("lightweight_preprocess", "raw")),
                lightweight_min_score=float(ocr.get("lightweight_min_score", 0.7)),
                lightweight_single_letter_min_score=float(
                    ocr.get("lightweight_single_letter_min_score", 0.85)
                ),
                lightweight_consensus_min_support_fraction=float(
                    ocr.get("lightweight_consensus_min_support_fraction", 0.6)
                ),
                lightweight_consensus_min_margin=int(
                    ocr.get("lightweight_consensus_min_margin", 1)
                ),
                vl_fallback=bool(ocr.get("vl_fallback", True)),
                textness_threshold=float(ocr.get("textness_threshold", 0.02)),
                min_anchor_observations=int(ocr.get("min_anchor_observations", 2)),
                min_anchor_keyframes=int(ocr.get("min_anchor_keyframes", 2)),
                assignment_margin_m=float(ocr.get("assignment_margin_m", 0.5)),
                preprocess=str(ocr.get("preprocess", "upscale4_autocontrast")),
                max_candidates=(
                    int(ocr["max_candidates"])
                    if ocr.get("max_candidates") is not None
                    else None
                ),
                max_candidates_per_cluster=(
                    int(ocr["max_candidates_per_cluster"])
                    if ocr.get("max_candidates_per_cluster") is not None
                    else None
                ),
                batch_size=int(ocr.get("batch_size", 4)),
                max_new_tokens=int(ocr.get("max_new_tokens", 32)),
            )
            if ocr
            else None
        ),
    )


def load_gdino_params(
    yaml_path: Path, *, experiment: Optional[str] = None
) -> GdinoParams:
    yaml_path = yaml_path.resolve()
    raw = _load_yaml(yaml_path)

    if experiment is not None:
        exps = raw.get("experiments")
        if not isinstance(exps, list):
            raise ValueError(
                "YAML with --experiment must contain an 'experiments' list"
            )
        found: Optional[Dict[str, Any]] = None
        for e in exps:
            if isinstance(e, dict) and e.get("name") == experiment:
                found = e
                break
        if found is None:
            raise ValueError(f"No experiment named {experiment!r} in {yaml_path}")
        prompt = _resolve_gdino_prompt_string(
            found, context=f"experiment {experiment!r}"
        )
        score = found.get("confidence_threshold")
        return GdinoParams(
            prompt=prompt,
            score=float(score) if score is not None else None,
            iou_threshold=DEFAULT_IOU_THRESHOLD,
            min_area_ratio=DEFAULT_MIN_AREA_RATIO,
            max_area_ratio=DEFAULT_MAX_AREA_RATIO,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
            crop_padding_px=DEFAULT_CROP_PADDING_PX,
            exclude_labels=[str(v) for v in found.get("exclude_labels", [])],
        )

    gp = raw.get("gdino_params")
    if not isinstance(gp, dict):
        raise ValueError(
            f"{yaml_path} must contain top-level 'gdino_params' (or use --experiment)"
        )
    prompt = _resolve_gdino_prompt_string(gp, context=f"gdino_params in {yaml_path}")

    def _f(key: str, default: float) -> float:
        v = gp.get(key)
        return float(v) if v is not None else default

    score_raw = gp.get("score")
    return GdinoParams(
        prompt=prompt,
        score=float(score_raw) if score_raw is not None else None,
        iou_threshold=_f("iou_threshold", DEFAULT_IOU_THRESHOLD),
        min_area_ratio=_f("min_area_ratio", DEFAULT_MIN_AREA_RATIO),
        max_area_ratio=_f("max_area_ratio", DEFAULT_MAX_AREA_RATIO),
        min_confidence=_f("min_confidence", DEFAULT_MIN_CONFIDENCE),
        crop_padding_px=int(gp.get("crop_padding_px", DEFAULT_CROP_PADDING_PX)),
        exclude_labels=[str(v) for v in gp.get("exclude_labels", [])],
    )
