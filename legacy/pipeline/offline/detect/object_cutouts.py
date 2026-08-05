"""GroundingDINO + MetaCLIP on pre-built cutout images (per-cutout ids)."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple, cast

import torch
from PIL import Image
from tqdm import tqdm

from pipeline.core.detectors import Detection, postprocess_detections
from pipeline.core.logging import logger
from pipeline.core.models.detection.grounding_dino import GroundingDINOModel
from pipeline.core.models.metaclip import MetaCLIP
from pipeline.offline.config.prompts_yaml import GdinoParams
from pipeline.offline.schema.cutout_schema import CutoutRecord


def _normalize_detection_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def _filter_excluded_labels(
    detections: List[Detection], excluded_labels: List[str] | None
) -> List[Detection]:
    if not excluded_labels:
        return detections
    excluded = {_normalize_detection_label(label) for label in excluded_labels}
    return [
        detection
        for detection in detections
        if _normalize_detection_label(detection.label) not in excluded
    ]


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    padding_px: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    padded_x0 = max(0, int(x0) - padding_px)
    padded_y0 = max(0, int(y0) - padding_px)
    padded_x1 = min(image_width, int(x1) + padding_px)
    padded_y1 = min(image_height, int(y1) + padding_px)
    if padded_x1 <= padded_x0:
        padded_x1 = min(image_width, padded_x0 + 1)
    if padded_y1 <= padded_y0:
        padded_y1 = min(image_height, padded_y0 + 1)
    return padded_x0, padded_y0, padded_x1, padded_y1


@torch.no_grad()
def detect_and_extract_object_features_for_cutouts(
    detector: GroundingDINOModel,
    clip_model: MetaCLIP,
    cutouts: List[CutoutRecord],
    text_prompt: str,
    batch_size: int = 4,
    max_detections_per_image: int | None = None,
    object_crop_batch_size: int | None = None,
) -> Dict[Tuple[int, int], Tuple[int, Tuple[float, float, float, float], torch.Tensor]]:
    """Single-prompt convenience wrapper."""
    gdino = GdinoParams(prompt=text_prompt)
    return cast(
        Dict[
            Tuple[int, int],
            Tuple[int, Tuple[float, float, float, float], torch.Tensor],
        ],
        detect_and_extract_object_features_multi_prompt(
            detector,
            clip_model,
            cutouts,
            gdino,
            batch_size=batch_size,
            max_detections_per_image=max_detections_per_image,
            object_crop_batch_size=object_crop_batch_size,
        ),
    )


@torch.no_grad()
def detect_and_extract_object_features_multi_prompt(
    detector: GroundingDINOModel,
    clip_model: MetaCLIP,
    cutouts: List[CutoutRecord],
    gdino: GdinoParams,
    batch_size: int = 4,
    max_detections_per_image: int | None = None,
    object_crop_batch_size: int | None = None,
    show_progress: bool = True,
    log_summary: bool = True,
    detection_timing_callback: Callable[[float, int], None] | None = None,
) -> Dict[Tuple[int, int], Tuple[int, Tuple[float, float, float, float], torch.Tensor]]:
    """GroundingDINO detect → merge raw boxes per image → one
    `_postprocess_detections` pass → MetaCLIP crops.

    Keys: (cutout_id, obj_idx) after merge and filtering, per cutout image.
    """
    features_dict: Dict[
        Tuple[int, int], Tuple[int, Tuple[float, float, float, float], torch.Tensor]
    ] = {}

    with tqdm(
        total=len(cutouts), desc="Detecting objects", disable=not show_progress
    ) as pbar:
        for i in range(0, len(cutouts), batch_size):
            batch = cutouts[i : i + batch_size]
            batch_images = [c.image for c in batch]
            batch_cut_ids = [c.cutout_id for c in batch]
            batch_kf_ids = [c.keyframe_id for c in batch]

            merged_by_idx: List[List[Detection]] = [[] for _ in range(len(batch))]
            if detection_timing_callback is None:
                detections_list = detector.detect(batch_images)
            else:
                if (
                    str(detector.device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.synchronize()
                started_at = time.perf_counter()
                detections_list = detector.detect(batch_images)
                if (
                    str(detector.device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.synchronize()
                detection_timing_callback(
                    time.perf_counter() - started_at, len(batch_images)
                )
            for j, dets in enumerate(detections_list):
                merged_by_idx[j].extend(dets)

            crop_batch: List[Image.Image] = []
            crop_meta: List[Tuple[int, int, Tuple[float, float, float, float], int]] = (
                []
            )

            for img, cut_id, kf_id, combined in zip(
                batch_images, batch_cut_ids, batch_kf_ids, merged_by_idx
            ):
                kept = postprocess_detections(
                    combined,
                    img.width,
                    img.height,
                    min_confidence=gdino.min_confidence,
                    min_area_ratio=gdino.min_area_ratio,
                    max_area_ratio=gdino.max_area_ratio,
                    iou_threshold=gdino.iou_threshold,
                )
                kept = _filter_excluded_labels(kept, gdino.exclude_labels)
                if max_detections_per_image is not None:
                    kept = kept[:max_detections_per_image]

                for obj_idx, det in enumerate(kept):
                    x0, y0, x1, y1 = _expand_bbox(
                        det.bbox,
                        padding_px=gdino.crop_padding_px,
                        image_width=img.width,
                        image_height=img.height,
                    )
                    crop_batch.append(img.crop((x0, y0, x1, y1)))
                    bbox_f = (float(x0), float(y0), float(x1), float(y1))
                    crop_meta.append((cut_id, obj_idx, bbox_f, kf_id))

            if crop_batch:
                crop_chunk_size = max(1, int(object_crop_batch_size or len(crop_batch)))
                for start in range(0, len(crop_batch), crop_chunk_size):
                    end = start + crop_chunk_size
                    crop_features = clip_model.get_image_features(crop_batch[start:end])
                    for (cut_id, obj_idx, bbox_f, kf_id), feat in zip(
                        crop_meta[start:end], crop_features
                    ):
                        features_dict[(cut_id, obj_idx)] = (kf_id, bbox_f, feat)

            pbar.update(len(batch))

    if not features_dict:
        logger.warning(
            "No objects detected for gdino prompt — object rows will be empty",
        )
        return features_dict

    if log_summary:
        logger.info("Extracted features for %d object crops", len(features_dict))
    return dict(sorted(features_dict.items()))
