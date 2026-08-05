"""Shared Paddle OCR merge logic, bbox helpers, and memory release (internal)."""

from __future__ import annotations

import gc
import re
from typing import Any, cast

import numpy as np
from PIL import Image

from pipeline.core.logging import logger

from .identity import OcrIdentity


def release_ocr_memory(reason: str) -> None:
    gc.collect()
    try:
        import paddle

        if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
            empty_cache = getattr(paddle.device.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        elif hasattr(paddle, "framework"):
            empty_cache = getattr(
                getattr(paddle.framework, "core", None), "cuda_empty_cache", None
            )
            if callable(empty_cache):
                empty_cache()
    except Exception as exc:
        logger.warning("Paddle memory release after %s failed: %s", reason, exc)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Torch memory release after %s failed: %s", reason, exc)
    logger.info("Released OCR memory after %s", reason)


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, dict):
        return cast(dict[str, Any], prediction.get("res", prediction))

    json_attr = getattr(prediction, "json", None)
    if callable(json_attr):
        data = json_attr()
        if isinstance(data, dict):
            return cast(dict[str, Any], data.get("res", data))
    elif isinstance(json_attr, dict):
        return cast(dict[str, Any], json_attr.get("res", json_attr))

    res_attr = getattr(prediction, "res", None)
    if isinstance(res_attr, dict):
        return cast(dict[str, Any], res_attr)

    to_dict = getattr(prediction, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return cast(dict[str, Any], data.get("res", data))

    return {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return cast(list[Any], value.tolist())
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def merge_paddleocr_vl_texts(outputs: Any) -> str:
    texts: list[str] = []

    def visit(node: Any, *, key: str = "") -> None:
        if isinstance(node, dict):
            block_content = node.get("block_content")
            if isinstance(block_content, str) and block_content.strip():
                texts.append(block_content.strip())
                return
            for child_key, child_value in node.items():
                visit(child_value, key=str(child_key))
            return
        if isinstance(node, list) or isinstance(node, tuple):
            for child in node:
                visit(child, key=key)
            return
        content = getattr(node, "content", None)
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
            return
        block_content = getattr(node, "block_content", None)
        if isinstance(block_content, str) and block_content.strip():
            texts.append(block_content.strip())
            return
        if hasattr(node, "to_dict") or hasattr(node, "json") or hasattr(node, "res"):
            result = prediction_to_dict(node)
            if result:
                visit(result, key=key)
            return
        if isinstance(node, str) and key in {
            "block_content",
            "content",
            "text",
            "rec_text",
        }:
            text = node.strip()
            if text:
                texts.append(text)

    for output in as_list(outputs):
        visit(prediction_to_dict(output))

    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return " ".join(deduped).strip()


def merge_lightweight_predictions(
    predictions: Any,
) -> tuple[str, float, tuple[float, float, float, float] | None]:
    texts: list[str] = []
    scores: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    for prediction in as_list(predictions):
        result = prediction_to_dict(prediction)
        rec_texts = [str(text).strip() for text in as_list(result.get("rec_texts"))]
        rec_scores = as_list(result.get("rec_scores"))
        rec_boxes = as_list(result.get("rec_boxes"))
        if not rec_boxes:
            rec_boxes = as_list(result.get("rec_polys"))
        if not rec_boxes:
            rec_boxes = as_list(result.get("dt_polys"))
        for idx, text in enumerate(rec_texts):
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                continue
            texts.append(normalized)
            if idx < len(rec_boxes):
                box = bbox_from_ocr_box(rec_boxes[idx])
                if box is not None:
                    boxes.append(box)
            if idx < len(rec_scores):
                try:
                    score = float(rec_scores[idx])
                except (TypeError, ValueError):
                    score = np.nan
            else:
                score = np.nan
            if np.isfinite(score):
                scores.append(score)
    return (
        " ".join(texts),
        (float(max(scores)) if scores else np.nan),
        union_bboxes(boxes),
    )


def bbox_from_ocr_box(box: Any) -> tuple[float, float, float, float] | None:
    try:
        arr = np.asarray(box, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size < 4:
        return None
    if arr.ndim == 1 and arr.size >= 4:
        x0, y0, x1, y1 = [float(v) for v in arr[:4]]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    arr = arr.reshape(-1, 2)
    return (
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 0])),
        float(np.max(arr[:, 1])),
    )


def union_bboxes(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def crop_expanded_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    pad_fraction: float = 0.25,
    min_pad_px: int = 4,
) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(float(min_pad_px), max(x1 - x0, y1 - y0) * float(pad_fraction))
    left = max(0, int(np.floor(x0 - pad)))
    top = max(0, int(np.floor(y0 - pad)))
    right = min(image.width, int(np.ceil(x1 + pad)))
    bottom = min(image.height, int(np.ceil(y1 + pad)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def is_lightweight_ocr_accepted(
    identity: OcrIdentity,
    *,
    score: float,
    min_score: float,
    single_letter_min_score: float,
) -> bool:
    if identity.is_empty or not np.isfinite(float(score)):
        return False
    has_mixed_alpha_numeric_token = any(
        re.search(r"[a-z]", token) and re.search(r"\d", token)
        for token in identity.tokens
    )
    if has_mixed_alpha_numeric_token:
        return False
    if identity.letters and any(len(letter) == 1 for letter in identity.letters):
        return float(score) >= float(single_letter_min_score)
    if identity.numbers:
        return float(score) >= float(min_score)
    if len(identity.letters) == 1 and len(identity.letters[0]) == 1:
        return float(score) >= float(single_letter_min_score)
    return float(score) >= float(min_score)


def default_lightweight_paddle_device(device: str) -> str:
    return "gpu:0" if str(device).startswith("cuda") else "cpu"
