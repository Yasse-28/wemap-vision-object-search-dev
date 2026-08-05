from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    label: str


def postprocess_detections(
    detections: List[Detection],
    img_w: int,
    img_h: int,
    min_confidence: float = 0.10,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.50,
    iou_threshold: float = 0.50,
) -> List[Detection]:
    image_area = float(img_w * img_h)

    def area(bbox: Tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def iou(
        a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
    ) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    clipped = []
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox
        detection.bbox = (
            max(0.0, min(float(img_w), x0)),
            max(0.0, min(float(img_h), y0)),
            max(0.0, min(float(img_w), x1)),
            max(0.0, min(float(img_h), y1)),
        )
        clipped.append(detection)

    filtered = [
        detection
        for detection in clipped
        if detection.bbox[2] > detection.bbox[0]
        and detection.bbox[3] > detection.bbox[1]
        and detection.confidence >= min_confidence
        and area(detection.bbox) > min_area_ratio * image_area
        and area(detection.bbox) < max_area_ratio * image_area
    ]
    filtered.sort(key=lambda detection: detection.confidence, reverse=True)

    kept: List[Detection] = []
    for detection in filtered:
        if all(
            iou(detection.bbox, previous.bbox) <= iou_threshold for previous in kept
        ):
            kept.append(detection)
    return kept
