"""Shared types, geometry helpers, and post-processing filters.

Used by both :class:`YoloWorldDetector` and :class:`GdinoVenueDetector` and
by the build pipeline loop in ``build_index.py``.

No model dependencies — safe to import without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.offline.shared.geometry import (
    CUBEMAP_FACE_ORDER,
    relative_pose_cubemap_face,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]

FACE_NAME_TO_LOCAL_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(CUBEMAP_FACE_ORDER)
}


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------


def _normalize_label(label: str) -> str:
    """Lowercase + collapse internal whitespace.

    Used as the NMS key and for people-label filtering.
    """
    return " ".join(label.lower().strip().split())


# ---------------------------------------------------------------------------
# ERP ↔ cubemap-face projection
#
# Convention matches pipeline.offline.localize.localize_3d exactly so that
# projected bbox centres can be fed straight into
# ``cubemap_pixel_to_equirect_uv`` for depth sampling.
#   lon = atan2(x, z),  lat = asin(y),  v_eq = lat/π + 0.5
# ---------------------------------------------------------------------------


def erp_pixel_to_pano_ray(
    x_erp: float,
    y_erp: float,
    erp_w: int,
    erp_h: int,
) -> np.ndarray:
    """ERP pixel → 3D unit ray in the panorama frame."""
    u_eq = float(x_erp) / max(1.0, float(erp_w - 1))
    v_eq = float(y_erp) / max(1.0, float(erp_h - 1))
    lon = (u_eq - 0.5) * 2.0 * np.pi
    lat = (v_eq - 0.5) * np.pi
    cos_lat = np.cos(lat)
    return np.array(
        [np.sin(lon) * cos_lat, np.sin(lat), np.cos(lon) * cos_lat],
        dtype=np.float64,
    )


def pano_ray_to_face_pixel(
    ray_pano: np.ndarray,
    face_name: str,
    face_size: int,
    fov_deg: float,
) -> Tuple[float, float]:
    """Project a panorama-frame ray onto a cubemap face (pixel coords).

    For rays behind the face plane, returns a far off-face value so the
    downstream AABB + clip step produces a truncated bbox rather than dropping
    the detection silently.
    """
    R_face = relative_pose_cubemap_face(face_name)[:3, :3]
    ray_cam = R_face @ ray_pano
    f = 0.5 * float(face_size) / np.tan(0.5 * np.radians(float(fov_deg)))
    c = (float(face_size) - 1.0) / 2.0
    if ray_cam[2] > 1e-6:
        u = ray_cam[0] / ray_cam[2] * f + c
        v = ray_cam[1] / ray_cam[2] * f + c
    else:
        big = 10.0 * float(face_size)
        u = c + big * (1.0 if ray_cam[0] >= 0 else -1.0)
        v = c + big * (1.0 if ray_cam[1] >= 0 else -1.0)
    return float(u), float(v)


def _best_face_for_ray(ray_pano: np.ndarray) -> Optional[str]:
    """Return the cubemap face whose +Z axis is most aligned with the ray."""
    best_face: Optional[str] = None
    best_z = -np.inf
    for face_name in CUBEMAP_FACE_ORDER:
        R_face = relative_pose_cubemap_face(face_name)[:3, :3]
        rz = float((R_face @ ray_pano)[2])
        if rz > best_z:
            best_z = rz
            best_face = face_name
    return best_face if best_z > 0.0 else None


def project_erp_bbox_to_face(
    bbox_erp: Sequence[float],
    erp_w: int,
    erp_h: int,
    *,
    face_size: int,
    fov_deg: float,
) -> Optional[Tuple[str, BBox]]:
    """Project an ERP-pixel bbox to the dominant cubemap face.

    Returns ``(face_name, (x1, y1, x2, y2))`` clipped to ``[0, face_size]``,
    or ``None`` if the projected bbox has < 1 px in any dimension.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_erp)
    cx_erp, cy_erp = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ray_center = erp_pixel_to_pano_ray(cx_erp, cy_erp, erp_w, erp_h)
    face_name = _best_face_for_ray(ray_center)
    if face_name is None:
        return None
    proj: List[Tuple[float, float]] = [
        pano_ray_to_face_pixel(
            erp_pixel_to_pano_ray(cu, cv, erp_w, erp_h),
            face_name,
            face_size,
            fov_deg,
        )
        for cu, cv in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    us = [p[0] for p in proj]
    vs = [p[1] for p in proj]
    fx1 = max(0.0, min(float(face_size), min(us)))
    fy1 = max(0.0, min(float(face_size), min(vs)))
    fx2 = max(0.0, min(float(face_size), max(us)))
    fy2 = max(0.0, min(float(face_size), max(vs)))
    if (fx2 - fx1) < 1.0 or (fy2 - fy1) < 1.0:
        return None
    return face_name, (fx1, fy1, fx2, fy2)


# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------


@dataclass
class HybridDetection:
    """Detection from YOLO-World or GroundingDINO in ERP pixel coordinates.

    ``bbox``      – (x1, y1, x2, y2) in ERP pixels.
    ``label``     – normalised string keyed for label-aware NMS.
    ``raw_label`` – original model output, preserved for debugging.
    ``source``    – ``"yolo"`` or ``"gdino"``.
    """

    bbox: BBox
    score: float
    label: str
    raw_label: str
    source: str


# ---------------------------------------------------------------------------
# Post-processing filters
# ---------------------------------------------------------------------------

# NMS for the pooled detections is the single class-agnostic spherical pass
# below (:func:`spherical_nms`). Planar per-label NMS was removed in favour of
# it to keep de-duplication in one place.


def erp_bbox_to_bfov(
    bbox_erp: Sequence[float],
    erp_w: int,
    erp_h: int,
) -> Tuple[float, float, float, float]:
    """Convert an ERP-pixel bbox to a bounding field-of-view (BFoV).

    In an equirectangular image longitude is linear in x and latitude is linear
    in y, so the angular centre and extents follow directly from the pixel box.

    Args:
        bbox_erp: ``(x1, y1, x2, y2)`` in ERP pixels.
        erp_w: ERP width in pixels.
        erp_h: ERP height in pixels.

    Returns:
        ``(theta, phi, fov_x, fov_y)`` in radians: longitude centre
        ``theta`` in ``[-π, π]``, latitude centre ``phi`` in ``[-π/2, π/2]``,
        and the angular width/height of the box.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_erp)
    w = max(1.0, float(erp_w))
    h = max(1.0, float(erp_h))
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    theta = (cx / w - 0.5) * 2.0 * np.pi
    phi = (cy / h - 0.5) * np.pi
    fov_x = (x2 - x1) / w * 2.0 * np.pi
    fov_y = (y2 - y1) / h * np.pi
    return theta, phi, fov_x, fov_y


def _wrapped_angle(delta: float) -> float:
    """Wrap an angle difference to ``[-π, π]``."""
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


def bfov_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Approximate spherical IoU (FoV-IoU) between two bounding fields of view.

    Unlike planar IoU on ERP pixels, this is correct across the longitude seam
    (``theta`` wraps at ±π) and de-biased at high latitude, where a fixed
    longitudinal span covers less surface (weighted by ``cos(phi)``). This is a
    pragmatic variant of FoV-IoU (Cao et al., 2022) suitable for de-duplication.

    Args:
        a: BFoV ``(theta, phi, fov_x, fov_y)`` in radians.
        b: BFoV ``(theta, phi, fov_x, fov_y)`` in radians.

    Returns:
        Overlap ratio in ``[0, 1]``.
    """
    theta_a, phi_a, fx_a, fy_a = a
    theta_b, phi_b, fx_b, fy_b = b
    hx_a, hy_a = 0.5 * fx_a, 0.5 * fy_a
    hx_b, hy_b = 0.5 * fx_b, 0.5 * fy_b

    # Latitude (phi) overlap — no wrap.
    lo = max(phi_a - hy_a, phi_b - hy_b)
    hi = min(phi_a + hy_a, phi_b + hy_b)
    inter_phi = max(0.0, hi - lo)
    if inter_phi <= 0.0:
        return 0.0

    # Longitude (theta) overlap on a circle (handles the ±π seam).
    d_theta = abs(_wrapped_angle(theta_a - theta_b))
    inter_theta = max(0.0, (hx_a + hx_b) - d_theta)
    inter_theta = min(inter_theta, 2.0 * hx_a, 2.0 * hx_b)
    if inter_theta <= 0.0:
        return 0.0

    # cos(phi) weighting: longitudinal surface extent shrinks toward the poles.
    cos_inter = max(0.0, float(np.cos(0.5 * (lo + hi))))
    cos_a = max(1e-6, float(np.cos(phi_a)))
    cos_b = max(1e-6, float(np.cos(phi_b)))
    area_a = fx_a * cos_a * fy_a
    area_b = fx_b * cos_b * fy_b
    inter = inter_theta * cos_inter * inter_phi
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def spherical_nms(
    detections: List[HybridDetection],
    erp_w: int,
    erp_h: int,
    iou_threshold: float,
) -> List[HybridDetection]:
    """Single, class-agnostic non-maximum suppression on the sphere.

    Greedy, highest-score-first suppression keyed on spherical (FoV) overlap.
    Replaces planar label-aware NMS on the pooled YOLO-World + GroundingDINO
    detections: being class-agnostic it removes same-label *and* cross-label
    duplicates in one pass (e.g. the same physical object detected by both
    detectors under different labels), and being spherical it is correct across
    the longitude seam and de-biased at the poles where planar ERP IoU fails.

    Args:
        detections: Pooled detections in ERP-pixel coordinates.
        erp_w: ERP width in pixels.
        erp_h: ERP height in pixels.
        iou_threshold: Spherical-IoU above which the lower-scoring detection is
            dropped. ``<= 0`` disables suppression (returns the input unchanged).

    Returns:
        The surviving detections, highest-score-first.
    """
    if iou_threshold <= 0.0 or len(detections) < 2:
        return list(detections)
    order = sorted(detections, key=lambda d: -d.score)
    bfovs = [erp_bbox_to_bfov(d.bbox, erp_w, erp_h) for d in order]
    kept: List[HybridDetection] = []
    kept_bfovs: List[Tuple[float, float, float, float]] = []
    for det, bf in zip(order, bfovs):
        if all(bfov_iou(bf, kb) <= iou_threshold for kb in kept_bfovs):
            kept.append(det)
            kept_bfovs.append(bf)
    return kept


def _clip_bboxes_to_image(
    detections: List[HybridDetection],
    img_w: int,
    img_h: int,
) -> List[HybridDetection]:
    out: List[HybridDetection] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        nx1 = max(0.0, min(float(img_w), x1))
        ny1 = max(0.0, min(float(img_h), y1))
        nx2 = max(0.0, min(float(img_w), x2))
        ny2 = max(0.0, min(float(img_h), y2))
        if nx2 > nx1 and ny2 > ny1:
            out.append(
                HybridDetection(
                    bbox=(nx1, ny1, nx2, ny2),
                    score=d.score,
                    label=d.label,
                    raw_label=d.raw_label,
                    source=d.source,
                )
            )
    return out


def _area_filter(
    detections: List[HybridDetection],
    img_w: int,
    img_h: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[HybridDetection]:
    image_area = float(img_w * img_h)
    if image_area <= 0:
        return list(detections)
    return [
        d
        for d in detections
        if min_area_ratio
        <= max(0.0, d.bbox[2] - d.bbox[0])
        * max(0.0, d.bbox[3] - d.bbox[1])
        / image_area
        <= max_area_ratio
    ]


def _aspect_ratio_filter(
    detections: List[HybridDetection],
    min_ratio: float,
    max_ratio: float,
) -> List[HybridDetection]:
    return [
        d
        for d in detections
        if min_ratio
        <= max(1e-6, d.bbox[2] - d.bbox[0]) / max(1e-6, d.bbox[3] - d.bbox[1])
        <= max_ratio
    ]


def compute_bbox_spherical(
    fx1: float,
    fy1: float,
    fx2: float,
    fy2: float,
    face_name: str,
    face_size: int,
    fov_deg: float,
) -> np.ndarray:
    """Compute spherical coordinates [theta, phi, fov_x, fov_y] in radians for
    a face-space bbox.

    Args:
        fx1, fy1, fx2, fy2: Bounding box in face pixel coordinates [0, face_size].
        face_name: Cubemap face name (e.g., "front", "right", "top").
        face_size: Pixel width/height of the face image.
        fov_deg: Horizontal field-of-view in degrees for the cubemap face.

    Returns:
        np.ndarray of shape (4,) with dtype float32:
        - theta: azimuth (atan2(x, z)), range [-π, π]
        - phi: elevation (asin(y)), range [-π/2, π/2]
        - fov_x: angular width of the bbox in radians
        - fov_y: angular height of the bbox in radians
    """
    # Compute bbox center in face pixels
    cx = 0.5 * (fx1 + fx2)
    cy = 0.5 * (fy1 + fy2)

    # Focal length derived from face size and FOV
    f = 0.5 * float(face_size) / np.tan(0.5 * np.radians(float(fov_deg)))

    # Principal point (center of face, 0-indexed)
    c = (float(face_size) - 1.0) / 2.0

    # Convert face pixel coordinates to camera-frame ray (z-forward, OpenCV convention)
    ray_cam = np.array([(cx - c) / f, (cy - c) / f, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    # Rotate from camera frame to panorama frame using the face's rotation matrix
    R_face = relative_pose_cubemap_face(face_name)[:3, :3]
    ray_pano = R_face.T @ ray_cam

    # Extract spherical coordinates: theta = azimuth, phi = elevation
    x, y, z = ray_pano
    theta = float(np.arctan2(x, z))
    phi = float(np.arcsin(float(np.clip(y, -1.0, 1.0))))

    # Compute angular field-of-view from pixel dimensions
    fov_x = float(2.0 * np.arctan(0.5 * abs(fx2 - fx1) / max(f, 1e-6)))
    fov_y = float(2.0 * np.arctan(0.5 * abs(fy2 - fy1) / max(f, 1e-6)))

    return np.array([theta, phi, fov_x, fov_y], dtype=np.float32)
