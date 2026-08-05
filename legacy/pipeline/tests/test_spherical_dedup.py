"""Tests for the spherical (FoV-IoU) cross-label de-duplication pass.

These exercise the pure-numpy geometry in ``pipeline.offline.detect.common``
without importing any model code. ``cv2`` (pulled in transitively by the
geometry module) is stubbed so the suite runs without OpenCV installed.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from pipeline.offline.detect.common import (  # noqa: E402
    HybridDetection,
    bfov_iou,
    erp_bbox_to_bfov,
    spherical_nms,
)

W, H = 4000, 2000


def _det(bbox, score, label, source="yolo"):
    return HybridDetection(
        bbox=bbox, score=score, label=label, raw_label=label, source=source
    )


def test_erp_bbox_to_bfov_center_and_extent():
    # A box centred on the image maps to (theta=0, phi=0).
    theta, phi, fov_x, fov_y = erp_bbox_to_bfov((1900, 900, 2100, 1100), W, H)
    assert theta == pytest.approx(0.0, abs=1e-6)
    assert phi == pytest.approx(0.0, abs=1e-6)
    assert fov_x == pytest.approx((200 / W) * 2 * np.pi, rel=1e-6)
    assert fov_y == pytest.approx((200 / H) * np.pi, rel=1e-6)


def test_identical_boxes_have_iou_one():
    bf = erp_bbox_to_bfov((1000, 900, 1100, 1100), W, H)
    assert bfov_iou(bf, bf) == pytest.approx(1.0, abs=1e-6)


def test_disjoint_boxes_have_zero_iou():
    a = erp_bbox_to_bfov((1000, 1000, 1100, 1200), W, H)
    b = erp_bbox_to_bfov((3000, 1000, 3100, 1200), W, H)
    assert bfov_iou(a, b) == 0.0


def test_cross_label_duplicate_is_merged():
    # Same physical object, two detectors, two labels, overlapping boxes.
    a = _det((1000, 900, 1100, 1100), 0.9, "trash can", "yolo")
    b = _det((1010, 910, 1105, 1095), 0.6, "gdino_venue", "gdino")
    kept = spherical_nms([a, b], W, H, 0.55)
    assert len(kept) == 1
    assert kept[0].label == "trash can"  # higher score survives


def test_same_label_duplicate_is_merged():
    # The single pass also subsumes same-label NMS (no separate stage needed).
    a = _det((1000, 900, 1100, 1100), 0.9, "trash can", "yolo")
    b = _det((1008, 908, 1102, 1098), 0.5, "trash can", "gdino")
    kept = spherical_nms([a, b], W, H, 0.55)
    assert len(kept) == 1


def test_distinct_objects_are_kept():
    a = _det((1000, 900, 1100, 1100), 0.9, "trash can")
    c = _det((3000, 1000, 3080, 1200), 0.7, "sign")
    kept = spherical_nms([a, c], W, H, 0.55)
    assert len(kept) == 2


def test_threshold_zero_disables_pass():
    a = _det((1000, 900, 1100, 1100), 0.9, "trash can")
    b = _det((1010, 910, 1105, 1095), 0.6, "gdino_venue", "gdino")
    kept = spherical_nms([a, b], W, H, 0.0)
    assert len(kept) == 2


def test_longitude_wrap_uses_cyclic_distance():
    # Two BFoVs straddling the +/-pi seam. With cyclic distance they overlap;
    # a naive planar distance (|theta_a - theta_b| ~ 6 rad) would report none.
    a = (-3.00, 0.0, 0.40, 0.40)
    b = (3.05, 0.0, 0.40, 0.40)
    assert bfov_iou(a, b) > 0.0


def test_high_latitude_boxes_do_not_crash_and_stay_bounded():
    # Near the pole, cos(phi) -> 0; IoU must stay finite in [0, 1].
    a = (0.0, 1.50, 0.30, 0.20)
    b = (0.2, 1.50, 0.30, 0.20)
    iou = bfov_iou(a, b)
    assert 0.0 <= iou <= 1.0
