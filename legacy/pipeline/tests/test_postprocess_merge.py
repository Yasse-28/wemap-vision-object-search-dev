"""Multi-prompt merge is: concatenate detections then postprocess (sort + NMS)."""

from pipeline.core.detectors import Detection, postprocess_detections


def test_combined_prompt_detections_nms_keeps_highest_iou_overlap() -> None:
    """Two highly overlapping boxes (as from two prompts) collapse to one."""
    high = Detection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.95, label="p1")
    low = Detection(bbox=(12.0, 12.0, 48.0, 48.0), confidence=0.40, label="p2")
    combined = [low, high]
    kept = postprocess_detections(combined, img_w=100, img_h=100, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.95


def test_non_overlapping_boxes_both_kept() -> None:
    a = Detection(bbox=(5.0, 5.0, 20.0, 20.0), confidence=0.9, label="a")
    b = Detection(bbox=(60.0, 60.0, 90.0, 90.0), confidence=0.85, label="b")
    kept = postprocess_detections([a, b], img_w=100, img_h=100, iou_threshold=0.5)
    assert len(kept) == 2
