"""Unit tests for the per-map analysis tool."""

from __future__ import annotations

import math

import numpy as np

from toolbox.benchmark.map_analysis import (
    Detections,
    GroundTruth,
    _angular_delta,
    _conditional_table,
    _horizontal_anisotropy,
    _mutual_information,
    _overlapping_pairs,
    _percentiles,
    _rank_auc,
    _uv_to_theta_phi,
    attach_to_ground_truth,
    seen_in_own_keyframe,
)
from toolbox.benchmark.map_layers import _colour


def _detections(
    theta: list[float],
    phi: list[float],
    width: list[float],
    keyframes: list[int],
    positions: list[tuple[float, float, float]] | None = None,
) -> Detections:
    count = len(theta)
    placed = positions is not None
    points = np.asarray(positions if placed else [(0.0, 0.0, 0.0)] * count, dtype=float)
    return Detections(
        row_index=np.arange(count),
        keyframe_id=np.asarray(keyframes, dtype=np.int64),
        theta=np.asarray(theta, dtype=float),
        phi=np.asarray(phi, dtype=float),
        angular_width=np.asarray(width, dtype=float),
        angular_height=np.asarray(width, dtype=float),
        source=np.array(["yolo"] * count),
        label=np.array(["chair"] * count),
        score=np.full(count, 0.5),
        depth=np.full(count, 5.0),
        position_eus=points,
        origin_eus=np.zeros((count, 3)),
        direction_eus=np.tile(np.asarray([0.0, 0.0, 1.0]), (count, 1)),
        placed=np.full(count, placed),
    )


def _ground_truth(
    positions: list[tuple[float, float, float]],
    classes: list[str],
    keyframes: list[str],
    uv: list[tuple[float, float]] | None = None,
) -> GroundTruth:
    count = len(classes)
    return GroundTruth(
        class_name=np.array(classes),
        position_eus=np.asarray(positions, dtype=float),
        accuracy_m=np.full(count, 5.0),
        source_keyframe_id=np.array(keyframes),
        erp_uv=np.asarray(uv if uv else [(0.5, 0.5)] * count, dtype=float),
        depth_m=np.full(count, 3.0),
        level=np.array([""] * count),
    )


def test_the_erp_ratio_maps_to_the_angles_the_parquet_stores() -> None:
    theta, phi = _uv_to_theta_phi(np.asarray([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]]))

    assert math.isclose(theta[0], 0.0, abs_tol=1e-12)
    assert math.isclose(phi[0], 0.0, abs_tol=1e-12)
    assert math.isclose(theta[1], -math.pi)
    assert math.isclose(phi[1], math.pi / 2)
    assert math.isclose(phi[2], -math.pi / 2)


def test_azimuths_wrap_instead_of_running_the_long_way_round() -> None:
    delta = _angular_delta(np.asarray([3.1]), np.asarray([-3.1]))

    assert abs(float(delta[0])) < 0.1


def test_an_annotation_inside_a_box_of_its_own_keyframe_counts_as_seen() -> None:
    detections = _detections([0.0], [0.0], [0.4], [7])
    ground_truth = _ground_truth([(0.0, 0.0, 0.0)], ["chaise"], ["7"], [(0.5, 0.5)])

    covered, nearest = seen_in_own_keyframe(detections, ground_truth)

    assert bool(covered[0])
    assert math.isclose(float(nearest[0]), 0.0, abs_tol=1e-9)


def test_a_box_in_another_keyframe_does_not_count() -> None:
    detections = _detections([0.0], [0.0], [0.4], [9])
    ground_truth = _ground_truth([(0.0, 0.0, 0.0)], ["chaise"], ["7"], [(0.5, 0.5)])

    covered, nearest = seen_in_own_keyframe(detections, ground_truth)

    assert not bool(covered[0])
    assert not math.isfinite(float(nearest[0]))


def test_an_annotation_outside_every_box_is_not_seen_but_reports_its_distance() -> None:
    detections = _detections([1.0], [0.0], [0.1], [7])
    ground_truth = _ground_truth([(0.0, 0.0, 0.0)], ["chaise"], ["7"], [(0.5, 0.5)])

    covered, nearest = seen_in_own_keyframe(detections, ground_truth)

    assert not bool(covered[0])
    assert math.isclose(float(nearest[0]), math.degrees(1.0), rel_tol=1e-6)


def test_attachment_takes_the_nearest_annotation_of_any_class() -> None:
    detections = _detections(
        [0.0, 0.0], [0.0, 0.0], [0.1, 0.1], [1, 1], [(0.0, 0, 0), (9.0, 0, 0)]
    )
    ground_truth = _ground_truth(
        [(0.2, 0, 0), (9.1, 0, 0)], ["chaise", "table"], ["1", "1"]
    )

    attachment = attach_to_ground_truth(detections, ground_truth)

    assert attachment.nearest.tolist() == [0, 1]
    assert list(attachment.nearest_class) == ["chaise", "table"]
    assert attachment.attached(0.5).tolist() == [True, True]
    assert attachment.attached(0.05).tolist() == [False, False]


def test_an_unplaced_detection_attaches_to_nothing() -> None:
    detections = _detections([0.0], [0.0], [0.1], [1])
    ground_truth = _ground_truth([(0.0, 0, 0)], ["chaise"], ["1"])

    attachment = attach_to_ground_truth(detections, ground_truth)

    assert attachment.nearest[0] == -1
    assert not attachment.attached(100.0)[0]


def test_overlapping_boxes_of_a_view_are_the_ones_below_the_margin() -> None:
    detections = _detections(
        [0.0, 0.05, 1.5], [0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [1, 1, 1]
    )

    left, right = _overlapping_pairs(detections, np.asarray([0, 1, 2]))

    assert list(zip(left.tolist(), right.tolist(), strict=True)) == [(0, 1)]


def test_mutual_information_is_one_for_a_bijection_and_zero_for_independence() -> None:
    left = np.array(["a", "a", "b", "b"])
    generator = np.random.default_rng(0)
    independent = generator.choice(["x", "y"], 4000)

    assert math.isclose(_mutual_information(left, np.array(["x", "x", "y", "y"])), 1.0)
    assert _mutual_information(generator.choice(["a", "b"], 4000), independent) < 0.05


def test_the_conditional_table_reports_shares_that_sum_to_one() -> None:
    condition = np.array(["p", "p", "p", "q"])
    outcome = np.array(["a", "a", "b", "b"])

    table = _conditional_table(condition, outcome)

    assert table["p"][0] == ("a", 2 / 3, 2)
    assert table["q"] == [("b", 1.0, 1)]


def test_a_perfectly_separating_cue_scores_one() -> None:
    high = np.array([3.0, 4.0])
    low = np.array([1.0, 2.0])

    assert _rank_auc(high, low, higher_is_same=True) == 1.0
    assert _rank_auc(low, high, higher_is_same=True) == 0.0


def test_percentiles_ignore_the_values_that_are_not_finite() -> None:
    stats = _percentiles(np.array([1.0, np.nan, 3.0, np.inf]))

    assert stats["n"] == 2.0
    assert stats["median"] == 2.0


def test_the_colour_scale_saturates_instead_of_indexing_out_of_range() -> None:
    assert _colour(-5.0) == _colour(0.0)
    assert _colour(99.0) == _colour(1.0)
    assert _colour(float("nan")) == "#808080"


def test_a_straight_capture_has_no_transverse_baseline() -> None:
    line = np.column_stack(
        [np.linspace(0.0, 10.0, 20), np.zeros(20), np.zeros(20)]
    )

    assert _horizontal_anisotropy(line) < 1e-6


def test_a_capture_that_circled_the_object_is_isotropic() -> None:
    angles = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.zeros(24), np.sin(angles)])

    assert _horizontal_anisotropy(ring) > 0.95


def test_anisotropy_needs_three_viewpoints_to_mean_anything() -> None:
    assert math.isnan(_horizontal_anisotropy(np.zeros((2, 3))))


def test_height_is_ignored_because_the_baseline_that_matters_is_on_the_ground() -> None:
    line = np.column_stack(
        [np.linspace(0.0, 10.0, 20), np.linspace(0.0, 5.0, 20), np.zeros(20)]
    )

    assert _horizontal_anisotropy(line) < 1e-6
