"""Unit tests for the T2, T3, T4 and T8 refinement passes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from toolbox.benchmark.depth_policies import (
    _box_pixel_grid,
    _nearest_mode,
    _trimmed_mean,
    sample_box,
)
from toolbox.benchmark.matching_baskets import GroupLabel, Resolved
from toolbox.benchmark.pose_offsets import (
    apply_offsets,
    estimate_offsets,
    transfer_coverage,
)
from toolbox.benchmark.ray_refinement import _foot_on_ray
from toolbox.benchmark.sigma_calibration import fit_sigma, radial_residuals


@dataclass
class _Pose:
    position: tuple[float, float, float]


@dataclass
class _Candidate:
    id: int
    video_keyframe_id: int
    theta_center: float
    phi_center: float
    eus_xyz: tuple[float, float, float]
    geokeyframe_pose: _Pose
    angular_width: float = 0.1
    angular_height: float = 0.1


def _resolved(
    group: str,
    keyframe: int,
    point: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Resolved:
    return Resolved(
        GroupLabel(str(keyframe), 0.0, 0.0, group),
        "prompt",
        _Candidate(1, keyframe, 0.0, 0.0, point, _Pose(origin)),  # type: ignore[arg-type]
        1,
    )


def test_the_near_mode_wins_over_a_heavier_background() -> None:
    depths = np.concatenate([np.full(30, 3.0), np.full(70, 9.0)])

    assert _nearest_mode(depths) == 3.0


def test_a_fringe_too_thin_to_be_a_surface_is_not_chosen() -> None:
    depths = np.concatenate([np.full(5, 1.0), np.full(95, 9.0)])

    assert _nearest_mode(depths) == 9.0


def test_the_trimmed_mean_ignores_both_tails() -> None:
    depths = np.asarray([0.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 99.0])

    assert math.isclose(_trimmed_mean(depths), 5.0)


def test_a_box_crossing_the_seam_wraps_instead_of_clipping() -> None:
    _, columns = _box_pixel_grid(math.pi - 0.01, 0.0, 0.2, 0.2, 3600, 1800)

    assert columns.min() >= 0
    assert columns.max() < 3600
    assert len(set(columns.tolist())) > 1


def test_the_centre_policy_reads_the_centre_pixel() -> None:
    depth_map = np.full((180, 360), 9.0)
    depth_map[90, 180] = 3.0

    sample = sample_box(depth_map, 0.0, 0.0, 0.1, 0.1, "center")

    assert sample is not None and sample.depth_m == 3.0


def test_a_box_of_only_invalid_pixels_returns_nothing() -> None:
    depth_map = np.full((180, 360), np.nan)

    assert sample_box(depth_map, 0.0, 0.0, 0.1, 0.1, "median") is None


def test_the_foot_on_the_ray_never_lands_behind_the_camera() -> None:
    origin = np.zeros(3)
    direction = np.asarray([0.0, 0.0, 1.0])

    behind = _foot_on_ray(origin, direction, np.asarray([5.0, 0.0, -4.0]))
    assert np.allclose(behind, 0.0)
    assert np.allclose(
        _foot_on_ray(origin, direction, np.asarray([5.0, 0.0, 4.0])), [0, 0, 4]
    )


def test_a_single_shifted_keyframe_is_pulled_back_towards_its_group() -> None:
    members = {
        "g": [
            _resolved("g", 1, (2.0, 0.0, 5.0)),
            _resolved("g", 2, (0.0, 0.0, 5.0)),
            _resolved("g", 3, (0.0, 0.0, 5.0)),
            _resolved("g", 4, (0.0, 0.0, 5.0)),
        ]
    }

    offsets = estimate_offsets(members, ridge=0.0)

    assert offsets["1"][0] < 0.0
    assert offsets["2"][0] > 0.0


def test_a_heavier_ridge_shrinks_every_offset() -> None:
    members = {
        "g": [_resolved("g", index, (float(index), 0.0, 5.0)) for index in range(4)]
    }

    light = estimate_offsets(members, ridge=1.0)
    heavy = estimate_offsets(members, ridge=100.0)

    for keyframe in light:
        assert np.linalg.norm(heavy[keyframe]) < np.linalg.norm(light[keyframe])


def test_offsets_move_only_the_keyframes_they_name() -> None:
    rows = [_resolved("g", 1, (0.0, 0.0, 5.0)), _resolved("g", 2, (0.0, 0.0, 5.0))]

    moved = apply_offsets(rows, {"1": np.asarray([1.0, 0.0, 0.0])})

    assert moved[0].candidate.eus_xyz == (1.0, 0.0, 5.0)
    assert moved[1].candidate.eus_xyz == (0.0, 0.0, 5.0)


def test_transfer_coverage_counts_the_keyframes_two_groups_share() -> None:
    members = {
        "a": [_resolved("a", 1, (0.0, 0.0, 5.0)), _resolved("a", 2, (0.0, 0.0, 5.0))],
        "b": [_resolved("b", 1, (9.0, 0.0, 5.0)), _resolved("b", 3, (9.0, 0.0, 5.0))],
    }

    assert transfer_coverage(members) == 0.5


def test_sigma_is_fitted_in_sigma_units_not_in_absolute_residuals() -> None:
    generator = np.random.default_rng(11)
    ranges = generator.uniform(1.0, 20.0, size=20000)
    residuals = np.abs(generator.normal(scale=2.0, size=ranges.size))

    base, slope = fit_sigma(ranges, residuals)

    assert abs(base - 2.0) < 0.1
    assert abs(slope) < 0.02


def test_a_group_seen_from_one_side_is_dropped_from_the_sigma_fit() -> None:
    members = {
        "g": [
            _resolved("g", 1, (0.0, 0.0, 5.0), origin=(0.0, 0.0, 0.0)),
            _resolved("g", 2, (0.0, 0.0, 6.0), origin=(0.0, 0.0, 0.1)),
        ]
    }

    ranges, _ = radial_residuals(members)

    assert ranges.size == 0
