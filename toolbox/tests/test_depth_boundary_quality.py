"""Unit tests for the depth-boundary measurement."""

from __future__ import annotations

import numpy as np
import pytest

from toolbox.benchmark.depth_boundary_quality import (
    _box_slice,
    scene_rates,
    surface_masks,
)


def _flat(value: float = 5.0, shape: tuple[int, int] = (9, 9)) -> np.ndarray:
    return np.full(shape, value)


def test_a_flat_wall_has_no_border_and_nothing_flying() -> None:
    edge, flying = surface_masks(_flat())

    assert not edge.any()
    assert not flying.any()


def test_a_step_between_two_surfaces_is_a_border_but_is_not_flying() -> None:
    depth = _flat()
    depth[:, 5:] = 12.0

    edge, flying = surface_masks(depth)

    # Every interior pixel on either side of the step sees the jump.
    assert edge[1:-1, 4:6].all()
    # Both surfaces are real, so no pixel sits in the gap between them.
    assert not flying.any()


def test_a_pixel_stranded_between_two_surfaces_is_flying() -> None:
    depth = _flat()
    depth[:, 5:] = 12.0
    depth[4, 4] = 8.5

    _, flying = surface_masks(depth)

    assert flying[4, 4]
    assert flying.sum() == 1


def test_the_invalid_sentinel_is_not_a_surface() -> None:
    depth = _flat()
    depth[4, 4] = 0.0

    edge, flying = surface_masks(depth)

    assert not edge[4, 4]
    assert not flying[4, 4]


def test_a_patch_needs_a_neighbourhood_at_all() -> None:
    with pytest.raises(ValueError):
        surface_masks(np.zeros((2, 9)))


def test_a_box_at_the_seam_wraps_around_the_panorama() -> None:
    depth = np.arange(20, dtype=np.float64).reshape(4, 5)

    patch = _box_slice(depth, 1, 0, 0, 1)

    # Column 0's left neighbour is the last column of the same row, not a clamp.
    assert patch.tolist() == [[9.0, 5.0, 6.0]]


def test_a_box_at_the_pole_is_clamped_instead() -> None:
    depth = np.arange(20, dtype=np.float64).reshape(4, 5)

    patch = _box_slice(depth, 0, 2, 1, 0)

    assert patch.tolist() == [[2.0], [2.0], [7.0]]


def test_the_scene_pass_counts_every_band_once() -> None:
    depth = _flat(shape=(1200, 40))
    depth[:, 20:] = 12.0

    valid, edge, flying = scene_rates(depth)

    # Only the first and last row are left out, having no neighbourhood.
    assert valid == 1198 * 40
    assert edge == 1198 * 2
    assert flying == 0
