from __future__ import annotations

import numpy as np
import pytest

from toolbox.bricks.merge_score import (
    DEFAULT_OBJECT_EXTENT_M,
    EXTENT_BOUNDS_M,
    cluster_extent_m,
    latent_cost,
    observed_extent_m,
    score_1v2,
)

_POINTS = np.asarray(
    [[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [3.0, 0.2, 0.0]], dtype=np.float64
)
_ORIGINS = np.asarray(
    [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float64
)


def test_uniform_extent_array_matches_the_scalar_bit_for_bit() -> None:
    scalar = latent_cost(_POINTS, _ORIGINS, object_extent_m=DEFAULT_OBJECT_EXTENT_M)
    per_point = latent_cost(
        _POINTS,
        _ORIGINS,
        object_extent_m=np.full(len(_POINTS), DEFAULT_OBJECT_EXTENT_M),
    )

    assert per_point == scalar


def test_score_without_per_side_extents_is_unchanged() -> None:
    left, right = _POINTS[:2], _POINTS[2:]
    baseline = score_1v2(left, right, _ORIGINS[:2], _ORIGINS[2:])
    explicit = score_1v2(
        left,
        right,
        _ORIGINS[:2],
        _ORIGINS[2:],
        left_extent_m=DEFAULT_OBJECT_EXTENT_M,
        right_extent_m=DEFAULT_OBJECT_EXTENT_M,
    )

    assert explicit == baseline


def test_extent_mismatch_raises_rather_than_broadcasting() -> None:
    with pytest.raises(ValueError, match="one value per point"):
        latent_cost(_POINTS, _ORIGINS, object_extent_m=np.asarray([1.0, 1.0]))


def test_a_larger_extent_makes_the_same_pair_easier_to_merge() -> None:
    left, right = _POINTS[:2], _POINTS[2:]
    small = score_1v2(
        left, right, _ORIGINS[:2], _ORIGINS[2:], left_extent_m=0.3, right_extent_m=0.3
    )
    large = score_1v2(
        left, right, _ORIGINS[:2], _ORIGINS[2:], left_extent_m=3.0, right_extent_m=3.0
    )

    assert large > small


def test_each_side_keeps_its_own_extent_in_the_union() -> None:
    # A wide cluster and a compact one: the compact side must not inherit the wide
    # side's tolerance just because they are being scored together.
    left, right = _POINTS[:2], _POINTS[2:]
    asymmetric = score_1v2(
        left, right, _ORIGINS[:2], _ORIGINS[2:], left_extent_m=3.0, right_extent_m=0.3
    )
    both_wide = score_1v2(
        left, right, _ORIGINS[:2], _ORIGINS[2:], left_extent_m=3.0, right_extent_m=3.0
    )

    assert asymmetric < both_wide


def test_observed_extent_scales_with_range_and_is_clipped() -> None:
    extents = observed_extent_m(
        np.asarray([6.0, 0.5, 200.0]),
        np.asarray([np.radians(30.0), np.radians(5.0), np.radians(40.0)]),
        np.asarray([0.0, 0.0, 0.0]),
    )

    assert extents[0] == pytest.approx(0.5 * 6.0 * np.radians(30.0))
    assert extents[1] == EXTENT_BOUNDS_M[0]
    assert extents[2] == EXTENT_BOUNDS_M[1]


def test_cluster_extent_is_the_median_not_the_mean() -> None:
    # One detection cut by the panorama edge under-reports the size; the median
    # ignores it where the mean would be dragged down.
    assert cluster_extent_m(np.asarray([0.2, 2.0, 2.2])) == pytest.approx(2.0)
    assert cluster_extent_m(np.asarray([])) == DEFAULT_OBJECT_EXTENT_M
