from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from toolbox.benchmark.matching_baskets import (
    GroupLabel,
    Resolved,
    _bias_ratio,
    _false_merges,
    _group_members,
    _labels_from_partition,
    _pooled_bias_ratio,
    build_baskets,
    keyframe_residuals,
    pair_counts,
    pair_f1,
    resolve_labels,
    spread_of,
    systematic_null,
)


@dataclass
class _Pose:
    position: tuple[float, float, float]


@dataclass
class _Candidate:
    """The `EnrichedCandidate` fields the harness reads, and nothing else."""

    id: int
    video_keyframe_id: int
    theta_center: float
    phi_center: float
    eus_xyz: tuple[float, float, float]
    geokeyframe_pose: _Pose
    angular_width: float = 0.1
    angular_height: float = 0.1


def _resolved(
    group: str | None,
    keyframe: int,
    point: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Resolved:
    label = GroupLabel(str(keyframe), 0.0, 0.0, group)
    candidate = _Candidate(1, keyframe, 0.0, 0.0, point, _Pose(origin))
    return Resolved(label, "prompt", candidate, 1)  # type: ignore[arg-type]


def test_the_angular_join_takes_the_nearest_box_and_counts_its_rivals() -> None:
    labels = [GroupLabel("7", 1.0, 0.5, "machine A")]
    candidates = {
        "machine": [
            _Candidate(1, 7, 1.0005, 0.5, (0.0, 0.0, 0.0), _Pose((0.0, 0.0, 0.0))),
            _Candidate(2, 7, 1.0001, 0.5, (0.0, 0.0, 0.0), _Pose((0.0, 0.0, 0.0))),
        ]
    }

    resolved, missed = resolve_labels(labels, candidates)  # type: ignore[arg-type]

    assert not missed
    assert resolved[0].candidate.id == 2
    assert resolved[0].rivals == 2


def test_a_label_no_box_matches_is_reported_rather_than_dropped() -> None:
    labels = [GroupLabel("7", 1.0, 0.5, "machine A")]
    candidates = {
        "machine": [_Candidate(1, 7, 2.0, 0.5, (0.0, 0.0, 0.0), _Pose((0.0, 0.0, 0.0)))]
    }

    resolved, missed = resolve_labels(labels, candidates)  # type: ignore[arg-type]

    assert not resolved
    assert missed == labels


def test_spread_along_the_viewing_ray_is_entirely_radial() -> None:
    rows = [
        _resolved("g", 1, (0.0, 0.0, 4.0)),
        _resolved("g", 2, (0.0, 0.0, 6.0)),
    ]

    spread = spread_of(rows)

    assert spread.radial == 1.0
    assert spread.tangential == 0.0


def test_spread_across_the_viewing_ray_is_entirely_tangential() -> None:
    rows = [
        _resolved("g", 1, (-1.0, 0.0, 5.0), origin=(-1.0, 0.0, 0.0)),
        _resolved("g", 2, (1.0, 0.0, 5.0), origin=(1.0, 0.0, 0.0)),
    ]

    spread = spread_of(rows)

    assert spread.tangential == 1.0
    assert spread.radial == 0.0


def test_the_two_components_compose_into_the_total() -> None:
    rows = [
        _resolved("g", 1, (0.0, 0.0, 4.0), origin=(0.0, 0.0, 0.0)),
        _resolved("g", 2, (1.0, 0.5, 6.0), origin=(2.0, 0.0, 0.0)),
        _resolved("g", 3, (-0.5, 0.2, 5.5), origin=(-2.0, 1.0, 0.0)),
    ]

    spread = spread_of(rows)

    assert math.isclose(
        spread.total**2, spread.radial**2 + spread.tangential**2, rel_tol=1e-9
    )


def test_a_keyframe_pushed_one_way_reads_as_a_bias_not_as_noise() -> None:
    biased = {
        "g": [
            _resolved("g", 1, (1.0, 0.0, 5.0)),
            _resolved("g", 1, (1.0, 0.0, 5.0)),
            _resolved("g", 2, (-1.0, 0.0, 5.0)),
            _resolved("g", 3, (-1.0, 0.0, 5.0)),
        ]
    }

    residuals, groups = keyframe_residuals(biased)["1"]

    assert _bias_ratio(residuals) == 1.0
    assert groups == {"g"}


def test_opposite_residuals_of_one_keyframe_read_as_noise() -> None:
    noisy = {
        "g": [
            _resolved("g", 1, (1.0, 0.0, 5.0)),
            _resolved("g", 1, (-1.0, 0.0, 5.0)),
        ]
    }

    residuals, _ = keyframe_residuals(noisy)["1"]

    assert _bias_ratio(residuals) == 0.0


def test_isotropic_noise_lands_on_its_own_shuffled_null() -> None:
    generator = np.random.default_rng(7)
    blocks = [generator.normal(size=(4, 3)) for _ in range(30)]

    observed = _pooled_bias_ratio(blocks)
    mean, deviation = systematic_null(blocks, draws=199, seed=1)

    assert abs(observed - mean) < 3 * deviation


def test_a_per_keyframe_offset_beats_its_shuffled_null() -> None:
    generator = np.random.default_rng(7)
    blocks = [
        generator.normal(scale=0.3, size=(4, 3)) + generator.normal(size=(1, 3))
        for _ in range(30)
    ]

    observed = _pooled_bias_ratio(blocks)
    mean, deviation = systematic_null(blocks, draws=199, seed=1)

    assert observed > mean + 5 * deviation


def test_an_unclaimed_detection_counts_as_its_own_fragment() -> None:
    result = {"hypotheses": [{"items": [0, 1]}], "unassigned_items": [2]}

    assert _labels_from_partition(result, 3) == [0, 0, 1]


def test_a_false_merge_needs_two_truths_in_one_cluster() -> None:
    assert _false_merges([0, 0, 0], [0, 0, 1]) == 2
    assert _false_merges([0, 0, 1], [0, 0, 1]) == 0


def test_a_group_under_the_minimum_builds_no_basket() -> None:
    rows = [_resolved("small", index, (0.0, 0.0, 5.0)) for index in range(3)]

    assert _group_members(rows) == {}
    assert build_baskets(rows) == []


def test_negatives_of_a_group_keyframe_become_a_mixed_basket() -> None:
    rows = [_resolved("g", index, (0.0, 0.0, 5.0)) for index in range(4)]
    rows.append(_resolved(None, 0, (3.0, 0.0, 5.0)))

    kinds = {basket.kind: basket for basket in build_baskets(rows)}

    assert set(kinds) == {"solo", "mixed"}
    assert kinds["mixed"].truth == [0, 0, 0, 0, -1]
    assert kinds["mixed"].expected_clusters == 2


def test_a_fixed_pair_list_ignores_where_the_detections_ended_up() -> None:
    """A moved detection must not change which groups count as confusable."""
    left = [_resolved("a", index, (0.0, 0.0, 5.0)) for index in range(4)]
    right = [_resolved("b", index + 10, (0.0, 0.0, 20.0)) for index in range(4)]

    followed = [b.name for b in build_baskets(left + right) if b.kind == "pair"]
    fixed = [
        b.name for b in build_baskets(left + right, [("a", "b")]) if b.kind == "pair"
    ]

    assert followed == []  # 15 m apart, past the confusion radius
    assert fixed == ["a + b"]  # kept anyway, because the baseline said so


def test_a_pair_named_for_a_group_that_vanished_is_skipped() -> None:
    rows = [_resolved("a", index, (0.0, 0.0, 5.0)) for index in range(4)]

    assert [b for b in build_baskets(rows, [("a", "gone")]) if b.kind == "pair"] == []


def test_pair_counts_cannot_be_lucky_on_the_cluster_count() -> None:
    """Two clusters over two objects, split in the wrong place, must not look right."""
    truth = [0, 0, 0, 1, 1, 1]
    wrong_place = [0, 0, 0, 0, 0, 1]  # right count, wrong partition

    true_positive, false_positive, false_negative = pair_counts(wrong_place, truth)

    assert (true_positive, false_positive, false_negative) == (4, 6, 2)
    assert pair_f1(*pair_counts(truth, truth)) == 1.0
    assert pair_f1(true_positive, false_positive, false_negative) < 0.6


def test_two_negatives_together_are_neither_a_hit_nor_a_miss() -> None:
    truth = [0, 0, -1, -1]

    assert pair_counts([0, 0, 1, 1], truth) == (1, 0, 0)
    assert pair_counts([0, 0, 0, 0], truth) == (1, 4, 0)
