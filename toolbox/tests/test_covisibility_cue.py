from __future__ import annotations

import math

import numpy as np

from toolbox.benchmark.covisibility_cue import (
    Fragment,
    PairCue,
    _auc_of,
    _direction_to_theta_phi,
    bootstrap_auc,
    pair_cues,
)
from toolbox.bricks.matching import _ray_direction_eus


class _Pose:
    """Just enough of `KeyframePose` for the direction round trip."""

    def __init__(self, orientation_wxyz: tuple[float, float, float, float]) -> None:
        self.orientation_wxyz = orientation_wxyz
        self.position = (0.0, 0.0, 0.0)


class _Oracle:
    """A visibility oracle whose answers are dictated by the test."""

    def __init__(self, answers: dict[int, bool | None]) -> None:
        self._answers = answers

    def covers(self, keyframe_id: int, point_eus: np.ndarray) -> bool | None:
        return self._answers.get(keyframe_id)


def _fragment(group: str, keyframes: set[str], x: float) -> Fragment:
    return Fragment(
        group=group,
        prompt="machine",
        centroid_eus=np.asarray([x, 0.0, 0.0]),
        keyframes=frozenset(keyframes),
        detections=3,
    )


def test_the_erp_direction_round_trips_through_the_world_frame() -> None:
    generator = np.random.default_rng(3)
    orientation = generator.normal(size=4)
    orientation /= np.linalg.norm(orientation)
    theta, phi = 1.2, -0.4

    direction = np.asarray(
        _ray_direction_eus(_Pose(tuple(orientation)), theta, phi)  # type: ignore[arg-type]
    )
    recovered = _direction_to_theta_phi(tuple(orientation), direction)

    assert math.isclose(recovered[0], theta, abs_tol=1e-9)
    assert math.isclose(recovered[1], phi, abs_tol=1e-9)


def test_a_shared_keyframe_switches_the_absence_indicator_off() -> None:
    fragments = [_fragment("A", {"1"}, 0.0), _fragment("B", {"1", "2"}, 1.0)]

    cues = pair_cues(fragments, _Oracle({1: True, 2: True}))  # type: ignore[arg-type]

    assert len(cues) == 1
    assert cues[0].shares_a_keyframe
    assert cues[0].absence == 0.0
    assert cues[0].coverage == 1.0


def test_covered_but_never_co_seen_fires_the_indicator() -> None:
    fragments = [_fragment("A", {"1"}, 0.0), _fragment("A", {"2"}, 1.0)]

    cues = pair_cues(fragments, _Oracle({1: True, 2: True}))  # type: ignore[arg-type]

    assert cues[0].same_object
    assert cues[0].absence == 1.0


def test_a_pair_no_keyframe_can_answer_for_is_dropped_not_defaulted() -> None:
    fragments = [_fragment("A", {"1"}, 0.0), _fragment("B", {"2"}, 1.0)]

    assert pair_cues(fragments, _Oracle({1: None, 2: None})) == []  # type: ignore[arg-type]


def test_pairs_beyond_the_merge_radius_are_not_in_the_population() -> None:
    fragments = [_fragment("A", {"1"}, 0.0), _fragment("B", {"2"}, 50.0)]

    assert pair_cues(fragments, _Oracle({1: True, 2: True})) == []  # type: ignore[arg-type]


def test_a_perfect_cue_scores_one_and_its_interval_stays_high() -> None:
    cues = [
        PairCue(True, 1.0, False, 1.0, 1.0),
        PairCue(True, 1.0, False, 1.0, 1.0),
        PairCue(False, 1.0, True, 1.0, 0.0),
        PairCue(False, 1.0, True, 1.0, 0.0),
    ]

    assert _auc_of(cues, "absence") == 1.0
    interval = bootstrap_auc(cues, "absence", draws=200)
    assert interval is not None and interval[0] == 1.0
