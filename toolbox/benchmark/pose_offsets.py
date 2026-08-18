"""Can a per-keyframe translation explain part of the spread? (T8)

T1 measures a pose bias but cannot size it: a keyframe's residuals lean the same way
more often than a reshuffle would explain, and that is all. This asks the question
that decides T7 — how much of the dispersion disappears if each view is allowed to
move by one vector.

The estimate is the alternation of *Data-Association-Free Landmark-based SLAM*
(arXiv:2302.13264) with the associations held fixed, which is what the hand labels
give: alternate "each group sits at the mean of its corrected detections" and "each
keyframe moves by the mean correction its detections ask for", ridge-pulled towards
no correction at all.

**The number this prints is cross-validated by group, and only that number counts.**
With free offsets and fixed associations the model can explain any dispersion by
moving the views; a training-set spread would fall to near zero and mean nothing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from toolbox.benchmark.association_sweep import DEFAULT_TIMEOUT_S
from toolbox.benchmark.matching_baskets import (
    Resolved,
    _group_members,
    load_resolved,
    spread_of,
)

#: Ridge weight, in "virtual detections pulling the offset back to zero". At 4 a
#: keyframe holding four detections is trusted half way, which is the scale of the
#: data here — most keyframes hold one or two.
DEFAULT_RIDGE = 4.0
#: Alternation is a contraction here; ten passes are far past convergence.
ITERATIONS = 10


def estimate_offsets(
    members: Mapping[str, list[Resolved]], *, ridge: float = DEFAULT_RIDGE
) -> dict[str, np.ndarray]:
    """Fit one translation per keyframe by alternating groups and views.

    Args:
        members: Hand-labelled groups, each holding its resolved detections.
        ridge: Pull towards the zero offset, in units of detection count.

    Returns:
        Keyframe id mapped to its EUS translation, metres.
    """
    points = {
        name: np.asarray([row.candidate.eus_xyz for row in rows], dtype=np.float64)
        for name, rows in members.items()
    }
    keyframes = {
        name: [row.label.keyframe_id for row in rows] for name, rows in members.items()
    }
    offsets: dict[str, np.ndarray] = {
        keyframe: np.zeros(3)
        for names in keyframes.values()
        for keyframe in names
    }
    for _ in range(ITERATIONS):
        centroids = {
            name: np.mean(
                [points[name][i] + offsets[keyframes[name][i]]
                 for i in range(len(points[name]))],
                axis=0,
            )
            for name in points
        }
        wanted: dict[str, list[np.ndarray]] = {key: [] for key in offsets}
        for name, rows in points.items():
            for index, keyframe in enumerate(keyframes[name]):
                wanted[keyframe].append(centroids[name] - rows[index])
        offsets = {
            keyframe: (
                np.sum(values, axis=0) / (len(values) + ridge)
                if values
                else np.zeros(3)
            )
            for keyframe, values in wanted.items()
        }
    return offsets


def apply_offsets(
    rows: Sequence[Resolved], offsets: Mapping[str, np.ndarray]
) -> list[Resolved]:
    """Move every detection by its keyframe's offset, zero when it has none."""
    moved = []
    for row in rows:
        shift = offsets.get(row.label.keyframe_id)
        if shift is None:
            moved.append(row)
            continue
        position = np.asarray(row.candidate.eus_xyz, dtype=np.float64) + shift
        moved.append(
            replace(row, candidate=replace(row.candidate, eus_xyz=tuple(position)))
        )
    return moved


def _pooled_total(members: Mapping[str, list[Resolved]]) -> float:
    """Mean total spread over groups — the number T8 has to move."""
    return statistics.fmean(spread_of(rows).total for rows in members.values())


def transfer_coverage(members: Mapping[str, list[Resolved]]) -> float:
    """Share of detections whose keyframe is also seen in another group.

    An offset can only be carried to a held-out group through a keyframe that appears
    in both. Below this share, a cross-validated gain of zero says the keyframes did
    not overlap — not that the pose is right.
    """
    groups_of: dict[str, set[str]] = {}
    for name, rows in members.items():
        for row in rows:
            groups_of.setdefault(row.label.keyframe_id, set()).add(name)
    detections = [row for rows in members.values() for row in rows]
    shared = sum(1 for row in detections if len(groups_of[row.label.keyframe_id]) >= 2)
    return shared / len(detections) if detections else 0.0


def cross_validated_gain(
    members: Mapping[str, list[Resolved]], *, ridge: float, folds: int
) -> tuple[float, float, float]:
    """Spread of held-out groups, before and after offsets fitted without them.

    Splitting by **group** is the only split that tests anything: a random split of
    detections would leave every keyframe's offset fitted on the very group it is then
    scored against, and the answer would be a foregone conclusion.

    Returns:
        Spread before, spread after, and the training-set spread after — the last one
        is printed only to show how far the honest number sits below the flattering
        one.
    """
    names = sorted(members)
    before, after, trained = [], [], []
    for fold in range(folds):
        held_out = {name for index, name in enumerate(names) if index % folds == fold}
        if not held_out or len(held_out) == len(names):
            continue
        train = {name: members[name] for name in names if name not in held_out}
        test = {name: members[name] for name in names if name in held_out}
        offsets = estimate_offsets(train, ridge=ridge)
        moved = {name: apply_offsets(rows, offsets) for name, rows in test.items()}
        before.append(_pooled_total(test))
        after.append(_pooled_total(moved))
        trained.append(
            _pooled_total(
                {name: apply_offsets(rows, offsets) for name, rows in train.items()}
            )
        )
    return (
        statistics.fmean(before),
        statistics.fmean(after),
        statistics.fmean(trained),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the T8 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Fit per-keyframe offsets and report the held-out spread they buy."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    members = _group_members(load_resolved(args))
    print(f"\n=== T8 — per-keyframe offsets, {len(members)} groups ===")
    print(
        f"  {transfer_coverage(members):.1%} of detections come from a keyframe seen "
        "in another group — the only ones a fitted offset can reach"
    )
    print(
        f"  {'ridge':>6s} {'held-out before':>16s} {'held-out after':>15s} "
        f"{'gain':>7s} {'(training after)':>17s} {'median |offset|':>16s}"
    )
    for ridge in args.ridge:
        before, after, trained = cross_validated_gain(
            members, ridge=ridge, folds=min(args.folds, len(members))
        )
        offsets = estimate_offsets(members, ridge=ridge)
        norms = [float(np.linalg.norm(value)) for value in offsets.values()]
        print(
            f"  {ridge:6.1f} {before:16.3f} {after:15.3f} "
            f"{(after - before) / before:+6.1%} {trained:17.3f} "
            f"{statistics.median(norms):16.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
