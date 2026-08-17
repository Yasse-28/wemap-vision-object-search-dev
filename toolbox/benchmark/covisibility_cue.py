"""Is the *absence* of co-visibility a cue that two fragments are one object?

Every cue measured so far is metric on a noisy quantity — depth distance, MetaCLIP
cosine, triangulation residual. This one is combinatorial: the detector put two boxes
in one panorama, so there are two objects; and the contrapositive, which is the new
part — if no view ever saw both fragments *while the views of one covered the
position of the other*, the occasion to see two objects existed and was not taken,
so the two fragments are one object seen twice.

"Covers" is where the depth maps finally earn their keep. They are poor estimators of
where a point is, and perfectly good oracles of whether a wall stands between a
camera and a point four metres away.

The population is fragment pairs, not group pairs: the fragments are what the
association actually produces, so a same-object pair is two clusters the current
method cut out of one hand-labelled group.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prepare.convention import theta_phi_to_uv

from toolbox.benchmark.association_sweep import DEFAULT_TIMEOUT_S
from toolbox.benchmark.matching_baskets import (
    Resolved,
    _group_members,
    load_resolved,
    matching_item,
)
from toolbox.benchmark.pair_cue_separability import _rank_auc
from toolbox.bricks import georef_source
from toolbox.bricks.georef_source import PoseSource
from toolbox.bricks.matching import GASP_PAIR_RADIUS_M, triangulate_items
from toolbox.bricks.vendored.depth_decode import (
    DepthMapNotFound,
    decode_uint16_meters,
    load_depth_map_from_path,
)
from toolbox.bricks.vendored.maths import quaternion

#: Beyond this a keyframe is not expected to have detected anything, so its silence
#: says nothing. Production's own depth trust ceiling.
MAX_VISIBLE_RANGE_M = 15.0
#: A cell counts as occluded when the depth map reads this much *closer* than it.
#: Loose on purpose: the question is "is there a wall in between", not "how far".
OCCLUSION_MARGIN_M = 1.0
#: Pairs of fragments further apart than this are never candidates for a merge, so
#: they are not the population the cue has to separate.
PAIR_RADIUS_M = GASP_PAIR_RADIUS_M
#: The geometry the conditional AUC is read at (see the depth-distance cue).
DEFAULT_CONDITIONAL_M = 2.0


@dataclass(frozen=True)
class Fragment:
    """A cluster the association produced inside one hand-labelled group."""

    group: str
    prompt: str
    centroid_eus: np.ndarray
    keyframes: frozenset[str]
    detections: int


@dataclass(frozen=True)
class PairCue:
    """One fragment pair, its truth, and the cue values measured on it."""

    same_object: bool
    distance_m: float
    shares_a_keyframe: bool
    #: Share of the *other* fragment's keyframes that geometrically cover this one.
    coverage: float
    #: The T1b indicator: no shared keyframe, and the coverage was there anyway.
    absence: float


class VisibilityOracle:
    """Answers "can keyframe f see the point p?" from f's own depth map.

    One decoded map is kept at a time: full-resolution ERP depth runs ~30 MB, and the
    query order here is by keyframe, so a single-slot cache is the whole win.
    """

    def __init__(self, map_path: Path, pose_source: PoseSource) -> None:
        self._depth_dir = map_path / "depths"
        self._poses = pose_source.poses
        self._names = pose_source.depth_filename_by_keyframe_id
        self._loaded: tuple[int, np.ndarray] | None = None
        self.missing: set[int] = set()

    def _depth_map(self, keyframe_id: int) -> np.ndarray | None:
        """The decoded depth map of one keyframe, in metres, NaN where invalid."""
        if self._loaded is not None and self._loaded[0] == keyframe_id:
            return self._loaded[1]
        name = self._names.get(keyframe_id)
        path = self._depth_dir / name if name else None
        if path is None or not path.is_file():
            self.missing.add(keyframe_id)
            return None
        try:
            raw, _ = load_depth_map_from_path(path)
        except (DepthMapNotFound, ValueError):
            self.missing.add(keyframe_id)
            return None
        metres = np.where(raw == 0, np.nan, decode_uint16_meters(raw))
        self._loaded = (keyframe_id, metres.astype(np.float64))
        return self._loaded[1]

    def covers(self, keyframe_id: int, point_eus: np.ndarray) -> bool | None:
        """Is `point_eus` in range and unoccluded from this keyframe?

        Returns None when the question cannot be answered — no pose, no depth map, or
        an invalid pixel. None is not False: a missing answer must not be counted as
        evidence, which is the whole reason the cue is scored on answered pairs only.
        """
        pose = self._poses.get(keyframe_id)
        if pose is None:
            return None
        origin = np.asarray(pose.position_eus, dtype=np.float64)
        offset = point_eus - origin
        distance = float(np.linalg.norm(offset))
        if distance > MAX_VISIBLE_RANGE_M:
            return False
        depth_map = self._depth_map(keyframe_id)
        if depth_map is None:
            return None
        theta, phi = _direction_to_theta_phi(pose.orientation_wxyz, offset / distance)
        height, width = depth_map.shape
        u, v = theta_phi_to_uv(
            np.asarray([theta]), np.asarray([phi]), width, height
        )
        column = int(np.mod(np.round(u[0]), width))
        row = int(np.clip(np.round(v[0]), 0, height - 1))
        depth = float(depth_map[row, column])
        if not math.isfinite(depth):
            return None
        return depth >= distance - OCCLUSION_MARGIN_M


def _direction_to_theta_phi(
    orientation_wxyz: Sequence[float], direction_eus: np.ndarray
) -> tuple[float, float]:
    """World direction back into the panorama's own `(theta, phi)`.

    The inverse of `matching._ray_direction_eus`: rotate by the conjugate, then read
    the ERP angles off the OpenGL ray convention (`-z` forward, `+y` up).
    """
    conjugate = (
        orientation_wxyz[0],
        -orientation_wxyz[1],
        -orientation_wxyz[2],
        -orientation_wxyz[3],
    )
    local = np.asarray(quaternion.rotate(conjugate, direction_eus), dtype=np.float64)
    theta = math.atan2(local[0], -local[2])
    phi = math.asin(float(np.clip(local[1], -1.0, 1.0)))
    return theta, phi


def fragments_of_group(
    name: str, rows: Sequence[Resolved], geo_transform: object, *, method: str
) -> list[Fragment]:
    """Partition one hand-labelled group into the clusters the association makes."""
    items = [matching_item(row) for row in rows]
    result = triangulate_items(items, geo_transform, partition_method=method)  # type: ignore[arg-type]
    if not result.get("available"):
        return []
    fragments: list[Fragment] = []
    for hypothesis in result["hypotheses"]:
        members = [rows[index] for index in hypothesis["items"]]
        if not members:
            continue
        fragments.append(
            Fragment(
                group=name,
                prompt=members[0].prompt,
                centroid_eus=np.asarray(
                    [row.candidate.eus_xyz for row in members], dtype=np.float64
                ).mean(axis=0),
                keyframes=frozenset(row.label.keyframe_id for row in members),
                detections=len(members),
            )
        )
    return fragments


def _coverage(
    oracle: VisibilityOracle, keyframes: frozenset[str], point_eus: np.ndarray
) -> float | None:
    """Share of those keyframes that cover the point; None if none could answer."""
    answers = [oracle.covers(int(keyframe), point_eus) for keyframe in keyframes]
    known = [answer for answer in answers if answer is not None]
    return float(np.mean(known)) if known else None


def pair_cues(
    fragments: Sequence[Fragment], oracle: VisibilityOracle
) -> list[PairCue]:
    """Measure the cue on every fragment pair of one prompt within reach.

    A pair whose coverage cannot be answered on either side is dropped rather than
    defaulted: the cue's applicability fraction is a number to report, not to fake.
    """
    cues: list[PairCue] = []
    for position, left in enumerate(fragments):
        for right in fragments[position + 1 :]:
            if left.prompt != right.prompt:
                continue
            distance = float(np.linalg.norm(left.centroid_eus - right.centroid_eus))
            if distance > PAIR_RADIUS_M:
                continue
            shared = bool(left.keyframes & right.keyframes)
            left_seen = _coverage(oracle, right.keyframes, left.centroid_eus)
            right_seen = _coverage(oracle, left.keyframes, right.centroid_eus)
            answered = [value for value in (left_seen, right_seen) if value is not None]
            if not answered:
                continue
            coverage = float(np.mean(answered))
            cues.append(
                PairCue(
                    same_object=left.group == right.group,
                    distance_m=distance,
                    shares_a_keyframe=shared,
                    coverage=coverage,
                    absence=0.0 if shared else coverage,
                )
            )
    return cues


def _auc_of(cues: Sequence[PairCue], attribute: str) -> float | None:
    """AUC of one cue field, "same object" being the positive class."""
    same = np.asarray(
        [getattr(cue, attribute) for cue in cues if cue.same_object], dtype=np.float64
    )
    other = np.asarray(
        [getattr(cue, attribute) for cue in cues if not cue.same_object],
        dtype=np.float64,
    )
    return _rank_auc(same, other, higher_is_same=True)


def bootstrap_auc(
    cues: Sequence[PairCue], attribute: str, *, draws: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile 90 % interval of one cue's AUC, resampling pairs with replacement.

    With a couple of dozen pairs the gate of T1b (0.70) sits well inside the sampling
    noise, so the point estimate alone would decide a four-to-six-day test on a coin
    flip. Resampling is stratified by class, since the class counts are the design.
    """
    same = [cue for cue in cues if cue.same_object]
    other = [cue for cue in cues if not cue.same_object]
    if not same or not other:
        return None
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        resampled = [
            same[index] for index in generator.integers(len(same), size=len(same))
        ] + [other[index] for index in generator.integers(len(other), size=len(other))]
        auc = _auc_of(resampled, attribute)
        if auc is not None:
            values.append(auc)
    if not values:
        return None
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def _print_report(cues: Sequence[PairCue], *, conditional_m: float) -> None:
    """Print the AUC table, its conditional column, and the applicability share."""
    same = sum(1 for cue in cues if cue.same_object)
    print(
        f"\n=== T1b — {len(cues)} fragment pairs within {PAIR_RADIUS_M:g} m "
        f"({same} same-object, {len(cues) - same} distinct) ==="
    )
    if not cues or same == 0 or same == len(cues):
        print("  one class is empty — no AUC to read")
        return
    near = [cue for cue in cues if cue.distance_m <= conditional_m]
    for label, subset in (("raw", cues), (f"within {conditional_m:g} m", near)):
        if not subset or not any(c.same_object for c in subset):
            print(f"  {label:16s} (empty)")
            continue
        print(
            f"  {label:16s} n={len(subset):4d}  "
            f"absence AUC={_fmt(_auc_of(subset, 'absence'))}  "
            f"coverage AUC={_fmt(_auc_of(subset, 'coverage'))}  "
            f"shared-keyframe AUC={_fmt(_auc_of(subset, 'shares_a_keyframe'))}"
        )
    interval = bootstrap_auc(cues, "absence")
    if interval is not None:
        print(
            f"  absence AUC 90% interval = [{interval[0]:.3f}, {interval[1]:.3f}] "
            f"— the 0.70 gate is "
            f"{'inside' if interval[0] <= 0.70 <= interval[1] else 'outside'} it"
        )
    applies = sum(1 for cue in cues if not cue.shares_a_keyframe and cue.coverage > 0)
    print(
        f"  the absence indicator fires on {applies}/{len(cues)} pairs "
        f"({applies / len(cues):.1%}) — a high AUC on a narrow slice buys nothing"
    )


def _fmt(value: float | None) -> str:
    """AUC as a fixed-width string, or a dash when a class was empty."""
    return "  n/a" if value is None else f"{value:.3f}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the T1b command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--method", default="gasp1v2")
    parser.add_argument("--conditional-m", type=float, default=DEFAULT_CONDITIONAL_M)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def collect_fragments(
    members: Mapping[str, list[Resolved]], geo_transform: object, method: str
) -> list[Fragment]:
    """Every group's fragments, and a line saying how much each group split."""
    fragments: list[Fragment] = []
    for name, rows in members.items():
        pieces = fragments_of_group(name, rows, geo_transform, method=method)
        print(f"  {name[:34]:34s} {len(rows):3d} detections -> {len(pieces)} fragments")
        fragments.extend(pieces)
    return fragments


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the absence cue on the hand-labelled groups of one map."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    resolved = load_resolved(args)
    members = _group_members(resolved)
    pose_source = georef_source.load_pose_source(map_path)
    print(f"\nfragmenting {len(members)} groups with {args.method}:")
    fragments = collect_fragments(members, pose_source.geo_transform, args.method)
    oracle = VisibilityOracle(map_path, pose_source)
    cues = pair_cues(fragments, oracle)
    _print_report(cues, conditional_m=args.conditional_m)
    if oracle.missing:
        print(f"  {len(oracle.missing)} keyframes had no readable depth map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
