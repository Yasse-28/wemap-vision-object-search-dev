"""Replay a hand-labelled basket offline, and decompose the spread that decides it.

`extent_baskets.py` builds its baskets by attaching detections to the nearest
annotation. That conditions on the result: a detection whose depth point landed
three metres away is exactly the one that failed to reach its annotation, so it is
dropped from the very basket meant to measure it. The hand labels of
`detection_group_label` are the parade — a human said "these boxes are that object"
by looking at the panorama, not at the 3D point.

What this module produces, per basket:

- the seven method rows of the toolbox's matching tab, so a change is read against
  the same line-up the UI shows;
- the spread of each group, split into the three components of T1 — **radial**
  (the depth is wrong), **tangential** (the depth is right but the bbox centre
  slides), **systematic** (the keyframe pose is off, so all its detections shift by
  one vector);
- the co-visibility matrix `(keyframe, group) -> count`, which is the raw material
  of T1b and T7 and costs nothing to emit here.

Baskets come in three families: **solo** (one group, one cluster expected), **pair**
(two nearby groups of one prompt, two clusters expected) and **mixed** (one group
plus the `group_name IS NULL` negatives of its keyframes, which must join nothing).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toolbox.benchmark.association_sweep import (
    DEFAULT_TIMEOUT_S,
    fetch_prompt_candidates,
)
from toolbox.bricks import georef_source
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.matching import (
    ANGLE_MATCH_TOL_RAD,
    MatchingItem,
    triangulate_items,
)
from toolbox.bricks.triangulate import DEFAULT_INLIER_THRESHOLD_DEG
from toolbox.bricks.vendored.geo_transform import GeoTransform
from toolbox.logging import logger

#: Fewer detections than this and a group says nothing about fragmentation.
MIN_BASKET_DETECTIONS = 4
#: Two groups further apart than this are not the confusable pair we want. Set to the
#: radius at which the association even considers a merge (`GASP_PAIR_RADIUS_M`), not
#: `extent_baskets`' 4 m: at 4 m vinci yields a *single* pair basket, the hardest one
#: that exists, and one basket cannot carry the half of every verdict that says what a
#: change costs in false merges.
PAIR_MAX_DISTANCE_M = 6.0
#: Under this parallax the radial and tangential components are not separable, so a
#: basket below it is reported but excluded from the T1 decomposition (plan T1, trap).
MIN_PARALLAX_DEG = 5.0
#: Range bands the per-band tables are cut on, metres.
RANGE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, math.inf),
)

#: The matching tab's comparison line-up (`MatchingPanel.tsx`, `COMPARED`), in order.
COMPARED: tuple[tuple[str, str, bool, float], ...] = (
    ("sequential RANSAC", "sequential", False, 0.0),
    ("J-linkage", "jlinkage", False, 0.0),
    ("T-linkage", "tlinkage", False, 0.0),
    ("GASP average", "gasp", False, 0.0),
    ("GASP + cannot-link", "gasp", True, 0.0),
    ("GASP 1v2 score", "gasp1v2", False, 0.0),
    ("GASP 1v2 + co-visibility", "gasp1v2", False, 1.0),
)


@dataclass(frozen=True)
class GroupLabel:
    """One row of `detection_group_label`. A null `group_name` is a negative."""

    keyframe_id: str
    theta_center: float
    phi_center: float
    group_name: str | None


@dataclass(frozen=True)
class Resolved:
    """A hand label matched to the enriched candidate it names."""

    label: GroupLabel
    prompt: str
    candidate: EnrichedCandidate
    #: How many candidates of this keyframe fell inside the angular tolerance. Above
    #: one the join picked the nearest, and the row is a duplicate proposal — the
    #: intra-view duplicate rate T1b needs before reading its own AUC.
    rivals: int


@dataclass(frozen=True)
class Basket:
    """Detections to partition, and the group each one truly belongs to."""

    name: str
    kind: str
    prompt: str
    resolved: list[Resolved]
    #: Group index per detection; -1 for an explicit negative, which belongs nowhere.
    truth: list[int]
    expected_clusters: int


@dataclass(frozen=True)
class Spread:
    """One group's dispersion, split into the three mechanisms of T1."""

    detections: int
    total: float
    radial: float
    tangential: float
    range_median: float
    max_parallax_deg: float

    @property
    def tangential_over_radial(self) -> float:
        """The ratio T1 gates on. Infinite when the radial part vanishes."""
        return self.tangential / self.radial if self.radial > 0 else math.inf


@dataclass(frozen=True)
class MethodOutcome:
    """One basket partitioned by one method."""

    method: str
    clusters: int
    false_merges: int
    error: str | None = None


@dataclass
class Covisibility:
    """`(keyframe, group) -> detection count`, plus the groups each keyframe sees."""

    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, keyframe_id: str, group_name: str) -> None:
        """Record one detection of `group_name` produced by `keyframe_id`."""
        key = (keyframe_id, group_name)
        self.counts[key] = self.counts.get(key, 0) + 1

    def keyframes_of(self, group_name: str) -> set[str]:
        """Every keyframe that produced at least one detection of this group."""
        return {kf for (kf, group) in self.counts if group == group_name}

    def groups_of(self, keyframe_id: str) -> set[str]:
        """Every group this keyframe produced a detection of."""
        return {group for (kf, group) in self.counts if kf == keyframe_id}


def load_group_labels(db_path: Path) -> list[GroupLabel]:
    """Read the hand-made basket labels out of the map's annotation database.

    Args:
        db_path: Path to `{map}/object-search-annotations.db`.

    Returns:
        Every row, negatives (`group_name IS NULL`) included.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT keyframe_id, theta_center, phi_center, group_name "
            "FROM detection_group_label"
        ).fetchall()
    finally:
        connection.close()
    return [
        GroupLabel(str(keyframe), float(theta), float(phi), group)
        for keyframe, theta, phi, group in rows
    ]


def _index_by_keyframe(
    candidates: Mapping[str, Sequence[EnrichedCandidate]],
) -> dict[str, list[tuple[str, EnrichedCandidate]]]:
    """Group every prompt's candidates by the keyframe that produced them."""
    index: dict[str, list[tuple[str, EnrichedCandidate]]] = defaultdict(list)
    for prompt, prompt_candidates in candidates.items():
        for candidate in prompt_candidates:
            index[str(candidate.video_keyframe_id)].append((prompt, candidate))
    return index


def resolve_labels(
    labels: Sequence[GroupLabel],
    candidates: Mapping[str, Sequence[EnrichedCandidate]],
    *,
    tolerance_rad: float = ANGLE_MATCH_TOL_RAD,
) -> tuple[list[Resolved], list[GroupLabel]]:
    """Match each label to the candidate it names, by keyframe and viewing angle.

    The plan expected `row_index` to carry the identity, but the toolbox writes it
    null — pgvector has no parquet row index to offer, which is the same reason
    `matching.resolve_items` joins on angles. So this is the angular join, run
    against the cached candidates instead of postgres.

    Args:
        labels: Rows of `detection_group_label`.
        candidates: Enriched candidates per prompt, from the offline cache.
        tolerance_rad: Half-width of the angular match, on both theta and phi.

    Returns:
        The resolved labels, and the labels no candidate matched.
    """
    index = _index_by_keyframe(candidates)
    resolved: list[Resolved] = []
    missed: list[GroupLabel] = []
    for label in labels:
        matches = [
            (max(abs(c.theta_center - label.theta_center),
                 abs(c.phi_center - label.phi_center)), prompt, c)
            for prompt, c in index.get(label.keyframe_id, ())
        ]
        near = [item for item in matches if item[0] < tolerance_rad]
        if not near:
            missed.append(label)
            continue
        _, prompt, candidate = min(near, key=lambda item: item[0])
        resolved.append(Resolved(label, prompt, candidate, len(near)))
    return resolved, missed


def matching_item(resolved: Resolved) -> MatchingItem:
    """Build the matching item straight from the candidate, without pgvector.

    Same construction as `extent_baskets._matching_items`: offline the candidate
    already carries every field the resolver would fetch.
    """
    candidate = resolved.candidate
    origin = np.asarray(candidate.geokeyframe_pose.position, dtype=np.float64)
    point = np.asarray(candidate.eus_xyz, dtype=np.float64)
    offset = point - origin
    norm = float(np.linalg.norm(offset))
    direction = (0.0, 0.0, -1.0) if norm == 0.0 else tuple(offset / norm)
    return MatchingItem(
        keyframe_id=str(candidate.video_keyframe_id),
        theta_center=float(candidate.theta_center),
        phi_center=float(candidate.phi_center),
        candidate_id=int(candidate.id),
        thumbnail=None,
        ray_origin_eus=(float(origin[0]), float(origin[1]), float(origin[2])),
        ray_direction_eus=(
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        ),
        stored_position_eus=(float(point[0]), float(point[1]), float(point[2])),
        embedding=None,
        angular_width=float(candidate.angular_width),
        angular_height=float(candidate.angular_height),
    )


def _group_members(resolved: Sequence[Resolved]) -> dict[str, list[Resolved]]:
    """Named groups with enough detections to say anything, in label order."""
    members: dict[str, list[Resolved]] = defaultdict(list)
    for item in resolved:
        if item.label.group_name is not None:
            members[item.label.group_name].append(item)
    return {
        name: rows
        for name, rows in members.items()
        if len(rows) >= MIN_BASKET_DETECTIONS
    }


def _negatives_of(
    resolved: Sequence[Resolved], keyframes: Iterable[str]
) -> list[Resolved]:
    """Explicit non-objects labelled on the given keyframes."""
    wanted = set(keyframes)
    return [
        item
        for item in resolved
        if item.label.group_name is None and item.label.keyframe_id in wanted
    ]


def _group_centroid(rows: Sequence[Resolved]) -> np.ndarray:
    """EUS centroid of a group's depth points."""
    return np.asarray(
        [row.candidate.eus_xyz for row in rows], dtype=np.float64
    ).mean(axis=0)


def build_baskets(
    resolved: Sequence[Resolved],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> list[Basket]:
    """Assemble the solo, pair and mixed baskets from the resolved labels.

    `pairs` fixes which groups are confusable. Callers that move the detections —
    every depth policy, every pose offset — must pass the baseline's pairs: letting
    the population follow the positions would compare two policies on two different
    sets of baskets.
    """
    members = _group_members(resolved)
    baskets: list[Basket] = [
        Basket(name, "solo", rows[0].prompt, list(rows), [0] * len(rows), 1)
        for name, rows in members.items()
    ]
    baskets.extend(_pair_baskets(members, pairs))
    baskets.extend(_mixed_baskets(members, resolved))
    return baskets


def confusable_pairs(members: Mapping[str, list[Resolved]]) -> list[tuple[str, str]]:
    """Group pairs of one prompt whose centroids sit within the confusion radius."""
    pairs: list[tuple[str, str]] = []
    names = list(members)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            if members[left][0].prompt != members[right][0].prompt:
                continue
            distance = float(
                np.linalg.norm(
                    _group_centroid(members[left]) - _group_centroid(members[right])
                )
            )
            if distance <= PAIR_MAX_DISTANCE_M:
                pairs.append((left, right))
    return pairs


def _pair_baskets(
    members: Mapping[str, list[Resolved]],
    pairs: Sequence[tuple[str, str]] | None,
) -> list[Basket]:
    """Two groups of one prompt, close enough to be confusable, as one basket."""
    wanted = confusable_pairs(members) if pairs is None else pairs
    baskets: list[Basket] = []
    for left, right in wanted:
        if left not in members or right not in members:
            continue
        rows = members[left] + members[right]
        truth = [0] * len(members[left]) + [1] * len(members[right])
        baskets.append(
            Basket(f"{left} + {right}", "pair", rows[0].prompt, rows, truth, 2)
        )
    return baskets


def _mixed_baskets(
    members: Mapping[str, list[Resolved]], resolved: Sequence[Resolved]
) -> list[Basket]:
    """One group plus the labelled negatives of its own keyframes."""
    baskets: list[Basket] = []
    for name, rows in members.items():
        negatives = _negatives_of(resolved, {row.label.keyframe_id for row in rows})
        if not negatives:
            continue
        members_and_negatives = list(rows) + negatives
        truth = [0] * len(rows) + [-1] * len(negatives)
        baskets.append(
            Basket(
                f"{name} + negatives",
                "mixed",
                rows[0].prompt,
                members_and_negatives,
                truth,
                1 + len(negatives),
            )
        )
    return baskets


def spread_of(rows: Sequence[Resolved]) -> Spread:
    """Decompose one group's dispersion around its own centroid.

    The residual of each detection is projected on its **viewing ray**: the part
    along the ray is a wrong depth, the part across it is a bbox centre that moved.
    Both are reported as RMS, so they compose — `radial² + tangential² = total²`.
    """
    points = np.asarray([row.candidate.eus_xyz for row in rows], dtype=np.float64)
    origins = np.asarray(
        [row.candidate.geokeyframe_pose.position for row in rows], dtype=np.float64
    )
    offsets = points - origins
    ranges = np.linalg.norm(offsets, axis=1)
    directions = offsets / np.where(ranges[:, None] == 0.0, 1.0, ranges[:, None])
    residuals = points - points.mean(axis=0)
    along = np.einsum("ij,ij->i", residuals, directions)
    across = np.linalg.norm(residuals - along[:, None] * directions, axis=1)
    return Spread(
        detections=len(rows),
        total=float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
        radial=float(np.sqrt(np.mean(along**2))),
        tangential=float(np.sqrt(np.mean(across**2))),
        range_median=float(np.median(ranges)),
        max_parallax_deg=_max_parallax_deg(directions),
    )


def _max_parallax_deg(directions: np.ndarray) -> float:
    """Widest angle between two viewing directions of a group, in degrees."""
    if len(directions) < 2:
        return 0.0
    cosines = np.clip(directions @ directions.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines.min())))


def covisibility_of(members: Mapping[str, list[Resolved]]) -> Covisibility:
    """Count, per keyframe and group, how many detections that view produced."""
    matrix = Covisibility()
    for name, rows in members.items():
        for row in rows:
            matrix.add(row.label.keyframe_id, name)
    return matrix


def _labels_from_partition(result: Mapping[str, Any], size: int) -> list[int]:
    """Turn a `triangulate_items` response into one cluster label per basket row.

    An item no hypothesis claimed is its own cluster, which is how the UI counts it:
    an unexplained detection is a fragment, not a free pass.
    """
    labels = [-1] * size
    for cluster, hypothesis in enumerate(result["hypotheses"]):
        for index in hypothesis["items"]:
            labels[index] = cluster
    next_label = len(result["hypotheses"])
    for index, label in enumerate(labels):
        if label < 0:
            labels[index] = next_label
            next_label += 1
    return labels


def pair_counts(
    labels: Sequence[int], truth: Sequence[int]
) -> tuple[int, int, int]:
    """True positives, false positives and false negatives over detection pairs.

    The cluster **count** can be right while the partition is wrong — two clusters of
    39 and 3 over two objects of 33 and 9 scores a perfect count and merges 270 pairs
    that belong apart. Pair counts cannot be lucky that way, which is why the verdict
    on a granularity change is read here and not on the count.

    Negatives (`truth == -1`) belong to no object, so a pair of two of them is neither
    a hit nor a miss — only their pairs with a real group are counted, as false
    positives when they land in it.
    """
    true_positive = false_positive = false_negative = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            same_truth = truth[left] == truth[right] and truth[left] >= 0
            both_negative = truth[left] < 0 and truth[right] < 0
            if both_negative:
                continue
            if labels[left] == labels[right]:
                if same_truth:
                    true_positive += 1
                else:
                    false_positive += 1
            elif same_truth:
                false_negative += 1
    return true_positive, false_positive, false_negative


def pair_f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    """Harmonic mean of pair precision and pair recall, 0 when both are empty."""
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    if not precision_denominator or not recall_denominator:
        return 0.0
    precision = true_positive / precision_denominator
    recall = true_positive / recall_denominator
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _false_merges(labels: Sequence[int], truth: Sequence[int]) -> int:
    """Pairs put in one cluster that the hand labels say are different objects."""
    return sum(
        1
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
        if labels[left] == labels[right] and truth[left] != truth[right]
    )


def score_basket(
    basket: Basket,
    geo_transform: GeoTransform,
    *,
    max_depth_m: float | None = None,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
) -> list[MethodOutcome]:
    """Partition one basket with each of the seven compared methods."""
    items = [matching_item(row) for row in basket.resolved]
    outcomes: list[MethodOutcome] = []
    for label, method, cannot_link, covis in COMPARED:
        try:
            result = triangulate_items(
                items,
                geo_transform,
                partition_method=method,
                max_depth_m=max_depth_m,
                inlier_threshold_deg=inlier_threshold_deg,
                use_cannot_link=cannot_link,
                covis_weight=covis,
            )
        except Exception as error:  # noqa: BLE001 - one method failing is a row, not a stop
            outcomes.append(MethodOutcome(label, 0, 0, str(error)))
            continue
        if not result.get("available"):
            outcomes.append(
                MethodOutcome(label, 0, 0, str(result.get("reason", "unavailable")))
            )
            continue
        labels = _labels_from_partition(result, len(items))
        outcomes.append(
            MethodOutcome(
                label,
                len(set(labels)),
                _false_merges(labels, basket.truth),
            )
        )
    return outcomes


def _band_of(range_m: float) -> str:
    """Name of the range band a distance falls in."""
    for low, high in RANGE_BANDS:
        if low <= range_m < high:
            return f"{low:g}-{high:g}m" if math.isfinite(high) else f"{low:g}m+"
    return "?"


def _spread_table(members: Mapping[str, list[Resolved]]) -> list[tuple[str, Spread]]:
    """Each group's spread, in decreasing order of dispersion."""
    rows = [(name, spread_of(group)) for name, group in members.items()]
    return sorted(rows, key=lambda row: row[1].total, reverse=True)


def _print_spreads(members: Mapping[str, list[Resolved]]) -> None:
    """Print the per-group T1 decomposition and its pooled reading."""
    print("\n=== T1 — spread decomposition, per group ===")
    header = (
        f"{'group':28s} {'n':>3s} {'range':>7s} {'par°':>6s} "
        f"{'total':>7s} {'radial':>7s} {'tang':>7s} {'t/r':>6s}"
    )
    print(header)
    usable: list[Spread] = []
    for name, spread in _spread_table(members):
        flag = "" if spread.max_parallax_deg >= MIN_PARALLAX_DEG else "  (low parallax)"
        print(
            f"{name[:28]:28s} {spread.detections:3d} {spread.range_median:7.2f} "
            f"{spread.max_parallax_deg:6.1f} {spread.total:7.3f} {spread.radial:7.3f} "
            f"{spread.tangential:7.3f} {spread.tangential_over_radial:6.2f}{flag}"
        )
        if spread.max_parallax_deg >= MIN_PARALLAX_DEG:
            usable.append(spread)
    _print_pooled_spread(usable)


def _capped(spread: Spread) -> float:
    """The t/r ratio, clipped so one radial-free group cannot own the mean."""
    return min(spread.tangential_over_radial, 99.0)


def _print_pooled_spread(spreads: Sequence[Spread]) -> None:
    """Pool the decomposition over groups, and cut it by range band."""
    if not spreads:
        print("  (no group above the parallax gate)")
        return
    print(
        f"\n  pooled over {len(spreads)} groups above {MIN_PARALLAX_DEG:g}°: "
        f"radial={statistics.fmean(s.radial for s in spreads):.3f} m  "
        f"tangential={statistics.fmean(s.tangential for s in spreads):.3f} m  "
        f"t/r={statistics.fmean(_capped(s) for s in spreads):.2f}"
    )
    by_band: dict[str, list[Spread]] = defaultdict(list)
    for spread in spreads:
        by_band[_band_of(spread.range_median)].append(spread)
    for band, rows in sorted(by_band.items()):
        print(
            f"    {band:8s} n={len(rows):2d}  "
            f"radial={statistics.fmean(s.radial for s in rows):.3f}  "
            f"tangential={statistics.fmean(s.tangential for s in rows):.3f}  "
            f"t/r={statistics.fmean(_capped(s) for s in rows):.2f}"
        )


def keyframe_residuals(
    members: Mapping[str, list[Resolved]],
) -> dict[str, tuple[np.ndarray, set[str]]]:
    """Per keyframe, its residual vectors and the groups they came from."""
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    groups_seen: dict[str, set[str]] = defaultdict(set)
    for name, rows in members.items():
        centroid = _group_centroid(rows)
        for row in rows:
            keyframe = row.label.keyframe_id
            vectors[keyframe].append(
                np.asarray(row.candidate.eus_xyz, dtype=np.float64) - centroid
            )
            groups_seen[keyframe].add(name)
    return {
        keyframe: (np.asarray(rows), groups_seen[keyframe])
        for keyframe, rows in vectors.items()
    }


def _bias_ratio(residuals: np.ndarray) -> float:
    """Norm of the mean residual over the mean norm — 1 for a bias, ~0 for noise."""
    norms = np.linalg.norm(residuals, axis=1)
    mean_norm = float(np.mean(norms))
    if not mean_norm:
        return 0.0
    return float(np.linalg.norm(residuals.mean(axis=0))) / mean_norm


def _pooled_bias_ratio(blocks: Sequence[np.ndarray]) -> float:
    """Mean bias ratio over keyframes, weighted by how many residuals each holds."""
    weights = [len(block) for block in blocks]
    return float(
        np.average([_bias_ratio(block) for block in blocks], weights=weights)
    )


def systematic_null(
    blocks: Sequence[np.ndarray], *, draws: int = 999, seed: int = 0
) -> tuple[float, float]:
    """Null mean and standard deviation of the pooled bias ratio, by permutation.

    The ratio of a keyframe holding `n` residuals is about `1/sqrt(n)` under isotropic
    noise, never zero, so the measured number means nothing on its own. Reshuffling
    the residual vectors between keyframes destroys any per-view bias while keeping
    every block size and the residual cloud itself — the honest reference.
    """
    pool = np.concatenate(blocks)
    sizes = [len(block) for block in blocks]
    generator = np.random.default_rng(seed)
    ratios = []
    for _ in range(draws):
        shuffled = pool[generator.permutation(len(pool))]
        offset = 0
        parts = []
        for size in sizes:
            parts.append(shuffled[offset : offset + size])
            offset += size
        ratios.append(_pooled_bias_ratio(parts))
    return float(np.mean(ratios)), float(np.std(ratios))


def _print_systematic(members: Mapping[str, list[Resolved]], *, seed: int = 0) -> None:
    """Print the third component: is a keyframe's residual a bias or noise?"""
    print("\n=== T1 — systematic component, per keyframe ===")
    stats = keyframe_residuals(members)
    scopes = (
        (">= 2 residuals", [r for r, _ in stats.values() if len(r) >= 2]),
        (
            ">= 2 residuals, >= 2 groups",
            [r for r, groups in stats.values() if len(r) >= 2 and len(groups) >= 2],
        ),
        (
            ">= 3 residuals, >= 2 groups",
            [r for r, groups in stats.values() if len(r) >= 3 and len(groups) >= 2],
        ),
    )
    print(
        f"  {len(stats)} keyframes; a keyframe with one residual has ratio 1 by "
        "construction and is excluded from every scope below"
    )
    for scope, blocks in scopes:
        if not blocks:
            print(f"  {scope:28s} (no keyframe)")
            continue
        observed = _pooled_bias_ratio(blocks)
        mean, deviation = systematic_null(blocks, seed=seed)
        z = (observed - mean) / deviation if deviation > 0 else math.inf
        print(
            f"  {scope:28s} k={len(blocks):3d}  observed={observed:.3f}  "
            f"shuffled null={mean:.3f}±{deviation:.3f}  z={z:+.2f}"
        )


def _viewing_alignment(members: Mapping[str, list[Resolved]]) -> list[float]:
    """Per keyframe, |cos| between its mean residual and its mean viewing direction.

    A pose error and a depth error both give a view's detections a common lean, so the
    bias ratio alone cannot tell them apart. Their directions can: a translation of
    the camera moves the points **any** way, while a depth error can only move them
    **along the ray**. Near 1 means the lean is a wrong depth wearing a pose error's
    clothes, and no camera offset will remove it.
    """
    residuals: dict[str, list[np.ndarray]] = defaultdict(list)
    directions: dict[str, list[np.ndarray]] = defaultdict(list)
    for rows in members.values():
        centroid = _group_centroid(rows)
        for row in rows:
            point = np.asarray(row.candidate.eus_xyz, dtype=np.float64)
            origin = np.asarray(
                row.candidate.geokeyframe_pose.position, dtype=np.float64
            )
            offset = point - origin
            norm = float(np.linalg.norm(offset))
            if norm == 0.0:
                continue
            residuals[row.label.keyframe_id].append(point - centroid)
            directions[row.label.keyframe_id].append(offset / norm)
    alignments = []
    for keyframe, vectors in residuals.items():
        if len(vectors) < 2:
            continue
        mean_residual = np.mean(vectors, axis=0)
        mean_direction = np.mean(directions[keyframe], axis=0)
        scale = float(np.linalg.norm(mean_residual)) * float(
            np.linalg.norm(mean_direction)
        )
        if scale > 0:
            alignments.append(abs(float(mean_residual @ mean_direction) / scale))
    return alignments


def _print_viewing_alignment(members: Mapping[str, list[Resolved]]) -> None:
    """Print whether the per-view lean points along the ray or across it."""
    alignments = _viewing_alignment(members)
    print("\n=== T1 — is the per-view lean a depth error or a pose error? ===")
    if not alignments:
        print("  (no keyframe with two residuals)")
        return
    print(
        f"  k={len(alignments)} keyframes  mean |cos(mean residual, viewing ray)| = "
        f"{statistics.fmean(alignments):.3f}  "
        f"median={statistics.median(alignments):.3f}"
        "\n  (1 = the lean is along the ray, so it is depth, not pose; "
        "an isotropic lean would sit near 0.5)"
    )


def _print_extent_control(members: Mapping[str, list[Resolved]]) -> None:
    """Control #4: does the tangential part grow with the boxes' angular extent?

    If the bbox centre slides because of parallax on an extended object, the slide
    grows with the angular extent — which is size over range. A flat correlation says
    the tangential part is not parallax.
    """
    rows = [
        (
            spread_of(group).tangential,
            float(
                np.mean(
                    [
                        math.hypot(
                            row.candidate.angular_width, row.candidate.angular_height
                        )
                        for row in group
                    ]
                )
            ),
        )
        for group in members.values()
    ]
    if len(rows) < 3:
        print(
            "\n=== T1 — control: tangential vs angular extent ==="
            "\n  (too few groups)"
        )
        return
    tangential = np.asarray([row[0] for row in rows])
    extent = np.asarray([row[1] for row in rows])
    correlation = float(np.corrcoef(tangential, extent)[0, 1])
    print(
        f"\n=== T1 — control: tangential vs angular extent ===\n"
        f"  n={len(rows)} groups, Pearson rho = {correlation:+.2f}"
    )


def _print_methods(baskets: Sequence[Basket], geo_transform: GeoTransform) -> None:
    """Print the seven-method comparison, pooled by basket family."""
    print("\n=== T0 — the seven methods, pooled by family ===")
    Scored = list[tuple[Basket, MethodOutcome]]
    pooled: dict[tuple[str, str], Scored] = defaultdict(list)
    for basket in baskets:
        for outcome in score_basket(basket, geo_transform):
            pooled[(basket.kind, outcome.method)].append((basket, outcome))
    for kind in ("solo", "pair", "mixed"):
        rows = {
            method: values
            for (family, method), values in pooled.items()
            if family == kind
        }
        if not rows:
            print(f"\n  {kind}: no basket")
            continue
        count = len(next(iter(rows.values())))
        print(f"\n  {kind} — {count} baskets")
        for label, _, _, _ in COMPARED:
            values = rows.get(label, [])
            good = [item for item in values if item[1].error is None]
            if not good:
                print(f"    {label:26s} (all failed)")
                continue
            exact = sum(
                1 for basket, outcome in good
                if outcome.clusters == basket.expected_clusters
            )
            print(
                f"    {label:26s} n={len(good):3d}  "
                f"mean clusters={statistics.fmean(o.clusters for _, o in good):5.2f}  "
                f"exact: {exact / len(good):5.1%}  "
                f"false merges/basket="
                f"{statistics.fmean(o.false_merges for _, o in good):5.2f}"
            )


def _family_score(
    baskets: Sequence[Basket],
    geo_transform: GeoTransform,
    kind: str,
    method: str,
    threshold_deg: float,
) -> tuple[int, float, float, float]:
    """Count, exact-cluster rate, mean clusters and false merges of one family.

    Only the named method is run — scoring all seven per threshold would make a sweep
    seven times the cost of the answer it gives.
    """
    partition, cannot_link, covis = next(
        (row[1], row[2], row[3]) for row in COMPARED if row[0] == method
    )
    exact = merges = clusters = scored = 0
    for basket in (item for item in baskets if item.kind == kind):
        items = [matching_item(row) for row in basket.resolved]
        result = triangulate_items(
            items,
            geo_transform,
            partition_method=partition,
            inlier_threshold_deg=threshold_deg,
            use_cannot_link=cannot_link,
            covis_weight=covis,
        )
        if not result.get("available"):
            continue
        labels = _labels_from_partition(result, len(items))
        scored += 1
        clusters += len(set(labels))
        exact += int(len(set(labels)) == basket.expected_clusters)
        merges += _false_merges(labels, basket.truth)
    if not scored:
        return 0, math.nan, math.nan, math.nan
    return scored, exact / scored, clusters / scored, merges / scored


def print_threshold_sweep(
    baskets: Sequence[Basket],
    geo_transform: GeoTransform,
    thresholds: Sequence[float],
    *,
    method: str = "GASP 1v2 score",
) -> None:
    """Sweep the ray inlier threshold, reading both families at every step.

    The threshold is a **granularity knob**: raising it accepts rays that point further
    apart, so it merges more. A solo-only reading would therefore reward it all the way
    to a single cluster per map (rule R1). The pair column is what makes the sweep
    interpretable, and the mean cluster count is what says whether a gain is a fix or a
    coarser cut.
    """
    print(f"\n=== inlier threshold sweep — {method} ===")
    print(
        f"  {'deg':>5s} | {'solo n':>6s} {'exact':>7s} {'clusters':>8s} | "
        f"{'pair n':>6s} {'exact':>7s} {'clusters':>8s} {'merges':>7s}"
    )
    for threshold in thresholds:
        solo = _family_score(baskets, geo_transform, "solo", method, threshold)
        pair = _family_score(baskets, geo_transform, "pair", method, threshold)
        print(
            f"  {threshold:5.1f} | {solo[0]:6d} {solo[1]:6.1%} {solo[2]:8.2f} | "
            f"{pair[0]:6d} {pair[1]:6.1%} {pair[2]:8.2f} {pair[3]:7.2f}"
        )


def export_covisibility(
    matrix: Covisibility, members: Mapping[str, list[Resolved]], path: Path
) -> None:
    """Write the co-visibility matrix and the group ranges, for T1b and T7."""
    payload = {
        "counts": [
            {"keyframe_id": keyframe, "group": group, "detections": count}
            for (keyframe, group), count in sorted(matrix.counts.items())
        ],
        "groups": {
            name: {
                "detections": len(rows),
                "prompt": rows[0].prompt,
                "centroid_eus": _group_centroid(rows).tolist(),
                "range_median": spread_of(rows).range_median,
            }
            for name, rows in members.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Co-visibility matrix written to %s", path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the harness's command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--covisibility-out", type=Path, default=None)
    parser.add_argument("--skip-methods", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="sweep the ray inlier threshold, in degrees, on both basket families",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def load_resolved(args: argparse.Namespace) -> list[Resolved]:
    """Read the hand labels and resolve them against the cached candidates."""
    map_path = args.map_path.expanduser().resolve()
    labels = load_group_labels(map_path / "object-search-annotations.db")
    if not labels:
        raise SystemExit(
            f"{map_path.name}: detection_group_label is empty — "
            "build baskets in the toolbox before running this harness."
        )
    candidates = fetch_prompt_candidates(
        map_path,
        args.ann_base_url,
        args.candidate_count,
        args.cache_dir,
        None,
        timeout_s=args.timeout,
    )
    resolved, missed = resolve_labels(labels, candidates)
    duplicates = sum(1 for item in resolved if item.rivals > 1)
    print(
        f"{map_path.name}: {len(labels)} labels, {len(resolved)} resolved, "
        f"{len(missed)} unmatched, {duplicates} with a rival box within "
        f"{ANGLE_MATCH_TOL_RAD} rad ({duplicates / max(len(resolved), 1):.1%} "
        "intra-view duplicates)"
    )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """Replay every hand-made basket and print the T0 and T1 tables."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    resolved = load_resolved(args)
    members = _group_members(resolved)
    negatives = [item for item in resolved if item.label.group_name is None]
    baskets = build_baskets(resolved)
    families = ", ".join(
        f"{kind}={sum(1 for basket in baskets if basket.kind == kind)}"
        for kind in ("solo", "pair", "mixed")
    )
    print(
        f"{len(members)} groups above {MIN_BASKET_DETECTIONS} detections, "
        f"{len(negatives)} explicit negatives, {len(baskets)} baskets "
        f"({families})"
    )
    _print_spreads(members)
    _print_systematic(members)
    _print_viewing_alignment(members)
    _print_extent_control(members)
    if args.covisibility_out:
        export_covisibility(covisibility_of(members), members, args.covisibility_out)
    if not args.skip_methods or args.thresholds:
        geo_transform = georef_source.load_pose_source(
            args.map_path.expanduser().resolve()
        ).geo_transform
        if args.thresholds:
            print_threshold_sweep(baskets, geo_transform, args.thresholds)
        if not args.skip_methods:
            _print_methods(baskets, geo_transform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
