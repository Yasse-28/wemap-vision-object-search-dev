"""Does an *observed* object extent recut baskets better than the 1 m constant?

The gate for porting `gasp1v2` into the association: run the matching tab's
`score1v2` agglomeration over baskets whose right answer is known, with the extent
assumed (`DEFAULT_OBJECT_EXTENT_M`) and then observed
(`0.5 * range * hypot(angular_width, angular_height)`), and compare.

Two basket kinds, because a single number cannot say whether a change is a fix or
just a coarser granularity:

- **solo** — the detections of one annotation. The right answer is one cluster, so
  this measures whether extended objects stop fragmenting;
- **pair** — the detections of two neighbouring annotations of one class. The right
  answer is two clusters, so this measures what the fix costs in false merges.

Baskets are built from the benchmark annotations rather than by hand in the toolbox:
the same construction, reproducible, and over every annotation instead of two.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolbox.benchmark.association_sweep import (
    DEFAULT_ACCURACY_M,
    DEFAULT_TIMEOUT_S,
    _prompt_annotations,
    fetch_prompt_candidates,
    nearest_annotation_labels,
)
from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    haversine_m,
    load_annotations,
)
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.matching import (
    MatchingItem,
    _agglomerate_by_score,
    _detection_extents_m,
    cannot_link_pairs,
)

#: Fewer detections than this and a basket says nothing about fragmentation.
MIN_BASKET_DETECTIONS = 4
#: Two annotations further apart than this are not the confusable pair we want.
PAIR_MAX_DISTANCE_M = 4.0


@dataclass(frozen=True)
class Basket:
    """Detection positions to partition, and the annotation each one belongs to."""

    indices: list[int]
    truth: list[int]


@dataclass(frozen=True)
class BasketOutcome:
    """One basket scored under one extent policy."""

    clusters: int
    false_merges: int
    truth_groups: int


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    """EUS triple, typed as one — every position here is three-dimensional."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _matching_items(detections: Sequence[EnrichedCandidate]) -> list[MatchingItem]:
    """Build matching items straight from enriched candidates, without pgvector.

    `resolve_items` exists to recover these fields from a `(keyframe, theta, phi)`
    triple the UI can produce. Offline the candidates already carry every field, so
    the database round trip would only be a way to reintroduce the angular join.
    """
    items: list[MatchingItem] = []
    for detection in detections:
        origin = np.asarray(detection.geokeyframe_pose.position, dtype=np.float64)
        point = np.asarray(detection.eus_xyz, dtype=np.float64)
        offset = point - origin
        norm = float(np.linalg.norm(offset))
        items.append(
            MatchingItem(
                keyframe_id=str(detection.video_keyframe_id),
                theta_center=float(detection.theta_center),
                phi_center=float(detection.phi_center),
                candidate_id=int(detection.id),
                thumbnail=None,
                ray_origin_eus=_triple(origin),
                ray_direction_eus=(
                    (0.0, 0.0, -1.0) if norm == 0.0 else _triple(offset / norm)
                ),
                stored_position_eus=_triple(point),
                embedding=None,
                angular_width=float(detection.angular_width),
                angular_height=float(detection.angular_height),
            )
        )
    return items


def score_basket(
    detections: Sequence[EnrichedCandidate],
    truth: Sequence[int],
    *,
    observed_extent: bool,
    use_cannot_link: bool = False,
) -> BasketOutcome:
    """Partition one basket and count its clusters and its false merges."""
    items = _matching_items(detections)
    indices = list(range(len(items)))
    blocked = set(cannot_link_pairs(items, indices)) if use_cannot_link else set()
    labels = _agglomerate_by_score(
        [np.asarray(item.stored_position_eus) for item in items],
        [np.asarray(item.ray_origin_eus) for item in items],
        [np.asarray(item.ray_direction_eus) for item in items],
        [
            0.5 * float(np.hypot(item.angular_width or 0.1, item.angular_height or 0.1))
            for item in items
        ],
        blocked,
        covis_weight=0.0,
        detection_extents_m=(
            _detection_extents_m(items, indices) if observed_extent else None
        ),
    )
    merged = sum(
        1
        for position, left in enumerate(labels)
        for right_position, right in enumerate(labels[position + 1 :], position + 1)
        if left == right and truth[position] != truth[right_position]
    )
    return BasketOutcome(
        clusters=len(set(labels)),
        false_merges=merged,
        truth_groups=len(set(truth)),
    )


def build_baskets(
    detections: Sequence[EnrichedCandidate],
    annotations: Sequence[Annotation],
    near_m: float,
) -> tuple[list[Basket], list[Basket]]:
    """Return the solo baskets and the neighbouring-pair baskets."""
    labels = nearest_annotation_labels(detections, annotations, near_m)
    members: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        if label >= 0:
            members.setdefault(int(label), []).append(position)
    covered = {
        index: rows
        for index, rows in sorted(members.items())
        if len(rows) >= MIN_BASKET_DETECTIONS
    }
    solo = [Basket(rows, [0] * len(rows)) for rows in covered.values()]

    pairs: list[Basket] = []
    ordered = list(covered)
    for position, left in enumerate(ordered):
        for right in ordered[position + 1 :]:
            distance = haversine_m(
                annotations[left].lat,
                annotations[left].lng,
                annotations[right].lat,
                annotations[right].lng,
            )
            if distance <= PAIR_MAX_DISTANCE_M:
                pairs.append(
                    Basket(
                        covered[left] + covered[right],
                        [0] * len(covered[left]) + [1] * len(covered[right]),
                    )
                )
    return solo, pairs


def _summarise(name: str, outcomes: Sequence[BasketOutcome], expected: int) -> str:
    if not outcomes:
        return f"  {name:26s} (no basket)"
    exact = sum(1 for item in outcomes if item.clusters == expected)
    return (
        f"  {name:26s} n={len(outcomes):3d}  mean clusters="
        f"{statistics.fmean(item.clusters for item in outcomes):5.2f}  "
        f"exactly {expected}: {exact / len(outcomes):5.1%}  "
        f"false merges/basket="
        f"{statistics.fmean(item.false_merges for item in outcomes):6.2f}"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse basket-gate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--near-m", type=float, default=1.0)
    parser.add_argument("--num-results", type=int, default=400)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--cannot-link", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score every prompt's baskets under both extent policies."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    annotations = load_annotations(
        map_path / "benchmark" / "annotations.geojson", DEFAULT_ACCURACY_M
    )
    by_prompt = _prompt_annotations(annotations)
    candidates = fetch_prompt_candidates(
        map_path,
        args.ann_base_url,
        args.candidate_count,
        args.cache_dir,
        args.prompts,
        timeout_s=args.timeout,
    )
    totals: dict[tuple[str, bool], list[BasketOutcome]] = {}
    for prompt, prompt_candidates in candidates.items():
        detections = prompt_candidates[: args.num_results]
        solo, pairs = build_baskets(detections, by_prompt[prompt], args.near_m)
        print(f"\n{prompt} — {len(solo)} solo, {len(pairs)} pair baskets")
        for kind, baskets, expected in (("solo", solo, 1), ("pair", pairs, 2)):
            for observed in (False, True):
                outcomes = [
                    score_basket(
                        [detections[index] for index in basket.indices],
                        basket.truth,
                        observed_extent=observed,
                        use_cannot_link=args.cannot_link,
                    )
                    for basket in baskets
                ]
                totals.setdefault((kind, observed), []).extend(outcomes)
                policy = "observed" if observed else "constant"
                print(_summarise(f"{kind} / {policy} extent", outcomes, expected))

    print("\n=== pooled over prompts ===")
    for kind, expected in (("solo", 1), ("pair", 2)):
        for observed in (False, True):
            policy = "observed" if observed else "constant"
            print(
                _summarise(
                    f"{kind} / {policy} extent",
                    totals.get((kind, observed), []),
                    expected,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
