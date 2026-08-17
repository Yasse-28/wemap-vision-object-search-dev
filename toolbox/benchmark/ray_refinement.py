"""Re-place each detection on its own ray, at the depth its cluster triangulates (T4).

The depth point decides *who goes with whom*; after that the rays alone can say
*where*. One pass: associate with the depth prior, triangulate each cluster from its
members' rays, slide every member along its ray to the foot of that point, associate
again. One pass only — iterating without a stopping rule merges everything.

Expectations are low and stated in advance. The rays carry bbox **centres**, so if the
spread were tangential this would triangulate the bias rather than remove it. T1 says
it is radial, which is the regime where sliding along the ray can actually help — and
the regime where the depth prior was contributing least.
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
    MIN_PARALLAX_DEG,
    Basket,
    Resolved,
    _false_merges,
    _group_members,
    _labels_from_partition,
    build_baskets,
    confusable_pairs,
    load_resolved,
    matching_item,
    spread_of,
)
from toolbox.bricks import georef_source
from toolbox.bricks.matching import triangulate_items
from toolbox.bricks.triangulate import DEFAULT_INLIER_THRESHOLD_DEG, _closest_point

#: A cluster whose rays span less than this cannot be triangulated — the point runs
#: off to infinity, and the "gain" would be a count of the cases quietly dropped.
MIN_CLUSTER_PARALLAX_DEG = MIN_PARALLAX_DEG


def _foot_on_ray(
    origin: np.ndarray, direction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Point of the ray closest to `target`, never behind the camera."""
    depth = float(np.dot(target - origin, direction))
    return origin + direction * max(depth, 0.0)


def _cluster_parallax_deg(directions: np.ndarray) -> float:
    """Widest angle between the rays of one cluster, in degrees."""
    if len(directions) < 2:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1).min())))


def refine_rows(
    rows: Sequence[Resolved],
    geo_transform: object,
    *,
    method: str = "gasp1v2",
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
) -> tuple[list[Resolved], int, int]:
    """Slide each detection to its cluster's triangulated point along its own ray.

    Returns:
        The moved rows, the number of clusters refined, and the number skipped for
        want of parallax — the second number is what stops a gain from being a
        silently narrowed population.
    """
    items = [matching_item(row) for row in rows]
    result = triangulate_items(
        items,
        geo_transform,  # type: ignore[arg-type]
        partition_method=method,
        inlier_threshold_deg=inlier_threshold_deg,
    )
    if not result.get("available"):
        return list(rows), 0, 0
    moved = list(rows)
    refined = skipped = 0
    for hypothesis in result["hypotheses"]:
        indices = list(hypothesis["items"])
        if len(indices) < 2:
            continue
        origins = np.asarray([items[i].ray_origin_eus for i in indices])
        directions = np.asarray([items[i].ray_direction_eus for i in indices])
        if _cluster_parallax_deg(directions) < MIN_CLUSTER_PARALLAX_DEG:
            skipped += 1
            continue
        point = _closest_point(origins, directions)
        if point is None:
            skipped += 1
            continue
        refined += 1
        target = np.asarray(point, dtype=np.float64)
        for position, index in enumerate(indices):
            placed = _foot_on_ray(origins[position], directions[position], target)
            moved[index] = replace(
                moved[index],
                candidate=replace(moved[index].candidate, eus_xyz=tuple(placed)),
            )
    return moved, refined, skipped


def _spread_line(members: Mapping[str, list[Resolved]], label: str) -> str:
    """One pooled spread line, over the groups above the parallax gate."""
    spreads = [
        spread for spread in (spread_of(rows) for rows in members.values())
        if spread.max_parallax_deg >= MIN_PARALLAX_DEG
    ]
    if not spreads:
        return f"  {label:12s} (no group above the parallax gate)"
    return (
        f"  {label:12s} total={statistics.fmean(s.total for s in spreads):.3f}  "
        f"radial={statistics.fmean(s.radial for s in spreads):.3f}  "
        f"tangential={statistics.fmean(s.tangential for s in spreads):.3f}"
    )


def _basket_line(
    baskets: Sequence[Basket],
    geo_transform: object,
    label: str,
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
) -> str:
    """Exact-cluster rate and false merges of one basket set, under gasp1v2."""
    exact = merges = scored = 0
    for basket in baskets:
        items = [matching_item(row) for row in basket.resolved]
        result = triangulate_items(
            items,
            geo_transform,  # type: ignore[arg-type]
            partition_method="gasp1v2",
            inlier_threshold_deg=inlier_threshold_deg,
        )
        if not result.get("available"):
            continue
        labels = _labels_from_partition(result, len(items))
        scored += 1
        exact += int(len(set(labels)) == basket.expected_clusters)
        merges += _false_merges(labels, basket.truth)
    if not scored:
        return f"  {label:12s} (no basket)"
    return (
        f"  {label:12s} n={scored:2d}  exact={exact / scored:5.1%}  "
        f"false merges/basket={merges / scored:6.2f}"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the T4 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--method", default="gasp1v2")
    parser.add_argument(
        "--inlier-threshold-deg", type=float, default=DEFAULT_INLIER_THRESHOLD_DEG
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single re-triangulation pass and compare it to the baseline."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    resolved = load_resolved(args)
    geo_transform = georef_source.load_pose_source(
        args.map_path.expanduser().resolve()
    ).geo_transform
    members = _group_members(resolved)
    pairs = confusable_pairs(members)
    # Refined basket by basket, not over the whole map: the basket *is* the population
    # the plan hands the association, and the merge score is quadratic in it.
    refined: list[Resolved] = []
    count = skipped = 0
    for rows in members.values():
        moved, refined_here, skipped_here = refine_rows(
            rows,
            geo_transform,
            method=args.method,
            inlier_threshold_deg=args.inlier_threshold_deg,
        )
        refined.extend(moved)
        count += refined_here
        skipped += skipped_here
    print(
        f"\n=== T4 — one re-triangulation pass at "
        f"{args.inlier_threshold_deg:g}° ===\n"
        f"  {count} clusters refined, {skipped} skipped under "
        f"{MIN_CLUSTER_PARALLAX_DEG:g}° of parallax"
    )
    print(_spread_line(members, "before"))
    print(_spread_line(_group_members(refined), "after"))
    for kind in ("solo", "pair"):
        print(f"  --- {kind} baskets, gasp1v2")
        print(
            _basket_line(
                [b for b in build_baskets(resolved, pairs) if b.kind == kind],
                geo_transform,
                "before",
                inlier_threshold_deg=args.inlier_threshold_deg,
            )
        )
        print(
            _basket_line(
                [b for b in build_baskets(refined, pairs) if b.kind == kind],
                geo_transform,
                "after",
                inlier_threshold_deg=args.inlier_threshold_deg,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
