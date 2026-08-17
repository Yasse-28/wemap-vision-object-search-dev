"""Does the noise model `sigma(r) = 0.5 + 0.05 r` have the right sign? (T3)

`merge_score._sigmas` widens the tolerance with range, on the reasoning that a far
detection is a less certain one. The measurements of 2026-08-15 said the observed
scatter *shrinks* with range on bbhotel. This fits the model on what the hand-labelled
groups actually show, one point per **detection** rather than per group: the groups
of one map cover two or three ranges, the detections cover the whole span, and it is
the detections the model is applied to.

The residual fitted is the **radial** one — the part along the viewing ray. That is
what `sigma` is supposed to describe: the tangential part is the box centre moving,
which no positional sigma models, and the two were separated in T1 precisely so they
would not be recalibrated as one.

`e_i` stays at its 1 m constant throughout, as the plan requires: fitting sigma and
the extent at once would measure their sum.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from toolbox.benchmark.association_sweep import DEFAULT_TIMEOUT_S
from toolbox.benchmark.matching_baskets import (
    MIN_PARALLAX_DEG,
    RANGE_BANDS,
    Resolved,
    _band_of,
    _group_members,
    load_resolved,
    spread_of,
)
from toolbox.benchmark.ray_refinement import refine_rows
from toolbox.bricks import georef_source
from toolbox.bricks.merge_score import BASE_SIGMA_M, SIGMA_PER_METRE

#: |N(0, s)| has mean s * sqrt(2/pi); the fit is on absolute residuals, so it has to
#: be undone before the coefficients are comparable to `_sigmas`.
HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)


def radial_residuals(
    members: Mapping[str, list[Resolved]], *, min_parallax_deg: float = MIN_PARALLAX_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Every detection's range and its residual along its own viewing ray.

    Groups below the parallax gate are dropped: with every view from one side, the
    radial and tangential parts are the same direction and the split means nothing.
    """
    ranges: list[float] = []
    residuals: list[float] = []
    for rows in members.values():
        if spread_of(rows).max_parallax_deg < min_parallax_deg:
            continue
        points = np.asarray([row.candidate.eus_xyz for row in rows], dtype=np.float64)
        origins = np.asarray(
            [row.candidate.geokeyframe_pose.position for row in rows], dtype=np.float64
        )
        offsets = points - origins
        distances = np.linalg.norm(offsets, axis=1)
        directions = offsets / distances[:, None]
        along = np.einsum("ij,ij->i", points - points.mean(axis=0), directions)
        ranges.extend(distances.tolist())
        residuals.extend(np.abs(along).tolist())
    return np.asarray(ranges), np.asarray(residuals)


def fit_sigma(ranges: np.ndarray, residuals: np.ndarray) -> tuple[float, float]:
    """Least-squares affine `sigma(r) = base + slope * r`, in sigma units."""
    design = np.column_stack([np.ones_like(ranges), ranges])
    coefficients, *_ = np.linalg.lstsq(design, residuals, rcond=None)
    return (
        float(coefficients[0]) / HALF_NORMAL_MEAN,
        float(coefficients[1]) / HALF_NORMAL_MEAN,
    )


def band_sigmas(
    ranges: np.ndarray, residuals: np.ndarray
) -> list[tuple[str, int, float, float]]:
    """Per range band: count, measured sigma, and what the model currently claims."""
    rows = []
    for low, high in RANGE_BANDS:
        mask = (ranges >= low) & (ranges < high)
        if not mask.any():
            continue
        measured = float(np.mean(residuals[mask])) / HALF_NORMAL_MEAN
        modelled = BASE_SIGMA_M + SIGMA_PER_METRE * float(np.mean(ranges[mask]))
        band = _band_of(float(np.mean(ranges[mask])))
        rows.append((band, int(mask.sum()), measured, modelled))
    return rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the T3 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="fit on the positions T4 leaves, not on the stored depth points",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Fit sigma(r) on the hand-labelled groups and compare it to the model in place."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    members = _group_members(load_resolved(args))
    if args.refine:
        geo_transform = georef_source.load_pose_source(
            args.map_path.expanduser().resolve()
        ).geo_transform
        members = {
            name: refine_rows(rows, geo_transform)[0] for name, rows in members.items()
        }
    ranges, residuals = radial_residuals(members)
    if ranges.size < 8:
        print(f"\n=== T3 — only {ranges.size} detections above the parallax gate ===")
        return 0
    base, slope = fit_sigma(ranges, residuals)
    correlation = float(np.corrcoef(ranges, residuals)[0, 1])
    print(
        f"\n=== T3 — sigma(r) on {ranges.size} detections, "
        f"range {ranges.min():.1f}-{ranges.max():.1f} m ===\n"
        f"  in place  sigma(r) = {BASE_SIGMA_M:.3f} + {SIGMA_PER_METRE:.3f} r\n"
        f"  measured  sigma(r) = {base:.3f} + {slope:+.3f} r   "
        f"(Pearson rho between range and |radial residual| = {correlation:+.2f})"
    )
    print(f"  {'band':10s} {'n':>4s} {'measured':>9s} {'model':>7s}")
    for band, count, measured, modelled in band_sigmas(ranges, residuals):
        print(f"  {band:10s} {count:4d} {measured:9.3f} {modelled:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
