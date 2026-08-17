"""Re-sample a detection's depth over its whole box, four ways, and re-score.

`sample_depths` reads **one pixel** — the box centre — of a sqrt-quantised uint16 ERP
depth map. That pixel is on the object most of the time and on a border, a hole or
the background the rest of the time, and T1 says the radial part of the spread is the
larger one. So the question is whether the box has a better answer in it.

Four policies, replayed offline over the hand-labelled groups:

- `center` — the existing single pixel, the baseline;
- `median` — median of the box's valid pixels;
- `nearest_mode` — the nearest depth mode holding enough mass. An object stands in
  front of its background, so under bimodality the near mode is the object;
- `trimmed` — 20/80 trimmed mean, a plain robustness control.

The box is projected corner by corner, never as "centre plus half a size": an ERP box
is not a pixel rectangle near the poles.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from prepare.convention import theta_phi_to_uv

from toolbox.benchmark.association_sweep import DEFAULT_TIMEOUT_S
from toolbox.benchmark.matching_baskets import (
    MIN_PARALLAX_DEG,
    Resolved,
    _group_members,
    _print_methods,
    build_baskets,
    confusable_pairs,
    load_resolved,
    spread_of,
)
from toolbox.bricks import georef_source
from toolbox.bricks.georef_source import PoseSource
from toolbox.bricks.matching import _ray_direction_eus
from toolbox.bricks.vendored.depth_decode import (
    DepthMapNotFound,
    decode_uint16_meters,
    load_depth_map_from_path,
)

#: Policies, in the order the tables print them. `center` first: it is the baseline.
POLICIES = ("center", "median", "nearest_mode", "trimmed")
#: Histogram bin of `nearest_mode`, metres. Coarser than the depth noise on purpose —
#: the question is "which surface", not "how far".
MODE_BIN_M = 0.25
#: Share of a box's valid pixels a mode must hold to count as a surface rather than a
#: fringe of border pixels.
MODE_MIN_MASS = 0.15
#: A box sampled on more pixels than this is strided down; a 5760x2880 ERP box can
#: otherwise reach millions of pixels for no gain in the statistic.
MAX_BOX_SAMPLES = 4096


@dataclass(frozen=True)
class DepthSample:
    """What one policy read on one box."""

    depth_m: float
    valid_fraction: float
    pixels: int


def _box_pixel_grid(
    theta: float,
    phi: float,
    width_rad: float,
    height_rad: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel rows and columns the box covers, from its four projected corners.

    Args:
        theta: Box centre azimuth, radians.
        phi: Box centre elevation, radians.
        width_rad: Angular width of the box.
        height_rad: Angular height of the box.
        width: Depth map width in pixels.
        height: Depth map height in pixels.

    Returns:
        Row indices and column indices, both already wrapped or clipped.
    """
    thetas = np.asarray([theta - width_rad / 2, theta + width_rad / 2] * 2)
    phis = np.asarray([phi - height_rad / 2] * 2 + [phi + height_rad / 2] * 2)
    phis = np.clip(phis, -math.pi / 2, math.pi / 2)
    u, v = theta_phi_to_uv(thetas, phis, width, height)
    rows: np.ndarray = np.arange(
        int(np.floor(v.min())), int(np.ceil(v.max())) + 1, dtype=np.int64
    )
    columns: np.ndarray = np.arange(
        int(np.floor(u.min())), int(np.ceil(u.max())) + 1, dtype=np.int64
    )
    return _strided(np.clip(rows, 0, height - 1)), np.mod(_strided(columns), width)


def _strided(values: np.ndarray) -> np.ndarray:
    """Thin an index range so a box never costs more than `MAX_BOX_SAMPLES` pixels."""
    limit = int(math.sqrt(MAX_BOX_SAMPLES))
    if values.size <= limit:
        return values
    return values[:: max(1, values.size // limit)]


def _nearest_mode(depths: np.ndarray) -> float:
    """The closest depth mode holding at least `MODE_MIN_MASS` of the valid pixels."""
    bins = np.floor(depths / MODE_BIN_M).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    heavy = values[counts >= MODE_MIN_MASS * depths.size]
    chosen = heavy.min() if heavy.size else values[counts.argmax()]
    return float(np.median(depths[bins == chosen]))


def _trimmed_mean(depths: np.ndarray) -> float:
    """Mean of the 20th-to-80th percentile band."""
    low = float(np.percentile(depths, 20.0))
    high = float(np.percentile(depths, 80.0))
    band = depths[(depths >= low) & (depths <= high)]
    return float(np.mean(band if band.size else depths))


def sample_box(
    depth_map: np.ndarray,
    theta: float,
    phi: float,
    width_rad: float,
    height_rad: float,
    policy: str,
) -> DepthSample | None:
    """Read one box's depth under one policy, or None when nothing valid is in it."""
    height, width = depth_map.shape
    rows, columns = _box_pixel_grid(theta, phi, width_rad, height_rad, width, height)
    patch = depth_map[np.ix_(rows, columns)]
    valid = patch[np.isfinite(patch)]
    fraction = float(valid.size) / float(patch.size) if patch.size else 0.0
    if policy == "center":
        u, v = theta_phi_to_uv(np.asarray([theta]), np.asarray([phi]), width, height)
        depth = float(
            depth_map[
                int(np.clip(round(float(v[0])), 0, height - 1)),
                int(np.mod(round(float(u[0])), width)),
            ]
        )
        return (
            DepthSample(depth, fraction, patch.size)
            if math.isfinite(depth)
            else None
        )
    if valid.size == 0:
        return None
    if policy == "median":
        depth = float(np.median(valid))
    elif policy == "nearest_mode":
        depth = _nearest_mode(valid)
    elif policy == "trimmed":
        depth = _trimmed_mean(valid)
    else:
        raise ValueError(f"Unknown depth policy: {policy!r}")
    return DepthSample(depth, fraction, patch.size)


class DepthMapReader:
    """One decoded depth map at a time, keyed by keyframe."""

    def __init__(self, map_path: Path, pose_source: PoseSource) -> None:
        self._depth_dir = map_path / "depths"
        self._names = pose_source.depth_filename_by_keyframe_id
        self._loaded: tuple[int, np.ndarray] | None = None

    def get(self, keyframe_id: int) -> np.ndarray | None:
        """The depth map in metres, NaN where the uint16 sentinel said invalid."""
        if self._loaded is not None and self._loaded[0] == keyframe_id:
            return self._loaded[1]
        name = self._names.get(keyframe_id)
        path = self._depth_dir / name if name else None
        if path is None or not path.is_file():
            return None
        try:
            raw, _ = load_depth_map_from_path(path)
        except (DepthMapNotFound, ValueError):
            return None
        metres = np.where(raw == 0, np.nan, decode_uint16_meters(raw)).astype(
            np.float64
        )
        self._loaded = (keyframe_id, metres)
        return metres


def repositioned(
    rows: Sequence[Resolved],
    reader: DepthMapReader,
    pose_source: PoseSource,
    policy: str,
) -> tuple[list[Resolved], list[float]]:
    """Re-place every detection at the depth its box gives under `policy`.

    A detection whose box holds no valid pixel keeps its stored position: dropping it
    would compare two different sets, and a policy that improves the spread by losing
    the hard detections has improved nothing.
    """
    moved: list[Resolved] = []
    fractions: list[float] = []
    for row in rows:
        candidate = row.candidate
        depth_map = reader.get(int(candidate.video_keyframe_id))
        pose = pose_source.poses.get(int(candidate.video_keyframe_id))
        sample = (
            None
            if depth_map is None or pose is None
            else sample_box(
                depth_map,
                float(candidate.theta_center),
                float(candidate.phi_center),
                float(candidate.angular_width),
                float(candidate.angular_height),
                policy,
            )
        )
        if sample is None or pose is None:
            moved.append(row)
            continue
        fractions.append(sample.valid_fraction)
        direction = np.asarray(
            _ray_direction_eus(pose, candidate.theta_center, candidate.phi_center),
            dtype=np.float64,
        )
        origin = np.asarray(pose.position_eus, dtype=np.float64)
        position = origin + direction * sample.depth_m
        moved.append(
            replace(row, candidate=replace(candidate, eus_xyz=tuple(position)))
        )
    return moved, fractions


def _spread_row(members: Mapping[str, list[Resolved]]) -> tuple[float, float, int]:
    """Pooled radial and tangential spread over the groups above the parallax gate."""
    spreads = [
        spread for spread in (spread_of(rows) for rows in members.values())
        if spread.max_parallax_deg >= MIN_PARALLAX_DEG
    ]
    if not spreads:
        return math.nan, math.nan, 0
    return (
        statistics.fmean(spread.radial for spread in spreads),
        statistics.fmean(spread.tangential for spread in spreads),
        len(spreads),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the T2 command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", default="http://unused")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--skip-methods", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Re-sample every hand-labelled detection under each policy and compare."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    resolved = load_resolved(args)
    pose_source = georef_source.load_pose_source(map_path)
    reader = DepthMapReader(map_path, pose_source)
    ordered = sorted(resolved, key=lambda row: int(row.candidate.video_keyframe_id))
    # Fixed once, on the stored positions: a policy must be scored on the baskets the
    # baseline defines, not on the ones its own displacement happens to create.
    pairs = confusable_pairs(_group_members(ordered))
    print(f"\n=== T2 — depth policies over {len(ordered)} boxes ===")
    print(
        f"  {'policy':14s} {'groups':>6s} {'radial':>8s} "
        f"{'tangential':>11s} {'valid px':>9s}"
    )
    for policy in args.policies:
        moved, fractions = repositioned(ordered, reader, pose_source, policy)
        members = _group_members(moved)
        radial, tangential, count = _spread_row(members)
        valid = statistics.fmean(fractions) if fractions else math.nan
        print(
            f"  {policy:14s} {count:6d} {radial:8.3f} {tangential:11.3f} {valid:9.1%}"
        )
        if not args.skip_methods:
            _print_methods(build_baskets(moved, pairs), pose_source.geo_transform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
