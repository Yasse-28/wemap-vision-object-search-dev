"""Depth quality where it decides a detection's position: at the object's border.

`depth_map_distribution.py` answers "how far is the scene". This answers a narrower
and more actionable question: **is the depth trustworthy under the boxes the detector
draws**. The two are not the same measurement, and the usual monocular-depth figures
(AbsRel, delta1) cannot separate them — those are dominated by large flat surfaces,
which is exactly the part of the image no detection is placed from.

A detection is placed by sampling one depth pixel at its box centre. When that pixel
sits on a depth discontinuity, the sample can land on the object, on the wall behind
it, or on neither — a **flying pixel**, a value interpolated across the jump that
corresponds to empty space. Two observations of one object that sample opposite sides
of the same border land metres apart, which is a split object rather than a small
error, and no association radius fixes it.

Three numbers come out, per detection and pooled:

- `edge_at_sample`: the sampled pixel is within `EDGE_MARGIN_PX` of a discontinuity;
- `flying_at_sample`: the sampled pixel *is* a flying pixel;
- `depth_ratio`: p90 over p10 of the depth inside the box, which says whether the box
  covers one surface or straddles two.

No ground truth is involved, and no pose: this is a statement about the depth maps and
the boxes alone, so it runs on any prepared map.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from prepare.convention import theta_phi_to_uv

from toolbox.bricks.georef_source import load_pose_source
from toolbox.bricks.prepare_postprocess import (
    DEFAULT_DEPTH_DIRNAME,
    _resolve_depth_path,
)
from toolbox.bricks.vendored.depth_decode import (
    DepthMapNotFound,
    decode_uint16_meters,
    load_depth_map_from_path,
)
from toolbox.logging import logger

#: A neighbouring pixel this much further away in relative terms is a different
#: surface, not the same one seen at an angle. Ten per cent is well above the
#: decoder's quantisation everywhere inside its trusted range.
EDGE_RELATIVE_JUMP = 0.10
#: A pixel is flying when it stands off *both* its nearest and its farthest
#: neighbour by this much: it is in the gap between two surfaces, on neither.
FLYING_RELATIVE_MARGIN = 0.05
#: How far from the sampled pixel a discontinuity still threatens it. The sample is
#: rounded to an integer pixel and the box centre carries its own error, so a border
#: two pixels away is a border the sample could have landed on.
EDGE_MARGIN_PX = 2
#: Rows per band of the scene-wide pass. The masks compare adjacent pixels, so the
#: pass runs at full resolution — subsampling the map first would compare pixels
#: several apart and report a different, larger quantity than `at_sample` does — and
#: bands keep the four shifted copies of a 16 Mpx map out of memory.
SCENE_BAND_ROWS = 512
#: Range bands the per-detection table is broken down by, metres.
RANGE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 15.0),
    (15.0, float("inf")),
)


@dataclass
class Measurement:
    """What one keyframe contributes, per detection and for the scene."""

    depth_m: np.ndarray
    edge_at_sample: np.ndarray
    flying_at_sample: np.ndarray
    depth_ratio: np.ndarray
    box_edge_share: np.ndarray
    box_flying_share: np.ndarray
    scene_valid: int = 0
    scene_edge: int = 0
    scene_flying: int = 0
    keyframes: int = 0

    @classmethod
    def empty(cls) -> Measurement:
        """A measurement carrying nothing, used as the accumulator's seed."""
        nothing = np.empty(0, dtype=np.float64)
        return cls(
            depth_m=nothing,
            edge_at_sample=np.empty(0, dtype=bool),
            flying_at_sample=np.empty(0, dtype=bool),
            depth_ratio=nothing,
            box_edge_share=nothing,
            box_flying_share=nothing,
        )

    def add(self, other: Measurement) -> None:
        """Concatenate another keyframe's rows into this accumulator."""
        self.depth_m = np.concatenate((self.depth_m, other.depth_m))
        self.edge_at_sample = np.concatenate(
            (self.edge_at_sample, other.edge_at_sample)
        )
        self.flying_at_sample = np.concatenate(
            (self.flying_at_sample, other.flying_at_sample)
        )
        self.depth_ratio = np.concatenate((self.depth_ratio, other.depth_ratio))
        self.box_edge_share = np.concatenate(
            (self.box_edge_share, other.box_edge_share)
        )
        self.box_flying_share = np.concatenate(
            (self.box_flying_share, other.box_flying_share)
        )
        self.scene_valid += other.scene_valid
        self.scene_edge += other.scene_edge
        self.scene_flying += other.scene_flying
        self.keyframes += other.keyframes


@dataclass(frozen=True)
class KeyframeWork:
    """One keyframe's depth map and the boxes to measure on it."""

    keyframe_id: int
    depth_path: Path
    theta: np.ndarray
    phi: np.ndarray
    angular_width: np.ndarray
    angular_height: np.ndarray


def surface_masks(metres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discontinuity and flying-pixel masks of a decoded depth patch.

    Both are read off the four-neighbour extremes. A pixel whose neighbourhood spans
    a large relative range sits on a border; one that additionally stands clear of
    *both* extremes is in the gap between the two surfaces rather than on either, and
    the point it projects to is in empty space.

    Args:
        metres: Decoded depth, zero marking the invalid sentinel.

    Returns:
        The edge mask and the flying mask, both the shape of ``metres``. Pixels on
        the array's own border are false in both: they have no full neighbourhood.

    Raises:
        ValueError: If ``metres`` is smaller than 3x3.
    """
    if metres.shape[0] < 3 or metres.shape[1] < 3:
        raise ValueError("A depth patch needs at least a 3x3 neighbourhood")
    values = np.where(metres > 0.0, metres, np.nan).astype(np.float64)
    shifted = np.stack(
        (
            values[:-2, 1:-1],
            values[2:, 1:-1],
            values[1:-1, :-2],
            values[1:-1, 2:],
        )
    )
    centre = values[1:-1, 1:-1]
    nearest = np.nanmin(shifted, axis=0)
    farthest = np.nanmax(shifted, axis=0)
    with np.errstate(invalid="ignore"):
        edge = (farthest - nearest) > EDGE_RELATIVE_JUMP * centre
        flying = (centre - nearest > FLYING_RELATIVE_MARGIN * centre) & (
            farthest - centre > FLYING_RELATIVE_MARGIN * centre
        )
    valid = np.isfinite(centre) & np.isfinite(nearest) & np.isfinite(farthest)
    edge_full = np.zeros(metres.shape, dtype=bool)
    flying_full = np.zeros(metres.shape, dtype=bool)
    edge_full[1:-1, 1:-1] = edge & valid
    flying_full[1:-1, 1:-1] = flying & valid
    return edge_full, flying_full


def _box_slice(
    metres: np.ndarray, row: int, column: int, half_rows: int, half_columns: int
) -> np.ndarray:
    """A rectangle of pixels around one centre, wrapping in azimuth.

    Columns wrap because an equirectangular map is a full turn: the last column
    neighbours the first. Rows are clamped instead — the poles are not adjacent to
    anything, and a box that reaches one is looking at the floor or the ceiling.
    """
    height, width = metres.shape
    rows = np.clip(np.arange(row - half_rows, row + half_rows + 1), 0, height - 1)
    columns = np.arange(column - half_columns, column + half_columns + 1) % width
    return metres[np.ix_(rows, columns)]


def scene_rates(metres: np.ndarray) -> tuple[int, int, int]:
    """Valid, edge and flying pixel counts over a whole depth map.

    The baseline the per-detection rates are read against: a map whose borders make
    up three per cent of its pixels says something different about a detection landing
    on one than a map where they make up twenty.

    Args:
        metres: Decoded depth of one full map, zero marking the invalid sentinel.

    Returns:
        Counts of valid, edge and flying pixels. The map's first and last row are
        left out, having no neighbourhood; both are the pole, seen by nothing.
    """
    valid = edge = flying = 0
    for start in range(0, max(metres.shape[0] - 2, 1), SCENE_BAND_ROWS):
        band = metres[start : start + SCENE_BAND_ROWS + 2]
        if band.shape[0] < 3:
            continue
        band_edge, band_flying = surface_masks(band)
        valid += int((band[1:-1] > 0.0).sum())
        edge += int(band_edge.sum())
        flying += int(band_flying.sum())
    return valid, edge, flying


def measure_one(work: KeyframeWork) -> Measurement:
    """Measure every box of one keyframe against its depth map."""
    result = Measurement.empty()
    try:
        codes, (height, width) = load_depth_map_from_path(work.depth_path)
    except (DepthMapNotFound, ValueError) as exc:
        logger.warning("Keyframe %s: unreadable depth map (%s)", work.keyframe_id, exc)
        return result
    metres = np.asarray(decode_uint16_meters(codes), dtype=np.float64)
    del codes

    valid, edge, flying = scene_rates(metres)
    result.scene_valid = valid
    result.scene_edge = edge
    result.scene_flying = flying
    result.keyframes = 1

    u, v = theta_phi_to_uv(work.theta, work.phi, width, height)
    columns = np.mod(np.round(u).astype(np.int64), width)
    rows = np.clip(np.round(v).astype(np.int64), 0, height - 1)
    # An angular half-extent in pixels: a full turn spans the width, half a turn the
    # height, which is the same convention `theta_phi_to_uv` inverts.
    half_columns = np.maximum(
        1, np.round(work.angular_width / (2 * np.pi) * width / 2).astype(np.int64)
    )
    half_rows = np.maximum(
        1, np.round(work.angular_height / np.pi * height / 2).astype(np.int64)
    )

    count = work.theta.size
    depth_m = np.full(count, np.nan)
    edge_at_sample = np.zeros(count, dtype=bool)
    flying_at_sample = np.zeros(count, dtype=bool)
    depth_ratio = np.full(count, np.nan)
    box_edge_share = np.full(count, np.nan)
    box_flying_share = np.full(count, np.nan)
    for index in range(count):
        sample = float(metres[rows[index], columns[index]])
        depth_m[index] = sample if sample > 0.0 else np.nan
        half_window = EDGE_MARGIN_PX + 1
        window = _box_slice(
            metres, rows[index], columns[index], half_window, half_window
        )
        window_edge, window_flying = surface_masks(window)
        edge_at_sample[index] = bool(window_edge[1:-1, 1:-1].any())
        flying_at_sample[index] = bool(window_flying[half_window, half_window])

        box = _box_slice(
            metres, rows[index], columns[index], half_rows[index], half_columns[index]
        )
        if box.shape[0] >= 3 and box.shape[1] >= 3:
            box_edge, box_flying = surface_masks(box)
            inside = box > 0.0
            if inside.any():
                values = box[inside]
                low = float(np.percentile(values, 10))
                depth_ratio[index] = (
                    float(np.percentile(values, 90)) / low if low > 0.0 else np.nan
                )
                box_edge_share[index] = float(box_edge.sum() / inside.sum())
                box_flying_share[index] = float(box_flying.sum() / inside.sum())

    result.depth_m = depth_m
    result.edge_at_sample = edge_at_sample
    result.flying_at_sample = flying_at_sample
    result.depth_ratio = depth_ratio
    result.box_edge_share = box_edge_share
    result.box_flying_share = box_flying_share
    return result


def build_work(map_path: Path, sample: int, seed: int = 0) -> list[KeyframeWork]:
    """Pick keyframes at random and gather their parquet boxes.

    A random sample rather than a stride: keyframes are numbered along the capture
    path, so every Nth one walks the venue in order and can miss a whole area.
    """
    table = pq.read_table(
        map_path / "object-search" / "metadata.parquet",
        columns=[
            "video_keyframe_id",
            "theta_center",
            "phi_center",
            "angular_width",
            "angular_height",
        ],
    )
    column = {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.column_names
    }
    keyframe_ids = column["video_keyframe_id"].astype(np.int64)
    source = load_pose_source(map_path)
    depth_dir = map_path / DEFAULT_DEPTH_DIRNAME
    available = np.unique(keyframe_ids)
    generator = np.random.default_rng(seed)
    if available.size > sample:
        available = np.sort(generator.choice(available, size=sample, replace=False))

    theta = column["theta_center"].astype(np.float64)
    phi = column["phi_center"].astype(np.float64)
    angular_width = column["angular_width"].astype(np.float64)
    angular_height = column["angular_height"].astype(np.float64)
    work: list[KeyframeWork] = []
    for keyframe_id in available:
        depth_path = _resolve_depth_path(
            depth_dir, source.depth_filename_by_keyframe_id.get(int(keyframe_id))
        )
        if depth_path is None:
            continue
        rows = np.flatnonzero(keyframe_ids == keyframe_id)
        if rows.size == 0:
            continue
        work.append(
            KeyframeWork(
                keyframe_id=int(keyframe_id),
                depth_path=depth_path,
                theta=theta[rows],
                phi=phi[rows],
                angular_width=angular_width[rows],
                angular_height=angular_height[rows],
            )
        )
    return work


def accumulate(work: Sequence[KeyframeWork], workers: int) -> Measurement:
    """Measure every keyframe, in parallel, reporting progress every 25 keyframes."""
    total = Measurement.empty()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(measure_one, work), start=1):
            total.add(result)
            if index % 25 == 0:
                logger.info("Measured %d/%d keyframe(s)", index, len(work))
    return total


def _share(mask: np.ndarray, among: np.ndarray) -> float:
    """Share of true values among the selected rows, NaN when none are selected."""
    selected = mask[among]
    return float(selected.mean()) if selected.size else float("nan")


def summarise(measurement: Measurement) -> dict:
    """Pooled rates overall and per range band, plus the scene-wide baselines."""
    placed = np.isfinite(measurement.depth_m)
    report: dict = {
        "keyframes": measurement.keyframes,
        "detections": int(measurement.depth_m.size),
        "placed": int(placed.sum()),
        "scene": {
            "edge_share": (
                measurement.scene_edge / measurement.scene_valid
                if measurement.scene_valid
                else float("nan")
            ),
            "flying_share": (
                measurement.scene_flying / measurement.scene_valid
                if measurement.scene_valid
                else float("nan")
            ),
        },
        "at_sample": {
            "edge_share": _share(measurement.edge_at_sample, placed),
            "flying_share": _share(measurement.flying_at_sample, placed),
        },
        "in_box": {
            "median_depth_ratio": _median(measurement.depth_ratio),
            "median_edge_share": _median(measurement.box_edge_share),
            "median_flying_share": _median(measurement.box_flying_share),
        },
        "by_range": {},
    }
    for low, high in RANGE_BANDS:
        band = placed & (measurement.depth_m >= low) & (measurement.depth_m < high)
        report["by_range"][f"{low:g}-{high:g}m"] = {
            "detections": int(band.sum()),
            "edge_at_sample": _share(measurement.edge_at_sample, band),
            "flying_at_sample": _share(measurement.flying_at_sample, band),
            "median_depth_ratio": _median(measurement.depth_ratio[band]),
        }
    return report


def _median(values: np.ndarray) -> float:
    """Median of the finite values, NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse depth-boundary arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True, action="append")
    parser.add_argument(
        "--sample",
        type=int,
        default=150,
        help="Keyframes drawn at random. Defaults to 150.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the depth under the detector's boxes and print a summary."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report: dict[str, dict] = {}
    for raw in args.map_path:
        map_path = raw.expanduser().resolve()
        work = build_work(map_path, args.sample)
        logger.info(
            "%s: %d keyframe(s), %d box(es)",
            map_path.name,
            len(work),
            sum(item.theta.size for item in work),
        )
        report[map_path.name] = summarise(accumulate(work, args.workers))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
