"""Depth distribution over a map's whole set of depth maps, not its detections.

`prepare_postprocess.sample_depths` answers "how far is each detection"; this answers
"how far is the scene", by reading every ERP depth map of a map. The two are different
questions and the second one bounds the first: a depth cap can only be judged against
what the sensor actually sees.

**Histogram the codes, not the metres.** The stored value is a uint16 code and
`decode_uint16_meters` is a monotone function of it alone, so accumulating
`bincount(code)` per file and converting the 65 536 bins once at the end is exact — and
about ten times cheaper than decoding 16.6 M float32 metres per file — which is what
makes a pass over 75 000 keyframes finish in minutes rather than hours.

**Weight by solid angle.** An equirectangular row near a pole covers far less of the
sphere than one at the horizon, so a raw pixel histogram over-counts the floor and the
ceiling — both of which are close. The report carries both the pixel-weighted and the
solid-angle-weighted distribution (`cos φ`), and the second is the one that describes
the scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolbox.bricks.georef_source import load_pose_source
from toolbox.bricks.prepare_postprocess import (
    DEFAULT_DEPTH_DIRNAME,
    _resolve_depth_path,
)
from toolbox.bricks.vendored.depth_decode import (
    decode_uint16_meters,
    load_depth_map_from_path,
)
from toolbox.logging import logger

CODES = 1 << 16
PERCENTILES = (5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
BEYOND = (10.0, 15.0, 20.0, 30.0, 50.0)
# The decoder's top level. It is not a measurement: sky, glass and anything past the
# model's range all land there, so it is reported on its own and excluded from the
# "usable" block rather than being averaged in as a 100 m surface.
CLAMP_MARGIN_M = 0.5
# Rows per solid-angle band. 32 keeps `cos φ` within 0.6 % of its mean inside a band on
# a 2880-row map, far below anything this measurement is used to decide.
BAND_ROWS = 32


@dataclass
class Histograms:
    """Code histograms for one or many depth maps, plus what was unusable."""

    pixels: np.ndarray  # counts per uint16 code
    solid_angle: np.ndarray  # cos(phi)-weighted counts per code
    files: int = 0
    invalid_pixels: int = 0
    total_pixels: int = 0

    @classmethod
    def empty(cls) -> Histograms:
        return cls(
            pixels=np.zeros(CODES, dtype=np.int64),
            solid_angle=np.zeros(CODES, dtype=np.float64),
        )

    def add(self, other: Histograms) -> None:
        self.pixels += other.pixels
        self.solid_angle += other.solid_angle
        self.files += other.files
        self.invalid_pixels += other.invalid_pixels
        self.total_pixels += other.total_pixels


def _band_weights(height: int) -> np.ndarray:
    """Mean `cos φ` for each band of `BAND_ROWS` rows, top to bottom."""
    rows = np.arange(height, dtype=np.float64)
    phi = (np.pi / 2.0) - (rows + 0.5) / height * np.pi
    cos_phi = np.cos(phi)
    bands = np.ceil(height / BAND_ROWS).astype(int)
    return np.asarray(
        [
            cos_phi[index * BAND_ROWS : (index + 1) * BAND_ROWS].mean()
            for index in range(bands)
        ]
    )


def histogram_one(path: Path) -> Histograms:
    """Accumulate the code histograms of a single depth map."""
    result = Histograms.empty()
    codes, (height, width) = load_depth_map_from_path(path)
    weights = _band_weights(height)
    for index, weight in enumerate(weights):
        band = codes[index * BAND_ROWS : (index + 1) * BAND_ROWS]
        if band.size == 0:
            continue
        counts = np.bincount(band.ravel(), minlength=CODES)
        result.pixels += counts
        result.solid_angle += weight * counts
    result.files = 1
    result.total_pixels = int(codes.size)
    result.invalid_pixels = int(result.pixels[0])
    return result


def depth_map_paths(map_path: Path, stride: int = 1) -> list[Path]:
    """Every depth map the manifest names, in keyframe order.

    Reads the manifest rather than globbing: a map directory can hold depth maps for
    keyframes that are not in it (78 665 files against 74 988 keyframes on vinci), and
    counting those would describe a different map.
    """
    source = load_pose_source(map_path)
    depth_dir = map_path / DEFAULT_DEPTH_DIRNAME
    paths: list[Path] = []
    for keyframe_id in sorted(source.depth_filename_by_keyframe_id):
        if keyframe_id % stride:
            continue
        resolved = _resolve_depth_path(
            depth_dir, source.depth_filename_by_keyframe_id[keyframe_id]
        )
        if resolved is not None:
            paths.append(resolved)
    return paths


def accumulate(paths: Sequence[Path], workers: int) -> Histograms:
    """Histogram every path, in parallel, reporting progress every 5 000 files."""
    total = Histograms.empty()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(
            pool.map(histogram_one, paths, chunksize=8), start=1
        ):
            total.add(result)
            if index % 5000 == 0:
                logger.info("Histogrammed %d/%d depth map(s)", index, len(paths))
    return total


def _metres_per_code() -> np.ndarray:
    """Depth in metres for every uint16 code, code 0 (invalid) excluded by callers."""
    return np.asarray(
        decode_uint16_meters(np.arange(CODES, dtype=np.uint16)), dtype=np.float64
    )


def summarise(histograms: Histograms) -> dict:
    """Percentiles, tail fractions and coverage, for both weightings."""
    metres = _metres_per_code()
    valid = np.ones(CODES, dtype=bool)
    valid[0] = False  # the invalid sentinel
    report: dict = {
        "files": histograms.files,
        "total_pixels": histograms.total_pixels,
        "invalid_fraction": (
            histograms.invalid_pixels / histograms.total_pixels
            if histograms.total_pixels
            else 0.0
        ),
    }
    far = float(metres[valid].max())
    report["clamp_m"] = far
    for name, counts in (
        ("pixel_weighted", histograms.pixels.astype(np.float64)),
        ("solid_angle_weighted", histograms.solid_angle),
        # Same weighting as the row above, minus the saturated top level: the
        # distribution of the geometry the model actually resolved.
        ("solid_angle_usable", histograms.solid_angle),
    ):
        keep = valid.copy()
        if name == "solid_angle_usable":
            keep &= metres < far - CLAMP_MARGIN_M
        weights = counts[keep]
        values = metres[keep]
        mass = weights.sum()
        if mass <= 0.0:
            report[name] = {}
            continue
        cumulative = np.cumsum(weights) / mass
        block = {
            f"p{int(percentile)}": float(
                values[int(np.searchsorted(cumulative, percentile / 100.0))]
            )
            for percentile in PERCENTILES
        }
        block["mean"] = float((values * weights).sum() / mass)
        block["mass_fraction"] = float(mass / max(counts[valid].sum(), 1e-12))
        for limit in BEYOND:
            block[f"beyond_{int(limit)}m"] = float(weights[values > limit].sum() / mass)
        report[name] = block
    return report


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse depth-distribution arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True, action="append")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep one keyframe in N. Defaults to every keyframe.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _iter_maps(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        yield path.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """Histogram every depth map of every requested map and print a summary."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report: dict[str, dict] = {}
    for map_path in _iter_maps(args.map_path):
        paths = depth_map_paths(map_path, args.stride)
        logger.info("%s: %d depth map(s)", map_path.name, len(paths))
        histograms = accumulate(paths, args.workers)
        report[map_path.name] = summarise(histograms)
        if args.out is not None:
            target = args.out.expanduser().resolve()
            np.savez_compressed(
                target.with_name(f"{target.stem}-{map_path.name}.npz"),
                pixels=histograms.pixels,
                solid_angle=histograms.solid_angle,
                metres=_metres_per_code(),
            )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
