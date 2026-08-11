"""Measure pairwise association-cue separability against nearby annotations.

For each prompt, every retrieved detection is labelled with its nearest annotation
within ``--near-m``. Upper-triangle pairs of labelled detections are then split into
same-object and different-object examples. The report contains distribution
percentiles and rank AUC for depth-point distance, forward ray-to-ray distance, and
cutout cosine. Ray pairs whose closest approach is behind either camera are reported
as unusable rather than silently counted as large distances.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from toolbox.benchmark.association_sweep import (
    DEFAULT_TIMEOUT_S,
    fetch_prompt_candidates,
)
from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    haversine_m,
    load_annotations,
)
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.localize import _ray_closest_approach_pairs

PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)


class CueReport(TypedDict):
    """Serializable summary for one pairwise cue."""

    same_count: int
    different_count: int
    same_percentiles: dict[str, float]
    different_percentiles: dict[str, float]
    auc: float | None
    unusable_same_fraction: float
    unusable_different_fraction: float


def _percentiles(values: np.ndarray) -> dict[str, float]:
    """Return named cue percentiles, or an empty mapping for no observations."""
    if values.size == 0:
        return {}
    measured = np.percentile(values, PERCENTILES)
    return {
        f"p{int(percentile)}": float(value)
        for percentile, value in zip(PERCENTILES, measured, strict=True)
    }


def _rank_auc(
    same_values: np.ndarray,
    different_values: np.ndarray,
    *,
    higher_is_same: bool,
) -> float | None:
    """Compute tie-aware binary AUC without a machine-learning dependency."""
    if same_values.size == 0 or different_values.size == 0:
        return None
    values = np.concatenate((same_values, different_values)).astype(np.float64)
    if not higher_is_same:
        values = -values
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (float(start + 1) + float(end)) / 2.0
    positive_count = same_values.size
    rank_sum = float(np.sum(ranks[:positive_count]))
    numerator = rank_sum - positive_count * (positive_count + 1) / 2.0
    return numerator / (positive_count * different_values.size)


def _cue_report(
    values: np.ndarray,
    same_mask: np.ndarray,
    usable_mask: np.ndarray,
    *,
    higher_is_same: bool,
) -> CueReport:
    """Summarize usable values and unusable fractions for both pair classes."""
    different_mask = ~same_mask
    same_values = values[same_mask & usable_mask]
    different_values = values[different_mask & usable_mask]
    same_total = int(np.count_nonzero(same_mask))
    different_total = int(np.count_nonzero(different_mask))
    return {
        "same_count": int(same_values.size),
        "different_count": int(different_values.size),
        "same_percentiles": _percentiles(same_values),
        "different_percentiles": _percentiles(different_values),
        "auc": _rank_auc(same_values, different_values, higher_is_same=higher_is_same),
        "unusable_same_fraction": (
            1.0 - same_values.size / same_total if same_total else 0.0
        ),
        "unusable_different_fraction": (
            1.0 - different_values.size / different_total if different_total else 0.0
        ),
    }


def _nearest_annotation_labels(
    candidates: Sequence[EnrichedCandidate],
    annotations: Sequence[Annotation],
    near_m: float,
) -> np.ndarray:
    """Return nearest annotation indices, with ``-1`` for detections beyond range."""
    labels = np.full(len(candidates), -1, dtype=np.int64)
    for candidate_index, candidate in enumerate(candidates):
        distances = np.asarray(
            [
                haversine_m(candidate.lat, candidate.lng, item.lat, item.lng)
                for item in annotations
            ],
            dtype=np.float64,
        )
        if distances.size:
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= near_m:
                labels[candidate_index] = nearest
    return labels


def evaluate_prompt_cues(
    candidates: Sequence[EnrichedCandidate],
    annotations: Sequence[Annotation],
    *,
    near_m: float,
) -> dict[str, CueReport | int]:
    """Evaluate all three cues for one prompt's retrieved candidate set."""
    labels = _nearest_annotation_labels(candidates, annotations, near_m)
    labelled_indices = np.flatnonzero(labels >= 0)
    if labelled_indices.size < 2:
        return {"labelled_detections": int(labelled_indices.size)}

    labelled = [candidates[int(index)] for index in labelled_indices]
    local_i, local_j = np.triu_indices(len(labelled), k=1)
    same_mask = labels[labelled_indices][local_i] == labels[labelled_indices][local_j]
    positions = np.asarray([item.eus_xyz for item in labelled], dtype=np.float64)
    origins = np.asarray(
        [item.geokeyframe_pose.position for item in labelled], dtype=np.float64
    )
    vectors = positions - origins
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / np.where(norms > 0.0, norms, 1.0)

    depth = np.linalg.norm(positions[local_i] - positions[local_j], axis=1)
    ray, t_i, t_j = _ray_closest_approach_pairs(
        origins[local_i],
        directions[local_i],
        origins[local_j],
        directions[local_j],
    )
    ray_usable = (t_i > 0.0) & (t_j > 0.0)

    embeddings_present = all(item.embedding is not None for item in labelled)
    cosine = np.zeros(local_i.size, dtype=np.float64)
    semantic_usable = np.zeros(local_i.size, dtype=bool)
    if embeddings_present:
        embeddings = np.vstack(
            [item.embedding for item in labelled if item.embedding is not None]
        ).astype(np.float64)
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(embedding_norms > 0.0, embedding_norms, 1.0)
        cosine = np.einsum("ij,ij->i", normalized[local_i], normalized[local_j])
        semantic_usable = np.ones(local_i.size, dtype=bool)

    all_usable = np.ones(local_i.size, dtype=bool)
    return {
        "labelled_detections": int(labelled_indices.size),
        "same_pairs": int(np.count_nonzero(same_mask)),
        "different_pairs": int(np.count_nonzero(~same_mask)),
        "depth_distance_m": _cue_report(
            depth, same_mask, all_usable, higher_is_same=False
        ),
        "ray_distance_m": _cue_report(ray, same_mask, ray_usable, higher_is_same=False),
        "cutout_cosine": _cue_report(
            cosine, same_mask, semantic_usable, higher_is_same=True
        ),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse pair-cue diagnostic command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--near-m", type=float, default=1.0)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch prompt candidates and emit a JSON separability report."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.near_m <= 0.0:
        raise ValueError("--near-m must be positive")
    map_path = args.map_path.expanduser().resolve()
    annotations = load_annotations(
        map_path / "benchmark" / "annotations.geojson", args.near_m
    )
    by_prompt_annotations: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        by_prompt_annotations.setdefault(annotation.prompt, []).append(annotation)
    prompt_candidates = fetch_prompt_candidates(
        map_path,
        args.ann_base_url,
        args.candidate_count,
        map_path / "benchmark" / "cache" / "pair-cue-separability",
        args.prompts,
        refresh=args.refresh,
        timeout_s=args.timeout,
    )
    report = {
        prompt: evaluate_prompt_cues(
            prompt_candidates[prompt], by_prompt_annotations[prompt], near_m=args.near_m
        )
        for prompt in prompt_candidates
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
