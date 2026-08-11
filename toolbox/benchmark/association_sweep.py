"""Offline configuration sweeps for object-search association and ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeAlias

from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    AnnotationGroup,
    Prediction,
    evaluate_prompt,
    evaluate_prompt_grouped,
    group_annotations,
    load_annotations,
    parse_predictions,
    post_json,
    precision_recall_curve,
    targets_from_annotations,
    targets_from_groups,
    write_csv,
)
from toolbox.bricks import candidates, db, georef_source, map_manifest, service
from toolbox.bricks.candidates import EnrichedCandidate
from toolbox.bricks.localize import (
    LocalizationParams,
    localize_from_enriched_candidates,
)
from toolbox.bricks.vendored.geo_transform import GeoTransform
from toolbox.logging import logger

DEFAULT_ACCURACY_M = 5.0
DEFAULT_TIMEOUT_S = 60.0
VERIFY_TOLERANCE = 1e-9
CSV_FILENAME = "association_sweep.csv"
JSON_FILENAME = "association_sweep.json"

PromptEvaluator: TypeAlias = Callable[[str, list[Prediction], float], float]


@dataclass(frozen=True)
class ThresholdMetrics:
    """Macro F1 at a fitted shared threshold and its LOO estimate."""

    macro_f1: float
    threshold: float
    loo_macro_f1: float


@dataclass(frozen=True)
class PromptDetail:
    """Threshold-free per-prompt metrics retained for per-class tables."""

    class_name: str
    strict_ap: float
    grouped_ap: float
    cluster_count: int


@dataclass(frozen=True)
class ConfigMetrics:
    """Metrics and granularity controls for one localization configuration."""

    label: str
    params: dict[str, Any]
    map_strict: float
    map_grouped: float
    macro_f1_strict: float
    best_threshold_strict: float
    loo_macro_f1_strict: float
    macro_f1_grouped: float
    best_threshold_grouped: float
    loo_macro_f1_grouped: float
    total_clusters: int
    median_observation_count: float
    median_spread_m: float
    median_clusters_per_prompt: float
    per_prompt: dict[str, PromptDetail]
    elapsed_s: float = 0.0


def _prompt_annotations(
    annotations: Sequence[Annotation],
) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.prompt, []).append(annotation)
    return grouped


def _class_name(annotations: Sequence[Annotation]) -> str:
    """Return the most common class using the HTTP benchmark's tie behaviour."""
    names = {annotation.class_name for annotation in annotations}
    return max(
        names,
        key=lambda name: sum(
            annotation.class_name == name for annotation in annotations
        ),
    )


def _cache_path(
    cache_dir: Path, map_id: str, prompt: str, candidate_count: int
) -> Path:
    key = json.dumps(
        {
            "map_id": map_id,
            "prompt": prompt,
            "candidate_count": candidate_count,
            "with_embeddings": True,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{map_id}-{digest}.pickle"


def fetch_prompt_candidates(
    map_path: str | Path,
    ann_base_url: str,
    candidate_count: int,
    cache_dir: str | Path,
    prompts: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, list[EnrichedCandidate]]:
    """Fetch and cache the enriched ANN candidate set once per benchmark prompt.

    Args:
        map_path: Map directory containing the manifest and benchmark annotations.
        ann_base_url: Base URL of the mirrored ANN service.
        candidate_count: Number of ANN hits requested for each prompt.
        cache_dir: Directory receiving one pickle per prompt and retrieval setup.
        prompts: Optional exact prompt subset. Annotation insertion order is retained.
        refresh: Replace matching cache entries when true.
        timeout_s: ANN request timeout in seconds.

    Returns:
        Prompt strings mapped to enriched candidates in retrieval order.

    Raises:
        ValueError: If the manifest has no geo reference or a prompt is unknown.
    """
    resolved_map_path = Path(map_path).expanduser().resolve()
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = map_manifest.load_map_manifest(resolved_map_path)
    if manifest.geo_ref_id is None:
        raise ValueError(f"{manifest.path}: manifest records no geo_ref_id")
    geo_ref_id = int(manifest.geo_ref_id)
    geo_transform = georef_source.load_pose_source(resolved_map_path).geo_transform
    annotations = load_annotations(
        resolved_map_path / "benchmark" / "annotations.geojson",
        DEFAULT_ACCURACY_M,
    )
    prompt_order = list(_prompt_annotations(annotations))
    if prompts is not None:
        requested = set(prompts)
        unknown = requested.difference(prompt_order)
        if unknown:
            raise ValueError(f"Unknown benchmark prompt(s): {sorted(unknown)!r}")
        prompt_order = [prompt for prompt in prompt_order if prompt in requested]

    result: dict[str, list[EnrichedCandidate]] = {}
    missing: list[tuple[str, Path]] = []
    for prompt in prompt_order:
        path = _cache_path(resolved_cache_dir, manifest.map_id, prompt, candidate_count)
        logger.info("Candidate cache for prompt %r: %s", prompt, path)
        if path.is_file() and not refresh:
            with path.open("rb") as stream:
                loaded = pickle.load(stream)  # noqa: S301 - trusted local cache
            if not isinstance(loaded, list):
                raise ValueError(f"Candidate cache {path} does not contain a list")
            result[prompt] = loaded
        else:
            missing.append((prompt, path))

    if missing:
        with db.connect() as conn:
            for prompt, path in missing:
                hits = service.query_by_text(
                    ann_base_url,
                    geo_ref_id,
                    prompt,
                    candidate_count,
                    timeout_s,
                )
                enriched = candidates.load_enriched_candidates(
                    conn,
                    geo_ref_id,
                    hits,
                    geo_transform,
                    with_embeddings=True,
                )
                with path.open("wb") as stream:
                    pickle.dump(enriched, stream, protocol=pickle.HIGHEST_PROTOCOL)
                result[prompt] = enriched

    return {prompt: result[prompt] for prompt in prompt_order}


def _threshold_candidates(
    predictions_by_prompt: Mapping[str, Sequence[Prediction]],
) -> list[float]:
    scores = {
        prediction.score
        for predictions in predictions_by_prompt.values()
        for prediction in predictions
    }
    return sorted(scores, reverse=True) or [0.0]


def _prompt_f1_table(
    predictions_by_prompt: Mapping[str, list[Prediction]],
    evaluator: PromptEvaluator,
) -> tuple[list[float], dict[str, list[float]]]:
    """Build F1 values at every global threshold without repeating accepted sets."""
    thresholds = _threshold_candidates(predictions_by_prompt)
    by_prompt: dict[str, list[float]] = {}
    for prompt, predictions in predictions_by_prompt.items():
        local_thresholds = sorted(
            {prediction.score for prediction in predictions}, reverse=True
        )
        local_f1 = {
            threshold: evaluator(
                prompt,
                predictions,
                math.nextafter(threshold, -math.inf),
            )
            for threshold in local_thresholds
        }
        values: list[float] = []
        local_index = 0
        current_f1 = 0.0
        for threshold in thresholds:
            while (
                local_index < len(local_thresholds)
                and local_thresholds[local_index] >= threshold
            ):
                current_f1 = local_f1[local_thresholds[local_index]]
                local_index += 1
            values.append(current_f1)
        by_prompt[prompt] = values
    return thresholds, by_prompt


def _fit_shared_threshold(
    prompt_names: Sequence[str],
    thresholds: Sequence[float],
    f1_by_prompt: Mapping[str, Sequence[float]],
) -> tuple[float, float, int]:
    if not prompt_names:
        return 0.0, 0.0, 0
    best_f1 = -1.0
    best_threshold = 0.0
    best_index = 0
    for index, threshold in enumerate(thresholds):
        macro_f1 = statistics.fmean(
            f1_by_prompt[prompt][index] for prompt in prompt_names
        )
        if (macro_f1, threshold) > (best_f1, best_threshold):
            best_f1 = macro_f1
            best_threshold = threshold
            best_index = index
    return best_f1, best_threshold, best_index


def shared_threshold_metrics(
    predictions_by_prompt: Mapping[str, list[Prediction]],
    annotations_by_prompt: Mapping[str, list[Annotation]],
    *,
    grouped: bool,
    group_radius_m: float,
) -> ThresholdMetrics:
    """Fit one acceptance threshold across prompts and compute prompt-wise LOO F1.

    Threshold values use the curve convention (score greater than or equal to the
    reported value), while the reused benchmark evaluators accept strictly greater
    than their argument. ``nextafter`` bridges those two exact semantics.

    Args:
        predictions_by_prompt: Predictions for every evaluated prompt.
        annotations_by_prompt: Ground truth for the same prompt keys.
        grouped: Evaluate grouped targets when true, strict annotations otherwise.
        group_radius_m: Single-linkage radius used for grouped targets.

    Returns:
        Fitted macro F1, its shared threshold, and held-out macro F1.
    """
    prompt_names = list(annotations_by_prompt)
    groups_by_prompt: dict[str, list[AnnotationGroup]] = {
        prompt: group_annotations(annotations, group_radius_m)
        for prompt, annotations in annotations_by_prompt.items()
    }

    def evaluator(
        prompt: str, predictions: list[Prediction], acceptance_threshold: float
    ) -> float:
        prompt_annotations = annotations_by_prompt[prompt]
        class_name = _class_name(prompt_annotations)
        if grouped:
            return evaluate_prompt_grouped(
                class_name=class_name,
                prompt=prompt,
                annotation_groups=groups_by_prompt[prompt],
                predictions=predictions,
                acceptance_threshold=acceptance_threshold,
            ).f1
        return evaluate_prompt(
            class_name=class_name,
            prompt=prompt,
            class_annotations=prompt_annotations,
            predictions=predictions,
            acceptance_threshold=acceptance_threshold,
        ).f1

    thresholds, f1_by_prompt = _prompt_f1_table(predictions_by_prompt, evaluator)
    macro_f1, threshold, threshold_index = _fit_shared_threshold(
        prompt_names, thresholds, f1_by_prompt
    )
    held_out_f1: list[float] = []
    for held_out in prompt_names:
        training = [prompt for prompt in prompt_names if prompt != held_out]
        if not training:
            held_out_index = threshold_index
        else:
            _, _, held_out_index = _fit_shared_threshold(
                training, thresholds, f1_by_prompt
            )
        held_out_f1.append(f1_by_prompt[held_out][held_out_index])
    return ThresholdMetrics(
        macro_f1=macro_f1,
        threshold=threshold,
        loo_macro_f1=statistics.fmean(held_out_f1) if held_out_f1 else 0.0,
    )


def evaluate_config(
    candidates_by_prompt: Mapping[str, list[EnrichedCandidate]],
    annotations: Sequence[Annotation],
    geo_transform: GeoTransform,
    params: LocalizationParams,
    *,
    group_radius_m: float,
    num_results: int,
) -> ConfigMetrics:
    """Evaluate one in-process localization configuration on all prompts.

    Args:
        candidates_by_prompt: Cached enriched candidates keyed by prompt.
        annotations: Benchmark annotations.
        geo_transform: Map coordinate transform used by localization.
        params: Association, filtering, and ranking parameters.
        group_radius_m: Radius for the grouped ground-truth view.
        num_results: Maximum returned clusters, applied consistently to the params.

    Returns:
        Strict/grouped metrics, shared-threshold estimates, and granularity controls.
    """
    effective_params = replace(params, num_results=num_results)
    annotations_by_prompt = _prompt_annotations(annotations)
    if set(candidates_by_prompt) != set(annotations_by_prompt):
        raise ValueError("Candidate prompts must exactly match annotation prompts")

    predictions_by_prompt: dict[str, list[Prediction]] = {}
    details: dict[str, PromptDetail] = {}
    observation_counts: list[float] = []
    spreads: list[float] = []
    clusters_per_prompt: list[int] = []
    strict_aps: list[float] = []
    grouped_aps: list[float] = []

    for prompt, prompt_annotations in annotations_by_prompt.items():
        localizations = localize_from_enriched_candidates(
            candidates_by_prompt[prompt], geo_transform, effective_params
        )
        predictions = parse_predictions({"localizations": localizations}, "match_score")
        predictions_by_prompt[prompt] = predictions
        groups = group_annotations(prompt_annotations, group_radius_m)
        strict_curve = precision_recall_curve(
            predictions, targets_from_annotations(prompt_annotations)
        )
        grouped_curve = precision_recall_curve(predictions, targets_from_groups(groups))
        strict_aps.append(strict_curve.average_precision)
        grouped_aps.append(grouped_curve.average_precision)
        clusters_per_prompt.append(len(localizations))
        observation_counts.extend(
            float(item.get("observation_count", 0)) for item in localizations
        )
        spreads.extend(float(item.get("spread_m", 0.0)) for item in localizations)
        details[prompt] = PromptDetail(
            class_name=_class_name(prompt_annotations),
            strict_ap=strict_curve.average_precision,
            grouped_ap=grouped_curve.average_precision,
            cluster_count=len(localizations),
        )

    strict = shared_threshold_metrics(
        predictions_by_prompt,
        annotations_by_prompt,
        grouped=False,
        group_radius_m=group_radius_m,
    )
    grouped_metrics = shared_threshold_metrics(
        predictions_by_prompt,
        annotations_by_prompt,
        grouped=True,
        group_radius_m=group_radius_m,
    )
    return ConfigMetrics(
        label="",
        params=asdict(effective_params),
        map_strict=statistics.fmean(strict_aps) if strict_aps else 0.0,
        map_grouped=statistics.fmean(grouped_aps) if grouped_aps else 0.0,
        macro_f1_strict=strict.macro_f1,
        best_threshold_strict=strict.threshold,
        loo_macro_f1_strict=strict.loo_macro_f1,
        macro_f1_grouped=grouped_metrics.macro_f1,
        best_threshold_grouped=grouped_metrics.threshold,
        loo_macro_f1_grouped=grouped_metrics.loo_macro_f1,
        total_clusters=sum(clusters_per_prompt),
        median_observation_count=(
            float(statistics.median(observation_counts)) if observation_counts else 0.0
        ),
        median_spread_m=float(statistics.median(spreads)) if spreads else 0.0,
        median_clusters_per_prompt=(
            float(statistics.median(clusters_per_prompt))
            if clusters_per_prompt
            else 0.0
        ),
        per_prompt=details,
    )


def _params_from_grid_entry(
    entry: Mapping[str, Any], base_params: LocalizationParams
) -> tuple[str, LocalizationParams]:
    allowed = {field.name for field in fields(LocalizationParams)}
    unknown = set(entry).difference(allowed | {"label"})
    if unknown:
        raise ValueError(f"Unknown LocalizationParams grid key(s): {sorted(unknown)!r}")
    label_value = entry.get("label", "")
    if not isinstance(label_value, str):
        raise ValueError("Grid entry 'label' must be a string")
    overrides = {key: value for key, value in entry.items() if key != "label"}
    return label_value, replace(base_params, **overrides)


def _csv_row(metrics: ConfigMetrics) -> dict[str, Any]:
    row = asdict(metrics)
    row["params"] = json.dumps(row["params"], ensure_ascii=False, sort_keys=True)
    row.pop("per_prompt")
    return row


def _write_sweep_outputs(results: Sequence[ConfigMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_csv_row(result) for result in results]
    write_csv(out_dir / CSV_FILENAME, rows)
    (out_dir / JSON_FILENAME).write_text(
        json.dumps(
            [asdict(result) for result in results], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def sweep(
    grid: Sequence[Mapping[str, Any]],
    candidates_by_prompt: Mapping[str, list[EnrichedCandidate]],
    annotations: Sequence[Annotation],
    geo_transform: GeoTransform,
    *,
    group_radius_m: float,
    num_results: int,
    out_dir: str | Path,
    base_params: LocalizationParams | None = None,
) -> list[ConfigMetrics]:
    """Evaluate a parameter grid and write one summary CSV plus detailed JSON.

    Args:
        grid: Localization parameter overrides with an optional free-form label.
        candidates_by_prompt: Cached enriched candidates keyed by prompt.
        annotations: Benchmark annotations.
        geo_transform: Map coordinate transform.
        group_radius_m: Radius for grouped annotation evaluation.
        num_results: Default result cap for every configuration.
        out_dir: Output directory for CSV and JSON artifacts.
        base_params: Base parameters before grid overrides.

    Returns:
        Metrics in grid order.
    """
    base = base_params or LocalizationParams()
    results: list[ConfigMetrics] = []
    for index, entry in enumerate(grid, start=1):
        label, params = _params_from_grid_entry(entry, base)
        started = time.perf_counter()
        metrics = evaluate_config(
            candidates_by_prompt,
            annotations,
            geo_transform,
            params,
            group_radius_m=group_radius_m,
            num_results=int(entry.get("num_results", num_results)),
        )
        elapsed = time.perf_counter() - started
        metrics = replace(metrics, label=label, elapsed_s=elapsed)
        results.append(metrics)
        print(
            f"[{index}/{len(grid)}] {label or '(unlabelled)'}: "
            f"{elapsed:.3f}s mAP={metrics.map_strict:.3f} "
            f"F1={metrics.macro_f1_strict:.3f}",
            flush=True,
        )
    _write_sweep_outputs(results, Path(out_dir).expanduser().resolve())
    return results


def verify_against_http(
    candidates_by_prompt: Mapping[str, list[EnrichedCandidate]],
    geo_transform: GeoTransform,
    params: LocalizationParams,
    *,
    map_id: str,
    bricks_base_url: str,
    timeout_s: float,
) -> float:
    """Compare offline localizations with the live bricks endpoint exactly.

    Args:
        candidates_by_prompt: Cached candidates used by the offline path.
        geo_transform: Map coordinate transform.
        params: Parameters sent to both implementations.
        map_id: Map identifier in the bricks service route.
        bricks_base_url: Base URL of the running bricks service.
        timeout_s: HTTP request timeout.

    Returns:
        Worst absolute coordinate or match-score deviation.

    Raises:
        RuntimeError: If counts differ or any deviation exceeds ``1e-9``.
    """
    worst = 0.0
    params_payload = asdict(params)
    for prompt, prompt_candidates in candidates_by_prompt.items():
        offline = localize_from_enriched_candidates(
            prompt_candidates, geo_transform, params
        )
        response = post_json(
            f"{bricks_base_url.rstrip('/')}/{map_id}/object-search/localize",
            {"text": prompt, "search_type": "object", **params_payload},
            timeout_s,
        )
        live_raw = response.get("localizations")
        if not isinstance(live_raw, list):
            raise RuntimeError(
                f"HTTP response for {prompt!r} has no localizations list"
            )
        if len(offline) != len(live_raw):
            raise RuntimeError(
                f"Verification failed for {prompt!r}: offline returned {len(offline)} "
                f"clusters, HTTP returned {len(live_raw)}"
            )
        for index, (offline_item, live_value) in enumerate(zip(offline, live_raw)):
            if not isinstance(live_value, dict):
                raise RuntimeError(
                    f"Verification failed for {prompt!r} cluster {index}: not an object"
                )
            offline_coordinates = offline_item.get("coordinates")
            live_coordinates = live_value.get("coordinates")
            if not isinstance(offline_coordinates, list) or not isinstance(
                live_coordinates, list
            ):
                raise RuntimeError(
                    f"Verification failed for {prompt!r} cluster {index}: "
                    "missing coordinates"
                )
            if len(offline_coordinates) != len(live_coordinates):
                raise RuntimeError(
                    f"Verification failed for {prompt!r} cluster {index}: "
                    "coordinate dimensions differ"
                )
            deviations = [
                abs(float(left) - float(right))
                for left, right in zip(offline_coordinates, live_coordinates)
            ]
            deviations.append(
                abs(
                    float(offline_item["match_score"])
                    - float(live_value["match_score"])
                )
            )
            if not all(math.isfinite(deviation) for deviation in deviations):
                raise RuntimeError(
                    f"Verification failed for {prompt!r} cluster {index}: "
                    "non-finite deviation"
                )
            cluster_worst = max(deviations, default=0.0)
            worst = max(worst, cluster_worst)
            if cluster_worst > VERIFY_TOLERANCE:
                raise RuntimeError(
                    f"Verification failed for {prompt!r} cluster {index}: "
                    f"deviation {cluster_worst:.17g} exceeds {VERIFY_TOLERANCE}"
                )
    print(f"Verification passed; worst deviation={worst:.17g}", flush=True)
    return worst


def _load_grid(path: Path) -> list[dict[str, Any]]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or any(not isinstance(entry, dict) for entry in raw):
        raise ValueError("Grid JSON must be a list of objects")
    return [dict(entry) for entry in raw]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse association sweep command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--ann-base-url", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--num-results", type=int, default=400)
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--group-annotation-radius-m", type=float, default=2.0)
    parser.add_argument("--default-accuracy", type=float, default=DEFAULT_ACCURACY_M)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bricks-base-url", default="http://127.0.0.1:45679")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run candidate fetching, optional verification, and the requested sweep."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    annotations = load_annotations(
        map_path / "benchmark" / "annotations.geojson", args.default_accuracy
    )
    prompt_candidates = fetch_prompt_candidates(
        map_path,
        args.ann_base_url,
        args.candidate_count,
        args.cache_dir,
        refresh=args.refresh,
        timeout_s=args.timeout,
    )
    pose_source = georef_source.load_pose_source(map_path)
    base_params = replace(
        LocalizationParams(),
        candidate_count=args.candidate_count,
        num_results=args.num_results,
        min_similarity=args.min_similarity,
    )
    if args.verify:
        verify_against_http(
            prompt_candidates,
            pose_source.geo_transform,
            base_params,
            map_id=map_manifest.load_map_manifest(map_path).map_id,
            bricks_base_url=args.bricks_base_url,
            timeout_s=args.timeout,
        )
    sweep(
        _load_grid(args.grid),
        prompt_candidates,
        annotations,
        pose_source.geo_transform,
        group_radius_m=args.group_annotation_radius_m,
        num_results=args.num_results,
        out_dir=args.out_dir,
        base_params=base_params,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
