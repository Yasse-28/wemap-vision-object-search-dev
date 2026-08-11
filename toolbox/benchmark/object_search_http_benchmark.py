#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://localhost:45678"
DEFAULT_API_STYLE = "standalone"
DEFAULT_REMOTE_LOCALIZE_URL_TEMPLATE = "https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize"
TP_COLOR = "#22c55e"
FP_COLOR = "#ef4444"
FN_COLOR = "#9ca3af"
REFERENCE_COLOR = "#2563eb"


@dataclass(frozen=True)
class Annotation:
    id: str
    class_name: str
    lat: float
    lng: float
    accuracy_m: float
    prompt: str = ""  # properties.prompt if set, otherwise class_name
    level: str | None = None


@dataclass(frozen=True)
class AnnotationGroup:
    id: str
    class_name: str
    prompt: str
    lat: float
    lng: float
    match_radius_m: float
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True)
class Prediction:
    id: str
    lat: float
    lng: float
    score: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    prediction_id: str
    annotation_id: str
    distance_m: float
    score: float


@dataclass(frozen=True)
class Target:
    """One thing that ought to be found: a lone annotation, or a group of them.

    Carries every member point because a grouped target is matched on the distance to
    its *nearest* member, not to the group's barycentre — a bank of screens is found
    when a cluster lands on any of its screens.
    """

    id: str
    points: tuple[tuple[float, float], ...]
    radius_m: float


@dataclass(frozen=True)
class CurvePoint:
    """Confusion counts when accepting every prediction scoring >= `threshold`."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CurveMetrics:
    average_precision: float
    best_f1: float
    best_f1_threshold: float
    best_f1_precision: float
    best_f1_recall: float
    points: list[CurvePoint] = field(default_factory=list)


EMPTY_CURVE = CurveMetrics(0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class PromptMetrics:
    class_name: str
    prompt: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accepted_predictions: int
    rejected_predictions: int
    ground_truth: int
    matches: list[Match] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    # Threshold-free summaries of the same predictions — see `precision_recall_curve`.
    # They are what two runs should be compared on: a change that shifts the score
    # distribution (the review-feedback boost does exactly that) moves the fixed
    # threshold's P/R/F1 without the ranking having improved or worsened.
    average_precision: float = 0.0
    best_f1: float = 0.0
    best_f1_threshold: float = 0.0
    best_f1_precision: float = 0.0
    best_f1_recall: float = 0.0


def _polygon_centroid(coordinates: Any) -> tuple[float, float] | None:
    if not coordinates or not isinstance(coordinates, list):
        return None
    ring = coordinates[0] if coordinates and isinstance(coordinates[0], list) else None
    if not ring:
        return None
    points = [
        (float(point[0]), float(point[1]))
        for point in ring
        if isinstance(point, list) and len(point) >= 2
    ]
    if not points:
        return None
    return (
        sum(lng for lng, _lat in points) / len(points),
        sum(lat for _lng, lat in points) / len(points),
    )


def load_annotations(path: Path, default_accuracy_m: float) -> list[Annotation]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features")
    if data.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError(f"{path} must be a GeoJSON FeatureCollection")

    annotations: list[Annotation] = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")

        lng_lat: tuple[float, float] | None = None
        if (
            geometry_type == "Point"
            and isinstance(coordinates, list)
            and len(coordinates) >= 2
        ):
            lng_lat = (float(coordinates[0]), float(coordinates[1]))
        elif geometry_type == "Polygon":
            lng_lat = _polygon_centroid(coordinates)

        class_name = properties.get("class") or properties.get("name")
        if lng_lat is None or not class_name:
            continue

        accuracy = properties.get("accuracy", default_accuracy_m)
        try:
            accuracy_m = float(accuracy)
        except (TypeError, ValueError):
            accuracy_m = default_accuracy_m

        lng, lat = lng_lat
        annotation_prompt = str(properties.get("prompt") or class_name)
        annotations.append(
            Annotation(
                id=str(feature.get("id") or idx),
                class_name=str(class_name),
                lat=lat,
                lng=lng,
                accuracy_m=accuracy_m,
                prompt=annotation_prompt,
                level=(
                    str(properties["level"])
                    if properties.get("level") is not None
                    else None
                ),
            )
        )
    return annotations


def build_localize_url(base_url: str, map_id: str, api_style: str, online: bool) -> str:
    base = base_url.rstrip("/")
    if api_style == "standalone":
        suffix = "localize" if online else "localize-offline"
        return f"{base}/{map_id}/object-search/{suffix}"
    if online:
        raise ValueError("--online is only supported with --api-style standalone")
    return f"{base}/{map_id}/geopose/object-search/text/localize"


def resolve_localize_url(args: argparse.Namespace) -> str:
    if args.localize_url:
        return str(args.localize_url).format(map_id=args.map_id)
    return build_localize_url(args.base_url, args.map_id, args.api_style, args.online)


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Request to {url} failed: {exc.reason}. "
            "Check --base-url/--api-style or --localize-url. The sandbox "
            "object-search service usually runs on --base-url http://localhost:45678 "
            "--api-style standalone."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Response from {url} was not JSON: {raw[:500]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Response from {url} must be a JSON object")
    return data


def _prediction_score(item: dict[str, Any], score_field: str) -> float:
    value = item.get(score_field)
    if value is None and score_field != "match_score":
        value = item.get("match_score")
    if value is None:
        value = item.get("similarity_score", item.get("confidence", 0.0))
    return float(value)


def parse_predictions(response: dict[str, Any], score_field: str) -> list[Prediction]:
    localizations = response.get("localizations", [])
    if not isinstance(localizations, list):
        raise RuntimeError("Response field 'localizations' must be a list")

    predictions: list[Prediction] = []
    for idx, item in enumerate(localizations):
        if not isinstance(item, dict):
            continue
        lat = item.get("lat")
        lng = item.get("lng", item.get("lon"))
        if lat is None or lng is None:
            # Standalone /localize and /localize-offline return
            # "coordinates": [lat, lng, alt] instead of lat/lng fields.
            coordinates = item.get("coordinates")
            if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
                lat, lng = coordinates[0], coordinates[1]
        if lat is None or lng is None:
            continue
        try:
            score = _prediction_score(item, score_field)
            prediction_id = item.get("cluster_id")
            if prediction_id is None:
                prediction_id = item.get("id")
            if prediction_id is None:
                prediction_id = idx
            predictions.append(
                Prediction(
                    id=str(prediction_id),
                    lat=float(lat),
                    lng=float(lng),
                    score=score,
                    raw=item,
                )
            )
        except (TypeError, ValueError):
            continue
    return predictions


def prediction_level(prediction: Prediction) -> Any:
    if "level" in prediction.raw:
        return prediction.raw.get("level")
    return prediction.raw.get("cluster_level")


def load_cluster_levels_from_artifact(map_path: Path) -> tuple[Any, Any] | None:
    artifact_path = map_path / "object-search.npz"
    if not artifact_path.is_file():
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        bundle = np.load(artifact_path, allow_pickle=False)
    except OSError:
        return None
    if "object_cluster_ids" not in bundle or "cluster_levels" not in bundle:
        return None
    return bundle["object_cluster_ids"], bundle["cluster_levels"]


def enrich_prediction_levels_from_artifact(
    predictions: list[Prediction],
    cluster_level_arrays: tuple[Any, Any] | None,
) -> None:
    if cluster_level_arrays is None:
        return
    object_cluster_ids, cluster_levels = cluster_level_arrays
    for prediction in predictions:
        if prediction_level(prediction) is not None:
            continue
        observations = prediction.raw.get("observations", [])
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, dict) or "object_idx" not in observation:
                continue
            try:
                object_idx = int(observation["object_idx"])
            except (TypeError, ValueError):
                continue
            if object_idx < 0 or object_idx >= len(object_cluster_ids):
                continue
            cluster_id = int(object_cluster_ids[object_idx])
            if cluster_id < 0 or cluster_id >= len(cluster_levels):
                continue
            level = int(cluster_levels[cluster_id])
            if level < 0:
                continue
            prediction.raw["level"] = level
            prediction.raw["level_source"] = "object-search.npz"
            prediction.raw["artifact_cluster_id"] = cluster_id
            break


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def match_predictions(
    predictions: list[Prediction],
    annotations: list[Annotation],
) -> list[Match]:
    candidates: list[tuple[float, Prediction, Annotation]] = []
    for prediction in predictions:
        for annotation in annotations:
            distance_m = haversine_m(
                prediction.lat,
                prediction.lng,
                annotation.lat,
                annotation.lng,
            )
            if distance_m <= annotation.accuracy_m:
                candidates.append((distance_m, prediction, annotation))

    candidates.sort(key=lambda item: (item[0], -item[1].score))
    used_predictions: set[str] = set()
    used_annotations: set[str] = set()
    matches: list[Match] = []
    for distance_m, prediction, annotation in candidates:
        if prediction.id in used_predictions or annotation.id in used_annotations:
            continue
        used_predictions.add(prediction.id)
        used_annotations.add(annotation.id)
        matches.append(
            Match(
                prediction_id=prediction.id,
                annotation_id=annotation.id,
                distance_m=distance_m,
                score=prediction.score,
            )
        )
    return matches


def group_annotations(
    annotations: list[Annotation],
    radius_m: float,
) -> list[AnnotationGroup]:
    """Group nearby annotations of the same class with single-linkage distance."""
    if radius_m <= 0:
        return [
            AnnotationGroup(
                id=f"group-{annotation.id}",
                class_name=annotation.class_name,
                prompt=annotation.prompt,
                lat=annotation.lat,
                lng=annotation.lng,
                match_radius_m=annotation.accuracy_m,
                annotations=(annotation,),
            )
            for annotation in annotations
        ]

    parents = list(range(len(annotations)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(annotations):
        for right_index in range(left_index + 1, len(annotations)):
            right = annotations[right_index]
            if left.class_name != right.class_name:
                continue
            if haversine_m(left.lat, left.lng, right.lat, right.lng) <= radius_m:
                union(left_index, right_index)

    by_root: dict[int, list[Annotation]] = {}
    for index, annotation in enumerate(annotations):
        by_root.setdefault(find(index), []).append(annotation)

    groups: list[AnnotationGroup] = []
    for group_index, members in enumerate(by_root.values(), start=1):
        lat = sum(annotation.lat for annotation in members) / len(members)
        lng = sum(annotation.lng for annotation in members) / len(members)
        match_radius_m = max(
            max(annotation.accuracy_m for annotation in members), radius_m
        )
        first = members[0]
        group_id = first.id if len(members) == 1 else f"group-{group_index:03d}"
        groups.append(
            AnnotationGroup(
                id=group_id,
                class_name=first.class_name,
                prompt=first.prompt,
                lat=lat,
                lng=lng,
                match_radius_m=match_radius_m,
                annotations=tuple(members),
            )
        )
    return groups


def match_predictions_to_groups(
    predictions: list[Prediction],
    groups: list[AnnotationGroup],
) -> list[Match]:
    candidates: list[tuple[float, Prediction, AnnotationGroup]] = []
    for prediction in predictions:
        for group in groups:
            distances = [
                haversine_m(
                    prediction.lat, prediction.lng, annotation.lat, annotation.lng
                )
                for annotation in group.annotations
            ]
            if not distances:
                continue
            distance_m = min(distances)
            if distance_m <= group.match_radius_m:
                candidates.append((distance_m, prediction, group))

    candidates.sort(key=lambda item: (item[0], -item[1].score))
    used_predictions: set[str] = set()
    used_groups: set[str] = set()
    matches: list[Match] = []
    for distance_m, prediction, group in candidates:
        if prediction.id in used_predictions or group.id in used_groups:
            continue
        used_predictions.add(prediction.id)
        used_groups.add(group.id)
        matches.append(
            Match(
                prediction_id=prediction.id,
                annotation_id=group.id,
                distance_m=distance_m,
                score=prediction.score,
            )
        )
    return matches


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_prompt(
    *,
    class_name: str,
    prompt: str,
    class_annotations: list[Annotation],
    predictions: list[Prediction],
    acceptance_threshold: float,
) -> PromptMetrics:
    accepted = [item for item in predictions if item.score > acceptance_threshold]
    rejected_count = len(predictions) - len(accepted)
    matches = match_predictions(accepted, class_annotations)
    tp = len(matches)
    fp = len(accepted) - tp
    fn = len(class_annotations) - tp
    precision, recall, f1 = compute_prf(tp, fp, fn)
    return PromptMetrics(
        class_name=class_name,
        prompt=prompt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accepted_predictions=len(accepted),
        rejected_predictions=rejected_count,
        ground_truth=len(class_annotations),
        matches=matches,
    )


def evaluate_prompt_grouped(
    *,
    class_name: str,
    prompt: str,
    annotation_groups: list[AnnotationGroup],
    predictions: list[Prediction],
    acceptance_threshold: float,
) -> PromptMetrics:
    accepted = [item for item in predictions if item.score > acceptance_threshold]
    rejected_count = len(predictions) - len(accepted)
    matches = match_predictions_to_groups(accepted, annotation_groups)
    tp = len(matches)
    fp = len(accepted) - tp
    fn = len(annotation_groups) - tp
    precision, recall, f1 = compute_prf(tp, fp, fn)
    return PromptMetrics(
        class_name=class_name,
        prompt=prompt,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accepted_predictions=len(accepted),
        rejected_predictions=rejected_count,
        ground_truth=len(annotation_groups),
        matches=matches,
    )


def targets_from_annotations(annotations: list[Annotation]) -> list[Target]:
    return [
        Target(
            annotation.id,
            ((annotation.lat, annotation.lng),),
            annotation.accuracy_m,
        )
        for annotation in annotations
    ]


def targets_from_groups(groups: list[AnnotationGroup]) -> list[Target]:
    return [
        Target(
            group.id,
            tuple((a.lat, a.lng) for a in group.annotations),
            group.match_radius_m,
        )
        for group in groups
    ]


def _distance_to_target(prediction: Prediction, target: Target) -> float:
    return min(
        haversine_m(prediction.lat, prediction.lng, lat, lng)
        for lat, lng in target.points
    )


def rank_predictions_against_targets(
    predictions: list[Prediction], targets: list[Target]
) -> list[tuple[Prediction, bool]]:
    """Predictions by descending score, each flagged true positive or not.

    Matching is **score-ordered**: the best-scoring prediction picks its nearest free
    target, and so on down. `match_predictions` instead sorts every admissible pair by
    distance, which minimises total distance but is unusable for a curve — lowering the
    threshold admits a new prediction that can *steal* an already-matched target, so
    recall would not be monotone in the threshold.

    Score order is also the detection-benchmark convention (PASCAL VOC, COCO), which
    is what makes the resulting AP comparable to numbers from anywhere else.
    """
    used: set[str] = set()
    ranked: list[tuple[Prediction, bool]] = []
    for prediction in sorted(predictions, key=lambda item: -item.score):
        best_target: Target | None = None
        best_distance = float("inf")
        for target in targets:
            if target.id in used:
                continue
            distance_m = _distance_to_target(prediction, target)
            if distance_m <= target.radius_m and distance_m < best_distance:
                best_target, best_distance = target, distance_m
        if best_target is not None:
            used.add(best_target.id)
        ranked.append((prediction, best_target is not None))
    return ranked


def _average_precision(points: list[CurvePoint]) -> float:
    """All-point interpolated AP: sum of precision(interpolated) x recall increments.

    Interpolated — precision at recall r is the *best* precision achievable at recall
    >= r — because the raw curve saw-tooths on every false positive, and the raw
    average would then reward the accident of where those land.
    """
    interpolated: list[tuple[float, float]] = []
    running_max = 0.0
    for point in reversed(points):
        running_max = max(running_max, point.precision)
        interpolated.append((point.recall, running_max))
    interpolated.reverse()

    area = 0.0
    previous_recall = 0.0
    for recall, precision in interpolated:
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def precision_recall_curve(
    predictions: list[Prediction], targets: list[Target]
) -> CurveMetrics:
    """Sweep the acceptance threshold over the predictions' own scores.

    Each point answers "what if we accepted everything scoring at least this?", so the
    curve costs one matching pass, not one per threshold. `best_f1_threshold` is the
    score at the best point: accept predictions scoring **>= it** (the runner's
    `--acceptance-threshold` is a strict `>`, so pass a hair less).
    """
    if not predictions or not targets:
        return EMPTY_CURVE

    ranked = rank_predictions_against_targets(predictions, targets)
    total_targets = len(targets)
    points: list[CurvePoint] = []
    true_positives = false_positives = 0

    for index, (prediction, is_true_positive) in enumerate(ranked):
        true_positives += int(is_true_positive)
        false_positives += int(not is_true_positive)
        # One point per distinct score: a threshold cannot separate ties, so emitting a
        # point mid-run would describe a set no threshold can actually select.
        if index + 1 < len(ranked) and ranked[index + 1][0].score == prediction.score:
            continue
        false_negatives = total_targets - true_positives
        precision, recall, f1 = compute_prf(
            true_positives, false_positives, false_negatives
        )
        points.append(
            CurvePoint(
                threshold=float(prediction.score),
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    best = max(points, key=lambda point: (point.f1, point.threshold))
    return CurveMetrics(
        average_precision=_average_precision(points),
        best_f1=best.f1,
        best_f1_threshold=best.threshold,
        best_f1_precision=best.precision,
        best_f1_recall=best.recall,
        points=points,
    )


def attach_curve(metrics: PromptMetrics, curve: CurveMetrics) -> None:
    metrics.average_precision = curve.average_precision
    metrics.best_f1 = curve.best_f1
    metrics.best_f1_threshold = curve.best_f1_threshold
    metrics.best_f1_precision = curve.best_f1_precision
    metrics.best_f1_recall = curve.best_f1_recall


def _slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    slug = slug.strip("-._")
    if not slug:
        slug = "prompt"
    return slug[:max_length].strip("-._") or "prompt"


def prompt_geojson_path(
    output_dir: Path, class_name: str, prompt: str, prompt_index: int
) -> Path:
    class_slug = _slugify(class_name, max_length=60)
    prompt_slug = _slugify(prompt, max_length=90)
    return output_dir / f"{class_slug}__{prompt_index:02d}__{prompt_slug}.geojson"


def _point_feature(
    *,
    feature_id: str,
    lat: float,
    lng: float,
    color: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Point",
            "coordinates": [lng, lat],
        },
        "properties": {
            **properties,
            "marker-color": color,
        },
    }


def build_prompt_geojson(
    *,
    class_name: str,
    prompt: str,
    annotations: list[Annotation],
    predictions: list[Prediction],
    metrics: PromptMetrics,
    acceptance_threshold: float,
) -> dict[str, Any]:
    matched_prediction_ids = {match.prediction_id for match in metrics.matches}
    matched_annotation_ids = {match.annotation_id for match in metrics.matches}
    match_by_prediction = {match.prediction_id: match for match in metrics.matches}
    match_by_annotation = {match.annotation_id: match for match in metrics.matches}
    accepted_predictions = [
        prediction
        for prediction in predictions
        if prediction.score > acceptance_threshold
    ]

    features: list[dict[str, Any]] = []
    for prediction in accepted_predictions:
        is_tp = prediction.id in matched_prediction_ids
        match = match_by_prediction.get(prediction.id)
        features.append(
            _point_feature(
                feature_id=f"prediction-{prediction.id}",
                lat=prediction.lat,
                lng=prediction.lng,
                color=TP_COLOR if is_tp else FP_COLOR,
                properties={
                    "role": "prediction",
                    "status": "TP" if is_tp else "FP",
                    "class": class_name,
                    "prompt": prompt,
                    "prediction_id": prediction.id,
                    "level": prediction_level(prediction),
                    "score": prediction.score,
                    "matched_annotation_id": match.annotation_id if match else None,
                    "distance_m": match.distance_m if match else None,
                },
            )
        )

    for annotation in annotations:
        is_matched = annotation.id in matched_annotation_ids
        match = match_by_annotation.get(annotation.id)
        features.append(
            _point_feature(
                feature_id=f"annotation-{annotation.id}",
                lat=annotation.lat,
                lng=annotation.lng,
                color=REFERENCE_COLOR if is_matched else FN_COLOR,
                properties={
                    "role": "annotation",
                    "status": "reference" if is_matched else "FN",
                    "class": class_name,
                    "prompt": prompt,
                    "annotation_id": annotation.id,
                    "accuracy_m": annotation.accuracy_m,
                    "level": annotation.level,
                    "matched_prediction_id": match.prediction_id if match else None,
                    "distance_m": match.distance_m if match else None,
                },
            )
        )

    return {
        "type": "FeatureCollection",
        "name": f"{class_name} - {prompt}",
        "properties": {
            "class": class_name,
            "prompt": prompt,
            "legend": {
                "TP": TP_COLOR,
                "FP": FP_COLOR,
                "FN": FN_COLOR,
                "reference": REFERENCE_COLOR,
            },
            "metrics": {
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "accepted_predictions": metrics.accepted_predictions,
                "rejected_predictions": metrics.rejected_predictions,
                "ground_truth": metrics.ground_truth,
            },
        },
        "features": features,
    }


def write_prompt_geojson(path: Path, geojson: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _emit_progress(args: argparse.Namespace, event: dict[str, Any]) -> None:
    if getattr(args, "progress_json", False):
        print(json.dumps(event, ensure_ascii=False), flush=True)


def _normalize_prompt(prompt: str) -> str:
    # Keep aligned with toolbox.bricks.feedback.normalize_query; this script stays
    # standalone.
    return prompt.casefold().strip()


class PromptNotFoundError(ValueError):
    """Raised when an only-prompt filter has no benchmark ground truth."""


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_annotations(args.annotations, args.default_accuracy)
    grouping_enabled = args.group_annotation_radius_m > 0

    # Group annotations by their resolved prompt (insertion order preserved).
    prompt_to_annotations: dict[str, list[Annotation]] = {}
    for ann in annotations:
        prompt_to_annotations.setdefault(ann.prompt, []).append(ann)

    if args.only_prompt:
        requested_prompts = {_normalize_prompt(prompt) for prompt in args.only_prompt}
        prompt_to_annotations = {
            prompt: prompt_annotations
            for prompt, prompt_annotations in prompt_to_annotations.items()
            if _normalize_prompt(prompt) in requested_prompts
        }
        if not prompt_to_annotations:
            requested = ", ".join(repr(prompt) for prompt in args.only_prompt)
            raise PromptNotFoundError(
                f"No ground truth for prompt(s) {requested} in {args.annotations}"
            )

    url = resolve_localize_url(args)
    prompt_metrics: list[PromptMetrics] = []
    grouped_prompt_metrics: list[PromptMetrics] = []
    prompt_geojson_paths: list[str] = []
    raw_by_prompt: list[dict[str, Any]] = []
    cluster_level_arrays = load_cluster_levels_from_artifact(args.map_path)

    _emit_progress(
        args,
        {
            "event": "start",
            "prompt_count": len(prompt_to_annotations),
            "annotation_count": len(annotations),
            "output_dir": str(args.output_dir) if args.output_dir else None,
        },
    )

    for prompt_index, (prompt, prompt_annotations) in enumerate(
        prompt_to_annotations.items(), start=1
    ):
        # Take the most common class_name among annotations for this prompt.
        class_name = max(
            {a.class_name for a in prompt_annotations},
            key=lambda cn: sum(1 for a in prompt_annotations if a.class_name == cn),
        )
        annotation_groups = (
            group_annotations(prompt_annotations, args.group_annotation_radius_m)
            if grouping_enabled
            else []
        )

        payload: dict[str, Any] = {
            "text": prompt,
            "num_results": args.num_results,
            "search_type": "object",
            "min_similarity": args.min_similarity,
        }
        if args.online:
            payload.update(
                {
                    "candidate_count": args.candidate_count,
                    "clustering_eps_m": args.clustering_eps_m,
                }
            )
        if args.feedback_alpha != 0.0:
            payload["feedback_alpha"] = args.feedback_alpha
        if args.feedback_beta != 0.0:
            payload["feedback_beta"] = args.feedback_beta
        # Sent only alongside a non-zero gain: with both gains at zero the term is
        # never consulted, and an unconditional field would make the request differ
        # from the baseline's for no behavioural reason.
        if args.feedback_normalization != "none" and (
            args.feedback_alpha != 0.0 or args.feedback_beta != 0.0
        ):
            payload["feedback_normalization"] = args.feedback_normalization
        if args.min_keyframes_per_cluster is not None:
            payload["min_keyframes_per_cluster"] = args.min_keyframes_per_cluster
        if args.max_observations_per_cluster is not None:
            payload["max_observations_per_cluster"] = args.max_observations_per_cluster
        if args.min_observations_per_cluster is not None:
            payload["min_observations_per_cluster"] = args.min_observations_per_cluster
        if args.max_cluster_spread_m is not None:
            payload["max_cluster_spread_m"] = args.max_cluster_spread_m
        if args.semantic_gate_threshold is not None:
            payload["semantic_gate_threshold"] = args.semantic_gate_threshold
        if args.association != "leader_canopy":
            payload.update(
                {
                    "association": args.association,
                    "combination": args.combination,
                    "association_sim_threshold": args.association_sim_threshold,
                    "descriptor": args.descriptor,
                }
            )

        started = time.perf_counter()
        try:
            response = post_json(url, payload, args.timeout)
            predictions = parse_predictions(response, args.score_field)
            enrich_prediction_levels_from_artifact(predictions, cluster_level_arrays)
            metrics = evaluate_prompt(
                class_name=class_name,
                prompt=prompt,
                class_annotations=prompt_annotations,
                predictions=predictions,
                acceptance_threshold=args.acceptance_threshold,
            )
            # Threshold-free, on the *whole* prediction list — deliberately not on the
            # accepted subset, since the sweep is what replaces the acceptance step.
            strict_curve = precision_recall_curve(
                predictions, targets_from_annotations(prompt_annotations)
            )
            attach_curve(metrics, strict_curve)
            grouped_metrics = (
                evaluate_prompt_grouped(
                    class_name=class_name,
                    prompt=prompt,
                    annotation_groups=annotation_groups,
                    predictions=predictions,
                    acceptance_threshold=args.acceptance_threshold,
                )
                if grouping_enabled
                else None
            )
            grouped_curve = (
                precision_recall_curve(
                    predictions, targets_from_groups(annotation_groups)
                )
                if grouping_enabled
                else EMPTY_CURVE
            )
            if grouped_metrics is not None:
                attach_curve(grouped_metrics, grouped_curve)
        except (
            Exception
        ) as exc:  # Keep long benchmark runs from aborting on one prompt.
            predictions = []
            strict_curve = grouped_curve = EMPTY_CURVE
            precision, recall, f1 = compute_prf(0, 0, len(prompt_annotations))
            metrics = PromptMetrics(
                class_name=class_name,
                prompt=prompt,
                true_positives=0,
                false_positives=0,
                false_negatives=len(prompt_annotations),
                precision=precision,
                recall=recall,
                f1=f1,
                accepted_predictions=0,
                rejected_predictions=0,
                ground_truth=len(prompt_annotations),
                error=str(exc),
            )
            if grouping_enabled:
                grouped_precision, grouped_recall, grouped_f1 = compute_prf(
                    0,
                    0,
                    len(annotation_groups),
                )
                grouped_metrics = PromptMetrics(
                    class_name=class_name,
                    prompt=prompt,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=len(annotation_groups),
                    precision=grouped_precision,
                    recall=grouped_recall,
                    f1=grouped_f1,
                    accepted_predictions=0,
                    rejected_predictions=0,
                    ground_truth=len(annotation_groups),
                    error=str(exc),
                )
            else:
                grouped_metrics = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics.elapsed_ms = elapsed_ms
        prompt_metrics.append(metrics)
        raw_prompt: dict[str, Any] = {
            "prompt": prompt,
            "class_name": class_name,
            "annotations": [asdict(annotation) for annotation in prompt_annotations],
            "predictions": [prediction.raw for prediction in predictions],
            "matches": [asdict(match) for match in metrics.matches],
            # The full sweep lives here rather than in `by_prompt`: it is one point per
            # distinct score, which would bloat metrics.json and the CSV for a series
            # nobody reads as a table.
            "precision_recall_curve": [asdict(p) for p in strict_curve.points],
            "error": metrics.error,
        }
        if grouped_metrics is not None:
            grouped_metrics.elapsed_ms = elapsed_ms
            grouped_prompt_metrics.append(grouped_metrics)
            raw_prompt["annotation_groups"] = [
                asdict(group) for group in annotation_groups
            ]
            raw_prompt["grouped_matches"] = [
                asdict(match) for match in grouped_metrics.matches
            ]
            raw_prompt["grouped_precision_recall_curve"] = [
                asdict(p) for p in grouped_curve.points
            ]
        raw_by_prompt.append(raw_prompt)
        if not args.no_prompt_geojson:
            geojson_path = prompt_geojson_path(
                args.prompt_geojson_dir,
                class_name,
                prompt,
                prompt_index,
            )
            write_prompt_geojson(
                geojson_path,
                build_prompt_geojson(
                    class_name=class_name,
                    prompt=prompt,
                    annotations=prompt_annotations,
                    predictions=predictions,
                    metrics=metrics,
                    acceptance_threshold=args.acceptance_threshold,
                ),
            )
            prompt_geojson_paths.append(str(geojson_path))
        if args.progress_json:
            _emit_progress(
                args,
                {
                    "event": "prompt",
                    "index": prompt_index,
                    "total": len(prompt_to_annotations),
                    "class_name": class_name,
                    "prompt": prompt,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "true_positives": metrics.true_positives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives,
                    "elapsed_ms": metrics.elapsed_ms,
                    "error": metrics.error,
                    **(
                        {
                            "grouped_precision": grouped_metrics.precision,
                            "grouped_recall": grouped_metrics.recall,
                            "grouped_f1": grouped_metrics.f1,
                            "grouped_true_positives": grouped_metrics.true_positives,
                            "grouped_false_positives": grouped_metrics.false_positives,
                            "grouped_false_negatives": grouped_metrics.false_negatives,
                            "annotation_group_count": grouped_metrics.ground_truth,
                        }
                        if grouped_metrics is not None
                        else {}
                    ),
                },
            )
        else:
            print(
                f"{class_name} | {prompt}: "
                f"P={metrics.precision:.3f} R={metrics.recall:.3f} F1={metrics.f1:.3f} "
                f"TP={metrics.true_positives} FP={metrics.false_positives} "
                f"FN={metrics.false_negatives}"
                + (f" ERROR={metrics.error}" if metrics.error else ""),
                flush=True,
            )

    prompt_rows = [asdict(item) for item in prompt_metrics]
    class_rows = aggregate_by_class(prompt_metrics)
    summary = aggregate_summary(prompt_metrics)
    config = {
        "map_path": str(args.map_path),
        "base_url": args.base_url,
        "map_id": args.map_id,
        "api_style": args.api_style,
        "online": args.online,
        "url": url,
        "localize_url": args.localize_url,
        "annotations": str(args.annotations),
        "run_started_at": args.run_started_at,
        "acceptance_threshold": args.acceptance_threshold,
        "num_results": args.num_results,
        "min_similarity": args.min_similarity,
        "candidate_count": args.candidate_count,
        "clustering_eps_m": args.clustering_eps_m,
        "feedback_alpha": args.feedback_alpha,
        "feedback_beta": args.feedback_beta,
        "feedback_normalization": args.feedback_normalization,
        "min_keyframes_per_cluster": args.min_keyframes_per_cluster,
        "max_observations_per_cluster": args.max_observations_per_cluster,
        "min_observations_per_cluster": args.min_observations_per_cluster,
        "max_cluster_spread_m": args.max_cluster_spread_m,
        "semantic_gate_threshold": args.semantic_gate_threshold,
        "association": args.association,
        "combination": args.combination,
        "association_sim_threshold": args.association_sim_threshold,
        "descriptor": args.descriptor,
        "score_field": args.score_field,
        "group_annotation_radius_m": args.group_annotation_radius_m,
        "default_accuracy": args.default_accuracy,
        "prompt_geojson_dir": str(args.prompt_geojson_dir),
    }
    result: dict[str, Any] = {
        "config": config,
        "summary": summary,
        "by_class": class_rows,
        "by_prompt": prompt_rows,
        "prompt_geojson_files": prompt_geojson_paths,
        "raw_by_prompt": raw_by_prompt,
    }
    if grouping_enabled:
        grouped = {
            "group_annotation_radius_m": args.group_annotation_radius_m,
            "summary": aggregate_summary(grouped_prompt_metrics),
            "by_class": aggregate_by_class(grouped_prompt_metrics),
            "by_prompt": [asdict(item) for item in grouped_prompt_metrics],
        }
        result["grouped"] = grouped
    done_event: dict[str, Any] = {"event": "done", "summary": summary}
    if grouping_enabled:
        done_event["grouped_summary"] = result["grouped"]["summary"]
    _emit_progress(args, done_event)
    return result


def aggregate_by_class(rows: list[PromptMetrics]) -> list[dict[str, Any]]:
    by_class: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_class.setdefault(
            row.class_name,
            {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "prompt_count": 0,
                "errors": 0,
            },
        )
        bucket["true_positives"] += row.true_positives
        bucket["false_positives"] += row.false_positives
        bucket["false_negatives"] += row.false_negatives
        bucket["prompt_count"] += 1
        bucket["errors"] += 1 if row.error else 0

    result = []
    for class_name, bucket in sorted(by_class.items()):
        precision, recall, f1 = compute_prf(
            bucket["true_positives"],
            bucket["false_positives"],
            bucket["false_negatives"],
        )
        result.append(
            {
                "class_name": class_name,
                **bucket,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return result


def aggregate_summary(rows: list[PromptMetrics]) -> dict[str, Any]:
    tp = sum(row.true_positives for row in rows)
    fp = sum(row.false_positives for row in rows)
    fn = sum(row.false_negatives for row in rows)
    precision, recall, f1 = compute_prf(tp, fp, fn)
    # Averaged over prompts, NOT pooled over predictions. `match_score` is normalised by
    # the best cluster of its own query, so a 0.93 under one prompt and a 0.93 under
    # another are not the same evidence — pooling them into one ranking would sort
    # incomparable numbers. Every prompt therefore weighs the same here, whatever its
    # number of annotations.
    scored = [row for row in rows if not row.error and row.ground_truth]
    return {
        "prompt_count": len(rows),
        "error_count": sum(1 for row in rows if row.error),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "scored_prompt_count": len(scored),
        "mean_average_precision": (
            sum(row.average_precision for row in scored) / len(scored)
            if scored
            else 0.0
        ),
        "mean_best_f1": (
            sum(row.best_f1 for row in scored) / len(scored) if scored else 0.0
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
        )
    if not scalar_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark object-search localization prompts against GeoJSON annotations. "
            "Predicted clusters are accepted when the selected score is > threshold."
        )
    )
    parser.add_argument(
        "--map-path",
        "--map_path",
        dest="map_path",
        type=Path,
        required=True,
        help=(
            "Map directory. Benchmark inputs are read from "
            "{map_path}/benchmark/annotations.geojson by default. "
            "Prompts are derived from each annotation's 'prompt' property when "
            "present, otherwise from its 'class' property."
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Override annotations GeoJSON path. Defaults to "
        "{map_path}/benchmark/annotations.geojson.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--localize-url",
        default=None,
        help=(
            "Override the full object-search localize URL. May include a "
            "{map_id} placeholder, for example "
            f"{DEFAULT_REMOTE_LOCALIZE_URL_TEMPLATE}."
        ),
    )
    parser.add_argument(
        "--map-id",
        default=None,
        help="Service map id. Defaults to the map directory name.",
    )
    parser.add_argument(
        "--api-style", choices=("geopose", "standalone"), default=DEFAULT_API_STYLE
    )
    parser.add_argument(
        "--online", action="store_true", help="Use standalone online /localize endpoint"
    )
    parser.add_argument("--num-results", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--acceptance-threshold", type=float, default=0.9)
    parser.add_argument("--score-field", default="match_score")
    parser.add_argument(
        "--default-accuracy",
        type=float,
        default=5.0,
        help=(
            "Match radius, in metres, for annotations carrying no 'accuracy' property. "
            "It must stay below half the spacing between distinct annotations of the "
            "same class, otherwise the 1-1 assignment picks between neighbouring "
            "targets on sub-metre differences and the metrics stop meaning anything."
        ),
    )
    parser.add_argument(
        "--group-annotation-radius-m",
        type=float,
        default=0.0,
        help=(
            "Optional radius for grouped metrics. When > 0, nearby annotations "
            "of the same class within this radius are counted as one GT target "
            "in an additional grouped result block."
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--clustering-eps-m", type=float, default=1.5)
    parser.add_argument(
        "--only-prompt",
        action="append",
        default=[],
        help=(
            "Evaluate only this prompt (repeatable, matched case-insensitively after "
            "trimming). Exit status 2 means none matched the annotations file."
        ),
    )
    parser.add_argument("--feedback-alpha", type=float, default=0.0)
    parser.add_argument("--feedback-beta", type=float, default=0.0)
    parser.add_argument(
        "--feedback-normalization",
        choices=("none", "center", "standardize"),
        default="none",
        help=(
            "How the review-feedback prototype similarities are rescaled across the "
            "retrieved candidates before the gains apply. 'none' is the raw term; "
            "'center' subtracts the median, 'standardize' also divides by a robust "
            "sigma. Ignored when both gains are zero."
        ),
    )
    parser.add_argument("--min-keyframes-per-cluster", type=int, default=None)
    parser.add_argument("--max-observations-per-cluster", type=int, default=None)
    parser.add_argument(
        "--min-observations-per-cluster",
        type=int,
        default=None,
        help=(
            "Geometric filter: drop clusters with fewer detections than this. Off by "
            "default. It is a filter and not a score term on purpose — blending "
            "cluster size into match_score cost ranking quality when measured."
        ),
    )
    parser.add_argument(
        "--semantic-gate-threshold",
        type=float,
        default=None,
        help=(
            "Two-gate association (ConceptGraphs): a detection joins a cluster only "
            "if it is within the spatial radius AND its cutout embedding is at least "
            "this cosine-similar to the cluster seed. Off by default, which is "
            "production's geometry-only rule."
        ),
    )
    parser.add_argument(
        "--association",
        choices=("leader_canopy", "incremental"),
        default="leader_canopy",
        help=(
            "Detection association algorithm. 'leader_canopy' preserves the current "
            "production-compatible path; 'incremental' enables greedy best-match "
            "association and requires cutout embeddings."
        ),
    )
    parser.add_argument(
        "--combination",
        choices=("conjunctive", "sum"),
        default="sum",
        help="Incremental eligibility rule; ignored by leader_canopy.",
    )
    parser.add_argument(
        "--association-sim-threshold",
        type=float,
        default=1.1,
        help="Minimum semantic-plus-geometric score for incremental sum association.",
    )
    parser.add_argument(
        "--descriptor",
        choices=("seed", "running_mean"),
        default="running_mean",
        help="Incremental cluster descriptor update; ignored by leader_canopy.",
    )
    parser.add_argument(
        "--max-cluster-spread-m",
        type=float,
        default=None,
        help=(
            "Geometric filter: drop clusters whose member positions have a mean "
            "per-axis standard deviation above this, in metres. Off by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Write all outputs into this directory: metrics.json, raw_results.json, "
            "results.csv and prompt_geojson/. Explicit --output-json/--output-csv/"
            "--prompt-geojson-dir values still take precedence."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Override summary JSON path. Defaults to "
        "{map_path}/benchmark/results/results.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override per-prompt CSV path. Defaults to "
        "{map_path}/benchmark/results/results.csv.",
    )
    parser.add_argument(
        "--prompt-geojson-dir",
        type=Path,
        default=None,
        help="Directory for one GeoJSON file per evaluated text prompt.",
    )
    parser.add_argument(
        "--no-prompt-geojson",
        action="store_true",
        help="Disable per-prompt GeoJSON exports.",
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit machine-readable JSON progress lines on stdout instead of "
        "human-readable text.",
    )
    args = parser.parse_args(argv)
    args.map_path = args.map_path.expanduser().resolve()
    benchmark_dir = args.map_path / "benchmark"
    results_dir = benchmark_dir / "results"
    if args.map_id is None:
        args.map_id = args.map_path.name
    if args.annotations is None:
        args.annotations = benchmark_dir / "annotations.geojson"
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_json is None:
            args.output_json = args.output_dir / "metrics.json"
        if args.output_csv is None:
            args.output_csv = args.output_dir / "results.csv"
        if args.prompt_geojson_dir is None:
            args.prompt_geojson_dir = args.output_dir / "prompt_geojson"
    if args.output_json is None:
        args.output_json = results_dir / "results.json"
    if args.output_csv is None:
        args.output_csv = results_dir / "results.csv"
    if args.prompt_geojson_dir is None:
        args.prompt_geojson_dir = results_dir / "prompt_geojson"
    args.run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.group_annotation_radius_m < 0:
        parser.error("--group-annotation-radius-m must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_benchmark(args)
    except PromptNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    raw_by_prompt = result.pop("raw_by_prompt")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, result["by_prompt"])
    grouped = result.get("grouped")
    if isinstance(grouped, dict):
        grouped_csv_path = args.output_csv.with_name("grouped_results.csv")
        write_csv(grouped_csv_path, grouped["by_prompt"])

    if args.output_dir is not None:
        raw_results_path = args.output_dir / "raw_results.json"
        raw_results_path.parent.mkdir(parents=True, exist_ok=True)
        raw_results_path.write_text(
            json.dumps(
                {"config": result["config"], "prompts": raw_by_prompt},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = result["summary"]
    if not args.progress_json:
        print(
            "\nSummary: "
            f"P={summary['precision']:.3f} R={summary['recall']:.3f} "
            f"F1={summary['f1']:.3f} "
            f"TP={summary['true_positives']} FP={summary['false_positives']} "
            f"FN={summary['false_negatives']} "
            f"errors={summary['error_count']}"
        )
        print(
            "Threshold-free: "
            f"mAP={summary['mean_average_precision']:.3f} "
            f"mean best F1={summary['mean_best_f1']:.3f} "
            f"over {summary['scored_prompt_count']} prompt(s) — compare runs on these, "
            "not on the fixed-threshold line above."
        )
        grouped = result.get("grouped")
        if isinstance(grouped, dict):
            grouped_summary = grouped["summary"]
            print(
                "Grouped summary: "
                f"P={grouped_summary['precision']:.3f} "
                f"R={grouped_summary['recall']:.3f} "
                f"F1={grouped_summary['f1']:.3f} "
                f"TP={grouped_summary['true_positives']} "
                f"FP={grouped_summary['false_positives']} "
                f"FN={grouped_summary['false_negatives']}"
            )
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.output_csv}")
        if isinstance(result.get("grouped"), dict):
            print(f"Wrote {args.output_csv.with_name('grouped_results.csv')}")
        if not args.no_prompt_geojson:
            print(f"Wrote prompt GeoJSON files under {args.prompt_geojson_dir}")
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
