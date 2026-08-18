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
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from toolbox.benchmark import vlm_scores as vlm_scores_module
from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    AnnotationGroup,
    Prediction,
    evaluate_prompt,
    evaluate_prompt_grouped,
    group_annotations,
    haversine_m,
    load_annotations,
    match_predictions,
    parse_predictions,
    post_json,
    precision_recall_curve,
    targets_from_annotations,
    targets_from_groups,
    write_csv,
)
from toolbox.bricks import (
    candidates,
    db,
    georef_source,
    map_manifest,
    service,
)
from toolbox.bricks import (
    feedback as feedback_module,
)
from toolbox.bricks.candidates import (
    EnrichedCandidate,
    apply_feedback_boost,
    normalize_prototype_similarities,
)
from toolbox.bricks.ingest_cli import EMBEDDING_DIM
from toolbox.bricks.localize import (
    LocalizationParams,
    localize_from_enriched_candidates,
)
from toolbox.bricks.rescoring import RescoreInput, build_rescorer
from toolbox.bricks.vendored.geo_transform import GeoTransform
from toolbox.bricks.vlm_gate import GateConfig, VlmYesNoScorer
from toolbox.logging import logger

DEFAULT_ACCURACY_M = 5.0
DEFAULT_TIMEOUT_S = 60.0
VERIFY_TOLERANCE = 1e-9
CSV_FILENAME = "association_sweep.csv"
JSON_FILENAME = "association_sweep.json"
DEFAULT_NEAR_M = 1.0
#: Localisation thresholds HOTA averages over. `near_m` alone would turn the score
#: into a statement about that one radius; the same radii the map report attaches
#: ground truth at keep the two tools reading the same scale.
HOTA_ALPHAS_M: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)

PromptEvaluator: TypeAlias = Callable[[str, list[Prediction], float], float]


@dataclass(frozen=True)
class ThresholdMetrics:
    """Macro F1 at a fitted shared threshold and its LOO estimate."""

    macro_f1: float
    threshold: float
    loo_macro_f1: float
    #: Spread of the thresholds each prompt would have picked for itself, p90 minus
    #: p10. It is the cost of sharing one threshold, stated before the LOO estimate
    #: has to pay it: a wide spread means the scores are not on a common scale.
    prompt_threshold_spread: float = 0.0


@dataclass(frozen=True)
class CalibrationMetrics:
    """How far a cluster's score is from the probability that it is correct."""

    #: Expected calibration error: the bin-weighted gap between mean score and
    #: observed correctness. Zero means a score of 0.7 is right 70 % of the time.
    ece: float
    #: The worst single bin's gap. A small ECE can still hide one ruinous bin.
    mce: float
    #: Mean score minus observed accuracy. Positive is overconfident, which is the
    #: direction open-vocabulary scoring is known to fail in.
    overconfidence: float
    #: `(mean score, observed accuracy, count)` per bin, for the reliability diagram.
    bins: tuple[tuple[float, float, int], ...]
    #: Predictions the estimate rests on.
    scored: int
    #: Mean score and observed accuracy over all of them, so the two halves of
    #: `overconfidence` can be read apart.
    mean_score: float = 0.0
    accuracy: float = 0.0
    #: The highest accuracy any scoring could reach here, `annotations / predictions`
    #: capped at one, because the match is one-to-one. Returning four clusters per
    #: object makes three of them wrong whatever the score says, so an ECE above this
    #: gap is granularity, not calibration, and `ass_a` is where to read it.
    accuracy_ceiling: float = 1.0


@dataclass(frozen=True)
class PromptDetail:
    """Threshold-free per-prompt metrics retained for per-class tables."""

    class_name: str
    strict_ap: float
    grouped_ap: float
    cluster_count: int
    mean_clusters_per_annotation: float = 0.0
    covered_annotations: int = 0
    hota: float = 0.0
    det_a: float = 0.0
    ass_a: float = 0.0


@dataclass(frozen=True)
class ClassFragmentation:
    """Fragmentation of one annotation class, pooled over the prompts using it."""

    mean_clusters_per_annotation: float
    covered_annotations: int
    # The control the fragmentation number cannot be read without: an annotation with
    # forty labelled detections has more opportunities to be split than one with two,
    # so a class that is merely detected more often looks fragmented for free.
    mean_detections_per_annotation: float = 0.0


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
    pair_precision: float
    pair_recall: float
    pair_f1: float
    rand_index: float
    labelled_detections: float
    total_clusters: int
    median_observation_count: float
    median_spread_m: float
    median_clusters_per_prompt: float
    per_prompt: dict[str, PromptDetail]
    # Fragmentation of well-covered annotations, pooled over every prompt: how many
    # distinct clusters hold the detections of one annotation. 1.0 means every such
    # annotation ended up whole. See `fragmentation_metrics`.
    mean_clusters_per_annotation: float = 0.0
    covered_annotations: int = 0
    # Same quantity per annotation class, JSON only — the CSV stays one row wide.
    fragmentation_by_class: dict[str, ClassFragmentation] = field(default_factory=dict)
    # HOTA and its two halves, averaged over prompts. Unlike `map_strict`, these stay
    # comparable when a configuration changes the granularity: see `hota_at`.
    det_a: float = 0.0
    ass_a: float = 0.0
    hota: float = 0.0
    # Score calibration, and what it costs: `threshold_spread_strict` is how far
    # apart the prompts' own best thresholds are, which is what a shared threshold
    # has to paper over. See `calibration_metrics`.
    ece: float = 0.0
    mce: float = 0.0
    overconfidence: float = 0.0
    accuracy_ceiling: float = 1.0
    threshold_spread_strict: float = 0.0
    # Reliability diagram, JSON only — the CSV stays one row wide.
    reliability: tuple[tuple[float, float, int], ...] = ()
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class HotaMetrics:
    """Detection and association accuracy, averaged over localisation thresholds.

    The pair is the point. `det_a` moves when the retrieval finds more or fewer of
    the right detections, `ass_a` moves when the association splits or merges them,
    and `hota` is the geometric mean that refuses to trade one for the other. A
    configuration that only splits objects differently changes `ass_a` alone.
    """

    det_a: float
    ass_a: float
    hota: float


@dataclass(frozen=True)
class PartitionMetrics:
    """Pairwise agreement metrics for one prompt's labelled detections."""

    pair_precision: float
    pair_recall: float
    pair_f1: float
    rand_index: float
    labelled_detections: int


def nearest_annotation_distances(
    detections: Sequence[EnrichedCandidate],
    annotations: Sequence[Annotation],
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest annotation of each detection, and how far away it is.

    Split out of `nearest_annotation_labels` so a metric can re-threshold the same
    assignment at several radii without recomputing the distances.

    Args:
        detections: Selected detections whose depth-projected coordinates are used.
        annotations: Prompt annotations eligible to label the detections.

    Returns:
        Nearest annotation index per detection (``-1`` when there is no annotation
        at all) and the matching distance in metres (``inf`` in that case).
    """
    nearest = np.full(len(detections), -1, dtype=np.int64)
    distance = np.full(len(detections), np.inf, dtype=np.float64)
    if not annotations:
        return nearest, distance
    for detection_index, detection in enumerate(detections):
        distances = np.asarray(
            [
                haversine_m(
                    detection.lat, detection.lng, annotation.lat, annotation.lng
                )
                for annotation in annotations
            ],
            dtype=np.float64,
        )
        nearest_index = int(np.argmin(distances))
        nearest[detection_index] = nearest_index
        distance[detection_index] = float(distances[nearest_index])
    return nearest, distance


def nearest_annotation_labels(
    detections: Sequence[EnrichedCandidate],
    annotations: Sequence[Annotation],
    near_m: float,
) -> np.ndarray:
    """Label detections by their nearest in-range annotation.

    Args:
        detections: Selected detections whose depth-projected coordinates are used.
        annotations: Prompt annotations eligible to label the detections.
        near_m: Maximum detection-to-annotation distance in metres.

    Returns:
        Annotation indices aligned with ``detections``; ``-1`` means unlabelled.
    """
    nearest, distance = nearest_annotation_distances(detections, annotations)
    return np.where(distance <= near_m, nearest, -1)


def partition_metrics(
    cluster_labels: np.ndarray, annotation_labels: np.ndarray
) -> PartitionMetrics:
    """Score an induced detection partition using a contingency table.

    Unlabelled detections (negative annotation labels) are excluded. Negative cluster
    labels are treated as distinct singleton clusters, rather than one shared noise
    cluster. Metrics with no applicable denominator are zero.

    Args:
        cluster_labels: Association labels aligned with the detections.
        annotation_labels: Nearest-annotation labels, negative when out of range.

    Returns:
        Pair precision, recall, F1, Rand index, and proxy coverage count.

    Raises:
        ValueError: If the two label vectors have different shapes.
    """
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    annotations = np.asarray(annotation_labels, dtype=np.int64)
    if clusters.shape != annotations.shape:
        raise ValueError("Cluster and annotation labels must have the same shape")

    labelled_mask = annotations >= 0
    clusters = clusters[labelled_mask].copy()
    annotations = annotations[labelled_mask]
    labelled_count = int(annotations.size)
    negative_indices = np.flatnonzero(clusters < 0)
    if negative_indices.size:
        next_label = int(np.max(clusters, initial=-1)) + 1
        clusters[negative_indices] = next_label + np.arange(negative_indices.size)

    _, cluster_inverse = np.unique(clusters, return_inverse=True)
    _, annotation_inverse = np.unique(annotations, return_inverse=True)
    contingency = np.zeros(
        (
            int(cluster_inverse.max(initial=-1)) + 1,
            int(annotation_inverse.max(initial=-1)) + 1,
        ),
        dtype=np.int64,
    )
    np.add.at(contingency, (cluster_inverse, annotation_inverse), 1)

    def choose_two(counts: np.ndarray) -> int:
        """Sum C(n, 2) over a vector or matrix of counts."""
        return int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))

    true_positive = choose_two(contingency)
    predicted_positive = choose_two(np.sum(contingency, axis=1))
    actual_positive = choose_two(np.sum(contingency, axis=0))
    total_pairs = labelled_count * (labelled_count - 1) // 2
    false_positive = predicted_positive - true_positive
    false_negative = actual_positive - true_positive
    true_negative = total_pairs - true_positive - false_positive - false_negative

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    pair_f1 = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    rand_index = (true_positive + true_negative) / total_pairs if total_pairs else 0.0
    return PartitionMetrics(
        pair_precision=precision,
        pair_recall=recall,
        pair_f1=pair_f1,
        rand_index=rand_index,
        labelled_detections=labelled_count,
    )


def _split_negative_clusters(clusters: np.ndarray) -> np.ndarray:
    """Give every dropped detection a cluster of its own, as `partition_metrics` does.

    A post-association filter that shatters an object must not be rewarded for it, so
    the negative label is not one shared noise cluster.
    """
    result = np.asarray(clusters, dtype=np.int64).copy()
    negative_indices = np.flatnonzero(result < 0)
    if negative_indices.size:
        next_label = int(np.max(result, initial=-1)) + 1
        result[negative_indices] = next_label + np.arange(negative_indices.size)
    return result


def hota_at(
    cluster_labels: np.ndarray,
    nearest: np.ndarray,
    distance: np.ndarray,
    annotation_count: int,
    alpha_m: float,
) -> tuple[float, float]:
    """Detection and association accuracy of one prompt at one distance threshold.

    Detection accuracy is `TP / (TP + FP + FN)`: a detection within ``alpha_m`` of an
    annotation is a true positive, one that is not is a false positive, and an
    annotation no detection reached is a false negative. The two halves are counted
    in different units — detections against annotations — exactly as HOTA counts
    predicted detections against ground-truth detections; it stays a fair comparison
    between configurations because every configuration is scored the same way.

    Association accuracy is the mean over true positives of the Jaccard overlap
    between the cluster a detection landed in and the set of detections belonging to
    its annotation. Splitting an object costs recall of that overlap, merging two
    costs its precision, and no reshuffling of cluster boundaries can raise both.
    That is the property `map_strict`/`map_grouped` lack, and the reason two
    granularities cannot be ranked without it.

    Args:
        cluster_labels: Association labels aligned with the detections.
        nearest: Nearest annotation index per detection, from
            `nearest_annotation_distances`.
        distance: Distance to that annotation, metres.
        annotation_count: How many annotations the prompt has.
        alpha_m: Localisation threshold.

    Returns:
        `(DetA, AssA)` at this threshold. Both are zero when there is no true
        positive to score.
    """
    clusters = _split_negative_clusters(cluster_labels)
    matched = (nearest >= 0) & (distance <= alpha_m)
    true_positive = int(matched.sum())
    false_positive = int(matched.size - true_positive)
    reached = np.unique(nearest[matched]).size if true_positive else 0
    false_negative = max(annotation_count - reached, 0)
    denominator = true_positive + false_positive + false_negative
    det_a = true_positive / denominator if denominator else 0.0
    if not true_positive:
        return det_a, 0.0

    # Jaccard per (cluster, annotation) pair: the shared detections over the union of
    # the cluster and the annotation's detections. `cluster_size` counts unmatched
    # detections too — they are what a cluster is contaminated with.
    _, cluster_index = np.unique(clusters, return_inverse=True)
    cluster_size = np.bincount(cluster_index)
    annotation_index = nearest[matched]
    annotation_size = np.bincount(annotation_index, minlength=annotation_count)
    pair_keys = cluster_index[matched] * (annotation_count + 1) + annotation_index
    unique_pairs, pair_inverse = np.unique(pair_keys, return_inverse=True)
    shared = np.bincount(pair_inverse)
    pair_cluster = unique_pairs // (annotation_count + 1)
    pair_annotation = unique_pairs % (annotation_count + 1)
    union = cluster_size[pair_cluster] + annotation_size[pair_annotation] - shared
    overlap = np.divide(
        shared, union, out=np.zeros(shared.shape, dtype=np.float64), where=union > 0
    )
    # Averaged over true-positive detections, not over pairs: a cluster holding forty
    # observations of an object weighs forty times one holding a single stray.
    ass_a = float((shared * overlap).sum() / true_positive)
    return det_a, ass_a


def hota_metrics(
    cluster_labels: np.ndarray,
    nearest: np.ndarray,
    distance: np.ndarray,
    annotation_count: int,
    alphas_m: Sequence[float] = HOTA_ALPHAS_M,
) -> HotaMetrics:
    """Average `hota_at` over the localisation thresholds.

    A single threshold makes the score an argument about `near_m` rather than about
    the association, which is why HOTA integrates over several. See `hota_at` for
    what each half measures.

    Args:
        cluster_labels: Association labels aligned with the detections.
        nearest: Nearest annotation index per detection.
        distance: Distance to that annotation, metres.
        annotation_count: How many annotations the prompt has.
        alphas_m: Localisation thresholds to average over.

    Returns:
        Mean detection accuracy, mean association accuracy, and the mean over
        thresholds of their geometric mean.

    Raises:
        ValueError: If the three per-detection vectors have different shapes.
    """
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    if clusters.shape != nearest.shape or clusters.shape != distance.shape:
        raise ValueError("Cluster labels, nearest and distance must be aligned")
    scores = [
        hota_at(clusters, nearest, distance, annotation_count, alpha)
        for alpha in alphas_m
    ]
    return HotaMetrics(
        det_a=statistics.fmean(det for det, _ in scores),
        ass_a=statistics.fmean(ass for _, ass in scores),
        hota=statistics.fmean(math.sqrt(det * ass) for det, ass in scores),
    )


def fragmentation_counts(
    cluster_labels: np.ndarray, annotation_labels: np.ndarray
) -> dict[int, tuple[int, int]]:
    """Count the distinct clusters holding each well-covered annotation's detections.

    Reuses the labelling that feeds `partition_metrics`: a detection belongs to the
    nearest annotation within ``near_m``. Only annotations with at least two labelled
    detections are reported — a single detection is one cluster by construction and
    would dilute the mean towards 1. Negative cluster labels (detections dropped by a
    post-association filter) count as one distinct cluster each, as in
    `partition_metrics`, so a filter that shatters an object is not rewarded.

    Args:
        cluster_labels: Association labels aligned with the detections.
        annotation_labels: Nearest-annotation labels, negative when out of range.

    Returns:
        Annotation index mapped to its distinct-cluster and labelled-detection counts.

    Raises:
        ValueError: If the two label vectors have different shapes.
    """
    clusters = np.asarray(cluster_labels, dtype=np.int64)
    annotations = np.asarray(annotation_labels, dtype=np.int64)
    if clusters.shape != annotations.shape:
        raise ValueError("Cluster and annotation labels must have the same shape")

    labelled_mask = annotations >= 0
    clusters = clusters[labelled_mask].copy()
    annotations = annotations[labelled_mask]
    negative_indices = np.flatnonzero(clusters < 0)
    if negative_indices.size:
        next_label = int(np.max(clusters, initial=-1)) + 1
        clusters[negative_indices] = next_label + np.arange(negative_indices.size)

    counts: dict[int, tuple[int, int]] = {}
    for annotation_index in np.unique(annotations):
        members = clusters[annotations == annotation_index]
        if members.size < 2:
            continue
        counts[int(annotation_index)] = (
            int(np.unique(members).size),
            int(members.size),
        )
    return counts


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


@dataclass(frozen=True)
class PromptPrototypes:
    """Reviewed cutout embeddings for one prompt, plus what was asked for.

    `*_requested` is the number of review rows; the arrays hold only the ids that
    still resolve to a candidate in this georef. The two differ after a reingest,
    and that difference is the whole reason both are carried.
    """

    positive: np.ndarray
    negative: np.ndarray
    positive_requested: int
    negative_requested: int

    @property
    def resolved(self) -> int:
        return int(self.positive.shape[0] + self.negative.shape[0])


def _rescore(
    prompt_candidates: Sequence[EnrichedCandidate],
    params: LocalizationParams,
    prototypes: PromptPrototypes | None,
) -> list[EnrichedCandidate]:
    """Run a registered rescorer over the retrieved set and write its scores.

    The rescorer replaces the max-prototype boost entirely: it sees every candidate
    embedding and every reviewed embedding, and returns one score per candidate,
    which lands in `similarity_boosted`. `LocalizationParams.rescorer` makes
    `feedback_enabled` true, so ranking reads that column — nothing else in
    localization changes, and cluster geometry is untouched exactly as with the
    boost.

    Args:
        prompt_candidates: One prompt's cached candidates, in retrieval order.
        params: Configuration naming the rescorer and its parameters.
        prototypes: Reviewed embeddings for this prompt.

    Returns:
        The candidates carrying the rescorer's scores.

    Raises:
        ValueError: If prototypes or candidate embeddings are missing, which would
            otherwise degrade silently into "this method does nothing".
    """
    if prototypes is None:
        raise ValueError(
            f"Rescorer {params.rescorer!r} needs review prototypes; "
            "run the sweep with --with-feedback"
        )
    embeddings = [candidate.embedding for candidate in prompt_candidates]
    if any(embedding is None for embedding in embeddings):
        raise ValueError("Rescoring needs candidates cached with embeddings")
    rescorer = build_rescorer(params.rescorer, dict(params.rescorer_params or {}))
    assert rescorer is not None  # noqa: S101 - `params.rescorer` is not None here
    result = rescorer.score(
        RescoreInput(
            candidate_ids=np.asarray(
                [candidate.id for candidate in prompt_candidates], dtype=np.int64
            ),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            base_similarity=np.asarray(
                [candidate.similarity for candidate in prompt_candidates],
                dtype=np.float32,
            ),
            positive_embeddings=prototypes.positive,
            negative_embeddings=prototypes.negative,
        )
    )
    scores = np.asarray(result.scores, dtype=np.float64)
    if scores.shape != (len(prompt_candidates),):
        raise ValueError(
            f"Rescorer {params.rescorer!r} returned {scores.shape} scores for "
            f"{len(prompt_candidates)} candidates"
        )
    return [
        replace(candidate, similarity_boosted=float(score))
        for candidate, score in zip(prompt_candidates, scores.tolist())
    ]


def apply_detection_gate(
    prompt_candidates: Sequence[EnrichedCandidate],
    params: LocalizationParams,
    vlm_scores: Mapping[int, float],
) -> list[EnrichedCandidate]:
    """Fold `p(yes)` into each candidate's ranking similarity, before association.

    The gate is a *score*, not a filter: a candidate the VLM rejects is demoted, not
    dropped, so cluster geometry is identical with and without it and the comparison
    stays readable at fixed granularity. `feedback_normalization` rescales the column
    across the retrieved set first, for the reason its own docstring gives — a constant
    offset flattens the ratio instead of sharpening it.

    Candidates with no score (unreadable cutout) keep their raw similarity: the honest
    reading of "the gate saw nothing" is "no evidence", not "rejected".

    Args:
        prompt_candidates: One prompt's candidates, in retrieval order.
        params: Configuration carrying `vlm_alpha` and the normalization.
        vlm_scores: `p(yes)` by candidate id.

    Returns:
        The candidates with `similarity_boosted` set.
    """
    values = [
        float(vlm_scores.get(candidate.id, np.nan)) for candidate in prompt_candidates
    ]
    finite = [value for value in values if math.isfinite(value)]
    # Normalisation is fitted on the scored candidates only, then applied to them;
    # unscored ones are held out of both steps rather than imputed at the median.
    applied = dict(
        zip(
            [index for index, value in enumerate(values) if math.isfinite(value)],
            normalize_prototype_similarities(finite, params.feedback_normalization),
        )
    )
    # The base is `effective_similarity`, not the raw one, so the gate composes with a
    # rescorer instead of replacing it: run the review model first, verify on top.
    return [
        replace(
            candidate,
            similarity_boosted=(
                candidate.effective_similarity + params.vlm_alpha * applied[index]
                if index in applied
                else candidate.effective_similarity
            ),
        )
        for index, candidate in enumerate(prompt_candidates)
    ]


def apply_cluster_gate(
    localizations: Sequence[dict],
    params: LocalizationParams,
    vlm_scores: Mapping[int, float],
) -> list[dict]:
    """Re-rank returned clusters by the VLM verdict on their own observations.

    This is the shape the 3D-grounding literature uses (VLM-Grounder, SeqVLM,
    DRIVE-Nav): verify a *candidate object* from several of its views and let the views
    agree, rather than judging every detection in isolation. A cluster's observations
    are already its best-scoring views, capped by `max_observations_per_cluster`.

    Clusters keep their membership and their coordinates — only `match_score` moves —
    so this changes what a caller would accept, never where it would go.

    Args:
        localizations: Localization dicts, ranked, as `localize` returned them.
        params: Configuration carrying `vlm_alpha` and `vlm_aggregate`.
        vlm_scores: `p(yes)` by candidate id.

    Returns:
        The same clusters, re-scored and re-sorted by descending `match_score`.
    """
    aggregates: dict[str, Callable[[Sequence[float]], np.floating]] = {
        "max": np.max,
        "mean": np.mean,
        "min": np.min,
    }
    aggregate = aggregates[params.vlm_aggregate]
    gate_values: list[float] = []
    for localization in localizations:
        scored = [
            vlm_scores[int(observation["object_idx"])]
            for observation in localization.get("observations", [])
            if int(observation["object_idx"]) in vlm_scores
        ]
        gate_values.append(float(aggregate(scored)) if scored else float("nan"))
    finite = [value for value in gate_values if math.isfinite(value)]
    applied = dict(
        zip(
            [index for index, value in enumerate(gate_values) if math.isfinite(value)],
            normalize_prototype_similarities(finite, params.feedback_normalization),
        )
    )
    rescored: list[dict] = []
    for index, localization in enumerate(localizations):
        updated = dict(localization)
        if index in applied:
            updated["match_score"] = (
                float(localization["match_score"]) + params.vlm_alpha * applied[index]
            )
            updated["vlm_gate_score"] = gate_values[index]
        rescored.append(updated)
    rescored.sort(key=lambda item: float(item["match_score"]), reverse=True)
    return rescored


def apply_feedback(
    prompt_candidates: Sequence[EnrichedCandidate],
    params: LocalizationParams,
    prototypes: PromptPrototypes | None = None,
    vlm_scores: Mapping[int, float] | None = None,
) -> list[EnrichedCandidate]:
    """Recompute the review boost on cached candidates, for one configuration.

    The cached candidates carry the *raw* prototype columns (`pos_sim`/`neg_sim`),
    because `fetch_prompt_candidates` fetches them with both gains at zero. Since
    the boost is affine in those columns and `normalize_prototype_similarities` is a
    pure function of the retrieved set, `feedback_alpha`, `feedback_beta` and the
    normalization can all be swept from that single cache — reproducing exactly what
    `candidates.load_enriched_candidates` would have written, without a second
    ANN + database round trip per configuration.

    A named `rescorer` takes over instead: see `_rescore`.

    Args:
        prompt_candidates: One prompt's cached candidates, in retrieval order.
        params: Configuration whose feedback gains and normalization apply.
        prototypes: Reviewed embeddings, required only by the rescorer branch.
        vlm_scores: `p(yes)` by candidate id, required only by the detection gate.

    Returns:
        The candidates with `similarity_boosted` and the applied columns set, or the
        input unchanged when the configuration has feedback off.
    """
    if not params.feedback_enabled or not prompt_candidates:
        return list(prompt_candidates)
    staged = list(prompt_candidates)
    if params.rescorer is not None:
        staged = _rescore(staged, params, prototypes)
    if params.vlm_gate == "detection":
        if vlm_scores is None:
            raise ValueError(
                "The detection gate needs VLM scores; run the sweep with --with-vlm"
            )
        return apply_detection_gate(staged, params, vlm_scores)
    if params.rescorer is not None:
        return staged
    # Normalisation spans the whole retrieved set, exactly as at load time — not the
    # `candidate_count` truncation, and not the depth-capped subset.
    applied_pos = normalize_prototype_similarities(
        [candidate.pos_sim for candidate in prompt_candidates],
        params.feedback_normalization,
    )
    applied_neg = normalize_prototype_similarities(
        [candidate.neg_sim for candidate in prompt_candidates],
        params.feedback_normalization,
    )
    return [
        replace(
            candidate,
            similarity_boosted=apply_feedback_boost(
                candidate.similarity,
                pos_value,
                neg_value,
                params.feedback_alpha,
                params.feedback_beta,
            ),
            pos_sim_applied=pos_value,
            neg_sim_applied=neg_value,
        )
        for candidate, pos_value, neg_value in zip(
            prompt_candidates, applied_pos, applied_neg
        )
    ]


def _log_feedback_coverage(prompt: str, enriched: Sequence[EnrichedCandidate]) -> None:
    """Report how many candidates carry prototype evidence for this prompt.

    An inert boost and an unhelpful one produce the same sweep row, so the count is
    logged before any configuration runs. Zero here means the reviewed ids resolved
    to nothing — `object_search_candidate.id` is a BIGSERIAL that no reingest
    preserves — and every feedback row in the grid will be a copy of the baseline.
    """
    positives = sum(1 for candidate in enriched if candidate.pos_sim)
    negatives = sum(1 for candidate in enriched if candidate.neg_sim)
    if positives or negatives:
        logger.info(
            "Prompt %r: %d candidate(s) with positive and %d with negative prototype "
            "evidence, out of %d",
            prompt,
            positives,
            negatives,
            len(enriched),
        )
    else:
        logger.warning(
            "Prompt %r: no prototype evidence on any of %d candidates — every "
            "feedback configuration will reproduce the baseline exactly",
            prompt,
            len(enriched),
        )


def _cache_path(
    cache_dir: Path,
    map_id: str,
    prompt: str,
    candidate_count: int,
    *,
    with_feedback: bool,
    exact: bool = False,
) -> Path:
    payload: dict[str, Any] = {
        "map_id": map_id,
        "prompt": prompt,
        "candidate_count": candidate_count,
        "with_embeddings": True,
    }
    # Added only when set, so caches written before the prototype columns existed
    # stay addressable by the no-feedback path instead of silently going cold.
    if with_feedback:
        payload["with_feedback"] = True
    if exact:
        payload["exact"] = True
    key = json.dumps(
        payload,
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
    with_feedback: bool = False,
    exact: bool = False,
    exact_device: str = "cpu",
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
        exact: Rank by a brute-force scan instead of asking the online service. The
            only way to obtain more than 1000 candidates, since the service sets
            `hnsw.ef_search = max(k, 1000)` and pgvector caps that at 1000. Uses its own
            cache entries and embeds the query locally with the same checkpoint.
        exact_device: Where to run that text encoder. CPU by default and on purpose: it
            is one 1024-d vector per prompt (11 s to load, 0.12 s to embed), while the
            GPU is usually holding the online service's own copy of the same model and a
            second one does not fit in 8 GB.
        with_feedback: Also resolve the map's review prototypes, with both gains at
            zero, so a grid can sweep them offline through `apply_feedback`. The
            retrieved set, its order and every raw similarity are unaffected.

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
        path = _cache_path(
            resolved_cache_dir,
            manifest.map_id,
            prompt,
            candidate_count,
            with_feedback=with_feedback,
            exact=exact,
        )
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
        embedder = None
        if exact:
            from inference.embedder import (
                METACLIP_DTYPE,
                METACLIP_MODEL_ID,
                MetaClipEmbedder,
            )

            embedder = MetaClipEmbedder(
                METACLIP_MODEL_ID,
                exact_device,
                METACLIP_DTYPE if exact_device != "cpu" else "float32",
            )
        with db.connect() as conn:
            for prompt, path in missing:
                if embedder is not None:
                    hits = candidates.query_exact_top_k(
                        conn,
                        geo_ref_id,
                        np.asarray(embedder.embed_text(prompt), dtype=np.float64),
                        candidate_count,
                    )
                else:
                    hits = service.query_by_text(
                        ann_base_url,
                        geo_ref_id,
                        prompt,
                        candidate_count,
                        timeout_s,
                    )
                review_feedback = (
                    feedback_module.load_review_feedback(
                        manifest.map_id, prompt, resolved_map_path
                    )
                    if with_feedback
                    else None
                )
                enriched = candidates.load_enriched_candidates(
                    conn,
                    geo_ref_id,
                    hits,
                    geo_transform,
                    # Both gains stay at zero and the normalization at "none": the
                    # cache holds the raw prototype columns and `apply_feedback`
                    # rescales and weights them per configuration.
                    feedback=review_feedback,
                    with_embeddings=True,
                )
                with path.open("wb") as stream:
                    pickle.dump(enriched, stream, protocol=pickle.HIGHEST_PROTOCOL)
                result[prompt] = enriched

    if with_feedback:
        # Logged for cache hits too: a cache written against a since-rebuilt index is
        # exactly the case where the boost is inert and the rows look merely flat.
        for prompt in prompt_order:
            _log_feedback_coverage(prompt, result[prompt])

    return {prompt: result[prompt] for prompt in prompt_order}


def fetch_prompt_prototypes(
    map_path: str | Path,
    prompts: Sequence[str],
    *,
    geo_ref_id: int,
    map_id: str,
) -> dict[str, PromptPrototypes]:
    """Load each prompt's reviewed cutout embeddings from the annotation DB.

    Not cached: it is one indexed query per prompt against ids the reviews already
    name, which is nothing next to the ANN round trip the candidate cache exists for.

    Args:
        map_path: Map directory holding `object-search-annotations.db`.
        prompts: Prompts to resolve, in any order.
        geo_ref_id: Georef partition the reviews belong to.
        map_id: Toolbox map identifier, used only for logging.

    Returns:
        Prototype embeddings per prompt; prompts without reviews are absent.
    """
    resolved_map_path = Path(map_path).expanduser().resolve()
    out: dict[str, PromptPrototypes] = {}
    with db.connect() as conn:
        for prompt in prompts:
            review = feedback_module.load_review_feedback(
                map_id, prompt, resolved_map_path
            )
            if review is None:
                # An empty prototype set, not a missing entry: every rescorer is
                # specified to fall back to the base similarity with no evidence, and
                # a prompt nobody reviewed is exactly that case. Skipping it instead
                # would abort the sweep on maps where only some prompts are reviewed.
                logger.warning(
                    "Prompt %r: no reviews at all — every rescorer will return the "
                    "base similarity for it",
                    prompt,
                )
                empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
                out[prompt] = PromptPrototypes(empty, empty, 0, 0)
                continue
            by_id = candidates.load_prototype_embeddings(
                conn, geo_ref_id, review.positive_ids + review.negative_ids
            )
            positive = [by_id[i] for i in review.positive_ids if i in by_id]
            negative = [by_id[i] for i in review.negative_ids if i in by_id]
            # An empty prototype set still has to be (0, d): a rescorer is entitled
            # to multiply by it, and (0, 0) would raise instead of contributing
            # nothing.
            dimension = next(
                (int(embedding.shape[0]) for embedding in by_id.values()),
                EMBEDDING_DIM,
            )
            empty = np.empty((0, dimension), dtype=np.float32)
            out[prompt] = PromptPrototypes(
                positive=(
                    np.asarray(positive, dtype=np.float32) if positive else empty
                ),
                negative=(
                    np.asarray(negative, dtype=np.float32) if negative else empty
                ),
                positive_requested=len(review.positive_ids),
                negative_requested=len(review.negative_ids),
            )
            logger.info(
                "Prompt %r: %d/%d positive and %d/%d negative review(s) resolved to "
                "an embedding",
                prompt,
                len(positive),
                len(review.positive_ids),
                len(negative),
                len(review.negative_ids),
            )
    return out


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


def calibration_metrics(
    predictions_by_prompt: Mapping[str, list[Prediction]],
    annotations_by_prompt: Mapping[str, list[Annotation]],
    bin_count: int = 10,
) -> CalibrationMetrics:
    """Compare cluster scores against the rate at which those clusters are right.

    Fitting one acceptance threshold per map, as this sweep does, is what an
    uncalibrated score forces: if 0.9 meant the same thing everywhere, the threshold
    would transfer. This measures that directly, so the sweep can say whether the
    per-map threshold is a scoring defect worth fixing or an irreducible difference
    between the maps.

    **Bins hold equal counts, not equal widths.** Cluster scores pile up near the top
    of the range, so ten fixed-width bins would put almost every prediction in one of
    them and report an error of nearly zero whatever the scores did.

    Correctness is the threshold-free match: a prediction is right when the greedy
    matcher pairs it with an annotation inside that annotation's own accuracy radius,
    which is the same verdict the benchmark reaches above its threshold.

    Args:
        predictions_by_prompt: Predictions for every evaluated prompt.
        annotations_by_prompt: Ground truth for the same prompt keys.
        bin_count: Quantile bins the reliability diagram uses.

    Returns:
        Expected and maximum calibration error, the signed confidence bias, and the
        per-bin table. Everything is zero when no prompt produced a prediction.
    """
    scores: list[float] = []
    correct: list[float] = []
    for prompt, predictions in predictions_by_prompt.items():
        matched = {
            match.prediction_id
            for match in match_predictions(
                list(predictions), list(annotations_by_prompt.get(prompt, []))
            )
        }
        for prediction in predictions:
            scores.append(float(prediction.score))
            correct.append(1.0 if prediction.id in matched else 0.0)
    if not scores:
        return CalibrationMetrics(0.0, 0.0, 0.0, (), 0)
    target_count = sum(len(items) for items in annotations_by_prompt.values())

    score_array = np.asarray(scores, dtype=np.float64)
    correct_array = np.asarray(correct, dtype=np.float64)
    order = np.argsort(score_array)
    groups = np.array_split(order, min(bin_count, order.size))
    table: list[tuple[float, float, int]] = []
    error = 0.0
    worst = 0.0
    for group in groups:
        if group.size == 0:
            continue
        mean_score = float(score_array[group].mean())
        accuracy = float(correct_array[group].mean())
        gap = abs(mean_score - accuracy)
        error += gap * group.size / score_array.size
        worst = max(worst, gap)
        table.append((mean_score, accuracy, int(group.size)))
    return CalibrationMetrics(
        ece=error,
        mce=worst,
        overconfidence=float(score_array.mean() - correct_array.mean()),
        bins=tuple(table),
        scored=int(score_array.size),
        mean_score=float(score_array.mean()),
        accuracy=float(correct_array.mean()),
        accuracy_ceiling=min(1.0, target_count / float(score_array.size)),
    )


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
    own_thresholds = [
        thresholds[max(range(len(thresholds)), key=lambda i: f1_by_prompt[prompt][i])]
        for prompt in prompt_names
        if thresholds
    ]
    spread = (
        float(np.percentile(own_thresholds, 90) - np.percentile(own_thresholds, 10))
        if len(own_thresholds) > 1
        else 0.0
    )
    return ThresholdMetrics(
        macro_f1=macro_f1,
        threshold=threshold,
        loo_macro_f1=statistics.fmean(held_out_f1) if held_out_f1 else 0.0,
        prompt_threshold_spread=spread,
    )


def evaluate_config(
    candidates_by_prompt: Mapping[str, list[EnrichedCandidate]],
    annotations: Sequence[Annotation],
    geo_transform: GeoTransform,
    params: LocalizationParams,
    *,
    group_radius_m: float,
    num_results: int,
    near_m: float = DEFAULT_NEAR_M,
    prototypes_by_prompt: Mapping[str, PromptPrototypes] | None = None,
    vlm_scores_by_prompt: Mapping[str, Mapping[int, float]] | None = None,
) -> ConfigMetrics:
    """Evaluate one in-process localization configuration on all prompts.

    Args:
        candidates_by_prompt: Cached enriched candidates keyed by prompt.
        annotations: Benchmark annotations.
        geo_transform: Map coordinate transform used by localization.
        params: Association, filtering, and ranking parameters.
        group_radius_m: Radius for the grouped ground-truth view.
        num_results: Maximum returned clusters, applied consistently to the params.
        near_m: Maximum distance for assigning an annotation proxy to a detection.
        prototypes_by_prompt: Reviewed embeddings, needed only by a rescorer.
        vlm_scores_by_prompt: `p(yes)` per candidate, needed only by a VLM gate.

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
    partitions: list[PartitionMetrics] = []
    hotas: list[HotaMetrics] = []
    # Pooled over annotations, not averaged over prompts: one fragmented annotation
    # weighs the same wherever it is, which is what makes the per-class split below
    # comparable with the overall figure.
    fragmentation_all: list[int] = []
    fragmentation_by_class: dict[str, list[tuple[int, int]]] = {}

    for prompt, prompt_annotations in annotations_by_prompt.items():
        localizations, selected, cluster_labels = localize_from_enriched_candidates(
            apply_feedback(
                candidates_by_prompt[prompt],
                effective_params,
                (prototypes_by_prompt or {}).get(prompt),
                (vlm_scores_by_prompt or {}).get(prompt),
            ),
            geo_transform,
            effective_params,
            return_cluster_labels=True,
        )
        nearest, distance = nearest_annotation_distances(selected, prompt_annotations)
        annotation_labels = np.where(distance <= near_m, nearest, -1)
        partitions.append(partition_metrics(cluster_labels, annotation_labels))
        prompt_hota = hota_metrics(
            np.asarray(cluster_labels, dtype=np.int64),
            nearest,
            distance,
            len(prompt_annotations),
        )
        hotas.append(prompt_hota)
        if effective_params.vlm_gate == "cluster":
            prompt_scores = (vlm_scores_by_prompt or {}).get(prompt)
            if prompt_scores is None:
                raise ValueError(
                    "The cluster gate needs VLM scores; run the sweep with --with-vlm"
                )
            localizations = apply_cluster_gate(
                localizations, effective_params, prompt_scores
            )
        counts = fragmentation_counts(cluster_labels, annotation_labels)
        for annotation_index, entry in counts.items():
            fragmentation_all.append(entry[0])
            fragmentation_by_class.setdefault(
                prompt_annotations[annotation_index].class_name, []
            ).append(entry)
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
            mean_clusters_per_annotation=(
                statistics.fmean(entry[0] for entry in counts.values())
                if counts
                else 0.0
            ),
            covered_annotations=len(counts),
            hota=prompt_hota.hota,
            det_a=prompt_hota.det_a,
            ass_a=prompt_hota.ass_a,
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
    calibration = calibration_metrics(predictions_by_prompt, annotations_by_prompt)
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
        pair_precision=(
            statistics.fmean(item.pair_precision for item in partitions)
            if partitions
            else 0.0
        ),
        pair_recall=(
            statistics.fmean(item.pair_recall for item in partitions)
            if partitions
            else 0.0
        ),
        pair_f1=(
            statistics.fmean(item.pair_f1 for item in partitions) if partitions else 0.0
        ),
        rand_index=(
            statistics.fmean(item.rand_index for item in partitions)
            if partitions
            else 0.0
        ),
        labelled_detections=(
            statistics.fmean(item.labelled_detections for item in partitions)
            if partitions
            else 0.0
        ),
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
        mean_clusters_per_annotation=(
            statistics.fmean(fragmentation_all) if fragmentation_all else 0.0
        ),
        covered_annotations=len(fragmentation_all),
        fragmentation_by_class={
            class_name: ClassFragmentation(
                mean_clusters_per_annotation=statistics.fmean(
                    entry[0] for entry in values
                ),
                covered_annotations=len(values),
                mean_detections_per_annotation=statistics.fmean(
                    entry[1] for entry in values
                ),
            )
            for class_name, values in sorted(fragmentation_by_class.items())
        },
        ece=calibration.ece,
        mce=calibration.mce,
        overconfidence=calibration.overconfidence,
        accuracy_ceiling=calibration.accuracy_ceiling,
        threshold_spread_strict=strict.prompt_threshold_spread,
        reliability=calibration.bins,
        det_a=statistics.fmean(item.det_a for item in hotas) if hotas else 0.0,
        ass_a=statistics.fmean(item.ass_a for item in hotas) if hotas else 0.0,
        hota=statistics.fmean(item.hota for item in hotas) if hotas else 0.0,
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
    row.pop("fragmentation_by_class")
    row.pop("reliability")
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
    near_m: float = DEFAULT_NEAR_M,
    prototypes_by_prompt: Mapping[str, PromptPrototypes] | None = None,
    vlm_scores_by_prompt: Mapping[str, Mapping[int, float]] | None = None,
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
        near_m: Maximum distance for assigning an annotation proxy to a detection.
        prototypes_by_prompt: Reviewed embeddings, needed only by a rescorer.

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
            near_m=near_m,
            prototypes_by_prompt=prototypes_by_prompt,
            vlm_scores_by_prompt=vlm_scores_by_prompt,
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
            apply_feedback(prompt_candidates, params), geo_transform, params
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
    parser.add_argument("--near-m", type=float, default=DEFAULT_NEAR_M)
    parser.add_argument("--default-accuracy", type=float, default=DEFAULT_ACCURACY_M)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--exact-device", default="cpu")
    parser.add_argument(
        "--exact-retrieval",
        action="store_true",
        help=(
            "Rank candidates by a brute-force scan instead of the online service, the "
            "only way past its 1000-candidate ceiling (pgvector caps hnsw.ef_search)."
        ),
    )
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help=(
            "Score every cached candidate with the VLM gate model, once, and reuse "
            "the table for every grid row naming a `vlm_gate`."
        ),
    )
    parser.add_argument("--vlm-model", default=GateConfig().model_id)
    parser.add_argument(
        "--cutout-root",
        type=Path,
        default=None,
        help=(
            "Rendered cutouts for a converted v1 index, whose thumbnail keys are "
            "virtual. See toolbox.benchmark.render_benchmark_cutouts."
        ),
    )
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help=(
            "Resolve the map's review prototypes so a grid can sweep feedback_alpha, "
            "feedback_beta and feedback_normalization. Uses its own cache entries."
        ),
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bricks-base-url", default="http://127.0.0.1:45679")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run candidate fetching, optional verification, and the requested sweep."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.near_m <= 0.0:
        raise ValueError("--near-m must be positive")
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
        with_feedback=args.with_feedback,
        exact=args.exact_retrieval,
        exact_device=args.exact_device,
    )
    grid = _load_grid(args.grid)
    manifest = map_manifest.load_map_manifest(map_path)
    prototypes = None
    if any(entry.get("rescorer") for entry in grid):
        if manifest.geo_ref_id is None:
            raise ValueError(f"{manifest.path}: manifest records no geo_ref_id")
        prototypes = fetch_prompt_prototypes(
            map_path,
            list(prompt_candidates),
            geo_ref_id=int(manifest.geo_ref_id),
            map_id=manifest.map_id,
        )
    vlm_scores = None
    if args.with_vlm or any(entry.get("vlm_gate") for entry in grid):
        scorer = VlmYesNoScorer(GateConfig(model_id=args.vlm_model))
        vlm_scores = {
            prompt: vlm_scores_module.load_or_score(
                prompt,
                prompt_candidates[prompt],
                map_path=map_path,
                map_id=manifest.map_id,
                cache_dir=args.cache_dir.expanduser().resolve() / "vlm",
                scorer=scorer,
                cutout_root=(
                    args.cutout_root.expanduser().resolve()
                    if args.cutout_root is not None
                    else None
                ),
                refresh=args.refresh,
            )
            for prompt in prompt_candidates
        }
        for prompt, scores in vlm_scores.items():
            logger.info(
                "Prompt %r: VLM gate covers %.1f%% of the candidates",
                prompt,
                100.0 * vlm_scores_module.coverage(prompt_candidates[prompt], scores),
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
            map_id=manifest.map_id,
            bricks_base_url=args.bricks_base_url,
            timeout_s=args.timeout,
        )
    sweep(
        grid,
        prompt_candidates,
        annotations,
        pose_source.geo_transform,
        group_radius_m=args.group_annotation_radius_m,
        num_results=args.num_results,
        out_dir=args.out_dir,
        base_params=base_params,
        near_m=args.near_m,
        prototypes_by_prompt=prototypes,
        vlm_scores_by_prompt=vlm_scores,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
