"""Why a returned cluster was wrong, and what fixing each reason would be worth.

TIDE (Bolya et al., ECCV 2020) transposed to this pipeline. A recall figure says how
much was lost; this says *where*, by typing every wrong cluster and then measuring what
recall would be if that one type of mistake had not happened. The point is
prioritisation — six numbers that rank the next piece of work, which no aggregate score
can do.

**The base metric is not mAP, deliberately.** TIDE reports delta-mAP, but strict mAP on
this benchmark is *paid* for fragmentation: the best strict-mAP configuration ever
measured here shatters objects and scores 0.802 with a pair recall of 0.265. "Suppress
the duplicates" would therefore come out near zero and the tool would mislead exactly
where it is most useful. Recall is the base instead, on two axes:

- **`delta_recall_at_k`** — rank-aware, which the *duplicate* type needs to be definable
  at all ("a better-scoring cluster already took this object"), and not fooled by
  splitting: a fragment that ranks well still matches one object while its siblings
  occupy top-k slots and cost recall, which is the right sign. It is also where the
  pipeline actually loses, 0.759 retrieved down to 0.044 at k=1 on vinci.
- **`delta_recall_all`** — the same fix with every returned cluster allowed. The pair
  is the attribution: an error that costs `at_k` but not `all` is purely a ranking
  problem, one that costs both loses the object outright. This replaces the
  delta-det_a/delta-ass_a pair originally considered, which would need
  cluster-to-detection membership this module never sees, and which says less: these
  types are defined on clusters, so the stage they live in is "ranking" or "retrieval",
  not "detection" or "association".

**Two types are gated on measured separability.** Telling a cluster that landed on the
wrong class from one that landed on nothing needs annotations of different classes to
sit further apart than the matching radius. On bbhotel 98.8 % of annotations have a
different class inside their radius, so `classification` there is noise with a name.
`separability` computes that share and the report withholds those two columns rather
than printing them — by number, never by map name, so a better-annotated map is handled
with no code change. See `docs/adr/0009-ground-truth-annotation-contract.md`.

**Matching is greedy by rank here, not by distance.** The benchmark's
`match_predictions` sorts candidate pairs by distance, which is right for a
threshold-free score and wrong for this: a *duplicate* is defined by a better-scoring
cluster having already taken the object, so the walk has to follow the ranking. Recall
figures from this module are therefore close to, but not identical with, the sweep's
`recall_at`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from toolbox.benchmark.object_search_http_benchmark import (
    LOCALISATION_FACTOR,
    Annotation,
    Prediction,
    haversine_m,
    match_radius_m,
)

#: Cutoff the headline contribution is measured at. Ten is the shallowest cutoff whose
#: recall still has room to move on the current maps: R@1 is 0.044 and saturates every
#: comparison, R@20 is nearly flat across association configurations.
HEADLINE_K = 10
#: Above this share of annotations having a *different* class inside their own matching
#: radius, the classification columns are withheld. A third means one attribution in
#: three would be a coin toss, which is not a diagnosis.
SEPARABILITY_LIMIT = 1.0 / 3.0
#: Every verdict a prediction can receive, plus the one an unfound object receives.
#: `correct` is carried so the counts add up to what was actually returned.
ERROR_TYPES: tuple[str, ...] = (
    "correct",
    "duplicate",
    "classification",
    "localisation",
    "classification_localisation",
    "background",
    "missed",
)
#: The types `separability` can withdraw.
GATED_TYPES: tuple[str, ...] = ("classification", "classification_localisation")
#: The types whose fix is to delete the prediction — every way of being a false
#: positive.
FALSE_POSITIVE_TYPES: tuple[str, ...] = (
    "duplicate",
    "classification",
    "classification_localisation",
    "background",
)


@dataclass(frozen=True)
class Separability:
    """Whether this map's ground truth can attribute a wrong-class prediction at all."""

    #: Share of annotations with an annotation of another class inside their own radius.
    overlap_share: float
    #: Median distance to the nearest annotation of another class, metres.
    median_other_class_m: float
    #: Annotations the share rests on.
    annotations: int

    @property
    def classification_measurable(self) -> bool:
        """Is the classification split worth printing on this map."""
        return self.overlap_share <= SEPARABILITY_LIMIT

    @property
    def reason(self) -> str | None:
        """Why the gated columns were withheld, or None when they were not."""
        if self.classification_measurable:
            return None
        return (
            f"{self.overlap_share:.1%} des annotations ont une autre classe dans leur "
            f"propre rayon (médiane {self.median_other_class_m:.2f} m) — au-delà de "
            f"{SEPARABILITY_LIMIT:.0%} l'attribution serait un tirage au sort"
        )


def separability(annotations: Sequence[Annotation]) -> Separability:
    """Measure whether classes are further apart than the radius that matches them.

    Two annotations are only compared when they share a `level`: a different floor is
    not a confusable neighbour whatever the horizontal distance says.

    Args:
        annotations: Every annotation of the map, all prompts together.

    Returns:
        The overlap share and the median distance behind it. An empty or single-class
        ground truth reports a share of zero — nothing can be confused.
    """
    if len(annotations) < 2:
        return Separability(0.0, float("inf"), len(annotations))
    overlapping = 0
    nearest: list[float] = []
    for index, annotation in enumerate(annotations):
        radius = match_radius_m(annotation)
        distances = [
            haversine_m(annotation.lat, annotation.lng, other.lat, other.lng)
            for position, other in enumerate(annotations)
            if position != index
            and other.class_name != annotation.class_name
            and (
                annotation.level is None
                or other.level is None
                or other.level == annotation.level
            )
        ]
        if not distances:
            continue
        closest = min(distances)
        nearest.append(closest)
        if closest <= radius:
            overlapping += 1
    if not nearest:
        return Separability(0.0, float("inf"), len(annotations))
    return Separability(
        overlap_share=overlapping / len(annotations),
        median_other_class_m=float(np.median(nearest)),
        annotations=len(annotations),
    )


@dataclass(frozen=True)
class ErrorContribution:
    """One error type: how often it happened and what not happening would be worth."""

    error_type: str
    count: int
    #: Recall gained at `HEADLINE_K` if this type of mistake had not been made.
    delta_recall_at_k: float
    #: The same with every returned cluster allowed. Zero here with a positive
    #: `delta_recall_at_k` means the objects were found and merely ranked too low.
    delta_recall_all: float
    #: False when `separability` withheld this type; `reason` then says why.
    measurable: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class Decomposition:
    """The whole table, plus the baseline the deltas are against."""

    separability: Separability
    contributions: tuple[ErrorContribution, ...]
    baseline_recall_at_k: float
    baseline_recall_all: float
    #: Predictions typed, and annotations they were typed against.
    predictions: int
    annotations: int

    def by_type(self) -> dict[str, ErrorContribution]:
        """The contributions keyed by their error type."""
        return {item.error_type: item for item in self.contributions}


def _ranked(predictions: Sequence[Prediction]) -> list[Prediction]:
    """Predictions best first, ties broken by id so a run is reproducible."""
    return sorted(predictions, key=lambda item: (-item.score, item.id))


def _match_by_rank(
    predictions: Sequence[Prediction],
    annotations: Sequence[Annotation],
    *,
    radius_factor: float = 1.0,
) -> dict[str, str]:
    """Walk the ranking, giving each prediction the nearest free annotation it reaches.

    Args:
        predictions: Already ranked, best first.
        annotations: Candidate targets, one match each.
        radius_factor: Multiplies every annotation's radius. Above one it answers "would
            this prediction have matched had it been placed better", which is how the
            localisation fix is applied.

    Returns:
        Prediction id mapped to the annotation id it took.
    """
    taken: set[str] = set()
    matched: dict[str, str] = {}
    for prediction in predictions:
        best: tuple[float, str] | None = None
        for annotation in annotations:
            if annotation.id in taken:
                continue
            distance = haversine_m(
                prediction.lat, prediction.lng, annotation.lat, annotation.lng
            )
            if distance <= match_radius_m(annotation) * radius_factor and (
                best is None or distance < best[0]
            ):
                best = (distance, annotation.id)
        if best is not None:
            taken.add(best[1])
            matched[prediction.id] = best[1]
    return matched


def _nearest_other_class(
    prediction: Prediction, others: Sequence[Annotation]
) -> tuple[float, Annotation] | None:
    """The closest annotation of a different prompt, with its distance."""
    best: tuple[float, Annotation] | None = None
    for annotation in others:
        distance = haversine_m(
            prediction.lat, prediction.lng, annotation.lat, annotation.lng
        )
        if best is None or distance < best[0]:
            best = (distance, annotation)
    return best


def type_prompt_predictions(
    predictions: Sequence[Prediction],
    own: Sequence[Annotation],
    others: Sequence[Annotation],
) -> tuple[dict[str, str], list[str]]:
    """Give every prediction of one prompt a verdict, and list the unfound objects.

    Verdicts follow TIDE's precedence, so a prediction that could be described two ways
    gets the most specific one. In order: it matched a free object (`correct`); it
    reached an object another, better-scoring cluster had already taken (`duplicate`);
    it is near an object of its own class but outside its radius (`localisation`); it is
    on an object of another class (`classification`, or
    `classification_localisation` when it is merely near one); it is near nothing at all
    (`background`).

    Args:
        predictions: This prompt's returned clusters, any order.
        own: Annotations of this prompt.
        others: Annotations of every other prompt.

    Returns:
        Prediction id mapped to its verdict, and the ids of annotations no prediction
        matched — the `missed` objects.
    """
    ranked = _ranked(predictions)
    matched = _match_by_rank(ranked, own)
    verdicts: dict[str, str] = {}
    for prediction in ranked:
        if prediction.id in matched:
            verdicts[prediction.id] = "correct"
            continue
        own_distances = [
            (
                haversine_m(
                    prediction.lat, prediction.lng, annotation.lat, annotation.lng
                ),
                annotation,
            )
            for annotation in own
        ]
        inside_own = [
            annotation
            for distance, annotation in own_distances
            if distance <= match_radius_m(annotation)
        ]
        if inside_own:
            # It reached an object of its own class, but the walk found that object
            # already taken by something ranked higher.
            verdicts[prediction.id] = "duplicate"
            continue
        near_own = any(
            distance <= match_radius_m(annotation) * LOCALISATION_FACTOR
            for distance, annotation in own_distances
        )
        other = _nearest_other_class(prediction, others)
        if other is not None:
            distance, annotation = other
            radius = match_radius_m(annotation)
            if distance <= radius:
                verdicts[prediction.id] = "classification"
                continue
            if distance <= radius * LOCALISATION_FACTOR and not near_own:
                verdicts[prediction.id] = "classification_localisation"
                continue
        verdicts[prediction.id] = "localisation" if near_own else "background"
    taken = set(matched.values())
    missed = [annotation.id for annotation in own if annotation.id not in taken]
    return verdicts, missed


def _recall(
    predictions: Sequence[Prediction],
    annotations: Sequence[Annotation],
    k: int | None,
    *,
    radius_factor: float = 1.0,
    granted: int = 0,
) -> float:
    """Share of annotations matched by the top `k` predictions, `None` meaning all.

    `granted` counts objects a fix declares found without a prediction — the missed
    type's repair — and the total is capped at one, so granting cannot exceed the
    ground truth.
    """
    if not annotations:
        return 0.0
    ranked = _ranked(predictions)
    if k is not None:
        ranked = ranked[:k]
    found = len(
        set(_match_by_rank(ranked, annotations, radius_factor=radius_factor).values())
    )
    return min(1.0, (found + granted) / len(annotations))


def _fixed_recall(
    error_type: str,
    predictions: Sequence[Prediction],
    own: Sequence[Annotation],
    verdicts: Mapping[str, str],
    missed: Sequence[str],
    k: int | None,
) -> float:
    """Recall for one prompt once this one type of mistake is repaired.

    Each repair has one unambiguous meaning. Deleting a false positive lets everything
    below it move up, which is the whole reason a rank-aware base was chosen. The
    localisation repair widens the radius instead of deleting, because the cluster was
    about the right object. The missed repair grants those objects, since there is no
    prediction to move.
    """
    if error_type in FALSE_POSITIVE_TYPES:
        kept = [
            prediction
            for prediction in predictions
            if verdicts.get(prediction.id) != error_type
        ]
        return _recall(kept, own, k)
    if error_type == "localisation":
        return _recall(predictions, own, k, radius_factor=LOCALISATION_FACTOR)
    if error_type == "missed":
        return _recall(predictions, own, k, granted=len(missed))
    return _recall(predictions, own, k)


def decompose(
    predictions_by_prompt: Mapping[str, Sequence[Prediction]],
    annotations: Sequence[Annotation],
    k: int = HEADLINE_K,
) -> Decomposition:
    """Type every returned cluster and price every error type, over all prompts.

    Args:
        predictions_by_prompt: Returned clusters per prompt, any order.
        annotations: Every annotation of the map. A prompt's own targets are the ones
            whose `prompt` matches; the rest are what a classification error lands on,
            which is why the whole set is needed rather than one prompt's slice.
        k: Cutoff for the headline contribution.

    Returns:
        The counts and the two deltas per type, macro-averaged over prompts, plus the
        separability verdict that says whether two of them may be read.
    """
    by_prompt: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        by_prompt.setdefault(annotation.prompt, []).append(annotation)
    gate = separability(annotations)

    counts = dict.fromkeys(ERROR_TYPES, 0)
    baseline_at_k: list[float] = []
    baseline_all: list[float] = []
    fixed_at_k: dict[str, list[float]] = {name: [] for name in ERROR_TYPES}
    fixed_all: dict[str, list[float]] = {name: [] for name in ERROR_TYPES}
    typed = 0
    for prompt, own in by_prompt.items():
        predictions = list(predictions_by_prompt.get(prompt, []))
        others = [
            annotation for annotation in annotations if annotation.prompt != prompt
        ]
        verdicts, missed = type_prompt_predictions(predictions, own, others)
        typed += len(verdicts)
        for verdict in verdicts.values():
            counts[verdict] += 1
        counts["missed"] += len(missed)
        baseline_at_k.append(_recall(predictions, own, k))
        baseline_all.append(_recall(predictions, own, None))
        for name in ERROR_TYPES:
            if name == "correct":
                continue
            fixed_at_k[name].append(
                _fixed_recall(name, predictions, own, verdicts, missed, k)
            )
            fixed_all[name].append(
                _fixed_recall(name, predictions, own, verdicts, missed, None)
            )

    base_at_k = float(np.mean(baseline_at_k)) if baseline_at_k else 0.0
    base_all = float(np.mean(baseline_all)) if baseline_all else 0.0
    contributions: list[ErrorContribution] = []
    for name in ERROR_TYPES:
        if name == "correct":
            continue
        withheld = name in GATED_TYPES and not gate.classification_measurable
        contributions.append(
            ErrorContribution(
                error_type=name,
                count=counts[name],
                delta_recall_at_k=(
                    0.0
                    if withheld or not fixed_at_k[name]
                    else float(np.mean(fixed_at_k[name])) - base_at_k
                ),
                delta_recall_all=(
                    0.0
                    if withheld or not fixed_all[name]
                    else float(np.mean(fixed_all[name])) - base_all
                ),
                measurable=not withheld,
                reason=gate.reason if withheld else None,
            )
        )
    contributions.sort(key=lambda item: -item.delta_recall_at_k)
    return Decomposition(
        separability=gate,
        contributions=tuple(contributions),
        baseline_recall_at_k=base_at_k,
        baseline_recall_all=base_all,
        predictions=typed,
        annotations=len(annotations),
    )


def report_lines(decomposition: Decomposition, k: int = HEADLINE_K) -> list[str]:
    """The decomposition as fixed-width lines, most valuable fix first."""
    gate = decomposition.separability
    lines = [
        f"  {decomposition.predictions} clusters typés contre "
        f"{decomposition.annotations} annotations",
        f"  rappel de base : R@{k} {decomposition.baseline_recall_at_k:.3f}, "
        f"tous clusters {decomposition.baseline_recall_all:.3f}",
        f"  séparabilité des classes : {gate.overlap_share:.1%} de recouvrement, "
        f"médiane {gate.median_other_class_m:.2f} m",
        "",
        f"  {'type d erreur':30s} {'nombre':>8s} {f'dR@{k}':>9s} {'dR tous':>9s}",
    ]
    for item in decomposition.contributions:
        if not item.measurable:
            lines.append(f"  {item.error_type:30s} {item.count:8d}    (non mesurable)")
            continue
        lines.append(
            f"  {item.error_type:30s} {item.count:8d} "
            f"{item.delta_recall_at_k:+9.3f} {item.delta_recall_all:+9.3f}"
        )
    if gate.reason:
        lines.append("")
        lines.append(f"  colonnes retenues : {gate.reason}")
    lines.append("")
    lines.append(
        "  (un type qui coûte du R@k sans coûter de R tous clusters est un problème de"
        " classement seul ; un type qui coûte les deux perd l'objet)"
    )
    return lines
