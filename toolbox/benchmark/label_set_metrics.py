"""Score a label against a *set* of acceptable ones, not against a single string.

OpenLex3D's contribution, transposed. A benchmark that stores one class per object
scores `seat` against `chair` as a false positive, which measures the vocabulary rather
than the pipeline. The fix is to annotate each object with several label sets ordered by
precision — exact synonyms, images *of* the thing, things that merely look like it, and
labels the crop dragged in — and to score a prediction by which set its label falls in.
See `docs/adr/0009-ground-truth-annotation-contract.md` for the annotation contract and
`Annotation.synonyms` and friends for where the sets are read from.

Two metrics, both from the paper:

- **top-N frequency** — does any of the N labels a model proposes fall in category C.
  Object-level and binary, so it answers "was the object recognised at all, allowing for
  synonymy" and nothing more.
- **set ranking** — does the proposed *order* match the ideal order of the categories. A
  model that puts a synonym above a visually-similar label knows something a model that
  reverses them does not, and top-N frequency cannot tell them apart. Implemented as
  nDCG against the object's own best possible ordering, so it is bounded in `[0, 1]` and
  comparable across objects with different set sizes.

**This module never produces the labels.** It takes an already-ranked list per object,
because the ranking can come from the detector's own `label` column, from
`gdino_labels.encode_classes` scored against a cluster's embedding, or from a VLM — and
which of those is right is a separate question from how to score them. `rank_labels`
is offered for the embedding case and is pure numpy: no model is loaded here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from toolbox.benchmark.object_search_http_benchmark import Annotation

#: The categories, most precise first. This order *is* the metric's opinion, and it is
#: the paper's: a synonym names the object, a depiction names a picture of it, a
#: visually-similar label names a different object that looks like it, and clutter names
#: something the crop caught by accident.
LABEL_CATEGORIES: tuple[str, ...] = (
    "synonyms",
    "depictions",
    "visually_similar",
    "clutter",
)
#: Gain per category for the ranking metric. Clutter earns a little because it is not
#: *wrong* — the object is in the crop, the crop is just loose — while a label in no
#: category at all earns nothing. These four numbers are the knob; the order they impose
#: is not negotiable, it comes from `LABEL_CATEGORIES`.
LABEL_GAINS: Mapping[str, float] = {
    "synonyms": 3.0,
    "depictions": 2.0,
    "visually_similar": 1.0,
    "clutter": 0.5,
}
#: Labels proposed per object that the metrics look at.
DEFAULT_TOP_N = 5


def _normalise(label: str) -> str:
    """Labels compare case- and whitespace-insensitively, and nothing else."""
    return " ".join(label.strip().lower().split())


def _category_members(annotation: Annotation) -> dict[str, frozenset[str]]:
    """The annotation's four sets, normalised and blank-free.

    `class_name` joins the synonyms. A blank never becomes a member: it would match a
    prediction that carries no label at all.
    """

    def members(labels: Sequence[str]) -> frozenset[str]:
        return frozenset(
            normalised for normalised in map(_normalise, labels) if normalised
        )

    return {
        "synonyms": members(annotation.accepted_labels),
        "depictions": members(annotation.depictions),
        "visually_similar": members(annotation.visually_similar),
        "clutter": members(annotation.clutter),
    }


def category_of(label: str, annotation: Annotation) -> str | None:
    """Which category a label falls in for this object, most precise winning.

    A label listed in two categories is credited with the more precise one, so an
    annotator who repeats a word cannot lower a score by accident.

    Args:
        label: The proposed label.
        annotation: The object it is proposed for.

    Returns:
        A member of `LABEL_CATEGORIES`, or None when the label is in none of them.
    """
    members = _category_members(annotation)
    wanted = _normalise(label)
    for category in LABEL_CATEGORIES:
        if wanted in members[category]:
            return category
    return None


def has_label_sets(annotation: Annotation) -> bool:
    """Does this annotation carry anything beyond its class name.

    An annotation predating ADR 0009 has only `class_name`, which `accepted_labels`
    turns into a one-element synonym set. Scoring it would report the old single-string
    behaviour under a new name, so the aggregate excludes it and says how many it
    excluded.
    """
    return bool(
        annotation.synonyms
        or annotation.depictions
        or annotation.visually_similar
        or annotation.clutter
    )


def top_n_frequency(
    ranked_labels: Sequence[str], annotation: Annotation, n: int = DEFAULT_TOP_N
) -> dict[str, bool]:
    """Whether the top `n` labels reach each category, one flag per category."""
    reached = dict.fromkeys(LABEL_CATEGORIES, False)
    for label in list(ranked_labels)[:n]:
        category = category_of(label, annotation)
        if category is not None:
            reached[category] = True
    return reached


def set_ranking(
    ranked_labels: Sequence[str], annotation: Annotation, n: int = DEFAULT_TOP_N
) -> float:
    """How close the proposed order is to this object's ideal order, in `[0, 1]`.

    nDCG with the category gains: the discount punishes a synonym ranked below a
    visually-similar label without punishing the model for the size of the annotator's
    sets, because the denominator is that annotator's own best achievable ordering.

    Args:
        ranked_labels: Proposed labels, best first.
        annotation: The object they are proposed for.
        n: How many proposals count.

    Returns:
        1.0 for the ideal order, 0.0 when nothing proposed is in any category. NaN when
        the object has no non-empty category, since there is no ideal to compare to.
    """
    members = _category_members(annotation)
    ideal_gains = sorted(
        (
            LABEL_GAINS[category]
            for category in LABEL_CATEGORIES
            for _ in range(len(members[category]))
        ),
        reverse=True,
    )
    if not ideal_gains:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    gains = np.zeros(n)
    for position, label in enumerate(list(ranked_labels)[:n]):
        category = category_of(label, annotation)
        if category is not None:
            gains[position] = LABEL_GAINS[category]
    ideal = np.zeros(n)
    ideal[: min(n, len(ideal_gains))] = ideal_gains[:n]
    best = float((ideal * discounts).sum())
    return float((gains * discounts).sum() / best) if best > 0 else float("nan")


@dataclass(frozen=True)
class LabelSetReport:
    """Aggregate over the objects that carry label sets, and how many did not."""

    #: Share of scored objects whose top-N reached each category.
    top_n: dict[str, float] = field(default_factory=dict)
    #: Mean set ranking over the scored objects.
    mean_set_ranking: float = float("nan")
    #: Objects with label sets, and the total offered.
    scored: int = 0
    annotations: int = 0
    #: Objects that had label sets but no proposed labels at all.
    unproposed: int = 0
    n: int = DEFAULT_TOP_N

    @property
    def coverage(self) -> float:
        """Share of annotations carrying label sets. Zero means nothing was measured."""
        return self.scored / self.annotations if self.annotations else 0.0


def evaluate_label_sets(
    ranked_by_annotation: Mapping[str, Sequence[str]],
    annotations: Sequence[Annotation],
    n: int = DEFAULT_TOP_N,
) -> LabelSetReport:
    """Score proposed labels against the annotations that carry sets.

    Objects with no label sets are skipped rather than scored against their bare class
    name: that would report the old single-string behaviour under a new name. The
    returned `coverage` is what says whether the figures mean anything yet.

    Args:
        ranked_by_annotation: Proposed labels per annotation id, best first.
        annotations: Every annotation, with or without label sets.
        n: How many proposals count.

    Returns:
        Per-category top-N shares, the mean set ranking, and the coverage behind them.
    """
    scored = [item for item in annotations if has_label_sets(item)]
    if not scored:
        return LabelSetReport(annotations=len(annotations), n=n)
    reached = dict.fromkeys(LABEL_CATEGORIES, 0)
    rankings: list[float] = []
    unproposed = 0
    for annotation in scored:
        proposed = list(ranked_by_annotation.get(annotation.id, ()))
        if not proposed:
            unproposed += 1
        for category, hit in top_n_frequency(proposed, annotation, n).items():
            reached[category] += int(hit)
        value = set_ranking(proposed, annotation, n)
        if not np.isnan(value):
            rankings.append(value)
    return LabelSetReport(
        top_n={
            category: reached[category] / len(scored) for category in LABEL_CATEGORIES
        },
        mean_set_ranking=float(np.mean(rankings)) if rankings else float("nan"),
        scored=len(scored),
        annotations=len(annotations),
        unproposed=unproposed,
        n=n,
    )


def vocabulary(annotations: Sequence[Annotation]) -> tuple[str, ...]:
    """Every label any annotation mentions, in a stable order.

    The set a label ranking is produced *against*. It has to be the whole benchmark's
    vocabulary rather than one object's: proposing `chair` for a chair is only evidence
    if `table` was on offer too.
    """
    seen: dict[str, None] = {}
    for annotation in annotations:
        for category in LABEL_CATEGORIES:
            for label in _category_members(annotation)[category]:
                seen.setdefault(label, None)
    return tuple(sorted(seen))


def rank_labels(
    embedding: np.ndarray, vocabulary_vectors: np.ndarray, labels: Sequence[str]
) -> list[str]:
    """Order a vocabulary by cosine similarity to one object's embedding.

    Pure numpy on purpose: the text vectors come from
    `gdino_labels.encode_classes`, which loads MetaCLIP, and keeping that out of here
    means the metrics above can be tested and reasoned about without a model.

    Args:
        embedding: One object's embedding, any norm.
        vocabulary_vectors: Row-normalised label vectors, aligned with `labels`.
        labels: The vocabulary, aligned with `vocabulary_vectors`.

    Returns:
        The labels, most similar first.

    Raises:
        ValueError: If the vectors and the labels are not aligned.
    """
    if vocabulary_vectors.shape[0] != len(labels):
        raise ValueError("One vector per label is required")
    if not labels:
        return []
    unit = np.asarray(embedding, dtype=np.float32).ravel()
    unit = unit / max(float(np.linalg.norm(unit)), 1e-6)
    similarity = vocabulary_vectors @ unit
    return [labels[index] for index in np.argsort(-similarity)]


def report_lines(report: LabelSetReport) -> list[str]:
    """The label-set report as fixed-width lines, coverage stated first."""
    if not report.scored:
        return [
            f"  0/{report.annotations} annotations portent des ensembles de labels —"
            " rien à mesurer (voir ADR 0009)",
        ]
    lines = [
        f"  {report.scored}/{report.annotations} annotations portent des ensembles"
        f" ({report.coverage:.1%}), N={report.n}",
    ]
    if report.unproposed:
        lines.append(
            f"  {report.unproposed} sans aucun label proposé — comptées comme échecs"
        )
    for category in LABEL_CATEGORIES:
        lines.append(
            f"  top-{report.n} atteint {category:18s} {report.top_n[category]:6.1%}"
        )
    lines.append(
        f"  {'classement des ensembles (nDCG)':32s} {report.mean_set_ranking:6.3f}"
    )
    return lines
