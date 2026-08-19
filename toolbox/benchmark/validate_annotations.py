"""Check a map's annotations against the ADR 0009 contract, and say what they earn.

Run after an annotation session. It reports three things, in this order:

1. **what is missing** — per field, how many annotations carry it, because a field
   half-filled is worse than a field absent: the metrics reading it would silently score
   a biased slice of the map;
2. **what is inconsistent** — the same click recorded twice, one object under two
   extents, one class under contradictory synonym sets;
3. **what the map has earned** — whether the separability gate now opens the
   classification columns of `error_decomposition`, and whether the label-set metrics
   have anything to measure at all.

Nothing here judges the pipeline. It judges the ground truth, which is why it reads the
GeoJSON export and nothing else: no parquet, no poses, no service. See
`docs/plans/2026-08-18-cahier-des-charges-annotation.md` for the annotator's side and
`docs/adr/0009-ground-truth-annotation-contract.md` for why each field exists.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from toolbox.benchmark.annotation_store import (
    ANNOTATION_DB_FILENAME,
    annotation_database_path,
    load_store_annotations,
)
from toolbox.benchmark.error_decomposition import SEPARABILITY_LIMIT, separability
from toolbox.benchmark.label_set_metrics import LABEL_CATEGORIES, has_label_sets
from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    load_annotations,
    match_radius_m,
)

#: Where the toolbox exports the annotation store for the benchmark to read. Only
#: rewritten when a benchmark run starts, so it is not what this command reads by
#: default — see `toolbox.benchmark.annotation_store`.
DEFAULT_GEOJSON = Path("benchmark") / "annotations.geojson"
#: Accuracy given to a feature carrying none. It only affects annotations with no
#: `extent_m`, which is exactly the case the report is already complaining about.
DEFAULT_ACCURACY_M = 5.0
#: Below this share of annotations carrying a field, the field is reported as partial
#: rather than present: a metric reading it would be scoring a biased subset.
PARTIAL_LIMIT = 0.95
#: Degrees of latitude two clicks must differ by to be different clicks. 1e-7 is about a
#: centimetre — below any real gap between two objects, above float noise.
SAME_CLICK_DEG = 1e-7
#: An extent outside this range is a typo rather than an object.
PLAUSIBLE_EXTENT_M = (0.02, 20.0)
#: What each field unblocks, quoted back when it is missing.
FIELD_BLOCKS = {
    "extent_m": "les colonnes classification/localisation, et tout rayon honnête",
    "object_id": "le type doublon",
    "exhaustive_zone": "les types background et manqué",
    "labels.synonyms": "l'appariement par synonymes",
    "labels.depictions": "le set ranking (catégorie facultative)",
    "labels.visually_similar": "le set ranking (catégorie facultative)",
    "labels.clutter": "le set ranking (catégorie facultative)",
}


@dataclass
class Findings:
    """Everything the check has to say, grouped by how it should be read."""

    missing: list[str] = field(default_factory=list)
    inconsistent: list[str] = field(default_factory=list)
    earned: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """Is anything wrong enough that a measurement would be misread."""
        return bool(self.inconsistent)


def field_coverage(annotations: Sequence[Annotation]) -> dict[str, float]:
    """Share of annotations carrying each contract field."""
    total = len(annotations)
    if not total:
        return {}
    counted = {
        "object_id": sum(item.object_id is not None for item in annotations),
        "extent_m": sum(item.extent_m is not None for item in annotations),
        "exhaustive_zone": sum(
            item.exhaustive_zone is not None for item in annotations
        ),
        "labels.synonyms": sum(bool(item.synonyms) for item in annotations),
        "labels.depictions": sum(bool(item.depictions) for item in annotations),
        "labels.visually_similar": sum(
            bool(item.visually_similar) for item in annotations
        ),
        "labels.clutter": sum(bool(item.clutter) for item in annotations),
    }
    return {name: count / total for name, count in counted.items()}


def check_missing(annotations: Sequence[Annotation], findings: Findings) -> None:
    """Report each field's coverage, worst first, naming what its absence blocks."""
    for name, share in sorted(field_coverage(annotations).items(), key=lambda x: x[1]):
        if share >= PARTIAL_LIMIT:
            findings.missing.append(f"  {name:24s} {share:6.1%}  complet")
            continue
        state = "absent" if share == 0.0 else "PARTIEL — pire qu'absent"
        findings.missing.append(
            f"  {name:24s} {share:6.1%}  {state} → bloque {FIELD_BLOCKS[name]}"
        )


def duplicate_clicks(annotations: Sequence[Annotation]) -> list[list[Annotation]]:
    """Groups of annotations that are one click recorded more than once.

    Same class and the same position to the centimetre. Two real objects of one class
    are never that close, so this is the insertion defect rather than a dense scene, and
    `object_id` is what settles it: distinct ids on one position is a contradiction the
    annotator has to resolve, not something this tool may assume either way.
    """
    buckets: dict[tuple[str, int, int], list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        key = (
            annotation.class_name,
            round(annotation.lat / SAME_CLICK_DEG),
            round(annotation.lng / SAME_CLICK_DEG),
        )
        buckets[key].append(annotation)
    return [group for group in buckets.values() if len(group) > 1]


def check_duplicates(annotations: Sequence[Annotation], findings: Findings) -> None:
    """Report co-located same-class annotations and whether ids explain them."""
    groups = duplicate_clicks(annotations)
    if not groups:
        return
    affected = sum(len(group) for group in groups)
    unexplained = [
        group for group in groups if len({item.object_id for item in group}) > 1
    ]
    findings.inconsistent.append(
        f"  {affected} annotations dans {len(groups)} groupes co-localisés au"
        " centimètre et de même classe — un clic enregistré plusieurs fois"
    )
    for group in groups[:5]:
        ids = ", ".join(item.id for item in group)
        findings.inconsistent.append(
            f"    {group[0].class_name} @ {group[0].lat:.7f},{group[0].lng:.7f} : {ids}"
        )
    if len(groups) > 5:
        findings.inconsistent.append(f"    ... et {len(groups) - 5} autres groupes")
    if unexplained:
        findings.inconsistent.append(
            f"  dont {len(unexplained)} portent des object_id différents au même"
            " endroit — contradiction à trancher, ni le code ni ce script ne peuvent"
            " deviner s'il y a un ou deux objets"
        )


def check_object_ids(annotations: Sequence[Annotation], findings: Findings) -> None:
    """One object_id must describe one object: one class, one extent."""
    by_id: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.object_id is not None:
            by_id[annotation.object_id].append(annotation)
    for object_id, group in sorted(by_id.items()):
        classes = {item.class_name for item in group}
        if len(classes) > 1:
            findings.inconsistent.append(
                f"  object_id {object_id!r} porte {len(classes)} classes : "
                f"{', '.join(sorted(classes))}"
            )
        extents = {item.extent_m for item in group if item.extent_m is not None}
        if len(extents) > 1:
            findings.inconsistent.append(
                f"  object_id {object_id!r} porte {len(extents)} emprises : "
                f"{', '.join(f'{value:g}' for value in sorted(extents))}"
            )


def check_extents(annotations: Sequence[Annotation], findings: Findings) -> None:
    """An extent outside the plausible range is a typo, and it moves every radius."""
    low, high = PLAUSIBLE_EXTENT_M
    wrong = [
        item
        for item in annotations
        if item.extent_m is not None and not low <= item.extent_m <= high
    ]
    if not wrong:
        return
    findings.inconsistent.append(
        f"  {len(wrong)} emprises hors de [{low:g}, {high:g}] m — probablement"
        " une unité ou une virgule"
    )
    for item in wrong[:5]:
        findings.inconsistent.append(
            f"    {item.id} ({item.class_name}) : {item.extent_m:g} m"
        )


def check_synonyms(annotations: Sequence[Annotation], findings: Findings) -> None:
    """Two annotations of one class should not disagree about what the class is called.

    Reported as an inconsistency rather than a warning: a synonym set is a property of
    the class, so a per-annotation difference means one of the two was typed in a hurry,
    and whichever wins changes what counts as a correct answer.
    """
    by_class: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for annotation in annotations:
        if annotation.synonyms:
            by_class[annotation.class_name].add(
                tuple(sorted(item.strip().lower() for item in annotation.synonyms))
            )
    for class_name, variants in sorted(by_class.items()):
        if len(variants) > 1:
            findings.inconsistent.append(
                f"  classe {class_name!r} annotée avec {len(variants)} ensembles de"
                " synonymes différents"
            )
            for variant in sorted(variants)[:3]:
                findings.inconsistent.append(f"    {list(variant)}")


def check_zones(annotations: Sequence[Annotation], findings: Findings) -> None:
    """A partly-declared map is fine; the report only has to say how much is covered."""
    declared = [item for item in annotations if item.exhaustive_zone is not None]
    if not declared or len(declared) == len(annotations):
        return
    zones = sorted({str(item.exhaustive_zone) for item in declared})
    findings.earned.append(
        f"  {len(declared)}/{len(annotations)} annotations dans une zone exhaustive"
        f" ({len(zones)} zones : {', '.join(zones[:6])}) — les types background et"
        " manqué ne seront lus que dedans"
    )


def check_earned(annotations: Sequence[Annotation], findings: Findings) -> None:
    """Say which measurements the ground truth now supports."""
    gate = separability(annotations)
    findings.earned.append(
        f"  séparabilité : {gate.overlap_share:.1%} de recouvrement de classes dans"
        f" le rayon (limite {SEPARABILITY_LIMIT:.0%}), médiane"
        f" {gate.median_other_class_m:.2f} m"
    )
    if gate.classification_measurable:
        findings.earned.append(
            "  → colonnes classification et classification+localisation DISPONIBLES"
        )
    else:
        findings.earned.append(f"  → colonnes retenues : {gate.reason}")

    radii = sorted(match_radius_m(item) for item in annotations)
    if radii:
        middle = radii[len(radii) // 2]
        findings.earned.append(
            f"  rayon d'appariement médian : {middle:.2f} m"
            + (
                "  (toujours le accuracy plat — annoter extent_m)"
                if all(item.extent_m is None for item in annotations)
                else ""
            )
        )

    with_sets = [item for item in annotations if has_label_sets(item)]
    findings.earned.append(
        f"  ensembles de labels : {len(with_sets)}/{len(annotations)}"
        f" ({len(with_sets) / max(len(annotations), 1):.1%})"
    )
    if with_sets:
        complete = sum(
            all(
                getattr(item, category if category != "synonyms" else "synonyms")
                for category in LABEL_CATEGORIES
            )
            for item in with_sets
        )
        findings.earned.append(
            f"  → dont {complete} avec les quatre catégories remplies (set ranking"
            " complet)"
        )


def validate(annotations: Sequence[Annotation]) -> Findings:
    """Run every check over one map's annotations."""
    findings = Findings()
    if not annotations:
        findings.inconsistent.append("  aucune annotation dans le fichier")
        return findings
    check_missing(annotations, findings)
    check_duplicates(annotations, findings)
    check_object_ids(annotations, findings)
    check_extents(annotations, findings)
    check_synonyms(annotations, findings)
    check_zones(annotations, findings)
    check_earned(annotations, findings)
    return findings


def report_lines(annotations: Sequence[Annotation], findings: Findings) -> list[str]:
    """The findings as a report, in the order they should be acted on."""
    lines = [f"{len(annotations)} annotations lues", "", "===== champs du contrat"]
    lines.extend(findings.missing)
    lines.append("")
    lines.append("===== incohérences")
    lines.extend(findings.inconsistent or ["  aucune"])
    lines.append("")
    lines.append("===== ce que la carte a gagné")
    lines.extend(findings.earned)
    return lines


def read_annotations(
    map_path: Path, geojson: Path | None, default_accuracy_m: float
) -> tuple[Path, list[Annotation]]:
    """The annotations to validate, and the path they came from.

    The store wins over the export unless a file is named explicitly: the report tells
    the annotator what to fix next, and an export written by the last benchmark run
    describes a map that may already have been fixed.
    """
    if geojson is not None:
        if not geojson.is_file():
            raise SystemExit(f"{geojson}: fichier introuvable.")
        return geojson, load_annotations(geojson, default_accuracy_m)

    db_path = annotation_database_path(map_path)
    if db_path.is_file():
        return db_path, load_store_annotations(db_path, default_accuracy_m)

    export = map_path / DEFAULT_GEOJSON
    if export.is_file():
        return export, load_annotations(export, default_accuracy_m)
    raise SystemExit(
        f"{map_path}: aucune annotation — ni {ANNOTATION_DB_FILENAME} ni"
        f" {DEFAULT_GEOJSON}. Ouvrir l'onglet Annotation du toolbox une fois pour"
        " créer le magasin."
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the validation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument(
        "--geojson",
        type=Path,
        default=None,
        help=(
            "Read this exported GeoJSON instead of the map's annotation store. The"
            f" store ({ANNOTATION_DB_FILENAME}) is the default because"
            f" {DEFAULT_GEOJSON} is only rewritten when a benchmark run starts, and"
            " so lags behind what has just been annotated."
        ),
    )
    parser.add_argument("--default-accuracy", type=float, default=DEFAULT_ACCURACY_M)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one map's annotations; non-zero when something is contradictory."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    map_path = args.map_path.expanduser().resolve()
    source, annotations = read_annotations(
        map_path, args.geojson, args.default_accuracy
    )
    findings = validate(annotations)
    print(f"source : {source}")
    print("\n".join(report_lines(annotations, findings)))
    return 1 if findings.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
