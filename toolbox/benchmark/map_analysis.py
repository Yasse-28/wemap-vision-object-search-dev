"""Describe one prepared map: its detections, its free labels, its ground truth.

An analysis tool, not a solution. It answers "what is in this index and how do its
parts relate", so that a pipeline change is argued against measured structure rather
than against an intuition. One map in, one report and a set of layers out.

It reads the **whole parquet**, not a query's candidate set: the subject is the map,
not a retrieval result. Detections are re-placed in EUS from `depth` and the two ERP
angles exactly as ingestion does, so no database, no ANN service and no candidate
cache are needed.

**On the ground truth.** An annotation's position was built as
`ray(u, v) * depth_map(u, v)` — the same construction as a detection's. It cannot
judge *where* a thing is, because it shares the depth map's error. It judges whether
detections agree with each other and with the annotator's intent, which is what the
pipeline can actually reach. One measurement escapes this entirely and is reported
first: an annotation also records the panorama and the pixel it was clicked in, so
asking whether a detection box covers that pixel *in that same panorama* is purely
angular — no depth, no 3D, no shared error.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from toolbox.benchmark.annotation_store import (
    read_ground_truth_collection,
    source_field,
)
from toolbox.benchmark.label_set_metrics import (
    DEFAULT_TOP_N,
    evaluate_label_sets,
    has_label_sets,
)
from toolbox.benchmark.label_set_metrics import report_lines as label_set_report_lines
from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    parse_annotated_features,
)
from toolbox.bricks.georef_source import PoseSource
from toolbox.bricks.ingest_cli import EMBEDDING_DIM
from toolbox.bricks.vendored.erp import theta_phi_to_opengl_ray_batch

#: Radii the ground-truth attachment is reported at. A single radius would hide how
#: much every downstream label depends on it: `--near-m` is 1.0 by convention only.
DEFAULT_RADII_M = (0.5, 1.0, 2.0, 3.0, 5.0)
#: Range bands, metres.
RANGE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 15.0),
    (15.0, math.inf),
)
#: Beyond this a detection is past the depth map's trusted range.
MAX_TRUSTED_RANGE_M = 15.0
#: Detector score buckets, for the calibration table.
SCORE_BUCKETS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01)
#: Azimuth bins the viewpoint coverage around an annotation is counted on. Twelve
#: bins is 30 degrees each: coarse enough that a handful of keyframes can fill one,
#: fine enough that a capture which only walked past one side cannot reach half.
COVERAGE_BINS = 12
#: Two viewpoints closer together than this, as seen from the object, intersect at
#: too shallow an angle to resolve their disagreement about its depth. Below it a
#: second observation adds photometric evidence but no geometric evidence.
USEFUL_PARALLAX_DEG = 5.0
#: Highest `min_keyframes_per_cluster` the retention curve is reported at. The online
#: default is 2, and the curve exists to show what raising it costs: past a handful of
#: keyframes the surviving share is flat and the extra rows say nothing.
KEYFRAME_THRESHOLD_MAX = 8
#: Rows the hubness estimate samples. A k-occurrence count scales with how many rows
#: could have retrieved it, so the sample is a fixed size rather than a fixed share:
#: two maps only compare when every row had the same number of chances to be picked.
HUBNESS_SAMPLE = 20000
#: Neighbours each sampled row retrieves. The query path asks for far more, but
#: hubness is a property of the space rather than of the request, and it is sharpest
#: at small k — a large k averages the asymmetry away.
HUBNESS_K = 10


@dataclass
class Detections:
    """Every parquet row, re-placed in EUS. Arrays are aligned and full-length."""

    row_index: np.ndarray
    keyframe_id: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    angular_width: np.ndarray
    angular_height: np.ndarray
    source: np.ndarray
    label: np.ndarray
    score: np.ndarray
    depth: np.ndarray
    position_eus: np.ndarray
    origin_eus: np.ndarray
    direction_eus: np.ndarray
    #: False where no pose or no depth, so no position could be built at all.
    placed: np.ndarray

    def __len__(self) -> int:
        return int(self.row_index.size)

    @property
    def range_m(self) -> np.ndarray:
        """Distance from the keyframe to the depth point, NaN where unplaced."""
        return np.linalg.norm(self.position_eus - self.origin_eus, axis=1)

    @property
    def angular_diagonal(self) -> np.ndarray:
        """Angular size of the box, radians — the quantity extent estimates use."""
        return np.hypot(self.angular_width, self.angular_height)


@dataclass
class GroundTruth:
    """Benchmark annotations in EUS, with everything the GeoJSON recorded about them.

    `source_keyframe_id` and `erp_uv` are what make the depth-free measurement
    possible: they say which panorama the annotator clicked, and where.
    """

    class_name: np.ndarray
    position_eus: np.ndarray
    accuracy_m: np.ndarray
    source_keyframe_id: np.ndarray
    erp_uv: np.ndarray
    depth_m: np.ndarray
    level: np.ndarray
    #: The annotations these arrays were built from, in the same order. Kept because
    #: the ADR 0009 label sets are lists, which no aligned array can hold.
    annotations: tuple[Annotation, ...] = ()

    def __len__(self) -> int:
        return int(self.class_name.size)

    @property
    def object_key(self) -> np.ndarray:
        """One key per annotation naming the object it belongs to.

        `object_id` when the annotation declares one, otherwise the annotation's own id:
        two clicks are one object only where the annotator said so. Counting rows
        instead reads 14 clicks on one camera as 14 cameras, which is what made s5's
        co-visible bound look like a 14x undercount when it was exact.
        """
        return np.array(
            [item.object_id or f"#{item.id}" for item in self.annotations],
            dtype=object,
        )

    def object_count(self, mask: np.ndarray | None = None) -> int:
        """Distinct objects among the selected annotations."""
        keys = self.object_key
        return int(np.unique(keys if mask is None else keys[mask]).size)


@dataclass
class MapData:
    """Everything one map offers, loaded once and shared by every section."""

    path: Path
    pose_source: PoseSource
    detections: Detections
    ground_truth: GroundTruth
    #: `(query, status)` counts and the per-row verdicts when they can be joined.
    reviews: list[tuple[str, str, int]]
    group_labels: list[tuple[str, float, float, str | None]]
    embeddings: np.ndarray | None


@dataclass
class Report:
    """Text for the terminal and the same numbers as a JSON payload."""

    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def head(self, title: str) -> None:
        """Start a section."""
        self.lines.append("")
        self.lines.append(f"===== {title}")

    def say(self, text: str = "") -> None:
        """Add one line of report text."""
        self.lines.append(text)

    def text(self) -> str:
        """The whole report as one string."""
        return "\n".join(self.lines)


def _rotate_batch(orientations: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Rotate one local ray per row by that row's quaternion, `wxyz`, vectorised."""
    w = orientations[:, 0]
    xyz = orientations[:, 1:]
    t = 2.0 * np.cross(xyz, local)
    return local + w[:, None] * t + np.cross(xyz, t)


def _pose_arrays(
    keyframe_id: np.ndarray, pose_source: PoseSource
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row keyframe position and orientation, plus the rows that have a pose."""
    unique, inverse = np.unique(keyframe_id, return_inverse=True)
    known = np.array([int(value) in pose_source.poses for value in unique])
    positions = np.full((unique.size, 3), np.nan)
    orientations = np.zeros((unique.size, 4))
    orientations[:, 0] = 1.0
    for index, value in enumerate(unique.tolist()):
        pose = pose_source.poses.get(int(value))
        if pose is None:
            continue
        positions[index] = pose.position_eus
        orientations[index] = pose.orientation_wxyz
    return positions[inverse], orientations[inverse], known[inverse]


def load_detections(map_path: Path, pose_source: PoseSource) -> Detections:
    """Read the parquet and place every row in EUS the way ingestion does.

    A row without depth or without a pose keeps a NaN position rather than being
    dropped: how many rows cannot be placed at all is one of the numbers this tool
    exists to report.
    """
    table = pq.read_table(map_path / "object-search" / "metadata.parquet")
    column = {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.column_names
    }
    keyframe_id = column["video_keyframe_id"].astype(np.int64)
    theta = column["theta_center"].astype(np.float64)
    phi = column["phi_center"].astype(np.float64)
    depth = column["depth"].astype(np.float64)

    origins, orientations, has_pose = _pose_arrays(keyframe_id, pose_source)
    local = np.asarray(theta_phi_to_opengl_ray_batch(theta, phi), dtype=np.float64)
    world = _rotate_batch(orientations, local)
    norms = np.linalg.norm(world, axis=1, keepdims=True)
    directions = world / np.where(norms == 0.0, 1.0, norms)
    positions = origins + directions * depth[:, None]
    placed = has_pose & np.isfinite(depth) & np.isfinite(positions).all(axis=1)

    return Detections(
        row_index=column["row_index"].astype(np.int64),
        keyframe_id=keyframe_id,
        theta=theta,
        phi=phi,
        angular_width=column["angular_width"].astype(np.float64),
        angular_height=column["angular_height"].astype(np.float64),
        source=column["detector_source"].astype(str),
        label=column["label"].astype(str),
        score=column["detection_score"].astype(np.float64),
        depth=depth,
        position_eus=positions,
        origin_eus=origins,
        direction_eus=directions,
        placed=placed,
    )


def _as_float(value: object) -> float:
    """A float, or NaN for a property that is absent or unparseable."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _altitude(properties: Mapping[str, object]) -> float:
    """The click's altitude, 0.0 when it has none — the third coordinate is optional."""
    value = _as_float(properties.get("altitude"))
    return 0.0 if math.isnan(value) else value


def load_ground_truth(map_path: Path, pose_source: PoseSource) -> GroundTruth:
    """Annotations in EUS, keeping the panorama and pixel each one was clicked in.

    Read from `object-search-annotations.db` when it exists, not from
    `benchmark/annotations.geojson`: that export is only rewritten when a benchmark run
    starts, so s0 counted 9 annotations on vinci-st-domingue-zone-1 where the store held
    12. See `toolbox.benchmark.annotation_store`.
    """
    _, collection = read_ground_truth_collection(map_path)
    pairs = parse_annotated_features(collection, 5.0)
    if not pairs:
        # A freshly prepared map has no annotations yet, which is a state to describe
        # rather than an error: s0 still says what the index holds, and every
        # ground-truth section reports that it has nothing to compare against.
        empty = np.empty(0)
        return GroundTruth(
            class_name=np.empty(0, dtype=object),
            position_eus=np.empty((0, 3)),
            accuracy_m=empty,
            source_keyframe_id=np.empty(0, dtype=object),
            erp_uv=np.empty((0, 2)),
            depth_m=empty,
            level=np.empty(0, dtype=object),
        )
    annotations: Sequence[Annotation] = [item for _, item in pairs]
    wgs84 = []
    keyframes: list[str] = []
    uv = []
    depths = []
    for properties, annotation in pairs:
        wgs84.append([annotation.lng, annotation.lat, _altitude(properties)])
        keyframe_id = source_field(properties, "keyframe_id")
        # Not `or ""`: the store writes `keyframeId` as an integer and keyframe 0 is a
        # real keyframe, which a falsiness test would erase.
        keyframes.append("" if keyframe_id is None else str(keyframe_id))
        uv.append(
            [
                _as_float(source_field(properties, "erp_u")),
                _as_float(source_field(properties, "erp_v")),
            ]
        )
        depths.append(_as_float(source_field(properties, "depth_m")))
    eus = np.asarray(
        pose_source.geo_transform.wgs84_to_local_positions(
            np.asarray(wgs84, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return GroundTruth(
        class_name=np.array([item.class_name for item in annotations]),
        position_eus=eus,
        accuracy_m=np.asarray(
            [item.accuracy_m for item in annotations], dtype=np.float64
        ),
        source_keyframe_id=np.array(keyframes),
        erp_uv=np.asarray(uv, dtype=np.float64),
        depth_m=np.asarray(depths, dtype=np.float64),
        level=np.array([item.level or "" for item in annotations]),
        annotations=tuple(annotations),
    )


def load_embeddings(map_path: Path, expected_rows: int) -> np.ndarray | None:
    """Memory-map the cutout embeddings, or None when they are absent or truncated.

    Despite the `.npy` name the file is a headerless raw float16 dump written by
    `prepare.writer.StreamingWriter` with `tofile`, so `np.load` rejects it and
    production reads it with `frombuffer` for the same reason.
    """
    path = map_path / "object-search" / "embeddings.npy"
    if not path.is_file():
        return None
    rows = path.stat().st_size // (2 * EMBEDDING_DIM)
    if rows != expected_rows:
        return None
    return np.memmap(path, dtype=np.float16, mode="r", shape=(rows, EMBEDDING_DIM))


def _table_rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    """Read one query from the map's annotation store, tolerating an absent table."""
    if not path.is_file():
        return []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()


def load_map(map_path: Path) -> MapData:
    """Load every per-map source this tool knows how to read."""
    from toolbox.bricks import georef_source

    resolved = map_path.expanduser().resolve()
    pose_source = georef_source.load_pose_source(resolved)
    detections = load_detections(resolved, pose_source)
    database = resolved / "object-search-annotations.db"
    return MapData(
        path=resolved,
        pose_source=pose_source,
        detections=detections,
        ground_truth=load_ground_truth(resolved, pose_source),
        reviews=[
            (str(query), str(status), int(target))
            for query, status, target in _table_rows(
                database, "SELECT query, status, target_id FROM detection_review"
            )
        ],
        group_labels=[
            (str(keyframe), float(theta), float(phi), group)
            for keyframe, theta, phi, group in _table_rows(
                database,
                "SELECT keyframe_id, theta_center, phi_center, group_name "
                "FROM detection_group_label",
            )
        ],
        embeddings=load_embeddings(resolved, len(detections)),
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    """The five numbers every distribution in this report is summarised by."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0.0}
    return {
        "n": float(finite.size),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def _percentile_line(name: str, values: np.ndarray, unit: str = "") -> str:
    """One distribution as a fixed-width line."""
    stats = _percentiles(values)
    if not stats.get("n"):
        return f"  {name:26s} (aucune valeur)"
    return (
        f"  {name:26s} n={int(stats['n']):8d}  p10 {stats['p10']:8.2f}  "
        f"med {stats['median']:8.2f}  p90 {stats['p90']:8.2f}  max {stats['max']:8.2f}"
        f" {unit}"
    )


def _label_counts(labels: np.ndarray) -> list[tuple[str, int]]:
    """Distinct labels with their counts, most frequent first."""
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [(str(values[i]), int(counts[i])) for i in order]


def section_inventory(data: MapData, report: Report) -> None:
    """S0 — what this map holds, and what it does not.

    Read before anything else: several sections below are silently empty on a map
    whose prepare run did not write labels or scores, and a table of zeros looks the
    same as a measured zero.
    """
    report.head("S0 — inventaire et disponibilité")
    detections = data.detections
    manifest_keyframes = len(data.pose_source.poses)
    active = np.unique(detections.keyframe_id).size
    report.say(f"  carte                      {data.path.name}")
    report.say(
        f"  keyframes                  {manifest_keyframes} au manifeste, "
        f"{active} portent des détections ({active / max(manifest_keyframes, 1):.1%})"
    )
    report.say(
        f"  détections                 {len(detections)}, dont "
        f"{int(detections.placed.sum())} plaçables "
        f"({1 - detections.placed.mean():.1%} sans position)"
    )
    missing: list[str] = []
    for source in sorted(set(detections.source.tolist())):
        rows = detections.source == source
        labels = _label_counts(detections.label[rows])
        scores = detections.score[rows]
        nan_share = float(np.isnan(scores).mean())
        report.say(
            f"    {source:8s} n={int(rows.sum()):8d}  labels distincts "
            f"{len(labels):4d}  score NaN {nan_share:6.1%}"
        )
        if len(labels) == 1:
            missing.append(
                f"{source} ne porte qu'un label ({labels[0][0]!r}) — pas de sémantique"
            )
        if nan_share > 0.5:
            missing.append(f"{source} n'a pas de score de détection")
    ground_truth = data.ground_truth
    report.say(
        f"  vérités terrain            {ground_truth.object_count()} objets "
        f"({len(ground_truth)} clics) sur "
        f"{len(set(ground_truth.class_name.tolist()))} classes ; "
        f"{int((ground_truth.source_keyframe_id != '').sum())} avec keyframe source, "
        f"{int(np.isfinite(ground_truth.erp_uv).all(axis=1).sum())} avec pixel source"
    )
    if len(ground_truth):
        accuracy = _label_counts(ground_truth.accuracy_m.astype(str))
        report.say(f"    précisions déclarées     {accuracy}")
    else:
        missing.append("aucune vérité terrain (magasin d'annotations vide)")
    report.say(
        f"  verdicts humains           {len(data.reviews)} ; "
        f"étiquettes de partition {len(data.group_labels)}"
    )
    report.say(
        "  embeddings                 "
        + (
            "absents ou tronqués"
            if data.embeddings is None
            else f"{data.embeddings.shape}"
        )
    )
    if data.embeddings is None:
        missing.append("embeddings absents ou de longueur incohérente")
    if not data.group_labels:
        missing.append("aucune étiquette de partition à la main")
    report.say()
    report.say("  CE QUI MANQUE :" if missing else "  rien ne manque")
    for item in missing:
        report.say(f"    - {item}")
    report.data["s0"] = {
        "map": data.path.name,
        "keyframes_manifest": manifest_keyframes,
        "keyframes_with_detections": int(active),
        "detections": len(detections),
        "placed": int(detections.placed.sum()),
        "ground_truth": len(ground_truth),
        "reviews": len(data.reviews),
        "group_labels": len(data.group_labels),
        "embeddings": None if data.embeddings is None else list(data.embeddings.shape),
        "missing": missing,
    }


def _per_keyframe_counts(detections: Detections) -> np.ndarray:
    """How many proposals each panorama carries."""
    _, counts = np.unique(detections.keyframe_id, return_counts=True)
    return counts.astype(np.float64)


def _source_summary(detections: Detections, source: str) -> dict[str, Any]:
    """Per-detector summary kept in the JSON payload."""
    rows = detections.source == source
    return {
        "n": int(rows.sum()),
        "range_m": _percentiles(detections.range_m[rows & detections.placed]),
        "score": _percentiles(detections.score[rows]),
    }


def section_detections(data: MapData, report: Report) -> None:
    """S1 — what a detection looks like, split by the detector that produced it.

    Every later section is conditional on these, and none of them had ever been
    described. `theta` is already expressed in the panorama's own frame, so its
    distribution says directly whether the detector fires mostly ahead of the camera.
    """
    report.head("S1 — distributions par détection")
    detections = data.detections
    for source in ["*"] + sorted(set(detections.source.tolist())):
        rows = (
            np.ones(len(detections), dtype=bool)
            if source == "*"
            else detections.source == source
        )
        usable = rows & detections.placed
        report.say(f"  --- {source} ({int(rows.sum())} lignes)")
        report.say(_percentile_line("portée", detections.range_m[usable], "m"))
        report.say(
            _percentile_line(
                "diagonale angulaire",
                np.degrees(detections.angular_diagonal[rows]),
                "deg",
            )
        )
        report.say(
            _percentile_line(
                "rapport largeur/hauteur",
                detections.angular_width[rows]
                / np.where(
                    detections.angular_height[rows] > 0,
                    detections.angular_height[rows],
                    np.nan,
                ),
            )
        )
        report.say(
            _percentile_line("phi (élévation)", np.degrees(detections.phi[rows]), "deg")
        )
        report.say(
            _percentile_line(
                "|theta| (azimut)", np.degrees(np.abs(detections.theta[rows])), "deg"
            )
        )
        if np.isfinite(detections.score[rows]).any():
            report.say(_percentile_line("score détecteur", detections.score[rows]))
        report.say(
            _percentile_line(
                "taille physique implicite",
                detections.range_m[usable] * detections.angular_diagonal[usable],
                "m",
            )
        )
    report.say()
    report.say(
        _percentile_line("propositions par panorama", _per_keyframe_counts(detections))
    )
    report.say()
    report.say("  répartition par bande de portée :")
    ranges = detections.range_m
    for low, high in RANGE_BANDS:
        band = detections.placed & (ranges >= low) & (ranges < high)
        name = f"{low:g}-{high:g} m" if math.isfinite(high) else f"{low:g} m+"
        diagonal = (
            float(np.degrees(np.median(detections.angular_diagonal[band])))
            if band.any()
            else float("nan")
        )
        report.say(
            f"    {name:10s} {int(band.sum()):8d}  ({band.mean():5.1%})  "
            f"diagonale méd {diagonal:5.1f} deg"
        )
    report.data["s1"] = {
        "range_m": _percentiles(detections.range_m[detections.placed]),
        "angular_diagonal_deg": _percentiles(np.degrees(detections.angular_diagonal)),
        "phi_deg": _percentiles(np.degrees(detections.phi)),
        "per_keyframe": _percentiles(_per_keyframe_counts(detections)),
        "by_source": {
            source: _source_summary(detections, source)
            for source in sorted(set(detections.source.tolist()))
        },
    }


@dataclass
class Attachment:
    """Each placed detection related to the nearest ground truth, once and for all.

    Computed on a KD-tree over the annotations, so every section can ask "which
    object is this detection near, and how near" without paying for it again.
    """

    #: Index into the ground truth, -1 when the map has no annotation at all.
    nearest: np.ndarray
    #: Metres to that annotation, inf where the detection has no position.
    distance_m: np.ndarray
    #: Class of the nearest annotation, empty where there is none.
    nearest_class: np.ndarray

    def attached(self, radius_m: float) -> np.ndarray:
        """Detections within `radius_m` of some annotation."""
        return self.distance_m <= radius_m


def attach_to_ground_truth(
    detections: Detections, ground_truth: GroundTruth
) -> Attachment:
    """Nearest annotation of any class for every placed detection.

    Deliberately *any* class rather than the matching one: which label lands near
    which kind of object is exactly what the label sections are trying to find out,
    so restricting the search to the matching class would assume the answer.
    """
    count = len(detections)
    nearest = np.full(count, -1, dtype=np.int64)
    distance = np.full(count, np.inf)
    if len(ground_truth) == 0:
        return Attachment(nearest, distance, np.full(count, "", dtype=object))
    tree = cKDTree(ground_truth.position_eus)
    rows = np.flatnonzero(detections.placed)
    found_distance, found_index = tree.query(detections.position_eus[rows], k=1)
    nearest[rows] = found_index
    distance[rows] = found_distance
    classes = np.full(count, "", dtype=object)
    classes[rows] = ground_truth.class_name[found_index]
    return Attachment(nearest, distance, classes)


def _conditional_table(
    condition: np.ndarray, outcome: np.ndarray, top: int = 6
) -> dict[str, list[tuple[str, float, int]]]:
    """`P(outcome | condition)`, keeping the `top` most likely outcomes per key."""
    table: dict[str, list[tuple[str, float, int]]] = {}
    for key in sorted(set(condition.tolist())):
        rows = condition == key
        total = int(rows.sum())
        if total == 0:
            continue
        values, counts = np.unique(outcome[rows], return_counts=True)
        order = np.argsort(counts)[::-1][:top]
        table[str(key)] = [
            (str(values[i]), float(counts[i] / total), int(counts[i])) for i in order
        ]
    return table


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    """Normalised mutual information between two categorical arrays, in `[0, 1]`.

    Reported rather than a correlation because both sides are nominal: a label is
    not ordered, so a Pearson coefficient would be meaningless on it.
    """
    if left.size == 0:
        return float("nan")
    left_values, left_index = np.unique(left, return_inverse=True)
    right_values, right_index = np.unique(right, return_inverse=True)
    joint = np.zeros((left_values.size, right_values.size))
    np.add.at(joint, (left_index, right_index), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (px * py))
    information = float(np.nansum(terms))
    entropy_x = float(-np.nansum(px * np.log(px)))
    entropy_y = float(-np.nansum(py * np.log(py)))
    scale = math.sqrt(max(entropy_x, 1e-12) * max(entropy_y, 1e-12))
    return information / scale if scale > 0 else float("nan")


def section_labels(
    data: MapData, report: Report, radii: Sequence[float] = DEFAULT_RADII_M
) -> None:
    """S2 — the free labels, and what they say about the objects.

    `detector_source`, `label` and `detection_score` cost nothing: they are already
    in the index. The question is whether they carry information the geometry does
    not — which object a label names, and how strongly a label predicts that a
    detection sits on a real annotated object at all.
    """
    report.head("S2 — labels libres et PDF conditionnelles")
    detections = data.detections
    attachment = attach_to_ground_truth(detections, data.ground_truth)
    if len(data.ground_truth) == 0:
        report.say("  aucune vérité terrain : section vide")
        return

    base = {
        radius: float(attachment.attached(radius)[detections.placed].mean())
        for radius in radii
    }
    report.say(
        "  taux de base P(rattaché) toutes détections confondues : "
        + "  ".join(f"{radius:g} m {share:5.1%}" for radius, share in base.items())
    )
    report.say(
        "  (un label n'est informatif que s'il dépasse ce taux — une classe dense"
        " rattache n'importe quoi)"
    )
    report.say()
    report.say("  P(rattaché à un objet | label, source) par rayon, et lift à 1 m :")
    header = "  ".join(f"{radius:g} m" for radius in radii)
    report.say(f"    {'label':28s} {'source':7s} {'n':>7s}   {header}   lift")
    interesting = _label_counts(detections.label[detections.placed])[:12]
    for label, _ in interesting:
        carrying = detections.label == label
        for source in sorted(set(detections.source[carrying].tolist())):
            rows = detections.placed & carrying & (detections.source == source)
            if rows.sum() < 30:
                continue
            shares = "  ".join(
                f"{attachment.attached(radius)[rows].mean():5.1%}" for radius in radii
            )
            lift = attachment.attached(1.0)[rows].mean() / max(base.get(1.0, 0.0), 1e-9)
            report.say(
                f"    {label[:28]:28s} {source:7s} "
                f"{int(rows.sum()):7d}   {shares}   x{lift:4.1f}"
            )

    radius = 2.0
    close = detections.placed & attachment.attached(radius)
    report.say()
    report.say(f"  P(label | classe de vérité terrain), rattachement à {radius:g} m :")
    for gt_class, entries in _conditional_table(
        attachment.nearest_class[close], detections.label[close]
    ).items():
        rendered = ", ".join(f"{name} {share:.0%}" for name, share, _ in entries[:5])
        report.say(f"    {gt_class[:26]:26s} {rendered}")

    report.say()
    report.say(f"  P(classe de vérité terrain | label), rattachement à {radius:g} m :")
    given_label = _conditional_table(
        detections.label[close], attachment.nearest_class[close]
    )
    for label, entries in list(given_label.items())[:14]:
        rendered = ", ".join(f"{name} {share:.0%}" for name, share, _ in entries[:3])
        report.say(f"    {label[:26]:26s} {rendered}")

    information = _mutual_information(
        detections.label[close], attachment.nearest_class[close]
    )
    report.say()
    report.say(
        f"  information mutuelle normalisée label / classe VT : {information:.3f}"
    )

    if np.isfinite(detections.score).any():
        report.say()
        report.say(f"  P(rattaché à {radius:g} m | score, source) — calibration :")
        for source in sorted(set(detections.source.tolist())):
            rows = detections.placed & (detections.source == source)
            if not rows.any() or not np.isfinite(detections.score[rows]).any():
                continue
            cells = []
            for low, high in zip(SCORE_BUCKETS[:-1], SCORE_BUCKETS[1:], strict=True):
                bucket = rows & (detections.score >= low) & (detections.score < high)
                share = (
                    f"{attachment.attached(radius)[bucket].mean():5.1%}"
                    if bucket.sum() >= 30
                    else f"{'-':>5s}"
                )
                cells.append(f"[{low:.2f},{high:.2f}) {share}")
            report.say(f"    {source:7s} " + "  ".join(cells))

    purity = embedding_neighbourhood_purity(data, attachment, radius_m=radius)
    if purity:
        report.say()
        report.say(
            f"  pureté du voisinage d'embedding ({int(purity['neighbours'])} plus"
            f" proches voisins, {int(purity['queries'])} requêtes) :"
        )
        report.say(f"    même objet : {purity['same_object']:.1%}")
        report.say(f"    même classe : {purity['same_class']:.1%}")
        report.say(
            "  (l'espace sémantique regroupe-t-il déjà un objet, avant toute"
            " association ?)"
        )

    report.data["s2"] = {
        "attachment_radius_m": radius,
        "embedding_purity": purity,
        "base_rate": base,
        "mutual_information_label_gt_class": information,
        "p_label_given_gt_class": _conditional_table(
            attachment.nearest_class[close], detections.label[close]
        ),
        "p_gt_class_given_label": given_label,
    }


def embedding_neighbourhood_purity(
    data: MapData,
    attachment: Attachment,
    *,
    radius_m: float = 2.0,
    neighbours: int = 10,
    queries: int = 2000,
    references: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """Do a detection's nearest cutouts in embedding space sit on the same object?

    Retrieval quality at its root, before any prompt: if the semantic space already
    grouped an object's detections, an association would only have to confirm it.
    Measured on a sample — an exact k-NN over a million 1024-d rows would cost more
    than the answer is worth, and the share is an average, not a per-row need.

    Two readings, because they fail differently: sharing the *object* is what an
    association needs, sharing the *class* is what a search needs.
    """
    if data.embeddings is None or len(data.ground_truth) == 0:
        return {}
    detections = data.detections
    attached = np.flatnonzero(detections.placed & attachment.attached(radius_m))
    if attached.size < 50:
        return {}
    generator = np.random.default_rng(seed)
    pool = generator.choice(attached, min(references, attached.size), replace=False)
    asked = generator.choice(pool, min(queries, pool.size), replace=False)
    bank = _unit_rows(data.embeddings, pool)
    probe = _unit_rows(data.embeddings, asked)

    same_object = 0.0
    same_class = 0.0
    counted = 0
    for start in range(0, probe.shape[0], 256):
        chunk = probe[start : start + 256]
        rows = asked[start : start + 256]
        cosines = chunk @ bank.T
        for position in range(chunk.shape[0]):
            cosines[position, pool == rows[position]] = -2.0
        top = np.argpartition(-cosines, neighbours, axis=1)[:, :neighbours]
        for position in range(chunk.shape[0]):
            picked = pool[top[position]]
            query = rows[position]
            same_object += float(
                (attachment.nearest[picked] == attachment.nearest[query]).mean()
            )
            same_class += float(
                (
                    attachment.nearest_class[picked] == attachment.nearest_class[query]
                ).mean()
            )
            counted += 1
    return {
        "neighbours": float(neighbours),
        "queries": float(counted),
        "same_object": same_object / max(counted, 1),
        "same_class": same_class / max(counted, 1),
    }


def _unit_rows(embeddings: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """L2-normalised float32 copy of the requested embedding rows."""
    vectors = np.asarray(embeddings[rows], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
    return vectors


def _uv_to_theta_phi(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ERP ratios to the angles the parquet stores, the convention of `erp.py`."""
    theta = uv[:, 0] * 2.0 * np.pi - np.pi
    phi = np.pi / 2.0 - uv[:, 1] * np.pi
    return theta, phi


def _angular_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Signed difference of two azimuths, wrapped into `[-pi, pi]`."""
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def seen_in_own_keyframe(
    detections: Detections, ground_truth: GroundTruth
) -> tuple[np.ndarray, np.ndarray]:
    """Is each annotation covered by a detection box **in the panorama it was clicked**?

    The one measurement in this tool that owes nothing to depth. The annotator gave a
    keyframe and a pixel; asking whether some box of that same keyframe contains that
    pixel is a comparison of two angles in one frame. No 3D, no depth map, none of
    the error the annotation shares with the detections.

    Returns:
        A boolean per annotation, and the angular distance to the nearest box centre
        of that keyframe in degrees (`inf` when the keyframe produced nothing).
    """
    theta, phi = _uv_to_theta_phi(ground_truth.erp_uv)
    covered = np.zeros(len(ground_truth), dtype=bool)
    nearest_deg = np.full(len(ground_truth), np.inf)
    by_keyframe: dict[str, np.ndarray] = {}
    keys = detections.keyframe_id.astype(str)
    for keyframe in np.unique(keys):
        by_keyframe[str(keyframe)] = np.flatnonzero(keys == keyframe)
    for index in range(len(ground_truth)):
        rows = by_keyframe.get(str(ground_truth.source_keyframe_id[index]))
        if rows is None or rows.size == 0 or not np.isfinite(theta[index]):
            continue
        delta_theta = np.abs(_angular_delta(detections.theta[rows], theta[index]))
        delta_phi = np.abs(detections.phi[rows] - phi[index])
        inside = (delta_theta <= detections.angular_width[rows] / 2.0) & (
            delta_phi <= detections.angular_height[rows] / 2.0
        )
        covered[index] = bool(inside.any())
        nearest_deg[index] = float(np.degrees(np.hypot(delta_theta, delta_phi).min()))
    return covered, nearest_deg


def _keyframe_positions(pose_source: PoseSource) -> tuple[np.ndarray, np.ndarray]:
    """Every keyframe of the manifest as an array, with its identifier."""
    identifiers = np.array(sorted(pose_source.poses), dtype=np.int64)
    positions = np.asarray(
        [pose_source.poses[int(value)].position_eus for value in identifiers],
        dtype=np.float64,
    )
    return identifiers, positions


def _parallax_cosines(origins: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cosine of the angle between every pair of viewing rays onto one point."""
    offsets = target - origins
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    directions = offsets / np.where(norms == 0.0, 1.0, norms)
    cosines = np.clip(directions @ directions.T, -1.0, 1.0)
    left, right = np.triu_indices(origins.shape[0], k=1)
    return cosines[left, right]


def _max_parallax_deg(origins: np.ndarray, target: np.ndarray) -> float:
    """Widest angle under which a set of viewpoints sees one point."""
    if origins.shape[0] < 2:
        return 0.0
    return float(np.degrees(np.arccos(_parallax_cosines(origins, target).min())))


def _useful_parallax_share(
    origins: np.ndarray, target: np.ndarray, minimum_deg: float = USEFUL_PARALLAX_DEG
) -> float:
    """Share of viewpoint pairs whose parallax is wide enough to triangulate on.

    The maximum parallax is set by the two extreme viewpoints and says nothing about
    the rest. Twenty observations from one standstill and two from opposite ends of a
    room report the same maximum; this separates them, because only the second gives
    an association more than one geometric opinion to average.
    """
    if origins.shape[0] < 2:
        return float("nan")
    angles = np.degrees(np.arccos(_parallax_cosines(origins, target)))
    return float((angles >= minimum_deg).mean())


def _azimuth_coverage(
    origins: np.ndarray, target: np.ndarray, bins: int = COVERAGE_BINS
) -> tuple[float, float]:
    """How much of the ring around a point the viewpoints occupy, and its widest hole.

    Only positional coverage is measured, not rotational: every keyframe is an
    equirectangular panorama and therefore already looks in every direction, so
    "where was the camera pointing" carries no information here.

    Returns:
        Occupied share of the azimuth bins, and the widest empty angular gap in
        degrees. A capture that walked straight past an object occupies two opposite
        bins and leaves a gap near 180; one that circled it leaves a small gap.
    """
    if origins.shape[0] == 0:
        return float("nan"), float("nan")
    offsets = origins - target
    azimuth = np.arctan2(offsets[:, 0], -offsets[:, 2])
    occupied = np.unique(np.floor((azimuth + math.pi) / (2 * math.pi) * bins) % bins)
    ordered = np.sort(azimuth)
    if ordered.size == 1:
        return 1.0 / bins, 360.0
    gaps = np.diff(ordered)
    wrap = 2 * math.pi - (ordered[-1] - ordered[0])
    widest = max(float(gaps.max()), float(wrap))
    return float(occupied.size / bins), float(np.degrees(widest))


def _horizontal_anisotropy(positions: np.ndarray) -> float:
    """How round the ground footprint of a set of viewpoints is, in `[0, 1]`.

    The ratio of the two singular values of the centred `(x, z)` cloud. A corridor
    gives near 0 and no algorithm can triangulate across it, whatever the parallax
    angle says: a long straight baseline still resolves only one direction well.
    """
    if positions.shape[0] < 3:
        return float("nan")
    ground = positions[:, [0, 2]]
    centred = ground - ground.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] <= 0:
        return float("nan")
    return float(singular[1] / singular[0])


@dataclass
class Observability:
    """What the capture makes possible for one annotation, before any algorithm."""

    detections: np.ndarray
    keyframes: np.ndarray
    achieved_parallax_deg: np.ndarray
    available_keyframes: np.ndarray
    available_parallax_deg: np.ndarray
    nearest_keyframe_m: np.ndarray
    #: Horizontal spread of the nearby viewpoints, smallest axis over largest. Near 0
    #: the capture walked a straight line past the object and no baseline exists
    #: across the corridor; near 1 it circled it.
    trajectory_anisotropy: np.ndarray
    #: Share of the azimuth bins around the object occupied by the viewpoints that
    #: actually produced attached detections, and by every viewpoint in range.
    achieved_coverage: np.ndarray
    available_coverage: np.ndarray
    #: Widest azimuth sector around the object that no in-range viewpoint occupies.
    available_gap_deg: np.ndarray
    #: Share of achieved observation pairs wide enough to triangulate on.
    useful_pair_share: np.ndarray


def observability(
    data: MapData, attachment: Attachment, radius_m: float
) -> Observability:
    """Per annotation: what was observed, and what the trajectory made observable.

    The pair matters more than either half. A poor achieved parallax on a rich
    available one is an algorithm or a detector problem; a poor available one is a
    capture problem, and no association will ever fix it.
    """
    ground_truth = data.ground_truth
    detections = data.detections
    identifiers, keyframe_positions = _keyframe_positions(data.pose_source)
    tree = cKDTree(keyframe_positions)
    count = len(ground_truth)
    result = Observability(
        detections=np.zeros(count, dtype=np.int64),
        keyframes=np.zeros(count, dtype=np.int64),
        achieved_parallax_deg=np.zeros(count),
        available_keyframes=np.zeros(count, dtype=np.int64),
        available_parallax_deg=np.zeros(count),
        nearest_keyframe_m=np.full(count, np.inf),
        trajectory_anisotropy=np.full(count, np.nan),
        achieved_coverage=np.full(count, np.nan),
        available_coverage=np.full(count, np.nan),
        available_gap_deg=np.full(count, np.nan),
        useful_pair_share=np.full(count, np.nan),
    )
    attached = detections.placed & attachment.attached(radius_m)
    for index in range(count):
        target = ground_truth.position_eus[index]
        rows = np.flatnonzero(attached & (attachment.nearest == index))
        result.detections[index] = rows.size
        if rows.size:
            result.keyframes[index] = np.unique(detections.keyframe_id[rows]).size
            result.achieved_parallax_deg[index] = _max_parallax_deg(
                detections.origin_eus[rows], target
            )
            seen_from = np.unique(detections.origin_eus[rows], axis=0)
            result.achieved_coverage[index] = _azimuth_coverage(seen_from, target)[0]
            result.useful_pair_share[index] = _useful_parallax_share(seen_from, target)
        near = tree.query_ball_point(target, MAX_TRUSTED_RANGE_M)
        result.available_keyframes[index] = len(near)
        if near:
            result.available_parallax_deg[index] = _max_parallax_deg(
                keyframe_positions[near], target
            )
            result.trajectory_anisotropy[index] = _horizontal_anisotropy(
                keyframe_positions[near]
            )
            coverage, gap = _azimuth_coverage(keyframe_positions[near], target)
            result.available_coverage[index] = coverage
            result.available_gap_deg[index] = gap
        distance, _ = tree.query(target, k=1)
        result.nearest_keyframe_m[index] = float(distance)
    return result


def keyframe_threshold_curve(
    profile: Observability,
    covered: np.ndarray,
    indexed: np.ndarray,
    max_keyframes: int = KEYFRAME_THRESHOLD_MAX,
) -> list[dict[str, float]]:
    """What `min_keyframes_per_cluster` costs, read against the ground truth.

    The online path drops any cluster observed from fewer than
    `min_keyframes_per_cluster` distinct keyframes (default 2). That threshold buys
    precision by discarding evidence, and this curve prices it: at each candidate
    value, how many annotations still have enough observing keyframes to survive, and
    — among those — what share the depth-free measurement finds covered.

    The two series answer different questions. `retained_share` is a ceiling the
    threshold imposes before any ranking runs. `covered_share` says whether the
    annotations it keeps are the ones the detector actually found: a threshold that
    raises it is selecting for quality, one that leaves it flat is only losing recall.

    Args:
        profile: per-annotation observability; `keyframes` is the distinct-keyframe
            count of the detections attached to each annotation.
        covered: the depth-free result per annotation, from `seen_in_own_keyframe`.
        indexed: whether each annotation's source panorama is in the index at all —
            annotations without one are not measurable by `covered` and are excluded
            from `covered_share`, though they still count in `retained_share`.
        max_keyframes: highest threshold reported.

    Returns:
        One row per threshold from 1 upwards, ordered.
    """
    total = int(profile.keyframes.size)
    rows: list[dict[str, float]] = []
    for threshold in range(1, max_keyframes + 1):
        survives = profile.keyframes >= threshold
        measurable = survives & indexed
        rows.append(
            {
                "min_keyframes": float(threshold),
                "retained": float(survives.sum()),
                "retained_share": (
                    float(survives.sum() / total) if total else float("nan")
                ),
                "covered_share": (
                    float(covered[measurable].mean())
                    if measurable.any()
                    else float("nan")
                ),
                "measurable": float(measurable.sum()),
            }
        )
    return rows


def section_ground_truth(
    data: MapData, report: Report, radii: Sequence[float] = DEFAULT_RADII_M
) -> None:
    """S3 — the annotations against the detections, depth-free measurement first."""
    report.head("S3 — vérités terrain et détections")
    ground_truth = data.ground_truth
    if len(ground_truth) == 0:
        report.say("  aucune vérité terrain")
        return
    detections = data.detections
    covered, nearest_deg = seen_in_own_keyframe(detections, ground_truth)
    indexed = np.isin(
        ground_truth.source_keyframe_id, np.unique(detections.keyframe_id).astype(str)
    )
    report.say("  --- mesure SANS profondeur : boîte couvrant le pixel annoté")
    report.say(
        f"    panorama source absent de l'index : {int((~indexed).sum())}"
        f"/{len(ground_truth)} ({1 - indexed.mean():.1%}) — non mesurables ici,"
        " et c'est une cause d'échec à part entière"
    )
    measurable = int(indexed.sum())
    share = covered[indexed].mean() if measurable else float("nan")
    report.say(
        f"    parmi les {measurable} mesurables : {int(covered[indexed].sum())} "
        f"couvertes ({share:.1%}) — c'est le rappel du détecteur"
    )
    report.say(
        _percentile_line(
            "écart angulaire au plus proche",
            nearest_deg[np.isfinite(nearest_deg)],
            "deg",
        )
    )
    for name in sorted(set(ground_truth.class_name.tolist())):
        rows = (ground_truth.class_name == name) & indexed
        if not rows.any():
            continue
        report.say(
            f"    {name[:26]:26s} mesurables {int(rows.sum()):4d} clics"
            f" sur {ground_truth.object_count(rows):3d} objets  "
            f"couvertes {covered[rows].mean():6.1%}"
        )

    attachment = attach_to_ground_truth(detections, ground_truth)
    report.say()
    report.say("  --- rattachement 3D par rayon (secondaire : dépend de la profondeur)")
    for radius in radii:
        attached = detections.placed & attachment.attached(radius)
        reached = np.unique(attachment.nearest[attached])
        hit = np.zeros(len(ground_truth), dtype=bool)
        hit[reached] = True
        objects_reached = ground_truth.object_count(hit)
        objects_total = ground_truth.object_count()
        report.say(
            f"    {radius:4.1f} m : {int(attached.sum()):7d} détections rattachées, "
            f"{objects_reached:3d}/{objects_total} objets atteints "
            f"({objects_reached / objects_total:.1%}) — "
            f"{reached.size}/{len(ground_truth)} clics"
        )

    radius = 2.0
    profile = observability(data, attachment, radius)
    report.say()
    report.say(f"  --- observabilité par annotation (rattachement {radius:g} m)")
    report.say(_percentile_line("détections", profile.detections.astype(float)))
    report.say(_percentile_line("keyframes distincts", profile.keyframes.astype(float)))
    report.say(
        _percentile_line("parallaxe obtenue", profile.achieved_parallax_deg, "deg")
    )
    report.say(
        _percentile_line("parallaxe DISPONIBLE", profile.available_parallax_deg, "deg")
    )
    report.say(
        _percentile_line(
            "keyframes à portée", profile.available_keyframes.astype(float)
        )
    )
    report.say(
        _percentile_line("keyframe le plus proche", profile.nearest_keyframe_m, "m")
    )
    report.say(
        _percentile_line("anisotropie de trajectoire", profile.trajectory_anisotropy)
    )
    report.say(
        "  (0 = la capture est passée en ligne droite, aucune base transversale ;"
        " 1 = elle a tourné autour)"
    )
    report.say()
    report.say(f"  --- couverture angulaire ({COVERAGE_BINS} secteurs d'azimut)")
    report.say(_percentile_line("secteurs occupés OBTENUS", profile.achieved_coverage))
    report.say(
        _percentile_line("secteurs occupés DISPONIBLES", profile.available_coverage)
    )
    report.say(
        _percentile_line("plus grand secteur vide", profile.available_gap_deg, "deg")
    )
    report.say(
        _percentile_line(
            f"paires utiles (>={USEFUL_PARALLAX_DEG:g} deg)", profile.useful_pair_share
        )
    )
    report.say(
        "  (l'écart obtenu/disponible dit si c'est l'association ou la capture qui"
        " limite ; la part de paires utiles dit si les observations apportent une"
        " géométrie ou seulement une redondance photométrique)"
    )

    curve = keyframe_threshold_curve(profile, covered, indexed)
    report.say()
    report.say("  --- coût du seuil min_keyframes_per_cluster (défaut en ligne : 2)")
    for row in curve:
        report.say(
            f"    >={int(row['min_keyframes']):2d} keyframes : "
            f"{int(row['retained']):4d}/{len(ground_truth)} annotations retenues "
            f"({row['retained_share']:6.1%})  couvertes {row['covered_share']:6.1%}"
        )
    report.say(
        "  (la part retenue est un plafond que le seuil impose avant tout classement ;"
        " si la part couverte ne monte pas avec lui, il ne trie rien,"
        " il perd du rappel)"
    )

    degenerate = (profile.available_keyframes > 0) & (
        profile.available_parallax_deg < USEFUL_PARALLAX_DEG
    )
    report.say(
        f"  annotations vues sous une base dégénérée (<{USEFUL_PARALLAX_DEG:g} deg"
        f" disponibles) : {int(degenerate.sum())}/{len(ground_truth)}"
        " — aucune association ne les placera, c'est un plafond de capture"
    )

    unseen = profile.detections == 0
    never_looked = unseen & (profile.available_keyframes == 0)
    looked_missed = unseen & (profile.available_keyframes > 0)
    not_indexed = unseen & ~indexed
    report.say()
    report.say(f"  --- annotations sans détection à {radius:g} m : {int(unseen.sum())}")
    report.say(
        f"    jamais regardées (aucun keyframe à {MAX_TRUSTED_RANGE_M:g} m) : "
        f"{int(never_looked.sum())}"
    )
    report.say(
        f"    regardées et manquées                        : {int(looked_missed.sum())}"
    )
    report.say(
        f"      dont panorama source non indexé            : {int(not_indexed.sum())}"
    )
    report.say(
        f"      dont détectées en 2D mais mal placées      : "
        f"{int((looked_missed & covered).sum())}"
    )
    report.say(
        f"      dont non détectées dans un panorama indexé : "
        f"{int((looked_missed & indexed & ~covered).sum())}"
    )
    report.data["s3"] = {
        "source_keyframe_indexed": float(indexed.mean()),
        "covered_in_own_keyframe": (
            float(covered[indexed].mean()) if indexed.any() else float("nan")
        ),
        "covered_by_class": {
            name: float(covered[(ground_truth.class_name == name) & indexed].mean())
            for name in sorted(set(ground_truth.class_name.tolist()))
        },
        "attachment_by_radius": {
            str(radius): int((detections.placed & attachment.attached(radius)).sum())
            for radius in radii
        },
        "observability": {
            "detections": _percentiles(profile.detections.astype(float)),
            "achieved_parallax_deg": _percentiles(profile.achieved_parallax_deg),
            "available_parallax_deg": _percentiles(profile.available_parallax_deg),
            "nearest_keyframe_m": _percentiles(profile.nearest_keyframe_m),
            "trajectory_anisotropy": _percentiles(profile.trajectory_anisotropy),
            "achieved_coverage": _percentiles(profile.achieved_coverage),
            "available_coverage": _percentiles(profile.available_coverage),
            "available_gap_deg": _percentiles(profile.available_gap_deg),
            "useful_pair_share": _percentiles(profile.useful_pair_share),
        },
        "keyframe_threshold": curve,
        "degenerate_baseline": int(degenerate.sum()),
        "unseen": {
            "total": int(unseen.sum()),
            "never_looked": int(never_looked.sum()),
            "looked_and_missed": int(looked_missed.sum()),
        },
    }


def _overlapping_pairs(
    detections: Detections, rows: np.ndarray, margin: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of one panorama whose boxes are *not* angularly disjoint.

    Below `margin` combined half-extents apart the two boxes describe the same
    direction, so they are the same thing proposed twice rather than two objects.
    """
    if rows.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left, right = np.triu_indices(rows.size, k=1)
    theta = detections.theta[rows]
    phi = detections.phi[rows]
    cos_phi = np.cos(phi)
    directions = np.stack(
        (cos_phi * np.sin(theta), np.sin(phi), -cos_phi * np.cos(theta)), axis=1
    )
    cosines = np.einsum("ij,ij->i", directions[left], directions[right])
    separation = np.arccos(np.clip(cosines, -1.0, 1.0))
    extents = detections.angular_diagonal[rows]
    half = 0.5 * (extents[left] + extents[right])
    ratio = np.where(half > 0.0, separation / np.where(half > 0.0, half, 1.0), np.inf)
    keep = ratio < margin
    return left[keep], right[keep]


def section_counting(
    data: MapData, report: Report, sample_keyframes: int = 400
) -> None:
    """S5 — how many objects a single panorama can already prove.

    `max_f (co-visible boxes in f)` is a **lower** bound on the number of objects: a
    panorama that produced three angularly disjoint boxes saw at least three things.
    It is combinatorial, so none of the noise sources of the depth path applies to
    it — but it is only worth as much as the intra-view duplicate rate, which is what
    this section measures first, with the embeddings deciding rather than a guess.
    """
    report.head("S5 — comptage co-visible et doublons intra-vue")
    detections = data.detections
    keys = detections.keyframe_id
    unique = np.unique(keys)
    generator = np.random.default_rng(0)
    chosen = (
        unique
        if unique.size <= sample_keyframes
        else generator.choice(unique, sample_keyframes, replace=False)
    )
    report.say(f"  échantillon de {chosen.size} panoramas sur {unique.size}")

    overlapping = 0
    cross_source = 0
    cosines: list[float] = []
    disjoint_counts: list[int] = []
    for keyframe in chosen.tolist():
        rows = np.flatnonzero(keys == keyframe)
        left, right = _overlapping_pairs(detections, rows)
        overlapping += left.size
        if left.size:
            same = detections.source[rows][left] != detections.source[rows][right]
            cross_source += int(same.sum())
            if data.embeddings is not None and left.size:
                take = slice(0, min(left.size, 200))
                a = np.asarray(data.embeddings[rows[left[take]]], dtype=np.float32)
                b = np.asarray(data.embeddings[rows[right[take]]], dtype=np.float32)
                a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
                b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
                cosines.extend(np.einsum("ij,ij->i", a, b).tolist())
        total_pairs = rows.size * (rows.size - 1) // 2
        disjoint_counts.append(total_pairs - left.size)
    report.say(
        f"  paires recouvrantes dans une vue : {overlapping}, dont "
        f"{cross_source} entre détecteurs différents "
        f"({cross_source / max(overlapping, 1):.1%})"
    )
    if cosines:
        report.say(_percentile_line("cosinus des recouvrantes", np.asarray(cosines)))
        report.say(
            "  (proche de 1 = la même chose proposée deux fois ; c'est ce qui doit"
            " être retiré avant de compter)"
        )
    report.say(
        _percentile_line(
            "paires disjointes par vue", np.asarray(disjoint_counts, dtype=float)
        )
    )

    attachment = attach_to_ground_truth(detections, data.ground_truth)
    if len(data.ground_truth):
        report.say()
        report.say("  --- borne inférieure contre le nombre réel d'objets")
        report.say(
            "    par classe : max sur les vues du nombre de boîtes disjointes, contre"
            " le nombre d'OBJETS de la classe (object_id distincts, pas de clics)"
        )
        for name in sorted(set(data.ground_truth.class_name.tolist())):
            in_class = data.ground_truth.class_name == name
            objects = data.ground_truth.object_count(in_class)
            clicks = int(in_class.sum())
            near = detections.placed & attachment.attached(2.0)
            rows = near & (attachment.nearest_class == name)
            if not rows.any():
                continue
            best = 0
            for keyframe in np.unique(detections.keyframe_id[rows]).tolist():
                view = np.flatnonzero(rows & (detections.keyframe_id == keyframe))
                left, _ = _overlapping_pairs(detections, view)
                best = max(best, view.size - left.size)
            report.say(
                f"    {name[:26]:26s} borne {best:4d}   objets {objects:4d}"
                f"   ({clicks} clics)"
            )
    report.data["s5"] = {
        "sampled_keyframes": int(chosen.size),
        "overlapping_pairs": overlapping,
        "cross_source_share": float(cross_source / max(overlapping, 1)),
        "overlap_cosine": _percentiles(np.asarray(cosines)) if cosines else {},
    }


def _sample_pairs(
    rows: np.ndarray, generator: np.random.Generator, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random index pairs drawn from `rows`, without building the full triangle."""
    if rows.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left = generator.choice(rows, limit)
    right = generator.choice(rows, limit)
    keep = left != right
    return left[keep], right[keep]


def _stratified_pairs(
    detections: Detections,
    attachment: Attachment,
    attached: np.ndarray,
    generator: np.random.Generator,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the three populations explicitly instead of hoping a uniform sample hits.

    Uniform sampling over every placed detection collapses on a sparse map: with 258
    annotations spread over an airport, two random detections are never on the same
    one, and the same-object population comes out empty. Each stratum is therefore
    drawn on purpose — same object, different objects, and one side attached — and
    the report says how many of each it got.
    """
    third = max(limit // 3, 1)
    same_left: list[np.ndarray] = []
    same_right: list[np.ndarray] = []
    holders = np.flatnonzero(attached)
    if holders.size:
        order = np.argsort(attachment.nearest[holders], kind="stable")
        ordered = holders[order]
        keys = attachment.nearest[ordered]
        starts = np.r_[0, np.flatnonzero(np.diff(keys) != 0) + 1]
        ends = np.r_[starts[1:], keys.size]
        per_group = max(third // max(starts.size, 1), 1)
        for start, end in zip(starts, ends, strict=True):
            members = ordered[start:end]
            if members.size < 2:
                continue
            picked = generator.integers(0, members.size, (2, per_group))
            keep = picked[0] != picked[1]
            same_left.append(members[picked[0][keep]])
            same_right.append(members[picked[1][keep]])
    different_left, different_right = _sample_pairs(holders, generator, third)
    free = np.flatnonzero(detections.placed & ~attached)
    empty = np.empty(0, dtype=np.int64)
    mixed_left = generator.choice(holders, third) if holders.size else empty
    mixed_right = generator.choice(free, third) if free.size else empty
    size = min(mixed_left.size, mixed_right.size)
    left = np.concatenate([*same_left, different_left, mixed_left[:size]]).astype(
        np.int64
    )
    right = np.concatenate([*same_right, different_right, mixed_right[:size]]).astype(
        np.int64
    )
    return left, right


def _pair_cues(
    detections: Detections,
    embeddings: np.ndarray | None,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every pairwise quantity an association could consult, for one set of pairs."""
    positions = detections.position_eus
    cues = {
        "distance_m": np.linalg.norm(positions[left] - positions[right], axis=1),
        "delta_keyframe": np.abs(
            detections.keyframe_id[left] - detections.keyframe_id[right]
        ).astype(np.float64),
        "same_keyframe": (
            detections.keyframe_id[left] == detections.keyframe_id[right]
        ).astype(np.float64),
        "same_source": (detections.source[left] == detections.source[right]).astype(
            np.float64
        ),
        "same_label": (detections.label[left] == detections.label[right]).astype(
            np.float64
        ),
        "range_ratio": (
            detections.range_m[left] / np.maximum(detections.range_m[right], 1e-6)
        ),
    }
    if embeddings is not None:
        first = np.asarray(embeddings[left], dtype=np.float32)
        second = np.asarray(embeddings[right], dtype=np.float32)
        first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-6)
        second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-6)
        cues["cosine"] = np.einsum("ij,ij->i", first, second).astype(np.float64)
    return cues


def _rank_auc(same: np.ndarray, other: np.ndarray, *, higher_is_same: bool) -> float:
    """Tie-aware binary AUC, the same estimator `pair_cue_separability` uses."""
    if same.size == 0 or other.size == 0:
        return float("nan")
    values = np.concatenate((same, other))
    if not higher_is_same:
        values = -values
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (float(start + 1) + float(end)) / 2.0
    numerator = float(np.sum(ranks[: same.size])) - same.size * (same.size + 1) / 2.0
    return numerator / (same.size * other.size)


def section_pairs(
    data: MapData,
    report: Report,
    radius_m: float = 2.0,
    pair_limit: int = 200000,
    conditional_m: float = 2.0,
) -> None:
    """S4 — what separates a pair of detections of one object from any other pair.

    Labels come from the ground truth: two detections attached to the same annotation
    are the same object, two attached to different ones are not. The third population
    is the one the baskets never had — pairs where only one side is attached, which
    is what a real candidate set is full of.

    Every cue is reported three ways: raw, conditioned on the pair already being
    close, and with the share of pairs it even applies to. A high AUC on a narrow
    slice buys nothing, which this dossier has now learned twice.
    """
    report.head("S4 — indices par paire de détections")
    detections = data.detections
    attachment = attach_to_ground_truth(detections, data.ground_truth)
    if len(data.ground_truth) == 0:
        report.say("  aucune vérité terrain")
        return
    attached = detections.placed & attachment.attached(radius_m)
    generator = np.random.default_rng(0)
    left, right = _stratified_pairs(
        detections, attachment, attached, generator, pair_limit
    )
    if left.size == 0:
        report.say("  pas assez de détections")
        return

    both = attached[left] & attached[right]
    same_object = both & (attachment.nearest[left] == attachment.nearest[right])
    different = both & ~same_object
    one_sided = attached[left] ^ attached[right]
    report.say(
        f"  {left.size} paires échantillonnées : {int(same_object.sum())} même objet, "
        f"{int(different.sum())} objets distincts, {int(one_sided.sum())} contaminées "
        f"(un seul côté rattaché, {one_sided.mean():.1%})"
    )
    cues = _pair_cues(detections, data.embeddings, left, right)
    close = cues["distance_m"] <= conditional_m
    report.say(
        f"  {'indice':16s} {'AUC brute':>10s} {'AUC | proche':>13s} "
        f"{'part concernée':>15s}"
    )
    for name, values in cues.items():
        higher = name in {"cosine", "same_label", "same_source"}
        raw = _rank_auc(values[same_object], values[different], higher_is_same=higher)
        near = _rank_auc(
            values[same_object & close],
            values[different & close],
            higher_is_same=higher,
        )
        applies = float((values != 0.0).mean()) if name.startswith("same_") else 1.0
        report.say(f"  {name:16s} {raw:10.3f} {near:13.3f} {applies:15.1%}")
    report.say()
    report.say("  distances, par population :")
    for name, mask in (
        ("même objet", same_object),
        ("objets distincts", different),
        ("contaminées", one_sided),
    ):
        report.say(_percentile_line(name, cues["distance_m"][mask], "m"))
    report.data["s4"] = {
        "sampled_pairs": int(left.size),
        "same_object": int(same_object.sum()),
        "different_object": int(different.sum()),
        "contaminated": int(one_sided.sum()),
        "auc": {
            name: {
                "raw": _rank_auc(
                    values[same_object],
                    values[different],
                    higher_is_same=name in {"cosine", "same_label", "same_source"},
                ),
                "conditional": _rank_auc(
                    values[same_object & close],
                    values[different & close],
                    higher_is_same=name in {"cosine", "same_label", "same_source"},
                ),
            }
            for name, values in cues.items()
        },
    }


@dataclass
class Hubness:
    """How unevenly a set of embeddings shares out the role of nearest neighbour."""

    #: How many of the sampled rows retrieved this row in their top `k`.
    k_occurrences: np.ndarray
    #: Skewness of that count. Zero means every row is someone's neighbour as often
    #: as any other; above 1 a minority is absorbing the retrievals.
    skewness: float
    #: Share of rows retrieved at least five times as often as chance would give.
    hub_share: float
    #: Share of all retrievals absorbed by the busiest 1 % of rows.
    hub_mass: float
    #: Share of rows no sampled query ever retrieved. Nothing will ever find them.
    antihub_share: float


def _k_occurrences(unit: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """Count, per row, how many other rows hold it in their `k` nearest neighbours."""
    count = unit.shape[0]
    occurrences = np.zeros(count, dtype=np.int64)
    for start in range(0, count, block):
        stop = min(start + block, count)
        cosines = unit[start:stop] @ unit.T
        cosines[np.arange(stop - start), np.arange(start, stop)] = -2.0
        top = np.argpartition(-cosines, k, axis=1)[:, :k]
        np.add.at(occurrences, top.ravel(), 1)
    return occurrences


def hubness(unit: np.ndarray, k: int = HUBNESS_K) -> Hubness:
    """Measure the hub/antihub asymmetry of an already L2-normalised sample.

    In a high-dimensional space the nearest-neighbour relation stops being symmetric:
    a few vectors sit close to the centre of mass and are therefore close to almost
    everything, so they surface for queries that have nothing to do with them, while
    others are never anyone's neighbour. Both halves cost the pipeline something — a
    hub is a false positive that recurs across unrelated prompts, an antihub is a
    detection retrieval can never reach, whatever the threshold.

    Args:
        unit: Row-normalised embeddings, one sample per row.
        k: Neighbours each row retrieves.

    Returns:
        The k-occurrence counts and the four summaries read off them.
    """
    occurrences = _k_occurrences(unit, k)
    values = occurrences.astype(np.float64)
    deviation = values - values.mean()
    spread = float(np.sqrt((deviation**2).mean()))
    skewness = float((deviation**3).mean() / spread**3) if spread > 0 else 0.0
    ordered = np.sort(values)[::-1]
    top = max(1, int(round(0.01 * values.size)))
    total = float(values.sum())
    return Hubness(
        k_occurrences=occurrences,
        skewness=skewness,
        hub_share=float((values >= 5 * k).mean()),
        hub_mass=float(ordered[:top].sum() / total) if total > 0 else float("nan"),
        antihub_share=float((values == 0).mean()),
    )


def section_hubness(
    data: MapData, report: Report, sample: int = HUBNESS_SAMPLE, k: int = HUBNESS_K
) -> None:
    """S6 — the embedding space's own geometry, with no ground truth involved.

    The only section that needs neither annotations nor poses. It asks whether the
    index is a space retrieval can work in at all: if a handful of cutouts are the
    nearest neighbour of everything, every prompt inherits the same false positives,
    and no threshold, association or rescoring downstream can undo it.

    The centred column says what removing the cloud's centre of mass would do to
    **image-to-image** neighbour structure, and nothing more. It is *not* a prediction
    about text queries, and it was read that way once: an index ingested with
    `--center-embeddings` on vinci did drop exactly as measured here (skew 2.841 ->
    1.266, antihubs 5.5% -> 1.1%) and text retrieval collapsed 9x anyway, mAP 0.325 ->
    0.036. A text embedding does not sit inside this cloud — MetaCLIP's modality gap —
    so the centre subtracted here is not its centre, and removing it moves the query
    somewhere unrelated. Hubness among the cutouts is real; it is not what limits
    text-to-image retrieval.
    """
    report.head("S6 — hubness de l'espace d'embedding")
    embeddings = data.embeddings
    if embeddings is None:
        report.say("  aucun embedding lisible")
        return
    detections = data.detections
    total = embeddings.shape[0]
    generator = np.random.default_rng(0)
    rows = (
        np.sort(generator.choice(total, size=sample, replace=False))
        if total > sample
        else np.arange(total)
    )
    unit = _unit_rows(embeddings, rows)
    report.say(f"  {rows.size} lignes échantillonnées sur {total}, k={k}")

    raw = hubness(unit, k)
    centred = unit - unit.mean(axis=0, keepdims=True)
    centred /= np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-6)
    after = hubness(centred, k)
    report.say(f"  {'':22s} {'brut':>10s} {'centré':>10s}")
    for name, left, right in (
        ("asymétrie de N_k", raw.skewness, after.skewness),
        (f"part de hubs (>={5 * k})", raw.hub_share, after.hub_share),
        ("masse du 1 % le plus vu", raw.hub_mass, after.hub_mass),
        ("part d'antihubs (N_k=0)", raw.antihub_share, after.antihub_share),
    ):
        report.say(f"  {name:22s} {left:10.3f} {right:10.3f}")
    report.say(
        "  (asymétrie > 1 = une minorité absorbe les retrouvailles ; la colonne"
        " centrée est ce qu'un simple recentrage des embeddings rendrait)"
    )

    top = 20
    order = np.argsort(raw.k_occurrences)[::-1][:top]
    sample_labels = detections.label[rows]
    report.say()
    report.say(f"  --- ce que sont les {top} plus gros hubs (taux de base en regard)")
    for label, count in _label_counts(sample_labels[order])[:6]:
        base = float((sample_labels == label).mean())
        report.say(
            f"    {label[:30]:30s} {count:3d}/{top} ({count / top:5.1%})"
            f"  base {base:5.1%}  x{count / top / max(base, 1e-9):4.1f}"
        )
    report.say(
        _percentile_line("portée des hubs", detections.range_m[rows[order]], "m")
    )
    report.say(
        _percentile_line("portée de l'échantillon", detections.range_m[rows], "m")
    )
    report.data["s6"] = {
        "sampled": int(rows.size),
        "k": k,
        "raw": {
            "skewness": raw.skewness,
            "hub_share": raw.hub_share,
            "hub_mass": raw.hub_mass,
            "antihub_share": raw.antihub_share,
        },
        "centred": {
            "skewness": after.skewness,
            "hub_share": after.hub_share,
            "hub_mass": after.hub_mass,
            "antihub_share": after.antihub_share,
        },
        "hub_labels": {
            label: {
                "count": count,
                "base_rate": float((sample_labels == label).mean()),
            }
            for label, count in _label_counts(sample_labels[order])[:6]
        },
    }


def section_label_sets(
    data: MapData, report: Report, radius_m: float = 2.0, top_n: int = DEFAULT_TOP_N
) -> None:
    """S7 — the free labels against the annotation's *set* of acceptable ones.

    The proposals are the map's own detector labels, ordered by detector score, not a
    text-embedding ranking: this tool loads no model, and keeping that property is worth
    more than the sharper ranking a MetaCLIP pass would give. `gdino_labels.py` is where
    the embedding route lives when it is wanted, and `label_set_metrics.rank_labels`
    consumes its vectors.

    Until the annotations carry label sets this section reports that they do not, rather
    than a table of zeros — the same rule `s0` applies to everything else.
    """
    report.head("S7 — labels contre les ensembles annotés (ADR 0009)")
    ground_truth = data.ground_truth
    annotations = list(ground_truth.annotations)
    if not annotations:
        report.say("  aucune vérité terrain")
        return
    scored = [item for item in annotations if has_label_sets(item)]
    if not scored:
        report.say(
            f"  0/{len(annotations)} annotations portent des ensembles de labels —"
            " rien à mesurer (voir ADR 0009)"
        )
        report.data["s7"] = {"coverage": 0.0, "annotations": len(annotations)}
        return

    detections = data.detections
    attachment = attach_to_ground_truth(detections, ground_truth)
    attached = detections.placed & attachment.attached(radius_m)
    proposals: dict[str, list[str]] = {}
    for index, annotation in enumerate(annotations):
        rows = np.flatnonzero(attached & (attachment.nearest == index))
        if rows.size == 0:
            continue
        order = rows[np.argsort(-detections.score[rows])]
        seen: dict[str, None] = {}
        for label in detections.label[order].tolist():
            seen.setdefault(str(label), None)
        proposals[annotation.id] = list(seen)[:top_n]

    result = evaluate_label_sets(proposals, annotations, top_n)
    for line in label_set_report_lines(result):
        report.say(line)
    report.data["s7"] = {
        "coverage": result.coverage,
        "scored": result.scored,
        "annotations": result.annotations,
        "unproposed": result.unproposed,
        "top_n": result.top_n,
        "mean_set_ranking": result.mean_set_ranking,
    }


#: Every section, in the order the report prints them.
SECTIONS = {
    "s0": section_inventory,
    "s1": section_detections,
    "s2": section_labels,
    "s3": section_ground_truth,
    "s4": section_pairs,
    "s5": section_counting,
    "s6": section_hubness,
    "s7": section_label_sets,
}


def parse_args(argv: Sequence[str]) -> Any:
    """Parse the map-analysis command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", default=list(SECTIONS))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--layers-dir", type=Path, default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Analyse one map: print the report, and optionally write JSON and layers."""
    import sys

    from toolbox.benchmark.map_layers import write_layers

    args = parse_args(sys.argv[1:] if argv is None else argv)
    data = load_map(args.map_path)
    report = Report()
    for name in args.sections:
        if name not in SECTIONS:
            raise SystemExit(f"Unknown section {name!r}; known: {sorted(SECTIONS)}")
        SECTIONS[name](data, report)
    print(report.text())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.data, indent=2), encoding="utf-8")
        print(f"\nJSON écrit dans {args.json_out}")
    if args.layers_dir:
        written = write_layers(data, args.layers_dir, args.layers)
        print(f"couches écrites : {', '.join(path.name for path in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
