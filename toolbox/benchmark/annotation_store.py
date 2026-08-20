"""Read a map's annotations from the toolbox's SQLite store.

The benchmark reads `benchmark/annotations.geojson`, which the TypeScript backend
rewrites **only when a benchmark run starts** (`regenerateGroundTruth` in
`toolbox/backend/src/benchmark-runner.ts`). Any other reader of that file is reading
whatever the last run left behind: on vinci-st-domingue-zone-1 the export sat a day
behind the store, showing 9 annotations of 6 classes where the store held 12, and
reporting every ADR 0009 field as absent because the export predated them.

So a reader that must describe the *current* annotations — `validate_annotations`,
whose whole job is telling the annotator what to do next — reads the store instead.
The feature collection built here mirrors `buildGroundTruth` in
`toolbox/backend/src/annotation-store.ts`; keep the two in step.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from toolbox.benchmark.object_search_http_benchmark import (
    Annotation,
    parse_annotations,
)

#: Where a click's panorama, pixel and depth are recorded, newest spelling first. The
#: Annotation tab nests them under `source` (`source.erpU`); older writers flattened
#: them (`source_erp_u`), and the exports on disk still carry that. `_label_set` in the
#: benchmark already accepts both for labels — this is the same drift, for the three
#: fields `Annotation` does not carry.
SOURCE_FIELD_SPELLINGS: dict[str, tuple[str, ...]] = {
    "keyframe_id": ("source.keyframeId", "source_keyframe_id"),
    "erp_u": ("source.erpU", "source_erp_u"),
    "erp_v": ("source.erpV", "source_erp_v"),
    "depth_m": ("source.depthM", "depth_m"),
}

#: Mirrors `ANNOTATION_DB_FILENAME` in `toolbox/backend/src/annotation-store.ts`.
ANNOTATION_DB_FILENAME = "object-search-annotations.db"


def annotation_database_path(map_path: Path) -> Path:
    """Where the toolbox backend keeps this map's annotation store."""
    return map_path / ANNOTATION_DB_FILENAME


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the store read-only, so validating can never mutate an annotation.

    The backend usually holds this database open in WAL mode; a read-only connection
    sees its committed state without taking a write lock.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_extra_properties(value: str | None) -> dict[str, Any]:
    """`extra_properties` as a dict, or empty for null and unparseable JSON.

    Mirrors `parseExtraProperties`: a row with corrupt JSON keeps its geometry and
    class rather than disappearing from the ground truth.
    """
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _point_coordinates(lng: float, lat: float, alt: float | None) -> list[float]:
    """GeoJSON coordinates, carrying altitude only when the row has one."""
    return [lng, lat] if alt is None else [lng, lat, alt]


def _ground_truth_point_features(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Features from `ground_truth_point` — where the Annotation tab writes points.

    The ADR 0009 fields (`object_id`, `extent_m`, `exhaustive_zone`, `labels.*`) live
    inside `extra_properties`, which is why the columns alone do not carry them.
    """
    rows = connection.execute(
        """
        SELECT ground_truth_point_id, lng, lat, alt, level, class AS class_name,
          prompt, accuracy, extra_properties
        FROM ground_truth_point ORDER BY ground_truth_point_id
        """
    ).fetchall()
    features: list[dict[str, Any]] = []
    for row in rows:
        properties = _parse_extra_properties(row["extra_properties"])
        for key, column in (
            ("class", "class_name"),
            ("prompt", "prompt"),
            ("accuracy", "accuracy"),
            ("level", "level"),
            ("altitude", "alt"),
        ):
            if row[column] is not None:
                properties[key] = row[column]
        features.append(
            {
                "type": "Feature",
                "id": f"gt-point-{row['ground_truth_point_id']}",
                "geometry": {
                    "type": "Point",
                    "coordinates": _point_coordinates(
                        row["lng"], row["lat"], row["alt"]
                    ),
                },
                "properties": properties,
            }
        )
    return features


def _manual_detection_features(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Features from `manual_detection` rows flagged as ground truth.

    Empty on both real maps at the time of writing, but `buildGroundTruth` emits them,
    and a validator disagreeing with the benchmark about what the ground truth *is*
    would be worse than the staleness this module exists to fix.

    `source_erp_u`/`source_erp_v` are deliberately not recomputed here: the export
    carries them, `parse_annotations` never reads them, and decoding the stored
    geometry to produce a field nothing consumes would be dead weight.
    """
    rows = connection.execute(
        """
        SELECT md.manual_detection_id, md.keyframe_id, mdc.class AS class_name,
          md.query, md.used_as_ground_truth, md.lat, md.lng, md.alt, md.level
        FROM manual_detection md
        LEFT JOIN manual_detection_classes mdc ON mdc.class_id = md.class_id
        ORDER BY md.manual_detection_id
        """
    ).fetchall()
    features: list[dict[str, Any]] = []
    for row in rows:
        if not row["used_as_ground_truth"]:
            continue
        if row["lat"] is None or row["lng"] is None:
            continue
        query = (row["query"] or "").strip()
        class_name = (row["class_name"] or "").strip() or query
        if not class_name:
            continue
        properties: dict[str, Any] = {
            "class": class_name,
            "id": row["manual_detection_id"],
            "source": "annotation",
            "source_keyframe_id": str(row["keyframe_id"]),
        }
        if query:
            properties["prompt"] = query
        if row["level"] is not None and str(row["level"]).strip():
            properties["level"] = row["level"]
        if row["alt"] is not None:
            properties["altitude"] = row["alt"]
        features.append(
            {
                "type": "Feature",
                "id": f"annotation-{row['manual_detection_id']}",
                "geometry": {
                    "type": "Point",
                    "coordinates": _point_coordinates(
                        row["lng"], row["lat"], row["alt"]
                    ),
                },
                "properties": properties,
            }
        )
    return features


def build_ground_truth(db_path: Path) -> dict[str, Any]:
    """The store's ground truth as a GeoJSON FeatureCollection.

    Mirrors `buildGroundTruth`, manual detections first then stored points, so the
    order matches the file the benchmark reads.
    """
    with _connect_readonly(db_path) as connection:
        features = _manual_detection_features(connection)
        features.extend(_ground_truth_point_features(connection))
    return {"type": "FeatureCollection", "features": features}


def load_store_annotations(
    db_path: Path, default_accuracy_m: float
) -> list[Annotation]:
    """Annotations read from the SQLite store, parsed exactly as the export is."""
    return parse_annotations(build_ground_truth(db_path), default_accuracy_m)


def _dig(properties: Mapping[str, Any], dotted: str) -> Any:
    """Read a possibly nested property by dotted path, None when any step is missing."""
    current: Any = properties
    for step in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(step)
    return current


def source_field(properties: Mapping[str, Any], name: str) -> Any:
    """One of `SOURCE_FIELD_SPELLINGS`, whichever spelling this feature happens to use.

    Reading only the flat spelling is what made the analysis report NaN pixels for every
    annotation written by the Annotation tab: the store nests them.
    """
    for dotted in SOURCE_FIELD_SPELLINGS[name]:
        value = _dig(properties, dotted)
        if value is not None:
            return value
    return None


def read_ground_truth_collection(map_path: Path) -> tuple[Path | None, dict[str, Any]]:
    """The map's ground truth and where it came from: store first, export as fallback.

    An empty collection with a `None` source means the map has neither — a freshly
    prepared map, which callers describe rather than treat as an error.
    """
    db_path = annotation_database_path(map_path)
    if db_path.is_file():
        return db_path, build_ground_truth(db_path)

    export = map_path / "benchmark" / "annotations.geojson"
    if export.is_file():
        with export.open("r", encoding="utf-8") as handle:
            return export, json.load(handle)
    return None, {"type": "FeatureCollection", "features": []}
