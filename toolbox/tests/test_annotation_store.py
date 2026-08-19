"""Unit tests for reading annotations out of the toolbox's SQLite store.

The staleness these cover is not hypothetical: the exported GeoJSON is rewritten only
when a benchmark run starts, so it lagged a day behind the store on
vinci-st-domingue-zone-1 and reported every ADR 0009 field as absent.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from toolbox.benchmark.annotation_store import (
    ANNOTATION_DB_FILENAME,
    build_ground_truth,
    load_store_annotations,
    read_ground_truth_collection,
    source_field,
)
from toolbox.benchmark.object_search_http_benchmark import parse_annotated_features
from toolbox.benchmark.validate_annotations import read_annotations

POINT_SCHEMA = """
CREATE TABLE ground_truth_point (
  ground_truth_point_id INTEGER PRIMARY KEY,
  lng REAL NOT NULL, lat REAL NOT NULL, alt REAL, level TEXT, class TEXT,
  prompt TEXT, accuracy REAL, extra_properties TEXT, created_at TEXT NOT NULL
);
CREATE TABLE manual_detection_classes (class_id INTEGER PRIMARY KEY, class TEXT);
CREATE TABLE manual_detection (
  manual_detection_id INTEGER PRIMARY KEY, keyframe_id TEXT NOT NULL,
  geometry_type TEXT NOT NULL, geometry BLOB NOT NULL, class_id INTEGER,
  query TEXT NOT NULL, used_as_ground_truth INTEGER NOT NULL DEFAULT 1,
  lat REAL, lng REAL, alt REAL, level TEXT, created_at TEXT NOT NULL
);
"""


def _store(tmp_path: Path) -> Path:
    """An annotation store with the tables `build_ground_truth` reads."""
    db_path = tmp_path / ANNOTATION_DB_FILENAME
    connection = sqlite3.connect(db_path)
    connection.executescript(POINT_SCHEMA)
    connection.commit()
    connection.close()
    return db_path


def _add_point(db_path: Path, **columns: object) -> None:
    row: dict[str, object] = {
        "lng": 2.0,
        "lat": 48.0,
        "alt": None,
        "level": "2",
        "class": "maglock",
        "prompt": "maglock",
        "accuracy": 5.0,
        "extra_properties": None,
        "created_at": "2026-08-19T13:19:44Z",
    }
    row.update(columns)
    connection = sqlite3.connect(db_path)
    connection.execute(
        f"INSERT INTO ground_truth_point ({', '.join(row)}) "
        f"VALUES ({', '.join(':' + name for name in row)})",
        row,
    )
    connection.commit()
    connection.close()


def test_contract_fields_are_read_out_of_extra_properties(tmp_path: Path) -> None:
    """The columns carry none of ADR 0009 — the JSON blob is where it lives."""
    db_path = _store(tmp_path)
    _add_point(
        db_path,
        extra_properties=json.dumps(
            {
                "object_id": "maglock-001",
                "extent_m": 0.1,
                "exhaustive_zone": "zone-1",
                "labels": {"synonyms": ["ventouse"]},
            }
        ),
    )

    (annotation,) = load_store_annotations(db_path, 5.0)

    assert annotation.object_id == "maglock-001"
    assert annotation.extent_m == pytest.approx(0.1)
    assert annotation.exhaustive_zone == "zone-1"
    assert annotation.synonyms == ("ventouse",)


def test_columns_win_over_the_blob_for_the_fields_they_own(tmp_path: Path) -> None:
    """Mirrors `buildGroundTruth`, which overwrites class/prompt/accuracy/level."""
    db_path = _store(tmp_path)
    _add_point(
        db_path,
        extra_properties=json.dumps({"class": "porte", "accuracy": 99.0}),
    )

    (annotation,) = load_store_annotations(db_path, 5.0)

    assert annotation.class_name == "maglock"
    assert annotation.accuracy_m == pytest.approx(5.0)


def test_a_row_with_corrupt_json_keeps_its_class(tmp_path: Path) -> None:
    """Losing the blob must not drop the annotation from the ground truth."""
    db_path = _store(tmp_path)
    _add_point(db_path, extra_properties="{not json")

    (annotation,) = load_store_annotations(db_path, 5.0)

    assert annotation.class_name == "maglock"
    assert annotation.object_id is None


def test_altitude_rides_along_only_when_the_row_has_one(tmp_path: Path) -> None:
    db_path = _store(tmp_path)
    _add_point(db_path, alt=None)
    _add_point(db_path, alt=12.5, lat=48.001)

    features = build_ground_truth(db_path)["features"]

    assert features[0]["geometry"]["coordinates"] == [2.0, 48.0]
    assert features[1]["geometry"]["coordinates"] == [2.0, 48.001, 12.5]


def test_a_manual_detection_not_used_as_ground_truth_is_skipped(
    tmp_path: Path,
) -> None:
    db_path = _store(tmp_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO manual_detection (keyframe_id, geometry_type, geometry, query,"
        " used_as_ground_truth, lat, lng, created_at)"
        " VALUES ('kf-1', 'bbox', X'00', 'cctv', 0, 48.0, 2.0, 'now')"
    )
    connection.execute(
        "INSERT INTO manual_detection (keyframe_id, geometry_type, geometry, query,"
        " used_as_ground_truth, lat, lng, created_at)"
        " VALUES ('kf-2', 'bbox', X'00', 'cctv', 1, 48.001, 2.0, 'now')"
    )
    connection.commit()
    connection.close()

    annotations = load_store_annotations(db_path, 5.0)

    assert [item.id for item in annotations] == ["annotation-2"]


def test_the_store_is_preferred_over_a_stale_export(tmp_path: Path) -> None:
    """The bug this module exists for: the export described a fixed map as broken."""
    db_path = _store(tmp_path)
    _add_point(db_path, extra_properties=json.dumps({"object_id": "maglock-001"}))
    export = tmp_path / "benchmark" / "annotations.geojson"
    export.parent.mkdir()
    export.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    source, annotations = read_annotations(tmp_path, None, 5.0)

    assert source == db_path
    assert [item.object_id for item in annotations] == ["maglock-001"]


def test_naming_an_export_explicitly_still_reads_it(tmp_path: Path) -> None:
    """The escape hatch: comparing against what the last benchmark actually scored."""
    _store(tmp_path)
    export = tmp_path / "annotations.geojson"
    export.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "gt-point-1",
                        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                        "properties": {"class": "door"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source, annotations = read_annotations(tmp_path, export, 5.0)

    assert source == export
    assert [item.class_name for item in annotations] == ["door"]


def test_a_map_with_neither_source_says_so(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        read_annotations(tmp_path, None, 5.0)

    assert ANNOTATION_DB_FILENAME in str(error.value)


def test_the_source_fields_are_read_under_either_spelling() -> None:
    """The store nests them under `source`; the exports on disk are flat."""
    nested = {"source": {"keyframeId": 0, "erpU": 0.25, "erpV": 0.5, "depthM": 1.99}}
    flat = {
        "source_keyframe_id": "127",
        "source_erp_u": 0.25,
        "source_erp_v": 0.5,
        "depth_m": 1.99,
    }

    for properties in (nested, flat):
        assert source_field(properties, "erp_u") == pytest.approx(0.25)
        assert source_field(properties, "depth_m") == pytest.approx(1.99)
    assert source_field(nested, "keyframe_id") == 0
    assert source_field(flat, "keyframe_id") == "127"


def test_a_missing_source_field_is_none_not_a_crash() -> None:
    assert source_field({}, "erp_u") is None
    assert source_field({"source": "not a mapping"}, "erp_u") is None


def test_keyframe_zero_survives_being_read(tmp_path: Path) -> None:
    """`or ""` would erase it: 0 is a real panorama, and the store writes an int."""
    db_path = _store(tmp_path)
    _add_point(db_path, extra_properties=json.dumps({"source": {"keyframeId": 0}}))

    ((properties, _),) = parse_annotated_features(build_ground_truth(db_path), 5.0)

    assert source_field(properties, "keyframe_id") == 0
    assert source_field(properties, "keyframe_id") is not None


def test_features_are_paired_with_their_own_properties(tmp_path: Path) -> None:
    """Zipping the parsed list against `features` would misalign past a skipped one."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {  # dropped: no class
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                "properties": {"source": {"erpU": 0.1}},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [2.0, 48.001]},
                "properties": {"class": "maglock", "source": {"erpU": 0.9}},
            },
        ],
    }

    pairs = parse_annotated_features(collection, 5.0)

    assert len(pairs) == 1
    properties, annotation = pairs[0]
    assert annotation.class_name == "maglock"
    assert source_field(properties, "erp_u") == pytest.approx(0.9)


def test_a_map_with_no_annotations_at_all_reads_as_empty(tmp_path: Path) -> None:
    """What `map_analysis` relies on to describe a freshly prepared map."""
    source, collection = read_ground_truth_collection(tmp_path)

    assert source is None
    assert collection["features"] == []
