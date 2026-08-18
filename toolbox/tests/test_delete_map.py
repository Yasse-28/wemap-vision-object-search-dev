"""Deleting an exported map, and the gates that stop it deleting the wrong one.

`delete_map` is the only destructive path in the toolbox, and its blast radius is
a `DELETE ... WHERE geo_ref_id = %s` on two tables shared by every map. Each test
here pins one refusal, because every one of them protects a map the tool was not
asked to touch.

The database half is not exercised: it needs a live Postgres with pgvector and
PostGIS (`infra/postgres/compose.yml`). What is exercised is everything that
decides *whether* to run it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolbox.bricks.delete_map import (
    ConfiguredMap,
    DeletionRefused,
    children_of,
    plan_deletion,
)
from toolbox.bricks.export_roi import PROVENANCE_FILENAME


def _write_map(map_path: Path, map_id: str, geo_ref_id: int) -> None:
    map_path.mkdir(parents=True, exist_ok=True)
    (map_path / f"{map_id}_1_20260101_000000.json").write_text(
        json.dumps(
            {
                "local_origin": [2.3522, 48.8566, 35.0],
                "map": {"name": map_id, "uuid": "u", "venue_type": "rail"},
                "geo_levels": [
                    {
                        "value": 0,
                        "min_altitude": None,
                        "max_altitude": None,
                        "geo_ref": geo_ref_id,
                        "geometry": None,
                    }
                ],
                "geo_keyframes": [
                    {
                        "image_url": "https://example.test/images/kf0.jpg",
                        "depth_url": "https://example.test/depths/kf0.tif",
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _exported(map_path: Path, map_id: str, geo_ref_id: int) -> None:
    """A map that looks like one this toolbox produced."""
    _write_map(map_path, map_id, geo_ref_id)
    (map_path / PROVENANCE_FILENAME).write_text(
        json.dumps({"map_id": map_id, "geo_ref_id": geo_ref_id}), encoding="utf-8"
    )


def _configured(
    tmp_path: Path, *entries: tuple[str, str | None]
) -> list[ConfiguredMap]:
    return [
        ConfiguredMap(map_id=map_id, path=tmp_path / map_id, parent_map=parent)
        for map_id, parent in entries
    ]


def test_a_production_map_is_never_deletable(tmp_path: Path) -> None:
    """No `parent_map` means this repo did not make it and cannot remake it."""
    _write_map(tmp_path / "src", "src", 7)
    configured = _configured(tmp_path, ("src", None))

    with pytest.raises(DeletionRefused, match="no parent_map"):
        plan_deletion("src", configured)


def test_a_map_with_children_is_refused_and_names_them(tmp_path: Path) -> None:
    _exported(tmp_path / "src-roi", "src-roi", 8)
    _exported(tmp_path / "src-roi-north", "src-roi-north", 9)
    configured = _configured(
        tmp_path, ("src", None), ("src-roi", "src"), ("src-roi-north", "src-roi")
    )

    with pytest.raises(DeletionRefused, match="src-roi-north"):
        plan_deletion("src-roi", configured)

    assert children_of("src-roi", configured) == ["src-roi-north"]
    assert children_of("src-roi-north", configured) == []


def test_a_shared_geo_ref_id_is_refused(tmp_path: Path) -> None:
    """Purging a shared id would take the neighbour's rows with it."""
    _write_map(tmp_path / "src", "src", 7)
    _exported(tmp_path / "src-roi", "src-roi", 7)
    configured = _configured(tmp_path, ("src", None), ("src-roi", "src"))

    with pytest.raises(DeletionRefused, match="also used by src"):
        plan_deletion("src-roi", configured)


def test_a_directory_without_provenance_is_kept(tmp_path: Path) -> None:
    """The config entry and the rows go; a directory we did not write stays."""
    _write_map(tmp_path / "src-roi", "src-roi", 8)  # no provenance file
    configured = _configured(tmp_path, ("src", None), ("src-roi", "src"))

    plan = plan_deletion("src-roi", configured)

    assert plan["removes_directory"] is False
    assert any(PROVENANCE_FILENAME in warning for warning in plan["warnings"])


def test_the_plan_reports_what_the_dialog_must_show(tmp_path: Path) -> None:
    _exported(tmp_path / "src-roi", "src-roi", 8)
    configured = _configured(tmp_path, ("src", None), ("src-roi", "src"))

    plan = plan_deletion("src-roi", configured)

    assert plan["map_id"] == "src-roi"
    assert plan["parent_map"] == "src"
    assert plan["geo_ref_id"] == 8
    assert plan["keyframe_count"] == 1
    assert plan["removes_directory"] is True
    assert plan["index_name"] == "idx_object_search_candidate_hnsw_georef_8"
    assert plan["warnings"] == []


def test_a_half_deleted_map_stays_deletable(tmp_path: Path) -> None:
    """A missing directory is a warning, not a refusal — the rows may still exist."""
    configured = _configured(tmp_path, ("src", None), ("src-roi", "src"))

    plan = plan_deletion("src-roi", configured)

    assert plan["geo_ref_id"] is None
    assert plan["warnings"]
