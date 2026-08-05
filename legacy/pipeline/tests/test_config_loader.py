import json
from pathlib import Path

import pytest

from pipeline.online.config_loader import load_map_entries


def test_default_maps_subdir(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "map-a"}]}), encoding="utf-8")
    parent, entries = load_map_entries(cfg)
    assert parent == tmp_path.resolve()
    assert len(entries) == 1
    assert entries[0].map_id == "map-a"
    assert entries[0].map_path == (tmp_path / "maps" / "map-a").resolve()


def test_explicit_path_relative_to_config_parent(tmp_path: Path):
    cfg = tmp_path / "deploy" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"maps": [{"id": "x", "path": "maps/custom"}]}),
        encoding="utf-8",
    )
    parent, entries = load_map_entries(cfg)
    assert parent == cfg.parent.resolve()
    assert entries[0].map_path == (cfg.parent / "maps" / "custom").resolve()


def test_duplicate_id_errors(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"maps": [{"id": "same"}, {"id": "same"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_map_entries(cfg)


def test_json5_comments_are_accepted(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        """
        {
          // Station maps
          maps: [
            {
              id: "station-a",
              path: "maps/station-a",
            },
          ],
        }
        """,
        encoding="utf-8",
    )

    parent, entries = load_map_entries(cfg)

    assert parent == tmp_path.resolve()
    assert len(entries) == 1
    assert entries[0].map_id == "station-a"
    assert entries[0].map_path == (tmp_path / "maps" / "station-a").resolve()


def test_emmid_is_loaded_when_present(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "id": "station-a",
                        "path": "maps/station-a",
                        "emmid": 33128,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _parent, entries = load_map_entries(cfg)

    assert entries[0].emmid == 33128


def test_emmid_defaults_to_none(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")

    _parent, entries = load_map_entries(cfg)

    assert entries[0].emmid is None


def test_display_name_field_does_not_change_map_entry_id(tmp_path: Path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "id": "station-a",
                        "display_name": "Ignored display name",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _parent, entries = load_map_entries(cfg)

    assert entries[0].map_id == "station-a"
