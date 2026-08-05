"""Tests for checked-in hybrid build configs."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_hybrid_train_station_ocr_config_is_top_level_and_enabled() -> None:
    config_path = CONFIG_DIR / "config_hybrid_train_station.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["ocr"]["enabled"] is True
    assert cfg["ocr"]["candidate_chunk_size"] == 1000
    assert cfg["ocr"]["max_candidates"] == 0
    assert cfg["ocr"]["max_candidates_per_keyframe"] == 0
    assert "ocr" not in cfg.get("offline", {})
