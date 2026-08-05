"""Tests for offline GroundingDINO YAML loader."""

from pathlib import Path

import pytest

from pipeline.offline.config.prompts_yaml import (
    DEFAULT_CROP_PADDING_PX,
    DEFAULT_GDINO_MODEL_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_AREA_RATIO,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MIN_CONFIDENCE,
    load_gdino_params,
    load_keyframe_filter_config,
    load_offline_build_config,
)


def test_load_gdino_params_minimal(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
gdino_params:
  prompt: "a ."
""",
        encoding="utf-8",
    )
    g = load_gdino_params(p)
    assert g.prompt == "a ."
    assert g.score is None
    assert g.model_threshold_used() == DEFAULT_GDINO_MODEL_THRESHOLD
    assert g.iou_threshold == DEFAULT_IOU_THRESHOLD
    assert g.min_area_ratio == DEFAULT_MIN_AREA_RATIO
    assert g.max_area_ratio == DEFAULT_MAX_AREA_RATIO
    assert g.min_confidence == DEFAULT_MIN_CONFIDENCE
    assert g.crop_padding_px == DEFAULT_CROP_PADDING_PX
    assert g.exclude_labels == []


def test_load_gdino_params_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
gdino_params:
  prompt: "x ."
  score: 0.08
  iou_threshold: 0.4
  min_area_ratio: 0.02
  max_area_ratio: 0.4
  min_confidence: 0.15
  crop_padding_px: 32
  exclude_labels: ["person", "passenger"]
""",
        encoding="utf-8",
    )
    g = load_gdino_params(p)
    assert g.prompt == "x ."
    assert g.score == 0.08
    assert g.model_threshold_used() == 0.08
    assert g.iou_threshold == 0.4
    assert g.min_area_ratio == 0.02
    assert g.max_area_ratio == 0.4
    assert g.min_confidence == 0.15
    assert g.crop_padding_px == 32
    assert g.exclude_labels == ["person", "passenger"]


def test_load_gdino_params_legacy_single_prompts_list(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
gdino_params:
  prompts:
    - "legacy one ."
""",
        encoding="utf-8",
    )
    g = load_gdino_params(p)
    assert g.prompt == "legacy one ."


def test_load_gdino_params_experiment(tmp_path: Path) -> None:
    p = tmp_path / "eval.yaml"
    p.write_text(
        """
experiments:
  - name: station_a
    prompt: "sign . bench ."
    confidence_threshold: 0.11
    exclude_labels: ["person"]
""",
        encoding="utf-8",
    )
    g = load_gdino_params(p, experiment="station_a")
    assert g.prompt == "sign . bench ."
    assert g.score == 0.11
    assert g.iou_threshold == DEFAULT_IOU_THRESHOLD
    assert g.exclude_labels == ["person"]


def test_load_gdino_params_experiment_missing_raises(tmp_path: Path) -> None:
    p = tmp_path / "eval.yaml"
    p.write_text("experiments: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No experiment"):
        load_gdino_params(p, experiment="nope")


def test_to_manifest_json_roundtrip_keys(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        'gdino_params:\n  prompt: "z ."\n  score: 0.2\n',
        encoding="utf-8",
    )
    g = load_gdino_params(p)
    j = g.to_manifest_json()
    assert '"prompt"' in j
    assert '"model_threshold_used": 0.2' in j
    assert '"exclude_labels": []' in j


def test_load_keyframe_filter_config_defaults_to_no_polygon(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
gdino_params:
  prompt: "z ."
""",
        encoding="utf-8",
    )

    filters = load_keyframe_filter_config(p)

    assert filters.polygon_geojson_path is None


def test_load_keyframe_filter_config_resolves_relative_polygon_path(
    tmp_path: Path,
) -> None:
    p = tmp_path / "configs" / "p.yaml"
    p.parent.mkdir()
    p.write_text(
        """
gdino_params:
  prompt: "z ."
keyframe_filters:
  level_filter: 3
  polygon_geojson_path: ../polygons/area.geojson
""",
        encoding="utf-8",
    )

    filters = load_keyframe_filter_config(p)

    assert filters.level_filter == 3
    assert (
        filters.polygon_geojson_path
        == (tmp_path / "polygons" / "area.geojson").resolve()
    )


def test_load_offline_build_config_sections(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
offline:
  device: cuda
  skip_objects: false
  inference:
    batch_size: 8
    cutout_workers: 0
    object_crop_batch_size: 2
  geometry:
    geometry: horizon
    horizon_nb_of_cuts: 6
  localization:
    clustering_eps_m: 4.0
  visual_refinement:
    enabled: true
    batch_size: 7
  ocr_refinement:
    enabled: true
    lightweight_first: true
    batch_size: 5
gdino_params:
  prompt: "z ."
""",
        encoding="utf-8",
    )

    config = load_offline_build_config(p)

    assert config.device == "cuda"
    assert config.skip_objects is False
    assert config.inference is not None
    assert config.inference.batch_size == 8
    assert config.inference.cutout_workers == 0
    assert config.inference.object_crop_batch_size == 2
    assert config.geometry is not None
    assert config.geometry.geometry == "horizon"
    assert config.geometry.horizon_nb_of_cuts == 6
    assert config.localization is not None
    assert config.localization.clustering_eps_m == 4.0
    assert config.visual_refinement is not None
    assert config.visual_refinement.enabled is True
    assert config.visual_refinement.batch_size == 7
    assert config.ocr_refinement is not None
    assert config.ocr_refinement.enabled is True
    assert config.ocr_refinement.lightweight_first is True
    assert config.ocr_refinement.batch_size == 5


def test_load_offline_build_config_rejects_removed_flat_source(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
offline:
  geometry:
    source: cutout_flat
gdino_params:
  prompt: "z ."
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="flat cutouts"):
        load_offline_build_config(p)


def test_load_offline_build_config_rejects_misplaced_keyframe_filters(
    tmp_path: Path,
) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
offline:
  localization:
    level_filter: 1
    polygon_geojson_path: area.geojson
gdino_params:
  prompt: "z ."
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top-level keyframe_filters"):
        load_offline_build_config(p)


def test_full_example_config_is_parseable() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "legacy"
        / "config_example_all.yaml"
    )

    gdino = load_gdino_params(config_path)
    filters = load_keyframe_filter_config(config_path)
    offline = load_offline_build_config(config_path)

    assert gdino.prompt
    assert filters.level_filter is None
    assert filters.polygon_geojson_path is None
    assert offline.inference is not None
    assert offline.geometry is not None
    assert offline.localization is not None
    assert offline.visual_refinement is not None
    assert offline.ocr_refinement is not None
