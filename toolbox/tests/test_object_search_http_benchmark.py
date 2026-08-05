from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolbox.benchmark.object_search_http_benchmark import (
    FN_COLOR,
    FP_COLOR,
    REFERENCE_COLOR,
    TP_COLOR,
    Annotation,
    build_prompt_geojson,
    compute_prf,
    enrich_prediction_levels_from_artifact,
    evaluate_prompt,
    evaluate_prompt_grouped,
    group_annotations,
    haversine_m,
    load_annotations,
    parse_args,
    parse_predictions,
    prediction_level,
    resolve_localize_url,
)


def test_load_annotations_reads_point_geojson(tmp_path: Path) -> None:
    path = tmp_path / "annotations.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "ann-1",
                        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                        "properties": {
                            "class": "boite aux lettres",
                            "accuracy": 3.5,
                            "level": "0",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path, default_accuracy_m=5.0)

    assert annotations == [
        Annotation(
            id="ann-1",
            class_name="boite aux lettres",
            lat=48.0,
            lng=2.0,
            accuracy_m=3.5,
            prompt="boite aux lettres",
            level="0",
        )
    ]


def test_load_annotations_uses_prompt_field_when_present(tmp_path: Path) -> None:
    path = tmp_path / "annotations.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "ann-1",
                        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                        "properties": {
                            "class": "boite aux lettres",
                            "prompt": "boite aux lettres rouge",
                            "accuracy": 5.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path, default_accuracy_m=5.0)

    assert len(annotations) == 1
    assert annotations[0].class_name == "boite aux lettres"
    assert annotations[0].prompt == "boite aux lettres rouge"


def test_load_annotations_falls_back_to_class_when_no_prompt(tmp_path: Path) -> None:
    path = tmp_path / "annotations.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "ann-1",
                        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                        "properties": {"class": "défibrillateur", "accuracy": 5.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path, default_accuracy_m=5.0)

    assert annotations[0].prompt == annotations[0].class_name == "défibrillateur"


def test_parse_predictions_reads_coordinates_array() -> None:
    # Standalone /localize returns ObjectLocation entries with a
    # "coordinates": [lat, lng, alt] tuple and no lat/lng/cluster_id fields.
    response = {
        "localizations": [
            {
                "coordinates": [48.88, 2.355, 35.2],
                "confidence": 0.8,
                "observation_count": 4,
                "similarity_score": 0.31,
                "match_score": 0.93,
                "level": 0,
            }
        ]
    }

    predictions = parse_predictions(response, "match_score")

    assert len(predictions) == 1
    assert predictions[0].lat == pytest.approx(48.88)
    assert predictions[0].lng == pytest.approx(2.355)
    assert predictions[0].score == pytest.approx(0.93)


def test_parse_predictions_uses_match_score_threshold_field() -> None:
    response = {
        "localizations": [
            {"cluster_id": 7, "lat": 48.0, "lng": 2.0, "match_score": 0.95},
            {"cluster_id": 8, "lat": 48.1, "lng": 2.1, "similarity_score": 0.4},
        ]
    }

    predictions = parse_predictions(response, "match_score")

    assert [item.id for item in predictions] == ["7", "8"]
    assert [item.score for item in predictions] == [0.95, 0.4]


def test_prediction_level_accepts_level_and_cluster_level() -> None:
    predictions = parse_predictions(
        {
            "localizations": [
                {
                    "cluster_id": 1,
                    "lat": 48.0,
                    "lng": 2.0,
                    "match_score": 0.9,
                    "level": 0,
                },
                {
                    "cluster_id": 2,
                    "lat": 48.0,
                    "lng": 2.0,
                    "match_score": 0.9,
                    "cluster_level": 3,
                },
            ]
        },
        "match_score",
    )

    assert [prediction_level(item) for item in predictions] == [0, 3]


def test_enrich_prediction_levels_from_artifact_uses_observation_object_idx() -> None:
    predictions = parse_predictions(
        {
            "localizations": [
                {
                    "cluster_id": 1,
                    "lat": 48.0,
                    "lng": 2.0,
                    "match_score": 0.9,
                    "observations": [{"object_idx": 2}],
                }
            ]
        },
        "match_score",
    )

    enrich_prediction_levels_from_artifact(
        predictions,
        (
            [5, 6, 1],
            [-1, 4, 2, 3, 0, 1, 2],
        ),
    )

    assert prediction_level(predictions[0]) == 4
    assert predictions[0].raw["level_source"] == "object-search.npz"
    assert predictions[0].raw["artifact_cluster_id"] == 1


def test_evaluate_prompt_matches_predictions_within_accuracy() -> None:
    annotation = Annotation(
        id="ann-1",
        class_name="boite aux lettres",
        lat=48.0,
        lng=2.0,
        accuracy_m=5.0,
    )
    response = {
        "localizations": [
            {"cluster_id": "near", "lat": 48.0, "lng": 2.00001, "match_score": 0.91},
            {"cluster_id": "far", "lat": 48.01, "lng": 2.01, "match_score": 0.92},
            {"cluster_id": "rejected", "lat": 48.0, "lng": 2.0, "match_score": 0.9},
        ]
    }

    metrics = evaluate_prompt(
        class_name="boite aux lettres",
        prompt="boite aux lettres",
        class_annotations=[annotation],
        predictions=parse_predictions(response, "match_score"),
        acceptance_threshold=0.9,
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.accepted_predictions == 2
    assert metrics.rejected_predictions == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_grouped_metrics_count_close_annotations_as_one_target() -> None:
    annotations = [
        Annotation(
            id="ann-1",
            class_name="boite aux lettres",
            lat=48.0,
            lng=2.0,
            accuracy_m=0.05,
        ),
        Annotation(
            id="ann-2",
            class_name="boite aux lettres",
            lat=48.0,
            lng=2.0000001,
            accuracy_m=0.05,
        ),
    ]
    predictions = parse_predictions(
        {
            "localizations": [
                {"cluster_id": "near", "lat": 48.0, "lng": 2.0, "match_score": 0.91},
            ]
        },
        "match_score",
    )

    strict_metrics = evaluate_prompt(
        class_name="boite aux lettres",
        prompt="boite aux lettres",
        class_annotations=annotations,
        predictions=predictions,
        acceptance_threshold=0.9,
    )
    groups = group_annotations(annotations, radius_m=0.25)
    grouped_metrics = evaluate_prompt_grouped(
        class_name="boite aux lettres",
        prompt="boite aux lettres",
        annotation_groups=groups,
        predictions=predictions,
        acceptance_threshold=0.9,
    )

    assert len(groups) == 1
    assert strict_metrics.true_positives == 1
    assert strict_metrics.false_negatives == 1
    assert grouped_metrics.true_positives == 1
    assert grouped_metrics.false_negatives == 0
    assert grouped_metrics.ground_truth == 1


def test_group_annotations_does_not_merge_different_classes() -> None:
    annotations = [
        Annotation(
            id="ann-1",
            class_name="boite aux lettres",
            lat=48.0,
            lng=2.0,
            accuracy_m=0.05,
        ),
        Annotation(
            id="ann-2",
            class_name="défibrillateur",
            lat=48.0,
            lng=2.0000001,
            accuracy_m=0.05,
        ),
    ]

    groups = group_annotations(annotations, radius_m=0.25)

    assert len(groups) == 2
    assert {group.class_name for group in groups} == {
        "boite aux lettres",
        "défibrillateur",
    }


def test_build_prompt_geojson_marks_tp_fp_fn_and_reference() -> None:
    annotations = [
        Annotation(
            id="matched-ann",
            class_name="boite aux lettres",
            lat=48.0,
            lng=2.0,
            accuracy_m=5.0,
        ),
        Annotation(
            id="missed-ann",
            class_name="boite aux lettres",
            lat=48.02,
            lng=2.02,
            accuracy_m=5.0,
        ),
    ]
    predictions = parse_predictions(
        {
            "localizations": [
                {
                    "cluster_id": "near",
                    "lat": 48.0,
                    "lng": 2.00001,
                    "match_score": 0.91,
                    "level": 0,
                },
                {"cluster_id": "far", "lat": 48.01, "lng": 2.01, "match_score": 0.92},
                {"cluster_id": "rejected", "lat": 48.0, "lng": 2.0, "match_score": 0.1},
            ]
        },
        "match_score",
    )
    metrics = evaluate_prompt(
        class_name="boite aux lettres",
        prompt="boite aux lettres",
        class_annotations=annotations,
        predictions=predictions,
        acceptance_threshold=0.9,
    )

    geojson = build_prompt_geojson(
        class_name="boite aux lettres",
        prompt="boite aux lettres",
        annotations=annotations,
        predictions=predictions,
        metrics=metrics,
        acceptance_threshold=0.9,
    )

    by_id = {feature["id"]: feature for feature in geojson["features"]}

    assert by_id["prediction-near"]["properties"]["status"] == "TP"
    assert by_id["prediction-near"]["properties"]["marker-color"] == TP_COLOR
    assert by_id["prediction-near"]["properties"]["level"] == 0
    assert by_id["prediction-far"]["properties"]["status"] == "FP"
    assert by_id["prediction-far"]["properties"]["marker-color"] == FP_COLOR
    assert "prediction-rejected" not in by_id
    assert by_id["annotation-matched-ann"]["properties"]["status"] == "reference"
    assert (
        by_id["annotation-matched-ann"]["properties"]["marker-color"] == REFERENCE_COLOR
    )
    assert by_id["annotation-missed-ann"]["properties"]["status"] == "FN"
    assert by_id["annotation-missed-ann"]["properties"]["marker-color"] == FN_COLOR


def test_haversine_and_prf() -> None:
    assert haversine_m(48.0, 2.0, 48.0, 2.0) == pytest.approx(0.0)
    assert compute_prf(1, 1, 0) == pytest.approx((0.5, 1.0, 2 / 3))


def test_default_paths_use_map_path(tmp_path: Path) -> None:
    map_path = tmp_path / "test-map"
    args = parse_args(["--map-path", str(map_path)])

    assert args.map_path == map_path.resolve()
    assert args.map_id == "test-map"
    assert args.annotations == map_path.resolve() / "benchmark" / "annotations.geojson"
    assert (
        args.output_json
        == map_path.resolve() / "benchmark" / "results" / "results.json"
    )
    assert (
        args.output_csv == map_path.resolve() / "benchmark" / "results" / "results.csv"
    )
    assert (
        args.prompt_geojson_dir
        == map_path.resolve() / "benchmark" / "results" / "prompt_geojson"
    )
    assert args.group_annotation_radius_m == 0.0


def test_map_id_can_override_map_path_name(tmp_path: Path) -> None:
    map_path = tmp_path / "local-map-dir"
    args = parse_args(["--map_path", str(map_path), "--map-id", "service-map-id"])

    assert args.map_id == "service-map-id"


def test_localize_url_template_uses_selected_map_id(tmp_path: Path) -> None:
    map_path = tmp_path / "local-map-dir"
    args = parse_args(
        [
            "--map_path",
            str(map_path),
            "--map-id",
            "selected-map",
            "--localize-url",
            "https://example.test/{map_id}/object-search/localize",
        ]
    )

    assert (
        resolve_localize_url(args)
        == "https://example.test/selected-map/object-search/localize"
    )


def test_output_dir_sets_default_output_paths(tmp_path: Path) -> None:
    map_path = tmp_path / "test-map"
    output_dir = tmp_path / "runs" / "2026-06-10_14-32-05"
    args = parse_args(["--map-path", str(map_path), "--output-dir", str(output_dir)])

    resolved = output_dir.resolve()
    assert args.output_dir == resolved
    assert args.output_json == resolved / "metrics.json"
    assert args.output_csv == resolved / "results.csv"
    assert args.prompt_geojson_dir == resolved / "prompt_geojson"


def test_output_dir_respects_explicit_overrides(tmp_path: Path) -> None:
    map_path = tmp_path / "test-map"
    output_dir = tmp_path / "run-dir"
    explicit_json = tmp_path / "elsewhere" / "summary.json"
    args = parse_args(
        [
            "--map-path",
            str(map_path),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(explicit_json),
        ]
    )

    assert args.output_json == explicit_json
    assert args.output_csv == output_dir.resolve() / "results.csv"


def _write_annotations(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "ann-1",
                        "geometry": {"type": "Point", "coordinates": [2.0, 48.0]},
                        "properties": {"class": "défibrillateur", "accuracy": 5.0},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_run_benchmark_collects_raw_records_and_progress_events(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from toolbox.benchmark.object_search_http_benchmark import run_benchmark

    map_path = tmp_path / "test-map"
    annotations_path = map_path / "benchmark" / "annotations.geojson"
    _write_annotations(annotations_path)
    output_dir = map_path / "benchmark" / "run-1"

    # Unreachable base URL: the prompt errors out, but raw records and progress
    # events must still be produced.
    args = parse_args(
        [
            "--map-path",
            str(map_path),
            "--base-url",
            "http://127.0.0.1:1",
            "--timeout",
            "0.2",
            "--online",
            "--output-dir",
            str(output_dir),
            "--no-prompt-geojson",
            "--progress-json",
        ]
    )

    result = run_benchmark(args)

    raw = result["raw_by_prompt"]
    assert len(raw) == 1
    assert raw[0]["prompt"] == "défibrillateur"
    assert raw[0]["class_name"] == "défibrillateur"
    assert raw[0]["annotations"][0]["id"] == "ann-1"
    assert raw[0]["annotations"][0]["lat"] == 48.0
    assert raw[0]["predictions"] == []
    assert raw[0]["error"] is not None

    assert result["config"]["num_results"] == 100
    assert result["config"]["run_started_at"]

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [event["event"] for event in events] == ["start", "prompt", "done"]
    assert events[0]["prompt_count"] == 1
    assert events[0]["annotation_count"] == 1
    assert events[1]["index"] == 1
    assert events[1]["total"] == 1
    assert events[1]["error"] is not None
    assert events[2]["summary"]["prompt_count"] == 1


def test_main_writes_raw_results_in_output_dir(tmp_path: Path) -> None:
    from toolbox.benchmark.object_search_http_benchmark import main

    map_path = tmp_path / "test-map"
    _write_annotations(map_path / "benchmark" / "annotations.geojson")
    output_dir = map_path / "benchmark" / "run-1"

    exit_code = main(
        [
            "--map-path",
            str(map_path),
            "--base-url",
            "http://127.0.0.1:1",
            "--timeout",
            "0.2",
            "--online",
            "--output-dir",
            str(output_dir),
            "--no-prompt-geojson",
            "--progress-json",
        ]
    )

    assert exit_code == 1  # the single prompt errored (unreachable service)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "raw_by_prompt" not in metrics
    assert metrics["summary"]["error_count"] == 1
    raw = json.loads((output_dir / "raw_results.json").read_text(encoding="utf-8"))
    assert raw["config"]["map_path"] == str(map_path.resolve())
    assert len(raw["prompts"]) == 1
    assert raw["prompts"][0]["annotations"][0]["class_name"] == "défibrillateur"
    assert (output_dir / "results.csv").exists()
