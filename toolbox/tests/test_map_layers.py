"""The heat-map layers: what each cell claims, and what it refuses to claim.

These layers are read as pictures, and a picture is believed faster than a table. So
what is pinned here is mostly restraint: a cell with too few samples must not be drawn,
an annotation seen once must not report a spread of zero, and a map without ground
truth must produce an empty layer rather than a green field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolbox.benchmark.map_analysis import Detections, GroundTruth, MapData
from toolbox.benchmark.map_layers import (
    BLOWUP_RANGE_M,
    MIN_CELL_SAMPLES,
    _cell_collection,
    _cell_means,
    layer_depth_blowups,
    layer_depth_range,
    layer_depth_scatter,
    layer_detection_coverage,
    layer_parallax,
)


@dataclass
class _Pose:
    position_eus: np.ndarray


class _GeoTransform:
    """EUS metres to degrees at a fixed scale — the layers only need it to be affine.

    A real georeference would make every expected coordinate a magic number; this keeps
    the assertions about cell geometry rather than about projection arithmetic.
    """

    def local_positions_to_wgs84(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        return np.column_stack(
            [positions[:, 0] * 1e-5, positions[:, 2] * 1e-5, positions[:, 1]]
        )


class _PoseSource:
    def __init__(self, origins: list[tuple[float, float, float]]) -> None:
        self.poses = {
            index: _Pose(np.asarray(origin, dtype=float))
            for index, origin in enumerate(origins)
        }
        self.geo_transform = _GeoTransform()


def _detections(
    positions: list[tuple[float, float, float]],
    origins: list[tuple[float, float, float]] | None = None,
    keyframes: list[int] | None = None,
) -> Detections:
    count = len(positions)
    points = np.asarray(positions, dtype=float)
    starts = np.asarray(origins if origins else [(0.0, 0.0, 0.0)] * count, dtype=float)
    return Detections(
        row_index=np.arange(count),
        keyframe_id=np.asarray(keyframes or [0] * count, dtype=np.int64),
        theta=np.zeros(count),
        phi=np.zeros(count),
        angular_width=np.full(count, 0.2),
        angular_height=np.full(count, 0.2),
        source=np.array(["yolo"] * count),
        label=np.array(["chair"] * count),
        score=np.full(count, 0.5),
        depth=np.linalg.norm(points - starts, axis=1),
        position_eus=points,
        origin_eus=starts,
        direction_eus=np.tile(np.asarray([0.0, 0.0, 1.0]), (count, 1)),
        placed=np.full(count, True),
    )


def _ground_truth(
    positions: list[tuple[float, float, float]],
    keyframes: list[str] | None = None,
) -> GroundTruth:
    count = len(positions)
    return GroundTruth(
        class_name=np.array(["chair"] * count),
        position_eus=np.asarray(positions, dtype=float),
        accuracy_m=np.full(count, 5.0),
        source_keyframe_id=np.array(keyframes or ["0"] * count),
        erp_uv=np.asarray([(0.5, 0.5)] * count, dtype=float),
        depth_m=np.full(count, 3.0),
        level=np.array([""] * count),
    )


def _map(
    detections: Detections,
    ground_truth: GroundTruth | None = None,
    origins: list[tuple[float, float, float]] | None = None,
) -> MapData:
    return MapData(
        path=Path("/nowhere"),
        pose_source=_PoseSource(origins if origins else [(0.0, 0.0, 0.0)]),
        detections=detections,
        ground_truth=ground_truth if ground_truth else _ground_truth([]),
        reviews=[],
        group_labels=[],
        embeddings=None,
    )


def test_a_cell_is_a_closed_square_carrying_its_height() -> None:
    data = _map(_detections([(0.0, 1.5, 0.0)]))
    keys = np.array([[0, 0]])

    collection = _cell_collection(data, keys, np.array([1.5]), [{"value": 1}], 2.0)

    ring = collection["features"][0]["geometry"]["coordinates"][0]
    assert collection["features"][0]["geometry"]["type"] == "Polygon"
    assert len(ring) == 5
    assert ring[0] == ring[-1]
    assert collection["features"][0]["properties"]["altitude_m"] == 1.5


def test_an_empty_cell_list_is_an_empty_collection_not_a_malformed_one() -> None:
    data = _map(_detections([(0.0, 0.0, 0.0)]))

    collection = _cell_collection(data, np.empty((0, 2)), np.empty(0), [], 2.0)

    assert collection == {"type": "FeatureCollection", "features": []}


def test_cell_means_ignore_the_values_that_are_not_finite() -> None:
    values = np.array([1.0, np.nan, 3.0])
    inverse = np.array([0, 0, 0])

    means, counts = _cell_means(values, inverse, 1)

    assert means[0] == 2.0
    assert counts[0] == 2.0


def test_a_cell_with_no_finite_value_reports_nan_rather_than_zero() -> None:
    means, counts = _cell_means(np.array([np.nan]), np.array([0]), 1)

    assert math.isnan(means[0])
    assert counts[0] == 0.0


def test_a_cell_below_the_sample_floor_is_not_drawn() -> None:
    # Two detections in one cell, MIN_CELL_SAMPLES is above that.
    assert MIN_CELL_SAMPLES > 2
    data = _map(_detections([(0.5, 0.0, 0.5), (0.6, 0.0, 0.6)]))

    assert layer_depth_range(data)["features"] == []


def test_a_cell_says_what_share_of_it_is_past_the_trusted_range() -> None:
    # Four rows in one cell, three of them placed 20 m from their keyframe.
    positions = [(0.2, 0.0, 0.2), (0.4, 0.0, 0.4), (0.6, 0.0, 0.6), (0.8, 0.0, 0.8)]
    origins = [(0.0, 0.0, 0.0), (0.0, 0.0, -20.0), (0.0, 0.0, -20.0), (0.0, 0.0, -20.0)]
    data = _map(_detections(positions, origins=origins))

    properties = layer_depth_range(data)["features"][0]["properties"]

    assert properties["detections"] == 4
    assert properties["beyond_trusted_share"] == 0.75


def test_nothing_saturating_means_no_blow_up_layer_at_all() -> None:
    data = _map(_detections([(0.0, 0.0, 5.0)]))

    assert layer_depth_blowups(data)["features"] == []


def test_blow_ups_are_reported_worst_first() -> None:
    positions = [(0.0, 0.0, 40.0), (0.0, 0.0, 90.0), (0.0, 0.0, 60.0)]
    data = _map(_detections(positions))

    ranges = [
        feature["properties"]["range_m"]
        for feature in layer_depth_blowups(data)["features"]
    ]

    assert ranges == sorted(ranges, reverse=True)
    assert min(ranges) > BLOWUP_RANGE_M


def test_a_map_without_annotations_has_no_scatter_field() -> None:
    data = _map(_detections([(0.0, 0.0, 1.0), (0.1, 0.0, 1.1), (0.2, 0.0, 1.2)]))

    assert layer_depth_scatter(data)["features"] == []
    assert layer_detection_coverage(data)["features"] == []


def test_an_annotation_seen_once_reports_no_spread_not_a_zero() -> None:
    # One attached detection cannot disagree with anything: including it as 0 m would
    # paint the safest possible colour on the least evidence.
    data = _map(
        _detections([(0.0, 0.0, 0.0)]),
        ground_truth=_ground_truth([(0.0, 0.0, 0.0)]),
    )

    assert layer_depth_scatter(data)["features"] == []


def test_the_spread_is_the_mean_distance_to_the_cloud_centroid() -> None:
    # Two detections a metre either side of the annotation: each sits 1 m from their
    # common centroid, so the spread is 1 m — the quantity a 1.25 m clustering radius
    # is about to be asked to swallow.
    data = _map(
        _detections([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
        ground_truth=_ground_truth([(0.0, 0.0, 0.0)]),
    )

    properties = layer_depth_scatter(data)["features"][0]["properties"]

    assert properties["annotations"] == 1
    assert abs(properties["mean_spread_m"] - 1.0) < 1e-9


def test_coverage_leaves_out_the_annotations_whose_panorama_is_not_indexed() -> None:
    # Its panorama produced nothing, so it is not measurable depth-free. Counting it as
    # uncovered would blame the detector for an index-coverage failure.
    data = _map(
        _detections([(0.0, 0.0, 0.0)], keyframes=[7]),
        ground_truth=_ground_truth([(0.0, 0.0, 0.0)], keyframes=["999"]),
    )

    assert layer_detection_coverage(data)["features"] == []


def test_a_cell_no_keyframe_ever_came_near_has_no_available_parallax() -> None:
    # The detections sit 40 m from the only keyframe, past the trusted range: nothing
    # ever looked at that cell closely, which is a capture ceiling, not a bug.
    positions = [(0.2, 0.0, 40.0), (0.4, 0.0, 40.2), (0.6, 0.0, 40.4)]
    data = _map(_detections(positions), origins=[(0.0, 0.0, 0.0)])

    properties = layer_parallax(data)["features"][0]["properties"]

    assert properties["keyframes_in_range"] == 0
    assert properties["available_parallax_deg"] == 0.0
    assert properties["marker-color"] == "#d73027"


def test_two_keyframes_on_opposite_sides_offer_a_wide_baseline() -> None:
    positions = [(0.2, 0.0, 0.2), (0.4, 0.0, 0.4), (0.6, 0.0, 0.6)]
    data = _map(
        _detections(positions),
        origins=[(-5.0, 0.0, 1.0), (5.0, 0.0, 1.0)],
    )

    properties = layer_parallax(data)["features"][0]["properties"]

    assert properties["keyframes_in_range"] == 2
    assert properties["available_parallax_deg"] > 100.0
