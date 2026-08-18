"""GeoJSON layers for the livemap, built from one map's analysis.

Rendering is where a distribution stops being a number and becomes a place: "the
annotations nobody detected" is a table until it is a set of pins, and then it is
obviously a corridor, a floor, or a ceiling height.

Every layer is a WGS84 `FeatureCollection` of points whose properties are numeric
and whose `marker-color` is already resolved, so a viewer can render it without
knowing anything about this tool. Adding a layer means adding one function that
returns `(positions_eus, properties)`; `write_layers` does the rest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from toolbox.benchmark.map_analysis import (
    MapData,
    attach_to_ground_truth,
    observability,
    seen_in_own_keyframe,
)

#: Green to red, for a share or a normalised score.
COLOUR_SCALE = ("#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027")
#: Cell size of the aggregated detection grid, metres. Small enough to show a room,
#: large enough that a livemap is not asked to draw a million points.
GRID_CELL_M = 2.0


def _colour(value: float, low: float = 0.0, high: float = 1.0) -> str:
    """Pick a scale colour for a value, clamped to `[low, high]`."""
    if not np.isfinite(value):
        return "#808080"
    share = (value - low) / (high - low) if high > low else 0.0
    index = int(np.clip(share, 0.0, 1.0) * (len(COLOUR_SCALE) - 1))
    return COLOUR_SCALE[index]


def _mean_or_nan(values: np.ndarray) -> float:
    """Mean of the finite entries, NaN when there are none — `nanmean` warns there."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _feature_collection(
    wgs84: np.ndarray, properties: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Assemble one GeoJSON point layer."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(wgs84[index, 0]),
                        float(wgs84[index, 1]),
                        float(wgs84[index, 2]),
                    ],
                },
                "properties": dict(properties[index]),
            }
            for index in range(len(properties))
        ],
    }


def layer_ground_truth(data: MapData, radius_m: float = 2.0) -> dict[str, Any]:
    """One point per annotation: was it detected, was it reachable, how well seen.

    The layer to open first. A red pin is an annotation the pipeline never reached;
    its `failure` property says whether the detector missed it in its own panorama
    or whether the box existed and the position landed elsewhere.
    """
    ground_truth = data.ground_truth
    attachment = attach_to_ground_truth(data.detections, ground_truth)
    profile = observability(data, attachment, radius_m)
    covered, nearest_deg = seen_in_own_keyframe(data.detections, ground_truth)
    properties = []
    for index in range(len(ground_truth)):
        reached = profile.detections[index] > 0
        failure = (
            ""
            if reached
            else ("placement" if covered[index] else "détection")
        )
        properties.append(
            {
                "class": str(ground_truth.class_name[index]),
                "covered_2d": bool(covered[index]),
                "nearest_box_deg": float(nearest_deg[index]),
                "detections": int(profile.detections[index]),
                "keyframes": int(profile.keyframes[index]),
                "achieved_parallax_deg": float(profile.achieved_parallax_deg[index]),
                "available_parallax_deg": float(profile.available_parallax_deg[index]),
                "nearest_keyframe_m": float(profile.nearest_keyframe_m[index]),
                "failure": failure,
                "marker-color": "#1a9850" if reached else (
                    "#fc8d59" if covered[index] else "#d73027"
                ),
            }
        )
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            ground_truth.position_eus
        )
    )
    return _feature_collection(wgs84, properties)


def layer_capture_distance(data: MapData) -> dict[str, Any]:
    """One point per annotation, coloured by how close the capture ever came."""
    attachment = attach_to_ground_truth(data.detections, data.ground_truth)
    profile = observability(data, attachment, 2.0)
    properties = [
        {
            "class": str(data.ground_truth.class_name[index]),
            "nearest_keyframe_m": float(profile.nearest_keyframe_m[index]),
            "keyframes_in_range": int(profile.available_keyframes[index]),
            "marker-color": _colour(profile.nearest_keyframe_m[index], 0.0, 8.0),
        }
        for index in range(len(data.ground_truth))
    ]
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            data.ground_truth.position_eus
        )
    )
    return _feature_collection(wgs84, properties)


def layer_keyframes(data: MapData) -> dict[str, Any]:
    """One point per keyframe: how much it produced, and how varied it was."""
    detections = data.detections
    identifiers = np.unique(detections.keyframe_id)
    rows = []
    positions = []
    for keyframe in identifiers.tolist():
        pose = data.pose_source.poses.get(int(keyframe))
        if pose is None:
            continue
        selected = detections.keyframe_id == keyframe
        scores = detections.score[selected]
        positions.append(pose.position_eus)
        rows.append(
            {
                "keyframe_id": int(keyframe),
                "detections": int(selected.sum()),
                "labels": int(np.unique(detections.label[selected]).size),
                "mean_score": _mean_or_nan(scores),
                "marker-color": _colour(float(selected.sum()), 0.0, 120.0),
            }
        )
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            np.asarray(positions, dtype=np.float64)
        )
    )
    return _feature_collection(wgs84, rows)


def layer_detection_grid(data: MapData, cell_m: float = GRID_CELL_M) -> dict[str, Any]:
    """Detections aggregated on a ground grid — a heat map a livemap can draw.

    A million point features would kill a viewer; a cell count is the same
    information at the scale anyone reads it.
    """
    detections = data.detections
    placed = detections.placed
    positions = detections.position_eus[placed]
    if positions.size == 0:
        return {"type": "FeatureCollection", "features": []}
    cells = np.floor(positions[:, [0, 2]] / cell_m).astype(np.int64)
    keys, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    ranges = detections.range_m[placed]
    heights = positions[:, 1]
    means = np.zeros(len(keys))
    altitudes = np.zeros(len(keys))
    np.add.at(means, inverse, ranges)
    np.add.at(altitudes, inverse, heights)
    means /= counts
    altitudes /= counts
    centres = np.column_stack(
        [
            (keys[:, 0] + 0.5) * cell_m,
            altitudes,
            (keys[:, 1] + 0.5) * cell_m,
        ]
    )
    top = float(np.percentile(counts, 95))
    properties = [
        {
            "detections": int(counts[index]),
            "mean_range_m": float(means[index]),
            "marker-color": _colour(float(counts[index]), 0.0, max(top, 1.0)),
        }
        for index in range(len(keys))
    ]
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(centres)
    )
    return _feature_collection(wgs84, properties)


def layer_embedding_agreement(data: MapData, radius_m: float = 2.0) -> dict[str, Any]:
    """Per annotation, how alike the cutouts attached to it look.

    The semantic counterpart of the geometry: a low value means the detections the
    pipeline gathered on one object do not look like each other, which is what an
    embedding-based merge would have to overcome.
    """
    if data.embeddings is None:
        return {"type": "FeatureCollection", "features": []}
    detections = data.detections
    attachment = attach_to_ground_truth(detections, data.ground_truth)
    attached = detections.placed & attachment.attached(radius_m)
    properties = []
    positions = []
    for index in range(len(data.ground_truth)):
        rows = np.flatnonzero(attached & (attachment.nearest == index))
        if rows.size < 2:
            continue
        take = rows[: min(rows.size, 300)]
        vectors = np.asarray(data.embeddings[take], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
        prototype = vectors.mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-6)
        cosines = vectors @ prototype
        positions.append(data.ground_truth.position_eus[index])
        properties.append(
            {
                "class": str(data.ground_truth.class_name[index]),
                "detections": int(rows.size),
                "mean_cosine": float(cosines.mean()),
                "min_cosine": float(cosines.min()),
                "marker-color": _colour(1.0 - float(cosines.mean()), 0.0, 0.6),
            }
        )
    if not positions:
        return {"type": "FeatureCollection", "features": []}
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            np.asarray(positions, dtype=np.float64)
        )
    )
    return _feature_collection(wgs84, properties)


#: Every layer this tool knows how to build, by file name.
LAYERS = {
    "ground-truth": layer_ground_truth,
    "capture-distance": layer_capture_distance,
    "keyframes": layer_keyframes,
    "detection-grid": layer_detection_grid,
    "embedding-agreement": layer_embedding_agreement,
}


def write_layers(
    data: MapData, out_dir: Path, names: Sequence[str] | None = None
) -> list[Path]:
    """Build the requested layers and write them as GeoJSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names or list(LAYERS):
        collection = LAYERS[name](data)
        path = out_dir / f"{name}.geojson"
        path.write_text(json.dumps(collection), encoding="utf-8")
        written.append(path)
    return written
