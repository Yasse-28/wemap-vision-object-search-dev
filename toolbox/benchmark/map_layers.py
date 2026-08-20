"""GeoJSON layers for the livemap, built from one map's analysis.

Rendering is where a distribution stops being a number and becomes a place: "the
annotations nobody detected" is a table until it is a set of pins, and then it is
obviously a corridor, a floor, or a ceiling height.

Every layer is a WGS84 `FeatureCollection` whose properties are numeric and whose
`marker-color` is already resolved, so a viewer can render it without knowing anything
about this tool. Two geometries are used, and the choice is not cosmetic: a **point**
layer is one measurement at one place (an annotation, a keyframe), while a **polygon**
layer is a ground cell carrying an aggregate — squares tile, so a value read per cell
looks like the field it is, where dots at cell centres read as a scatter of pins.

A note on the word heat map. A viewer's own heat-map style weights *density*: it is
right for "how many detections are here" and wrong for "how deep are they here",
because it would turn a mean into a count. The cell layers below therefore carry the
value and its resolved colour, and are meant to be drawn as flat filled squares.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from toolbox.benchmark.map_analysis import (
    MAX_TRUSTED_RANGE_M,
    MapData,
    _horizontal_anisotropy,
    _keyframe_positions,
    _max_parallax_deg,
    attach_to_ground_truth,
    observability,
    seen_in_own_keyframe,
)

#: Green to red, for a share or a normalised score.
COLOUR_SCALE = ("#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027")
#: Cell size of the aggregated detection grid, metres. Small enough to show a room,
#: large enough that a livemap is not asked to draw a million points.
GRID_CELL_M = 2.0
#: Below this many samples a cell's aggregate is noise, and a heat map made of noise
#: is worse than a gap: it invites a reading the sample size does not support.
MIN_CELL_SAMPLES = 3
#: A range past this is not a measurement, it is the depth map saturating — sky, glass
#: or a mirror. Reported as its own layer because these rows are what drags a cluster
#: centroid across a room, and they sit on identifiable lines rather than everywhere.
BLOWUP_RANGE_M = 30.0
#: Cap on the blow-up points a layer will emit. They come in lines of hundreds; the
#: worst ones say where the problem is, and a viewer does not need the rest.
BLOWUP_LIMIT = 2000


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


def _cell_collection(
    data: MapData,
    keys: np.ndarray,
    altitudes: np.ndarray,
    properties: Sequence[Mapping[str, Any]],
    cell_m: float,
) -> dict[str, Any]:
    """Assemble a grid layer as square polygons, one per occupied cell.

    Squares rather than points at cell centres: a value aggregated over a cell IS the
    cell, and tiled squares read as a field where dots read as a scatter. The corners
    are converted through the same georeference as every other layer, so a cell lines
    up with the detections that produced it.
    """
    if len(properties) == 0:
        return {"type": "FeatureCollection", "features": []}
    corners = np.empty((len(properties), 4, 3), dtype=np.float64)
    for offset, (dx, dz) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        corners[:, offset, 0] = (keys[:, 0] + dx) * cell_m
        corners[:, offset, 1] = altitudes
        corners[:, offset, 2] = (keys[:, 1] + dz) * cell_m
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            corners.reshape(-1, 3)
        )
    ).reshape(len(properties), 4, 3)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [float(point[0]), float(point[1])]
                            for point in list(wgs84[index]) + [wgs84[index][0]]
                        ]
                    ],
                },
                "properties": {
                    **dict(properties[index]),
                    "altitude_m": float(altitudes[index]),
                },
            }
            for index in range(len(properties))
        ],
    }


def _ground_cells(
    positions: np.ndarray, cell_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell key per position, the unique keys, and the inverse index into them."""
    cells = np.floor(positions[:, [0, 2]] / cell_m).astype(np.int64)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    return cells, keys, inverse


def _cell_means(
    values: np.ndarray, inverse: np.ndarray, cell_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean of the finite values, and how many went into each."""
    finite = np.isfinite(values)
    totals = np.zeros(cell_count)
    counts = np.zeros(cell_count)
    np.add.at(totals, inverse[finite], values[finite])
    np.add.at(counts, inverse[finite], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    return means, counts


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


def layer_depth_range(data: MapData, cell_m: float = GRID_CELL_M) -> dict[str, Any]:
    """Per cell: how far away this map places the things it puts there.

    The depth magnitude field. Read it against the 15 m trusted range: a cell whose
    mean range is beyond it holds positions built by extrapolating a depth map, and
    every downstream distance — clustering radius included — is guesswork there.

    `beyond_trusted_share` is the property to filter on. A cell with a moderate mean
    and a high share is worse than its colour suggests: the mean is being held down by
    near rows while a minority of far ones drag the cluster centroids.
    """
    detections = data.detections
    placed = detections.placed
    positions = detections.position_eus[placed]
    if positions.size == 0:
        return {"type": "FeatureCollection", "features": []}
    ranges = detections.range_m[placed]
    _, keys, inverse = _ground_cells(positions, cell_m)
    means, counts = _cell_means(ranges, inverse, len(keys))
    altitudes, _ = _cell_means(positions[:, 1], inverse, len(keys))
    beyond, _ = _cell_means(
        (ranges > MAX_TRUSTED_RANGE_M).astype(float), inverse, len(keys)
    )
    keep = counts >= MIN_CELL_SAMPLES
    properties = [
        {
            "detections": int(counts[index]),
            "mean_range_m": float(means[index]),
            "beyond_trusted_share": float(beyond[index]),
            "marker-color": _colour(means[index], 0.0, MAX_TRUSTED_RANGE_M),
        }
        for index in np.flatnonzero(keep)
    ]
    return _cell_collection(
        data, keys[keep], altitudes[keep], properties, cell_m
    )


def layer_depth_blowups(data: MapData) -> dict[str, Any]:
    """Where the depth map saturated: one point per row placed absurdly far.

    A point layer, not a cell one, and deliberately: these do not form a field, they
    form rays. Seeing them line up along a window bay or a mirrored wall is the whole
    diagnostic — an aggregate would hide the very structure that identifies the cause.
    """
    detections = data.detections
    placed = detections.placed
    rows = np.flatnonzero(placed & (detections.range_m > BLOWUP_RANGE_M))
    if rows.size == 0:
        return {"type": "FeatureCollection", "features": []}
    # Worst first, then cap: the tail of a saturating line says nothing the head did
    # not, and a viewer asked to draw every one of them stops being usable.
    order = np.argsort(-detections.range_m[rows])[:BLOWUP_LIMIT]
    rows = rows[order]
    ranges = detections.range_m[rows]
    properties = [
        {
            "keyframe_id": int(detections.keyframe_id[row]),
            "label": str(detections.label[row]),
            "source": str(detections.source[row]),
            "range_m": float(ranges[index]),
            "depth_m": float(detections.depth[row]),
            "marker-color": _colour(
                ranges[index], BLOWUP_RANGE_M, BLOWUP_RANGE_M * 3.0
            ),
        }
        for index, row in enumerate(rows.tolist())
    ]
    wgs84 = np.asarray(
        data.pose_source.geo_transform.local_positions_to_wgs84(
            detections.position_eus[rows]
        )
    )
    return _feature_collection(wgs84, properties)


def layer_depth_scatter(
    data: MapData, cell_m: float = GRID_CELL_M, radius_m: float = 2.0
) -> dict[str, Any]:
    """Per cell: how far apart the observations of one object land.

    The fragmentation field, and the reason it is worth a map of its own. An object
    seen from several keyframes should collapse to one cluster; it does not when its
    detections' depths disagree, and the spread of those positions — not the object's
    size — is what decides whether clustering keeps it whole.

    Measured per annotation (the spread of the detections attached to it about their
    own centroid) and then averaged onto the cell, so it needs ground truth: a map
    without annotations gets an empty layer rather than a field of zeros.
    """
    ground_truth = data.ground_truth
    if len(ground_truth) == 0:
        return {"type": "FeatureCollection", "features": []}
    detections = data.detections
    attachment = attach_to_ground_truth(detections, ground_truth)
    attached = detections.placed & attachment.attached(radius_m)
    spreads = np.full(len(ground_truth), np.nan)
    observations = np.zeros(len(ground_truth))
    for index in range(len(ground_truth)):
        rows = np.flatnonzero(attached & (attachment.nearest == index))
        observations[index] = rows.size
        if rows.size < 2:
            continue
        cloud = detections.position_eus[rows]
        spreads[index] = float(
            np.linalg.norm(cloud - cloud.mean(axis=0), axis=1).mean()
        )
    _, keys, inverse = _ground_cells(ground_truth.position_eus, cell_m)
    means, counts = _cell_means(spreads, inverse, len(keys))
    altitudes, _ = _cell_means(ground_truth.position_eus[:, 1], inverse, len(keys))
    seen, _ = _cell_means(observations, inverse, len(keys))
    keep = counts > 0
    properties = [
        {
            "annotations": int(counts[index]),
            "mean_spread_m": float(means[index]),
            "mean_detections": float(seen[index]),
            # Against the clustering radius, because that is the decision the number
            # feeds: a spread near it cannot survive one cluster.
            "marker-color": _colour(means[index], 0.0, radius_m),
        }
        for index in np.flatnonzero(keep)
    ]
    return _cell_collection(
        data, keys[keep], altitudes[keep], properties, cell_m
    )


def layer_detection_coverage(
    data: MapData, cell_m: float = GRID_CELL_M
) -> dict[str, Any]:
    """Per cell: the share of annotations there that a box actually covered.

    The depth-free measurement, spatialised — it owes nothing to the depth map, so a
    red cell here is the detector failing in that part of the building, not geometry
    failing. Cells hold few annotations each, so `annotations` is the property to
    check before reading a colour.
    """
    ground_truth = data.ground_truth
    if len(ground_truth) == 0:
        return {"type": "FeatureCollection", "features": []}
    covered, _ = seen_in_own_keyframe(data.detections, ground_truth)
    indexed = np.isin(
        ground_truth.source_keyframe_id,
        np.unique(data.detections.keyframe_id).astype(str),
    )
    values = np.where(indexed, covered.astype(float), np.nan)
    _, keys, inverse = _ground_cells(ground_truth.position_eus, cell_m)
    means, counts = _cell_means(values, inverse, len(keys))
    altitudes, _ = _cell_means(ground_truth.position_eus[:, 1], inverse, len(keys))
    keep = counts > 0
    properties = [
        {
            "annotations": int(counts[index]),
            "covered_share": float(means[index]),
            # Inverted: red is where the detector misses, as everywhere else in the
            # tool red is the problem.
            "marker-color": _colour(1.0 - means[index]),
        }
        for index in np.flatnonzero(keep)
    ]
    return _cell_collection(
        data, keys[keep], altitudes[keep], properties, cell_m
    )


def layer_parallax(data: MapData, cell_m: float = GRID_CELL_M) -> dict[str, Any]:
    """Per cell: the geometry the capture made available there, whatever was detected.

    A capture ceiling, and the only field here that needs neither annotations nor
    detections to be right — it asks, of each cell the map has content in, what the
    keyframes within trusted range could ever have triangulated. Where this is red no
    association will place an object correctly, so it is the layer that says whether a
    failure is worth debugging or worth re-capturing.

    `anisotropy` is the companion reading: 0 means the capture went past in a straight
    line, 1 that it went around. A wide parallax with a low anisotropy is a corridor.
    """
    detections = data.detections
    placed = detections.placed
    positions = detections.position_eus[placed]
    if positions.size == 0:
        return {"type": "FeatureCollection", "features": []}
    _, keys, inverse = _ground_cells(positions, cell_m)
    altitudes, counts = _cell_means(positions[:, 1], inverse, len(keys))
    _, keyframe_positions = _keyframe_positions(data.pose_source)
    tree = cKDTree(keyframe_positions)
    keep = counts >= MIN_CELL_SAMPLES
    indices = np.flatnonzero(keep)
    properties = []
    for index in indices.tolist():
        centre = np.array(
            [
                (keys[index, 0] + 0.5) * cell_m,
                altitudes[index],
                (keys[index, 1] + 0.5) * cell_m,
            ]
        )
        near = tree.query_ball_point(centre, MAX_TRUSTED_RANGE_M)
        origins = keyframe_positions[near] if near else np.empty((0, 3))
        parallax = _max_parallax_deg(origins, centre) if len(near) else 0.0
        properties.append(
            {
                "detections": int(counts[index]),
                "keyframes_in_range": int(len(near)),
                "available_parallax_deg": float(parallax),
                "anisotropy": (
                    float(_horizontal_anisotropy(origins)) if len(near) > 2 else 0.0
                ),
                # 60 deg is generous: past it two views triangulate comfortably, and
                # the interesting distinction is all below that.
                "marker-color": _colour(60.0 - min(parallax, 60.0), 0.0, 60.0),
            }
        )
    return _cell_collection(
        data, keys[keep], altitudes[keep], properties, cell_m
    )


#: Every layer this tool knows how to build, by file name.
LAYERS = {
    "ground-truth": layer_ground_truth,
    "capture-distance": layer_capture_distance,
    "keyframes": layer_keyframes,
    "detection-grid": layer_detection_grid,
    "embedding-agreement": layer_embedding_agreement,
    "depth-range": layer_depth_range,
    "depth-blowups": layer_depth_blowups,
    "depth-scatter": layer_depth_scatter,
    "detection-coverage": layer_detection_coverage,
    "parallax": layer_parallax,
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
