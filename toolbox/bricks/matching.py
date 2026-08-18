"""Inspect a hand-picked set of detections: do they describe one object?

Answers the two questions the toolbox's matching tab asks, over the *same* set:
semantic (the MetaCLIP cosine matrix) and geometric (robust N-view triangulation of
the detection rays). Neither ranks anything — this is an inspection path, not a
retrieval one, and the numbers it returns are meant to be read next to a panorama.

The set arrives as `(keyframe_id, theta_center, phi_center)` triples, because that is
the only key the toolbox can produce for a detection: pgvector carries no parquet
`row_index`, and a cluster observation has no other identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from toolbox.bricks.candidates import _parse_embedding
from toolbox.bricks.localize import angular_gap_ratio
from toolbox.bricks.merge_score import (
    DEFAULT_OBJECT_EXTENT_M,
    cluster_extent_m,
    covisibility_conflict,
    observed_extent_m,
    score_1v2,
)
from toolbox.bricks.signed_clustering import average_linkage_clusters
from toolbox.bricks.triangulate import (
    DEFAULT_INLIER_THRESHOLD_DEG,
    Hypothesis,
    TriangulationResult,
    triangulate_linkage,
    triangulate_multi,
    triangulate_rays,
)
from toolbox.bricks.vendored.erp import theta_phi_to_opengl_ray
from toolbox.bricks.vendored.geo_transform import GeoTransform, Pose
from toolbox.bricks.vendored.maths import quaternion, vector3

#: Partition methods the endpoint accepts. "sequential" is RANSAC applied repeatedly;
#: the linkage pair is order-free and is the reference to compare it against.
PARTITION_METHODS = ("sequential", "jlinkage", "tlinkage", "gasp", "gasp1v2")

#: Multicut's own defaults, so a GASP run is comparable to the association in place.
GASP_PAIR_RADIUS_M = 6.0
GASP_GEO_PIVOT_M = 1.0
#: A cannot-link needs near-perfect precision to be worth its hardness, so two boxes
#: must be clearly apart — this many times their combined half-extents — before their
#: keyframe is taken as evidence of two objects rather than two proposals of one.
SAME_KEYFRAME_MARGIN = 1.5
#: How hard a co-visibility conflict pushes against a merge, in `score_1v2` units.
COVISIBILITY_SCALE = 10.0
# The requested angles and the stored ones are the same float16 values, so this is a
# rounding margin, not a search radius (see the toolbox's angular join).
ANGLE_MATCH_TOL_RAD = 1e-3

_MATCH_SQL = """
SELECT
    c.id,
    c.theta_center,
    c.phi_center,
    c.thumbnail,
    c.embedding,
    c.angular_width,
    c.angular_height,
    ST_X(c.object_position) AS px,
    ST_Y(c.object_position) AS py,
    ST_Z(c.object_position) AS pz,
    k.video_keyframe_id,
    ST_X(k.position) AS vkf_px,
    ST_Y(k.position) AS vkf_py,
    ST_Z(k.position) AS vkf_pz,
    k.orientation
FROM object_search_candidate AS c
JOIN geokeyframe AS k
  ON k.geo_ref_id = c.geo_ref_id
 AND k.id = c.geokeyframe_id
WHERE c.geo_ref_id = %s
  AND k.video_keyframe_id = ANY(%s)
"""


@dataclass(frozen=True)
class MatchingItem:
    """One requested detection, resolved against pgvector or not."""

    keyframe_id: str
    theta_center: float
    phi_center: float
    candidate_id: int | None
    thumbnail: str | None
    ray_origin_eus: tuple[float, float, float] | None
    ray_direction_eus: tuple[float, float, float] | None
    stored_position_eus: tuple[float, float, float] | None
    embedding: np.ndarray | None
    #: Angular extent of the box, radians — what makes "two boxes in one panorama are
    #: two objects" separable from "two proposals of the same object".
    angular_width: float | None = None
    angular_height: float | None = None


def _ray_direction_eus(
    pose: Pose, theta: float, phi: float
) -> tuple[float, float, float]:
    """World direction the detection is seen along, from the keyframe's own frame."""
    local = theta_phi_to_opengl_ray(float(theta), float(phi))
    direction = np.asarray(
        quaternion.rotate(pose.orientation_wxyz, local), dtype=np.float64
    )
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return (0.0, 0.0, -1.0)
    direction = direction / norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


def resolve_items(
    conn: Any,
    geo_ref_id: int,
    requested: Sequence[tuple[str, float, float]],
) -> list[MatchingItem]:
    """Look each requested direction up in pgvector, keeping unresolved ones.

    An item that does not resolve is returned with null fields rather than dropped:
    a basket that silently loses members would make the cosine matrix and the
    triangulation describe a different set than the one on screen.
    """
    keyframe_ids: list[int] = []
    for keyframe_id, _, _ in requested:
        try:
            keyframe_ids.append(int(keyframe_id))
        except (TypeError, ValueError):
            continue

    rows: list[dict] = []
    if keyframe_ids:
        # psycopg2, and the repo reads rows as dicts by zipping the description —
        # same shape `candidates.py` builds, so the row access below matches it.
        with conn.cursor() as cursor:
            cursor.execute(_MATCH_SQL, (geo_ref_id, sorted(set(keyframe_ids))))
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, values)) for values in cursor.fetchall()]

    by_keyframe: dict[str, list[dict]] = {}
    for row in rows:
        by_keyframe.setdefault(str(row["video_keyframe_id"]), []).append(row)

    items: list[MatchingItem] = []
    for keyframe_id, theta, phi in requested:
        best: dict | None = None
        best_delta = math.inf
        for row in by_keyframe.get(str(keyframe_id), ()):
            delta = max(
                abs(float(row["theta_center"]) - theta),
                abs(float(row["phi_center"]) - phi),
            )
            if delta < best_delta:
                best, best_delta = row, delta
        if best is None or best_delta > ANGLE_MATCH_TOL_RAD:
            items.append(
                MatchingItem(
                    str(keyframe_id), theta, phi, None, None, None, None, None, None
                )
            )
            continue
        pose = Pose.from_position_orientation(
            vector3.from_xyz(
                float(best["vkf_px"]), float(best["vkf_py"]), float(best["vkf_pz"])
            ),
            quaternion.cast(np.asarray(best["orientation"], dtype=np.float64)),
        )
        stored = (
            (float(best["px"]), float(best["py"]), float(best["pz"]))
            if best["px"] is not None
            else None
        )
        items.append(
            MatchingItem(
                keyframe_id=str(keyframe_id),
                theta_center=theta,
                phi_center=phi,
                candidate_id=int(best["id"]),
                thumbnail=best["thumbnail"] or None,
                ray_origin_eus=(
                    float(best["vkf_px"]),
                    float(best["vkf_py"]),
                    float(best["vkf_pz"]),
                ),
                ray_direction_eus=_ray_direction_eus(pose, theta, phi),
                stored_position_eus=stored,
                embedding=_parse_embedding(best.get("embedding")),
                angular_width=float(best["angular_width"]),
                angular_height=float(best["angular_height"]),
            )
        )
    return items


def _angular_gap(left: MatchingItem, right: MatchingItem) -> float | None:
    """How far apart two boxes of one keyframe are, in units of their own extent.

    Above 1 the boxes do not touch. Returns None when an extent is unknown, because a
    hard constraint must never be inferred from a missing value.
    """
    if left.angular_width is None or right.angular_width is None:
        return None
    ratio = float(
        angular_gap_ratio(
            np.array([left.theta_center, right.theta_center], dtype=np.float64),
            np.array([left.phi_center, right.phi_center], dtype=np.float64),
            np.array([left.angular_width, right.angular_width], dtype=np.float64),
            np.array(
                [left.angular_height or 0.0, right.angular_height or 0.0],
                dtype=np.float64,
            ),
            np.array([0]),
            np.array([1]),
        )[0]
    )
    return None if not math.isfinite(ratio) else ratio


def cannot_link_pairs(
    items: Sequence[MatchingItem],
    indices: Sequence[int],
    *,
    margin: float = SAME_KEYFRAME_MARGIN,
) -> list[tuple[int, int]]:
    """Pairs that a single panorama says are two different objects.

    Two boxes in the *same* keyframe, angularly disjoint by a comfortable margin,
    cannot be one object seen twice — the view is 360 deg, an object occupies one
    direction in it. Overlapping boxes are excluded on purpose: those are the
    duplicate proposals (YOLO and GDINO on one object) that the association is
    explicitly designed to merge.
    """
    pairs: list[tuple[int, int]] = []
    for position, left_index in enumerate(indices):
        for right_index in indices[position + 1 :]:
            left, right = items[left_index], items[right_index]
            if left.keyframe_id != right.keyframe_id:
                continue
            gap = _angular_gap(left, right)
            if gap is not None and gap >= margin:
                pairs.append((left_index, right_index))
    return pairs


def similarity_matrix(items: Sequence[MatchingItem]) -> list[list[float | None]]:
    """Cosine between every pair, `null` where an embedding is missing.

    The embeddings are unit vectors, so this is a dot product — and the same scale
    the retrieval similarity uses, which is what makes the numbers comparable to the
    ones the cluster list shows.
    """
    size = len(items)
    matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    for i, left in enumerate(items):
        if left.embedding is None:
            continue
        for j, right in enumerate(items):
            if right.embedding is None:
                continue
            matrix[i][j] = float(np.dot(left.embedding, right.embedding))
    return matrix


def _detection_range_m(item: MatchingItem) -> float | None:
    """Distance from the keyframe to the detection's depth point, when it has one."""
    if item.ray_origin_eus is None or item.stored_position_eus is None:
        return None
    return float(
        np.linalg.norm(
            np.asarray(item.stored_position_eus) - np.asarray(item.ray_origin_eus)
        )
    )


def _detection_extents_m(
    items: Sequence[MatchingItem], indices: Sequence[int]
) -> list[float]:
    """Observed half-size of each detection, falling back to the constant.

    A detection with no depth point has no range, so nothing is observed and the
    assumed extent is the honest answer for it.
    """
    ranges = [_detection_range_m(items[index]) for index in indices]
    extents = observed_extent_m(
        np.asarray([value or 0.0 for value in ranges], dtype=np.float64),
        np.asarray(
            [items[index].angular_width or 0.0 for index in indices], dtype=np.float64
        ),
        np.asarray(
            [items[index].angular_height or 0.0 for index in indices], dtype=np.float64
        ),
    )
    return [
        DEFAULT_OBJECT_EXTENT_M if value is None else float(extent)
        for value, extent in zip(ranges, extents, strict=True)
    ]


def _agglomerate_by_score(
    points: list[np.ndarray | None],
    origins: list[np.ndarray],
    directions: list[np.ndarray],
    half_extents: list[float],
    blocked: set[tuple[int, int]],
    *,
    covis_weight: float,
    detection_extents_m: list[float] | None = None,
) -> list[int]:
    """Merge the cluster pair with the best "one object" score, until none is positive.

    The criterion is evaluated on the **whole** pair of clusters at each step, which is
    the point: a merge has to improve the fit of everything it touches, so two members
    happening to be close is not enough. Cannot-link is inherited through merges, as
    in the average-linkage path.

    With `detection_extents_m`, each cluster's tolerance is the median of its members'
    *observed* sizes instead of `DEFAULT_OBJECT_EXTENT_M`. It is recomputed as members
    join, but never on the union being evaluated — see `score_1v2`.
    """
    clusters: dict[int, list[int]] = {index: [index] for index in range(len(points))}
    forbidden: dict[int, set[int]] = {index: set() for index in clusters}
    for i, j in blocked:
        forbidden[i].add(j)
        forbidden[j].add(i)

    def cluster_points(members: list[int]) -> np.ndarray | None:
        rows = [points[m] for m in members if points[m] is not None]
        return np.asarray(rows) if rows else None

    def cluster_origins(members: list[int]) -> np.ndarray:
        return np.asarray([origins[m] for m in members if points[m] is not None])

    def cluster_extent(members: list[int]) -> float | None:
        if detection_extents_m is None:
            return None
        return cluster_extent_m(
            np.asarray(
                [detection_extents_m[m] for m in members if points[m] is not None]
            )
        )

    while True:
        best: tuple[float, int, int] | None = None
        keys = list(clusters)
        for position, left in enumerate(keys):
            left_points = cluster_points(clusters[left])
            if left_points is None:
                continue
            for right in keys[position + 1 :]:
                if right in forbidden[left]:
                    continue
                right_points = cluster_points(clusters[right])
                if right_points is None:
                    continue
                score = score_1v2(
                    left_points,
                    right_points,
                    cluster_origins(clusters[left]),
                    cluster_origins(clusters[right]),
                    left_extent_m=cluster_extent(clusters[left]),
                    right_extent_m=cluster_extent(clusters[right]),
                )
                if covis_weight > 0.0:
                    conflict = max(
                        covisibility_conflict(
                            np.asarray([origins[m] for m in clusters[left]]),
                            np.asarray([directions[m] for m in clusters[left]]),
                            np.asarray([half_extents[m] for m in clusters[left]]),
                            right_points.mean(axis=0),
                        ),
                        covisibility_conflict(
                            np.asarray([origins[m] for m in clusters[right]]),
                            np.asarray([directions[m] for m in clusters[right]]),
                            np.asarray([half_extents[m] for m in clusters[right]]),
                            left_points.mean(axis=0),
                        ),
                    )
                    score -= covis_weight * conflict * COVISIBILITY_SCALE
                if score > 0 and (best is None or score > best[0]):
                    best = (score, left, right)
        if best is None:
            break
        _, left, right = best
        clusters[left].extend(clusters.pop(right))
        forbidden[left] |= forbidden.pop(right)
        for others in forbidden.values():
            if right in others:
                others.discard(right)
                others.add(left)

    labels = [0] * len(points)
    for label, members in enumerate(clusters.values()):
        for member in members:
            labels[member] = label
    return labels


def _gasp_partition(
    items: Sequence[MatchingItem],
    ray_index: Sequence[int],
    *,
    inlier_threshold_deg: float,
    depth_weight: float,
    use_cannot_link: bool,
    merge_criterion: str = "average",
    covis_weight: float = 0.0,
    observed_extent: bool = False,
) -> tuple[list[Hypothesis], list[int], list[tuple[int, int]]]:
    """Partition by agglomerating a signed graph, then triangulate each cluster.

    The edge cost mirrors the association in production — `1 - distance / pivot` on the
    depth-projected points, the only cue measured as separable (AUC 0.879) — so the
    comparison isolates the *accumulation rule* (mean here, sum in GAEC) rather than
    changing the evidence at the same time.
    """
    local_positions: list[np.ndarray | None] = [
        None if items[index].stored_position_eus is None
        else np.asarray(items[index].stored_position_eus)
        for index in ray_index
    ]
    edges: list[tuple[int, int, float]] = []
    for i in range(len(ray_index)):
        for j in range(i + 1, len(ray_index)):
            left, right = local_positions[i], local_positions[j]
            if left is None or right is None:
                continue
            distance = float(np.linalg.norm(left - right))
            if distance > GASP_PAIR_RADIUS_M:
                continue
            edges.append((i, j, 1.0 - distance / GASP_GEO_PIVOT_M))

    blocked = cannot_link_pairs(items, list(ray_index)) if use_cannot_link else []
    position_of = {index: position for position, index in enumerate(ray_index)}
    blocked_positions = {(position_of[i], position_of[j]) for i, j in blocked}
    if merge_criterion == "score1v2":
        labels = _agglomerate_by_score(
            local_positions,
            [np.asarray(items[index].ray_origin_eus) for index in ray_index],
            [np.asarray(items[index].ray_direction_eus) for index in ray_index],
            [
                0.5 * float(np.hypot(
                    items[index].angular_width or 0.1,
                    items[index].angular_height or 0.1,
                ))
                for index in ray_index
            ],
            blocked_positions,
            covis_weight=covis_weight,
            detection_extents_m=(
                _detection_extents_m(items, ray_index) if observed_extent else None
            ),
        )
    else:
        labels = average_linkage_clusters(len(ray_index), edges, blocked_positions)

    by_label: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        by_label.setdefault(label, []).append(position)

    hypotheses: list[Hypothesis] = []
    unassigned: list[int] = []
    for members in by_label.values():
        if len(members) < 2:
            unassigned.extend(members)
            continue
        origins = np.array([items[ray_index[m]].ray_origin_eus for m in members])
        directions = np.array([items[ray_index[m]].ray_direction_eus for m in members])
        depth_points = np.array(
            [
                local_positions[m] if local_positions[m] is not None
                else (np.nan, np.nan, np.nan)
                for m in members
            ]
        )
        result = triangulate_rays(
            origins,
            directions,
            inlier_threshold_deg=inlier_threshold_deg,
            depth_points=depth_points,
            depth_weight=depth_weight,
        )
        if result.point_eus is None:
            unassigned.extend(members)
            continue
        hypotheses.append(
            Hypothesis(
                point_eus=result.point_eus,
                member_indices=tuple(members),
                mean_residual_deg=result.mean_inlier_residual_deg,
                mean_residual_m=result.mean_inlier_residual_m,
                max_parallax_deg=result.max_parallax_deg,
            )
        )
    hypotheses.sort(key=lambda item: len(item.member_indices), reverse=True)
    return hypotheses, sorted(unassigned), blocked


def triangulate_items(
    items: Sequence[MatchingItem],
    geo_transform: GeoTransform,
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
    partition_method: str = "sequential",
    max_depth_m: float | None = None,
    depth_weight: float = 0.0,
    use_cannot_link: bool = False,
    covis_weight: float = 0.0,
    observed_extent: bool = False,
) -> dict[str, Any]:
    """Triangulate the resolved rays and report the result in WGS84 and EUS.

    `ray_index` maps a row of the triangulation back to its position in the basket:
    unresolved items contribute no ray, so the two lists are not the same length.

    `max_depth_m` drops detections whose depth-map point sits beyond it — far depth is
    where the estimate stops being trustworthy, and production already caps it for the
    same reason. A detection with no depth at all is kept: nothing says it is far.
    """
    beyond_depth = [
        index
        for index, item in enumerate(items)
        if max_depth_m is not None
        and (_detection_range_m(item) or 0.0) > max_depth_m
    ]
    excluded = set(beyond_depth)
    ray_index = [
        index
        for index, item in enumerate(items)
        if item.ray_origin_eus is not None
        and item.ray_direction_eus is not None
        and index not in excluded
    ]
    if len(ray_index) < 2:
        return {
            "available": False,
            "reason": "Fewer than two usable detections.",
            "ray_index": ray_index,
            "beyond_max_depth_items": beyond_depth,
        }
    origins = np.array([items[index].ray_origin_eus for index in ray_index])
    directions = np.array([items[index].ray_direction_eus for index in ray_index])
    depth_points = np.array(
        [
            items[index].stored_position_eus
            if items[index].stored_position_eus is not None
            else (np.nan, np.nan, np.nan)
            for index in ray_index
        ]
    )
    result: TriangulationResult = triangulate_rays(
        origins,
        directions,
        inlier_threshold_deg=inlier_threshold_deg,
        depth_points=depth_points,
        depth_weight=depth_weight,
    )
    if result.point_eus is None:
        return {
            "available": False,
            "reason": "No consensus: the rays do not meet within the threshold.",
            "ray_index": ray_index,
            "beyond_max_depth_items": beyond_depth,
            "residuals_deg": list(result.residuals_deg),
        }
    wgs84 = geo_transform.local_positions_to_wgs84(np.array([result.point_eus]))
    lng, lat, alt = float(wgs84[0, 0]), float(wgs84[0, 1]), float(wgs84[0, 2])
    # The floor the point lands on, resolved the way `localize` resolves a cluster's:
    # on a multi-level venue "which floor" is the question altitude is asked for.
    level_value = geo_transform.levels_for_altitudes(
        np.array([result.point_eus[1]], dtype=np.float64),
        lats=np.array([lat], dtype=np.float64),
        lngs=np.array([lng], dtype=np.float64),
    )[0]
    # The same rays read as a partition: how many objects are in this set, not just
    # where the biggest consensus sits. A basket built from two clusters is exactly
    # the case where one point is the wrong answer.
    blocked: list[tuple[int, int]] = []
    if partition_method in ("gasp", "gasp1v2"):
        hypotheses, unclaimed, blocked = _gasp_partition(
            items,
            ray_index,
            inlier_threshold_deg=inlier_threshold_deg,
            depth_weight=depth_weight,
            use_cannot_link=use_cannot_link,
            merge_criterion="score1v2" if partition_method == "gasp1v2" else "average",
            covis_weight=covis_weight,
            observed_extent=observed_extent,
        )
    elif partition_method == "sequential":
        hypotheses, unclaimed = triangulate_multi(
            origins,
            directions,
            inlier_threshold_deg=inlier_threshold_deg,
            depth_points=depth_points,
            depth_weight=depth_weight,
        )
    else:
        hypotheses, unclaimed = triangulate_linkage(
            origins,
            directions,
            inlier_threshold_deg=inlier_threshold_deg,
            method=partition_method,
            depth_points=depth_points,
            depth_weight=depth_weight,
        )

    serialized_hypotheses = []
    for hypothesis in hypotheses:
        point_wgs84 = geo_transform.local_positions_to_wgs84(
            np.array([hypothesis.point_eus])
        )
        hypothesis_level = geo_transform.levels_for_altitudes(
            np.array([hypothesis.point_eus[1]], dtype=np.float64),
            lats=np.array([float(point_wgs84[0, 1])], dtype=np.float64),
            lngs=np.array([float(point_wgs84[0, 0])], dtype=np.float64),
        )[0]
        serialized_hypotheses.append(
            {
                # Positions in the caller's item list, so the UI needs no mapping.
                "items": [ray_index[local] for local in hypothesis.member_indices],
                "lat": float(point_wgs84[0, 1]),
                "lng": float(point_wgs84[0, 0]),
                "alt": float(point_wgs84[0, 2]),
                "level": (
                    int(hypothesis_level)
                    if np.isfinite(hypothesis_level)
                    else None
                ),
                "mean_residual_deg": hypothesis.mean_residual_deg,
                "mean_residual_m": hypothesis.mean_residual_m,
                "max_parallax_deg": hypothesis.max_parallax_deg,
            }
        )

    return {
        "available": True,
        "ray_index": ray_index,
        "partition_method": partition_method,
        "beyond_max_depth_items": beyond_depth,
        # Returned so the caller can measure the cue's precision against its own
        # annotations before trusting it — a hard constraint is only worth its
        # hardness if it is almost never wrong.
        "cannot_link_pairs": [list(pair) for pair in blocked],
        "hypotheses": serialized_hypotheses,
        "unassigned_items": [ray_index[local] for local in unclaimed],
        "point_eus": list(result.point_eus),
        "lng": lng,
        "lat": lat,
        "alt": alt,
        "level": int(level_value) if np.isfinite(level_value) else None,
        # Positions in the basket, not indices into `ray_index` — the UI colours
        # basket rows with this and must not have to compose two mappings.
        "inlier_items": [ray_index[index] for index in result.inlier_indices],
        "residuals_deg": list(result.residuals_deg),
        "residuals_m": list(result.residuals_m),
        "mean_inlier_residual_deg": result.mean_inlier_residual_deg,
        "mean_inlier_residual_m": result.mean_inlier_residual_m,
        "max_parallax_deg": result.max_parallax_deg,
    }


def _wgs84(geo_transform: GeoTransform, eus: tuple[float, float, float]) -> list[float]:
    """`[lat, lng, alt]` — the order every toolbox map layer wants."""
    converted = geo_transform.local_positions_to_wgs84(np.array([eus]))
    return [float(converted[0, 1]), float(converted[0, 0]), float(converted[0, 2])]


def build_matching_response(
    conn: Any,
    geo_ref_id: int,
    geo_transform: GeoTransform,
    requested: Sequence[tuple[str, float, float]],
    *,
    inlier_threshold_deg: float = DEFAULT_INLIER_THRESHOLD_DEG,
    ray_length_m: float = 30.0,
    partition_method: str = "sequential",
    max_depth_m: float | None = None,
    depth_weight: float = 0.0,
    use_cannot_link: bool = False,
    covis_weight: float = 0.0,
    observed_extent: bool = False,
) -> dict[str, Any]:
    items = resolve_items(conn, geo_ref_id, requested)
    serialized: list[dict[str, Any]] = []
    for item in items:
        # The ray is drawn as a fixed-length segment rather than stopped at the
        # triangulated point: when triangulation *fails*, the diverging rays are the
        # only thing that explains why, and a segment that stops at a point that does
        # not exist would explain nothing.
        ray_end = (
            None
            if item.ray_origin_eus is None or item.ray_direction_eus is None
            else _wgs84(
                geo_transform,
                (
                    item.ray_origin_eus[0] + item.ray_direction_eus[0] * ray_length_m,
                    item.ray_origin_eus[1] + item.ray_direction_eus[1] * ray_length_m,
                    item.ray_origin_eus[2] + item.ray_direction_eus[2] * ray_length_m,
                ),
            )
        )
        serialized.append(
            {
                "keyframe_id": item.keyframe_id,
                "theta_center": item.theta_center,
                "phi_center": item.phi_center,
                "resolved": item.candidate_id is not None,
                "candidate_id": item.candidate_id,
                "thumbnail": item.thumbnail,
                "has_embedding": item.embedding is not None,
                "has_stored_position": item.stored_position_eus is not None,
                "keyframe_wgs84": (
                    None
                    if item.ray_origin_eus is None
                    else _wgs84(geo_transform, item.ray_origin_eus)
                ),
                "ray_end_wgs84": ray_end,
                "stored_wgs84": (
                    None
                    if item.stored_position_eus is None
                    else _wgs84(geo_transform, item.stored_position_eus)
                ),
            }
        )
    return {
        "items": serialized,
        "similarity": similarity_matrix(items),
        "triangulation": triangulate_items(
            items,
            geo_transform,
            inlier_threshold_deg=inlier_threshold_deg,
            partition_method=partition_method,
            max_depth_m=max_depth_m,
            depth_weight=depth_weight,
            use_cannot_link=use_cannot_link,
            covis_weight=covis_weight,
            observed_extent=observed_extent,
        ),
    }
