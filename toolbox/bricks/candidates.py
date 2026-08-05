"""Shared candidate loading for object-search v2 and v1.5 localize.

Ported from `backend/object_search/candidates.py`. Two changes:

- the ORM query with its `Func(F(...), function="ST_X")` annotations becomes one
  raw SQL statement (same columns, same `object_position IS NOT NULL` filter);
- `GeoRef` becomes a plain `geo_ref_id` plus a `GeoTransform` built from
  `georef.db`, and the S3 URL envelope is dropped — the toolbox serves files from
  the map directory.

The enrichment *order* is load-bearing and preserved: object positions → WGS84 →
levels, then the same for keyframe positions, then headings, then sort by
similarity descending.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from toolbox.bricks.vendored.geo_transform import GeoTransform, Pose
from toolbox.bricks.vendored.maths import quaternion, vector3
from toolbox.bricks.vendored.viewer360_headings import headings_from_orientations

K_INTERNAL = 1000  # candidates fetched from HNSW — effective hard cap
LOOSE_ALPHA = 0.3  # pre-filter: keep score >= LOOSE_ALPHA * max_score

_ENRICH_SQL = """
SELECT
    c.id,
    ST_X(c.object_position) AS px,
    ST_Y(c.object_position) AS py,
    ST_Z(c.object_position) AS pz,
    c.thumbnail,
    c.theta_center,
    c.phi_center,
    c.angular_width,
    c.angular_height,
    k.video_keyframe_id,
    ST_X(k.position) AS vkf_px,
    ST_Y(k.position) AS vkf_py,
    ST_Z(k.position) AS vkf_pz,
    k.orientation,
    k.image,
    k.depth_map
FROM object_search_candidate AS c
JOIN geokeyframe AS k ON k.id = c.geokeyframe_id
WHERE c.geo_ref_id = %s
  AND c.id = ANY(%s)
  AND c.object_position IS NOT NULL
"""


@dataclass(frozen=True)
class EnrichedCandidate:
    id: int
    similarity: float
    eus_xyz: tuple[float, float, float]
    lat: float
    lng: float
    alt: float
    level: int | None
    video_keyframe_id: int
    theta_center: float
    phi_center: float
    geokeyframe_pose: Pose
    thumbnail: str | None
    angular_width: float
    angular_height: float
    vkf_lat: float
    vkf_lng: float
    vkf_alt: float
    vkf_level: int | None
    video_keyframe_filename: str
    video_keyframe_heading: float
    video_keyframe_depth: str


def _prefilter_hnsw_results(hnsw_results: list[dict]) -> list[dict]:
    if not hnsw_results:
        return []
    max_sim = hnsw_results[0]["similarity"]
    n_above = sum(1 for r in hnsw_results if r["similarity"] >= LOOSE_ALPHA * max_sim)
    cap = n_above * 2
    return hnsw_results[:cap]


def load_enriched_candidates(
    conn,
    geo_ref_id: int,
    hnsw_results: list[dict],
    geo_transform: GeoTransform,
) -> list[EnrichedCandidate]:
    """Pre-filter HNSW hits, fetch DB rows, convert object positions to WGS84."""
    results = _prefilter_hnsw_results(hnsw_results)
    if not results:
        return []

    sim_by_id = {r["id"]: r["similarity"] for r in results}
    ids = list(sim_by_id.keys())

    with conn.cursor() as cursor:
        cursor.execute(_ENRICH_SQL, [geo_ref_id, ids])
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    if not rows:
        return []

    eus_xyz = np.array([[r["px"], r["py"], r["pz"]] for r in rows], dtype=np.float64)
    wgs84 = geo_transform.local_positions_to_wgs84(eus_xyz)
    # NOTE: levels are resolved from the EUS local-up coordinate (eus_xyz[:, 1]),
    # never the WGS84 altitude. georef.db's level bands are heights above the
    # georef origin, which is exactly what local-up measures. Passing wgs84[:, 2]
    # here would make every level None — silently disabling the level-compatibility
    # guard in clustering, which then merges objects across floors.
    levels_arr = geo_transform.levels_for_altitudes(
        eus_xyz[:, 1], lats=wgs84[:, 1], lngs=wgs84[:, 0]
    )

    vkf_eus = np.array(
        [[r["vkf_px"], r["vkf_py"], r["vkf_pz"]] for r in rows], dtype=np.float64
    )
    vkf_wgs84 = geo_transform.local_positions_to_wgs84(vkf_eus)
    vkf_levels_arr = geo_transform.levels_for_altitudes(
        vkf_eus[:, 1], lats=vkf_wgs84[:, 1], lngs=vkf_wgs84[:, 0]
    )

    orientations_wxyz = np.array([r["orientation"] for r in rows], dtype=np.float64)
    vkf_headings = headings_from_orientations(orientations_wxyz)

    enriched: list[EnrichedCandidate] = []
    for i, row in enumerate(rows):
        orientation = quaternion.cast(np.asarray(row["orientation"], dtype=np.float64))
        pose = Pose.from_position_orientation(
            vector3.from_xyz(row["vkf_px"], row["vkf_py"], row["vkf_pz"]),
            orientation,
        )
        level_val = levels_arr[i]
        vkf_level_val = vkf_levels_arr[i]
        enriched.append(
            EnrichedCandidate(
                id=row["id"],
                similarity=float(sim_by_id[row["id"]]),
                eus_xyz=(float(row["px"]), float(row["py"]), float(row["pz"])),
                lat=float(wgs84[i, 1]),
                lng=float(wgs84[i, 0]),
                alt=float(wgs84[i, 2]),
                level=int(level_val) if np.isfinite(level_val) else None,
                video_keyframe_id=row["video_keyframe_id"],
                theta_center=float(row["theta_center"]),
                phi_center=float(row["phi_center"]),
                geokeyframe_pose=pose,
                thumbnail=row["thumbnail"] or None,
                angular_width=float(row["angular_width"]),
                angular_height=float(row["angular_height"]),
                vkf_lat=float(vkf_wgs84[i, 1]),
                vkf_lng=float(vkf_wgs84[i, 0]),
                vkf_alt=float(vkf_wgs84[i, 2]),
                vkf_level=int(vkf_level_val) if np.isfinite(vkf_level_val) else None,
                video_keyframe_filename=os.path.basename(row["image"] or ""),
                video_keyframe_heading=float(vkf_headings[i]),
                video_keyframe_depth=os.path.basename(row["depth_map"] or ""),
            )
        )

    enriched.sort(key=lambda c: c.similarity, reverse=True)
    return enriched


def resolve_candidates_v2_response(
    conn,
    geo_ref_id: int,
    hnsw_results: list[dict],
    geo_transform: GeoTransform,
) -> dict:
    """Flat candidate list, in the v2 API shape.

    PORT NOTE: production also returns `base_url_video_keyframe_{images,depths}`
    and `base_url_thumbnails` pointing at S3. Locally the toolbox resolves those
    from the map directory, so the envelope is just `{"candidates": [...]}`.
    """
    enriched = load_enriched_candidates(conn, geo_ref_id, hnsw_results, geo_transform)
    return {
        "candidates": [
            {
                "id": c.id,
                "similarity": round(c.similarity, 4),
                "lat": round(c.lat, 8),
                "lng": round(c.lng, 8),
                "alt": round(c.alt, 3),
                "level": c.level,
                "thumbnail_key": c.thumbnail or "",
                "video_keyframe_id": c.video_keyframe_id,
                "theta_center": c.theta_center,
                "phi_center": c.phi_center,
                "fov_x": c.angular_width,
                "fov_y": c.angular_height,
                "video_keyframe_filename": c.video_keyframe_filename,
                "video_keyframe_heading": round(c.video_keyframe_heading, 4),
                "video_keyframe_depth": c.video_keyframe_depth,
                "video_keyframe_lat": round(c.vkf_lat, 8),
                "video_keyframe_lng": round(c.vkf_lng, 8),
                "video_keyframe_alt": round(c.vkf_alt, 3),
                "video_keyframe_level": c.vkf_level,
            }
            for c in enriched
        ]
    }
