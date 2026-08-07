"""Shared candidate loading for object-search v2 and v1.5 localize.

Ported from `backend/object_search/candidates.py`. Two changes:

- the ORM query with its `Func(F(...), function="ST_X")` annotations becomes one
  raw SQL statement (same columns, same `object_position IS NOT NULL` filter);
- `GeoRef` becomes a plain `geo_ref_id` plus a `GeoTransform` built from the v2
  map manifest, and the S3 URL envelope is dropped — the toolbox serves files from
  the map directory.

The enrichment *order* is load-bearing and preserved: object positions → WGS84 →
levels, then the same for keyframe positions, then headings, then sort by
similarity descending.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from toolbox.bricks.feedback import ReviewFeedback
from toolbox.bricks.vendored.geo_transform import GeoTransform, Pose
from toolbox.bricks.vendored.maths import quaternion, vector3
from toolbox.bricks.vendored.viewer360_headings import headings_from_orientations
from toolbox.logging import logger

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

# Optional review-feedback columns, spliced into _ENRICH_SQL's SELECT list only
# when a prototype set is non-empty. Two properties are load-bearing:
#
# - `1 - POWER(<->, 2)/2` is the *same* formula the online service's `_hnsw_query`
#   uses to turn an L2 distance into a similarity. It equals cosine only because
#   the embeddings are unit vectors (verified: norms 0.999428–1.000566). Using a
#   different scale here would make alpha/beta meaningless.
# - `p.geo_ref_id = %s` is not redundant with `p.id = ANY(%s)`: without it one
#   map could borrow another map's annotations through a stale target_id.
#
# MAX, not AVG: a single cluster-level "correct" click in the review UI fans out
# to one row per cutout, so the positive set contains rows nobody judged
# individually. MAX asks "is this near *any* endorsed cutout", which is the
# question the annotation actually answers.
_PROTOTYPE_SIM_SQL = """,
    (SELECT MAX(1 - POWER(c.embedding <-> p.embedding, 2) / 2)
       FROM object_search_candidate AS p
      WHERE p.id = ANY(%s) AND p.geo_ref_id = %s) AS {alias}"""

# Where the extra SELECT columns get spliced. Anchoring on this exact substring
# means the no-feedback path emits `_ENRICH_SQL` *unchanged* — not a regenerated
# equivalent — which is what makes "alpha = beta = 0 reproduces today's output"
# a property of the code rather than a hope.
_ENRICH_FROM_ANCHOR = "\nFROM object_search_candidate AS c"


def _build_enrich_query(
    geo_ref_id: int,
    ids: list[int],
    feedback: ReviewFeedback | None,
) -> tuple[str, list, bool, bool]:
    """`(sql, params, has_pos, has_neg)` for the enrichment fetch.

    The prototype subqueries live in the SELECT list, so their parameters come
    *before* the WHERE clause's — get that order wrong and psycopg2 silently binds
    a map id where an id array belongs.
    """
    if feedback is None:
        return _ENRICH_SQL, [geo_ref_id, ids], False, False

    has_pos = bool(feedback.positive_ids)
    has_neg = bool(feedback.negative_ids)
    if not has_pos and not has_neg:
        return _ENRICH_SQL, [geo_ref_id, ids], False, False

    extra = ""
    params: list = []
    if has_pos:
        extra += _PROTOTYPE_SIM_SQL.format(alias="pos_sim")
        params += [feedback.positive_ids, geo_ref_id]
    if has_neg:
        extra += _PROTOTYPE_SIM_SQL.format(alias="neg_sim")
        params += [feedback.negative_ids, geo_ref_id]

    sql = _ENRICH_SQL.replace(_ENRICH_FROM_ANCHOR, extra + _ENRICH_FROM_ANCHOR, 1)
    return sql, params + [geo_ref_id, ids], has_pos, has_neg


def _log_prototype_resolution(
    feedback: ReviewFeedback, rows: list[dict], has_pos: bool, has_neg: bool
) -> None:
    """Warn when annotations resolve to nothing — the silent-neutrality failure.

    `detection_review.target_id` is an `object_search_candidate.id`, a BIGSERIAL
    wiped and re-inserted by every ingest. After a reingest the stored ids match
    nothing, every `pos_sim`/`neg_sim` comes back NULL, and the boost quietly
    becomes a no-op that looks exactly like "the feature did not help". This is
    the only place that difference is observable, so it is logged loudly.
    """
    if has_pos and all(row.get("pos_sim") is None for row in rows):
        logger.warning(
            "None of the %d positive annotation(s) resolved to a candidate in this "
            "georef — the boost is inert. Ids are BIGSERIAL and do not survive a "
            "reingest; re-collect them or freeze the index.",
            len(feedback.positive_ids),
        )
    if has_neg and all(row.get("neg_sim") is None for row in rows):
        logger.warning(
            "None of the %d negative annotation(s) resolved to a candidate in this "
            "georef — the penalty is inert (same cause as above).",
            len(feedback.negative_ids),
        )


def apply_feedback_boost(
    similarity: float, pos_sim: float, neg_sim: float, alpha: float, beta: float
) -> float:
    """`similarity + alpha*pos_sim - beta*neg_sim`, clipped to [-1, 1].

    With `alpha = beta = 0` this returns `similarity` bit-for-bit — the term is
    multiplied to exactly 0.0, not merely made small — which is what "off by
    default" has to mean for the baseline to be the current code path.
    """
    boosted = float(similarity) + float(alpha) * float(pos_sim)
    boosted -= float(beta) * float(neg_sim)
    return max(-1.0, min(1.0, boosted))


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
    # Review-feedback terms. `pos_sim`/`neg_sim` are kept rather than folded away
    # so a tuning session can see which half of the term moved a candidate.
    #
    # `similarity_boosted` is None — not 0.0 — when no boost was computed, so a
    # candidate built without it (every existing test fixture, and any caller that
    # never heard of this feature) falls back to the raw similarity through
    # `effective_similarity`. A 0.0 default would instead read as "maximally
    # penalised" and silently sink every such candidate.
    similarity_boosted: float | None = None
    pos_sim: float = 0.0
    neg_sim: float = 0.0

    @property
    def effective_similarity(self) -> float:
        """The similarity ranking should use: boosted when present, raw otherwise."""
        return (
            self.similarity
            if self.similarity_boosted is None
            else self.similarity_boosted
        )


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
    feedback: ReviewFeedback | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> list[EnrichedCandidate]:
    """Pre-filter HNSW hits, fetch DB rows, convert object positions to WGS84.

    `feedback`/`alpha`/`beta` add `similarity_boosted` to every candidate. They do
    **not** change which rows are fetched, how many survive the prefilter, or the
    sort order — that stays on raw `similarity`, so the retrieved set is identical
    with and without feedback and only the ranking downstream can differ.
    """
    results = _prefilter_hnsw_results(hnsw_results)
    if not results:
        return []

    sim_by_id = {r["id"]: r["similarity"] for r in results}
    ids = list(sim_by_id.keys())

    sql, params, has_pos, has_neg = _build_enrich_query(geo_ref_id, ids, feedback)
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    if not rows:
        return []

    if feedback is not None:
        _log_prototype_resolution(feedback, rows, has_pos, has_neg)

    eus_xyz = np.array([[r["px"], r["py"], r["pz"]] for r in rows], dtype=np.float64)
    wgs84 = geo_transform.local_positions_to_wgs84(eus_xyz)
    # NOTE: levels are resolved from the EUS local-up coordinate (eus_xyz[:, 1]),
    # never the WGS84 altitude. The manifest's level bands are heights above the
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
        similarity = float(sim_by_id[row["id"]])
        # NULL when the prototype set is empty or every prototype was filtered out
        # by geo_ref_id — treat as "no evidence", i.e. no contribution.
        pos_sim = float(row.get("pos_sim") or 0.0)
        neg_sim = float(row.get("neg_sim") or 0.0)
        enriched.append(
            EnrichedCandidate(
                id=row["id"],
                similarity=similarity,
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
                similarity_boosted=apply_feedback_boost(
                    similarity, pos_sim, neg_sim, alpha, beta
                ),
                pos_sim=pos_sim,
                neg_sim=neg_sim,
            )
        )

    # Raw similarity, deliberately. Sorting on the boosted value here would move
    # the `select_top_candidates` truncation and the clustering seed order, which
    # changes cluster *geometry* rather than cluster ranking — a different, much
    # harder-to-reason-about intervention than the one this feature is.
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
