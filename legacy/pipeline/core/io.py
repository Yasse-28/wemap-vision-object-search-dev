from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

from pipeline.core.database import (
    INDEX_METADATA_PARAM_KEY,
    LEGACY_MANIFEST_PARAM_KEY,
    ensure_object_search_index_schema,
    migrate_legacy_manifest_param_to_index_metadata,
)
from pipeline.core.types import (
    OBJECT_SEARCH_INDEX_DB_FILENAME,
    UNRESOLVED_LEVEL_SENTINEL,
    LoadedIndex,
    ObjectSearchIndexMetadata,
)


def _resolve_index_db_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        return path / OBJECT_SEARCH_INDEX_DB_FILENAME
    raise FileNotFoundError(f"Object-search index not found: {path}")


# ---------------------------------------------------------------------------
# Intermediate NamedTuples returned by table-level load helpers
# ---------------------------------------------------------------------------


class _CutoutData(NamedTuple):
    ids: np.ndarray
    keyframe_ids: np.ndarray
    center_xy: np.ndarray
    rotation: np.ndarray
    embeddings: np.ndarray


class _ClusterData(NamedTuple):
    centroids_world: Optional[np.ndarray]
    centroids_geo: Optional[np.ndarray]
    obs_counts: Optional[np.ndarray]
    conf: Optional[np.ndarray]
    levels: Optional[np.ndarray]
    ocr_texts: Optional[np.ndarray]
    ocr_tokens: Optional[np.ndarray]
    ocr_keys: Optional[np.ndarray]
    ocr_obs_counts: Optional[np.ndarray]
    ocr_sources: Optional[np.ndarray]
    cutout_cluster_ids: Optional[np.ndarray]
    cutout_ids: Optional[np.ndarray]
    cutout_keyframe_ids: Optional[np.ndarray]
    cutout_levels: Optional[np.ndarray]
    cutout_obs_counts: Optional[np.ndarray]


class _ObjectData(NamedTuple):
    ids: np.ndarray
    keyframe_ids: np.ndarray
    cutout_ids: np.ndarray
    bboxes: np.ndarray
    embeddings: np.ndarray
    pos_keyframe: Optional[np.ndarray]
    pos_local: Optional[np.ndarray]
    pos_world: Optional[np.ndarray]
    depths: Optional[np.ndarray]
    loc_valid: Optional[np.ndarray]
    cluster_ids: Optional[np.ndarray]
    detection_levels: Optional[np.ndarray]
    bbox_spherical: Optional[np.ndarray]
    visual_sim: Optional[np.ndarray]
    visual_cand: Optional[np.ndarray]
    visual_assign: Optional[np.ndarray]
    textness: Optional[np.ndarray]
    ocr_texts: Optional[np.ndarray]
    ocr_tokens: Optional[np.ndarray]
    ocr_keys: Optional[np.ndarray]
    ocr_cand: Optional[np.ndarray]
    ocr_assign: Optional[np.ndarray]
    ocr_source: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    sources: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Alignment helpers (lifted to module level from load_index)
# ---------------------------------------------------------------------------


def _aligned_3d(lst: list, n: int) -> np.ndarray | None:
    """Build an (n, 3) float32 array from a list of 3-vectors-or-None."""
    if not any(x is not None for x in lst):
        return None
    arr = np.full((n, 3), np.nan, dtype=np.float32)
    for i, v in enumerate(lst):
        if v is not None:
            arr[i] = v
    return arr


def _aligned_1d(lst: list, dtype: Any, n: int) -> np.ndarray | None:
    """Build a length-n array from a list of scalars-or-None."""
    if not any(x is not None for x in lst):
        return None
    fill: Any = False if dtype is bool else np.nan
    arr = np.full(n, fill, dtype=dtype)
    for i, v in enumerate(lst):
        if v is not None:
            arr[i] = v
    return arr


def _aligned_4d(lst: list, n: int) -> np.ndarray | None:
    """Build an (n, 4) float32 array from a list of 4-vectors-or-None."""
    if not any(x is not None for x in lst):
        return None
    arr = np.full((n, 4), np.nan, dtype=np.float32)
    for i, v in enumerate(lst):
        if v is not None:
            arr[i] = v
    return arr


# ---------------------------------------------------------------------------
# Private table-level load helpers
# ---------------------------------------------------------------------------


def _load_cutouts(
    cursor: sqlite3.Cursor,
    metadata: ObjectSearchIndexMetadata,
) -> _CutoutData:
    cursor.execute("SELECT * FROM cutout ORDER BY cutout_id")
    rows = cursor.fetchall()

    if not rows:
        return _CutoutData(
            ids=np.array([], dtype=np.int64),
            keyframe_ids=np.array([], dtype=np.int64),
            center_xy=np.zeros((0, 2), dtype=np.float32),
            rotation=np.zeros((0, 4, 4), dtype=np.float32),
            embeddings=np.zeros((0, metadata.projection_dim), dtype=np.float32),
        )

    ids, kf_ids, xys, rots, embs = [], [], [], [], []
    for row in rows:
        ids.append(row["cutout_id"])
        kf_ids.append(row["keyframe_id"])
        cx = row["center_x"] if row["center_x"] is not None else float("nan")
        cy = row["center_y"] if row["center_y"] is not None else float("nan")
        xys.append([cx, cy])
        rots.append(np.frombuffer(row["rotation"], dtype=np.float32).reshape(4, 4))
        embs.append(np.frombuffer(row["embedding"], dtype=np.float32))

    return _CutoutData(
        ids=np.array(ids, dtype=np.int64),
        keyframe_ids=np.array(kf_ids, dtype=np.int64),
        center_xy=np.array(xys, dtype=np.float32),
        rotation=np.stack(rots, axis=0),
        embeddings=np.stack(embs, axis=0),
    )


def _load_clusters(cursor: sqlite3.Cursor) -> _ClusterData:
    # cluster_levels in the params table is a fallback; the cluster table's
    # per-row level column takes precedence when cluster rows exist.
    cursor.execute("SELECT value FROM params WHERE key = ?", ("cluster_levels",))
    param_row = cursor.fetchone()
    levels_from_params: Optional[np.ndarray] = None
    if param_row is not None and param_row[0] is not None:
        levels_from_params = np.frombuffer(param_row[0], dtype=np.int32)

    cursor.execute("SELECT * FROM cluster ORDER BY cluster_id")
    cluster_rows = cursor.fetchall()

    centroids_world = centroids_geo = obs_counts = conf = levels = None
    ocr_texts = ocr_tokens = ocr_keys = ocr_obs_counts = ocr_sources = None

    if cluster_rows:
        cw, cg, obs, confs, lvls = [], [], [], [], []
        ot, otok, ok, oobs, osrc = [], [], [], [], []
        for row in cluster_rows:
            cw.append(np.frombuffer(row["centroid_world"], dtype=np.float32))
            cg.append(np.frombuffer(row["centroid_geo"], dtype=np.float32))
            obs.append(row["observation_count"])
            confs.append(row["confidence"])
            lvls.append(row["level"])
            ot.append(row["ocr_text"] if row["ocr_text"] is not None else "")
            otok.append(row["ocr_tokens"] if row["ocr_tokens"] is not None else "")
            ok.append(row["ocr_key"] if row["ocr_key"] is not None else "")
            oobs.append(
                row["ocr_observation_count"]
                if row["ocr_observation_count"] is not None
                else 0
            )
            osrc.append(row["ocr_source"] if row["ocr_source"] is not None else 0)

        centroids_world = np.stack(cw, axis=0)
        centroids_geo = np.stack(cg, axis=0)
        obs_counts = np.array(obs, dtype=np.int32)
        conf = np.array(confs, dtype=np.float32)
        levels = np.array(lvls, dtype=np.int32)  # overrides levels_from_params
        ocr_texts = np.array(ot, dtype="<U512")
        ocr_tokens = np.array(otok, dtype="<U512")
        ocr_keys = np.array(ok, dtype="<U256")
        ocr_obs_counts = np.array(oobs, dtype=np.int32)
        ocr_sources = np.array(osrc, dtype=np.int16)
    else:
        levels = levels_from_params

    cursor.execute("SELECT * FROM cluster_cutout ORDER BY cluster_id, cutout_id")
    cc_rows = cursor.fetchall()

    cc_cluster_ids = cc_ids = cc_kf_ids = cc_levels = cc_obs = None
    if cc_rows:
        cc_cluster_ids = np.array([r["cluster_id"] for r in cc_rows], dtype=np.int32)
        cc_ids = np.array([r["cutout_id"] for r in cc_rows], dtype=np.int64)
        cc_kf_ids = np.array([r["keyframe_id"] for r in cc_rows], dtype=np.int64)
        cc_levels = np.array([r["level"] for r in cc_rows], dtype=np.int32)
        cc_obs = np.array([r["observation_count"] for r in cc_rows], dtype=np.int32)

    return _ClusterData(
        centroids_world=centroids_world,
        centroids_geo=centroids_geo,
        obs_counts=obs_counts,
        conf=conf,
        levels=levels,
        ocr_texts=ocr_texts,
        ocr_tokens=ocr_tokens,
        ocr_keys=ocr_keys,
        ocr_obs_counts=ocr_obs_counts,
        ocr_sources=ocr_sources,
        cutout_cluster_ids=cc_cluster_ids,
        cutout_ids=cc_ids,
        cutout_keyframe_ids=cc_kf_ids,
        cutout_levels=cc_levels,
        cutout_obs_counts=cc_obs,
    )


def _empty_object_data(projection_dim: int) -> _ObjectData:
    """An _ObjectData with empty id/bbox/embedding arrays and all optionals None."""
    empty = np.array([], dtype=np.int64)
    return _ObjectData(
        ids=empty,
        keyframe_ids=empty,
        cutout_ids=empty,
        bboxes=np.zeros((0, 4), dtype=np.float32),
        embeddings=np.zeros((0, projection_dim), dtype=np.float32),
        pos_keyframe=None,
        pos_local=None,
        pos_world=None,
        depths=None,
        loc_valid=None,
        cluster_ids=None,
        detection_levels=None,
        bbox_spherical=None,
        visual_sim=None,
        visual_cand=None,
        visual_assign=None,
        textness=None,
        ocr_texts=None,
        ocr_tokens=None,
        ocr_keys=None,
        ocr_cand=None,
        ocr_assign=None,
        ocr_source=None,
        labels=None,
        sources=None,
    )


def _load_objects(
    cursor: sqlite3.Cursor,
    load_embeddings: bool,
    projection_dim: int,
) -> _ObjectData:
    if load_embeddings:
        cursor.execute("SELECT * FROM object ORDER BY object_idx")
    else:
        cols = [
            r[1]
            for r in cursor.execute("PRAGMA table_info(object)").fetchall()
            if r[1] != "embedding"
        ]
        cursor.execute(f"SELECT {', '.join(cols)} FROM object ORDER BY object_idx")
    rows = cursor.fetchall()

    if not rows:
        return _empty_object_data(projection_dim)

    # Untyped accumulators: lists while walking rows, then converted to
    # ndarray-or-None. `Any` so that reassignment is allowed by mypy.
    obj_idx: Any = []
    obj_kf: Any = []
    obj_cut: Any = []
    obj_bb: Any = []
    obj_sph_bbox: Any = []
    obj_emb: Any = []
    obj_pos_kf: Any = []
    obj_pos_local: Any = []
    obj_pos_world: Any = []
    obj_depth: Any = []
    obj_loc_valid: Any = []
    obj_cluster: Any = []
    obj_level: Any = []
    obj_vis_sim: Any = []
    obj_vis_cand: Any = []
    obj_vis_assign: Any = []
    obj_textness: Any = []
    obj_ocr_text: Any = []
    obj_ocr_tokens: Any = []
    obj_ocr_key: Any = []
    obj_ocr_cand: Any = []
    obj_ocr_assign: Any = []
    obj_ocr_src: Any = []
    obj_label: Any = []
    obj_det_source: Any = []

    has_cluster = has_level = has_vis_sim = has_vis_cand = has_vis_assign = False
    has_textness = has_ocr_text = has_ocr_tokens = has_ocr_key = False
    has_ocr_cand = has_ocr_assign = has_ocr_src = False

    for row in rows:
        obj_idx.append(row["object_idx"])
        obj_kf.append(row["keyframe_id"])
        obj_cut.append(row["cutout_id"])
        obj_bb.append(np.frombuffer(row["bbox_coordinates"], dtype=np.float32))

        try:
            sph = row["bbox_spherical_coordinates"]
            obj_sph_bbox.append(
                np.frombuffer(sph, dtype=np.float32) if sph is not None else None
            )
        except (IndexError, KeyError):
            obj_sph_bbox.append(None)

        if load_embeddings:
            obj_emb.append(np.frombuffer(row["embedding"], dtype=np.float32))

        for lst, key in (
            (obj_pos_kf, "position_keyframe"),
            (obj_pos_local, "position_local"),
            (obj_pos_world, "position_world"),
        ):
            try:
                v = row[key]
                lst.append(
                    np.frombuffer(v, dtype=np.float32) if v is not None else None
                )
            except (IndexError, KeyError):
                lst.append(None)

        obj_depth.append(row["depth"] if row["depth"] is not None else None)
        obj_loc_valid.append(
            bool(row["localization_valid"])
            if row["localization_valid"] is not None
            else None
        )

        if row["cluster_id"] is not None:
            obj_cluster.append(int(row["cluster_id"]))
            has_cluster = True
        else:
            obj_cluster.append(-1)

        if row["level"] is not None:
            obj_level.append(int(row["level"]))
            has_level = True
        else:
            obj_level.append(UNRESOLVED_LEVEL_SENTINEL)

        if row["visual_similarity_score"] is not None:
            obj_vis_sim.append(row["visual_similarity_score"])
            has_vis_sim = True
        else:
            obj_vis_sim.append(None)

        if row["visual_candidate"] is not None:
            obj_vis_cand.append(bool(row["visual_candidate"]))
            has_vis_cand = True
        else:
            obj_vis_cand.append(None)

        if row["visual_assigned"] is not None:
            obj_vis_assign.append(bool(row["visual_assigned"]))
            has_vis_assign = True
        else:
            obj_vis_assign.append(None)

        if row["textness_score"] is not None:
            obj_textness.append(row["textness_score"])
            has_textness = True
        else:
            obj_textness.append(None)

        v_text = row["ocr_text"]
        obj_ocr_text.append(v_text if v_text is not None else "")
        has_ocr_text = has_ocr_text or (v_text is not None)

        v_tokens = row["ocr_tokens"]
        obj_ocr_tokens.append(v_tokens if v_tokens is not None else "")
        has_ocr_tokens = has_ocr_tokens or (v_tokens is not None)

        v_key = row["ocr_key"]
        obj_ocr_key.append(v_key if v_key is not None else "")
        has_ocr_key = has_ocr_key or (v_key is not None)

        if row["ocr_candidate"] is not None:
            obj_ocr_cand.append(bool(row["ocr_candidate"]))
            has_ocr_cand = True
        else:
            obj_ocr_cand.append(None)

        if row["ocr_assigned"] is not None:
            obj_ocr_assign.append(bool(row["ocr_assigned"]))
            has_ocr_assign = True
        else:
            obj_ocr_assign.append(None)

        if row["ocr_source"] is not None:
            obj_ocr_src.append(row["ocr_source"])
            has_ocr_src = True
        else:
            obj_ocr_src.append(None)

        try:
            obj_label.append(row["label"] or "")
            obj_det_source.append(row["detection_source"] or "")
        except (IndexError, KeyError):
            obj_label.append("")
            obj_det_source.append("")

    n = len(obj_kf)
    return _ObjectData(
        ids=np.array(obj_idx, dtype=np.int64),
        keyframe_ids=np.array(obj_kf, dtype=np.int64),
        cutout_ids=np.array(obj_cut, dtype=np.int64),
        bboxes=np.array(obj_bb, dtype=np.float32),
        embeddings=(
            np.stack(obj_emb, axis=0)
            if load_embeddings
            else np.zeros((n, 0), dtype=np.float32)
        ),
        pos_keyframe=_aligned_3d(obj_pos_kf, n),
        pos_local=_aligned_3d(obj_pos_local, n),
        pos_world=_aligned_3d(obj_pos_world, n),
        depths=_aligned_1d(obj_depth, np.float32, n),
        loc_valid=_aligned_1d(obj_loc_valid, bool, n),
        cluster_ids=(
            np.array(obj_cluster, dtype=np.int32)
            if has_cluster or any(x == -1 for x in obj_cluster)
            else None
        ),
        detection_levels=(
            np.array(obj_level, dtype=np.int32)
            if has_level or any(x == UNRESOLVED_LEVEL_SENTINEL for x in obj_level)
            else None
        ),
        bbox_spherical=_aligned_4d(obj_sph_bbox, n),
        visual_sim=(
            np.array([x for x in obj_vis_sim if x is not None], dtype=np.float32)
            if has_vis_sim
            else None
        ),
        visual_cand=(
            np.array([x for x in obj_vis_cand if x is not None], dtype=bool)
            if has_vis_cand
            else None
        ),
        visual_assign=(
            np.array([x for x in obj_vis_assign if x is not None], dtype=bool)
            if has_vis_assign
            else None
        ),
        textness=(
            np.array([x for x in obj_textness if x is not None], dtype=np.float32)
            if has_textness
            else None
        ),
        ocr_texts=np.array(obj_ocr_text, dtype="<U512") if has_ocr_text else None,
        ocr_tokens=np.array(obj_ocr_tokens, dtype="<U512") if has_ocr_tokens else None,
        ocr_keys=np.array(obj_ocr_key, dtype="<U256") if has_ocr_key else None,
        ocr_cand=(
            np.array([x for x in obj_ocr_cand if x is not None], dtype=bool)
            if has_ocr_cand
            else None
        ),
        ocr_assign=(
            np.array([x for x in obj_ocr_assign if x is not None], dtype=bool)
            if has_ocr_assign
            else None
        ),
        ocr_source=(
            np.array([x for x in obj_ocr_src if x is not None], dtype=np.int16)
            if has_ocr_src
            else None
        ),
        labels=np.array(obj_label, dtype="<U256") if any(obj_label) else None,
        sources=np.array(obj_det_source, dtype="<U16") if any(obj_det_source) else None,
    )


# ---------------------------------------------------------------------------
# Private cursor-level write helpers
# ---------------------------------------------------------------------------


def _write_params(
    cursor: sqlite3.Cursor,
    metadata: ObjectSearchIndexMetadata,
    state: dict[str, np.ndarray],
) -> None:
    cursor.execute(
        "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
        (INDEX_METADATA_PARAM_KEY, metadata.to_json().encode("utf-8")),
    )
    cursor.execute("DELETE FROM params WHERE key = ?", (LEGACY_MANIFEST_PARAM_KEY,))

    processed = state.get("object_processed_cutout_ids")
    if processed is not None:
        cursor.execute(
            "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
            ("processed_cutout_ids", np.asarray(processed, dtype=np.int64).tobytes()),
        )

    for key in ("cluster_levels",):
        arr = state.get(key)
        if arr is not None:
            typed = np.asarray(
                arr, dtype=np.int32 if key == "cluster_levels" else np.float32
            )
            cursor.execute(
                "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
                (key, typed.tobytes()),
            )


def _write_cutouts(
    cursor: sqlite3.Cursor,
    state: dict[str, np.ndarray],
) -> None:
    cutout_ids = np.asarray(state["cutout_ids"], dtype=np.int64).ravel()
    cutout_kf = np.asarray(state["cutout_keyframe_ids"], dtype=np.int64).ravel()
    cutout_xy = np.asarray(state["cutout_center_xy"], dtype=np.float32).reshape(-1, 2)
    cutout_rot = np.asarray(
        state["cutout_rotation_cutout_to_equirect"], dtype=np.float32
    ).reshape(-1, 4, 4)
    cutout_emb = np.asarray(state["cutout_embeddings"], dtype=np.float32)

    for i, cid in enumerate(cutout_ids):
        cx_f = float(cutout_xy[i, 0])
        cy_f = float(cutout_xy[i, 1])
        cx: float | None = cx_f if not np.isnan(cx_f) else None
        cy: float | None = cy_f if not np.isnan(cy_f) else None
        cursor.execute(
            """
            INSERT OR REPLACE INTO cutout
            (cutout_id, keyframe_id, center_x, center_y, rotation, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(cid),
                int(cutout_kf[i]),
                cx,
                cy,
                cutout_rot[i].astype(np.float32).tobytes(),
                cutout_emb[i].astype(np.float32).tobytes(),
            ),
        )


def _write_clusters(
    cursor: sqlite3.Cursor,
    state: dict[str, np.ndarray],
) -> None:
    cluster_ids = state.get("object_cluster_ids")
    if (
        cluster_ids is None
        or np.asarray(cluster_ids).size == 0
        or "cluster_centroids_world" not in state
        or "cluster_centroids_geo" not in state
    ):
        return

    centroids_world = np.asarray(state["cluster_centroids_world"], dtype=np.float32)
    centroids_geo = np.asarray(state["cluster_centroids_geo"], dtype=np.float32)
    obs_counts = np.asarray(state["cluster_observation_counts"], dtype=np.int32).ravel()
    conf = np.asarray(state["cluster_confidence"], dtype=np.float32).ravel()
    levels = np.asarray(
        state.get("cluster_levels", np.array([], dtype=np.int32)), dtype=np.int32
    ).ravel()

    ocr_texts = state.get("cluster_ocr_texts")
    ocr_tokens = state.get("cluster_ocr_tokens")
    ocr_keys = state.get("cluster_ocr_keys")
    ocr_obs_counts = state.get("cluster_ocr_observation_counts")
    ocr_sources = state.get("cluster_ocr_source")

    for cidx in range(centroids_world.shape[0]):
        cursor.execute(
            """
            INSERT OR REPLACE INTO cluster
            (cluster_id, centroid_world, centroid_geo, observation_count,
             confidence, level,
             ocr_text, ocr_tokens, ocr_key, ocr_observation_count, ocr_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cidx,
                centroids_world[cidx].astype(np.float32).tobytes(),
                centroids_geo[cidx].astype(np.float32).tobytes(),
                int(obs_counts[cidx]),
                float(conf[cidx]),
                int(levels[cidx]) if levels.size > cidx else 0,
                str(ocr_texts[cidx]) if ocr_texts is not None else None,
                str(ocr_tokens[cidx]) if ocr_tokens is not None else None,
                str(ocr_keys[cidx]) if ocr_keys is not None else None,
                int(ocr_obs_counts[cidx]) if ocr_obs_counts is not None else None,
                int(ocr_sources[cidx]) if ocr_sources is not None else None,
            ),
        )


def _write_cluster_cutouts(
    cursor: sqlite3.Cursor,
    state: dict[str, np.ndarray],
) -> None:
    cursor.execute("DELETE FROM cluster_cutout")
    cc_cluster_ids = state.get("cluster_cutout_cluster_ids")
    if cc_cluster_ids is None:
        return

    cc_ids_arr = np.asarray(cc_cluster_ids, dtype=np.int32).ravel()
    cc_cut_ids = np.asarray(state["cluster_cutout_ids"], dtype=np.int64).ravel()
    cc_kf_ids = np.asarray(state["cluster_cutout_keyframe_ids"], dtype=np.int64).ravel()
    cc_levels = np.asarray(state["cluster_cutout_levels"], dtype=np.int32).ravel()
    cc_obs = np.asarray(
        state["cluster_cutout_observation_counts"], dtype=np.int32
    ).ravel()

    for idx in range(cc_ids_arr.shape[0]):
        cursor.execute(
            """
            INSERT OR REPLACE INTO cluster_cutout
            (cluster_id, cutout_id, keyframe_id, level, observation_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(cc_ids_arr[idx]),
                int(cc_cut_ids[idx]),
                int(cc_kf_ids[idx]),
                int(cc_levels[idx]),
                int(cc_obs[idx]),
            ),
        )


def _write_objects(
    cursor: sqlite3.Cursor,
    state: dict[str, np.ndarray],
) -> None:
    obj_kf = np.asarray(state["object_keyframe_ids"], dtype=np.int64).ravel()
    obj_cut = np.asarray(state["object_cutout_ids"], dtype=np.int64).ravel()
    obj_bb = np.asarray(state["object_bboxes"], dtype=np.float32).reshape(-1, 4)
    obj_emb = np.asarray(state["object_embeddings"], dtype=np.float32)

    obj_pos_keyframe = state.get("object_positions_keyframe")
    obj_pos_local = state.get("object_positions_local")
    obj_pos_world = state.get("object_positions_world")
    obj_depth = state.get("object_depths")
    obj_loc_valid = state.get("object_localization_valid")
    obj_cluster = state.get("object_cluster_ids")
    obj_level = state.get("object_detection_levels")
    obj_bbox_spherical = state.get("object_bbox_spherical")
    obj_visual_sim = state.get("object_visual_similarity_scores")
    obj_visual_cand = state.get("object_visual_candidate_mask")
    obj_visual_assign = state.get("object_visual_assigned_mask")
    obj_textness = state.get("object_textness_scores")
    obj_ocr_text = state.get("object_ocr_texts")
    obj_ocr_tokens = state.get("object_ocr_tokens")
    obj_ocr_key = state.get("object_ocr_keys")
    obj_ocr_cand = state.get("object_ocr_candidate_mask")
    obj_ocr_assign = state.get("object_ocr_assigned_mask")
    obj_ocr_src = state.get("object_ocr_source")
    obj_labels = state.get("object_labels")
    obj_sources = state.get("object_sources")

    for oidx in range(len(obj_kf)):
        pos_keyframe = (
            obj_pos_keyframe[oidx].astype(np.float32).tobytes()
            if obj_pos_keyframe is not None
            else None
        )
        pos_local = (
            obj_pos_local[oidx].astype(np.float32).tobytes()
            if obj_pos_local is not None
            else None
        )
        pos_world = (
            obj_pos_world[oidx].astype(np.float32).tobytes()
            if obj_pos_world is not None
            else None
        )
        depth = float(obj_depth[oidx]) if obj_depth is not None else None
        loc_valid = int(obj_loc_valid[oidx]) if obj_loc_valid is not None else None
        cluster_id = None
        if obj_cluster is not None:
            raw = int(obj_cluster[oidx])
            cluster_id = raw if raw >= 0 else None
        level = int(obj_level[oidx]) if obj_level is not None else None
        bbox_sph = (
            obj_bbox_spherical[oidx].astype(np.float32).tobytes()
            if obj_bbox_spherical is not None
            else None
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO object
            (object_idx, keyframe_id, cutout_id,
             bbox_coordinates, bbox_spherical_coordinates, embedding,
             position_keyframe, position_local, position_world, depth,
             localization_valid, cluster_id, level,
             visual_similarity_score, visual_candidate, visual_assigned,
             textness_score, ocr_text, ocr_tokens, ocr_key, ocr_candidate,
             ocr_assigned, ocr_source,
             label, detection_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                oidx,
                int(obj_kf[oidx]),
                int(obj_cut[oidx]),
                obj_bb[oidx].astype(np.float32).tobytes(),
                bbox_sph,
                obj_emb[oidx].astype(np.float32).tobytes(),
                pos_keyframe,
                pos_local,
                pos_world,
                depth,
                loc_valid,
                cluster_id,
                level,
                float(obj_visual_sim[oidx]) if obj_visual_sim is not None else None,
                int(obj_visual_cand[oidx]) if obj_visual_cand is not None else None,
                int(obj_visual_assign[oidx]) if obj_visual_assign is not None else None,
                float(obj_textness[oidx]) if obj_textness is not None else None,
                str(obj_ocr_text[oidx]) if obj_ocr_text is not None else None,
                str(obj_ocr_tokens[oidx]) if obj_ocr_tokens is not None else None,
                str(obj_ocr_key[oidx]) if obj_ocr_key is not None else None,
                int(obj_ocr_cand[oidx]) if obj_ocr_cand is not None else None,
                int(obj_ocr_assign[oidx]) if obj_ocr_assign is not None else None,
                int(obj_ocr_src[oidx]) if obj_ocr_src is not None else None,
                str(obj_labels[oidx]) if obj_labels is not None else None,
                str(obj_sources[oidx]) if obj_sources is not None else None,
            ),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_index_to_db(
    db_path: Path,
    metadata: ObjectSearchIndexMetadata,
    state: dict[str, np.ndarray],
) -> None:
    tmp_path = db_path.with_name(f".{db_path.name}.tmp.db")
    if tmp_path.exists():
        os.remove(tmp_path)

    conn = sqlite3.connect(str(tmp_path))
    try:
        ensure_object_search_index_schema(conn)
        cursor = conn.cursor()
        _write_params(cursor, metadata, state)
        _write_cutouts(cursor, state)
        _write_clusters(cursor, state)
        _write_cluster_cutouts(cursor, state)
        _write_objects(cursor, state)
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, db_path)


def load_index(index_path: Path, *, load_object_embeddings: bool = True) -> LoadedIndex:
    """Load the object-search index into memory.

    ``load_object_embeddings=False`` skips the (large) per-object embedding blob
    — used by the pgvector online path, which keeps embeddings in Postgres. This
    avoids both the multi-GB ``object_embeddings`` matrix and the ``fetchall()``
    blob spike (the column is dropped from the SELECT, not just discarded).
    """
    db_path = _resolve_index_db_path(Path(index_path))
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing object-search index database: {db_path}")

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_object_search_index_schema(conn)
        migrate_legacy_manifest_param_to_index_metadata(conn)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT value FROM params WHERE key = ?", (INDEX_METADATA_PARAM_KEY,)
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(
                f"Missing {INDEX_METADATA_PARAM_KEY!r} in params table (and no legacy "
                f"{LEGACY_MANIFEST_PARAM_KEY!r} to migrate)"
            )
        metadata = ObjectSearchIndexMetadata.from_json(row[0].decode("utf-8"))

        cursor.execute(
            "SELECT value FROM params WHERE key = ?", ("processed_cutout_ids",)
        )
        proc_row = cursor.fetchone()
        obj_processed = None
        if proc_row is not None and proc_row[0] is not None:
            obj_processed = np.frombuffer(proc_row[0], dtype=np.int64)

        cutouts = _load_cutouts(cursor, metadata)
        clusters = _load_clusters(cursor)
        objects = _load_objects(cursor, load_object_embeddings, metadata.projection_dim)

        return LoadedIndex(
            metadata=metadata,
            cutout_embeddings=cutouts.embeddings,
            cutout_ids=cutouts.ids,
            cutout_keyframe_ids=cutouts.keyframe_ids,
            cutout_center_xy=cutouts.center_xy,
            cutout_rotation_cutout_to_equirect=cutouts.rotation,
            object_embeddings=objects.embeddings,
            object_ids=objects.ids,
            object_keyframe_ids=objects.keyframe_ids,
            object_cutout_ids=objects.cutout_ids,
            object_bboxes=objects.bboxes,
            object_processed_cutout_ids=obj_processed,
            object_positions_keyframe=objects.pos_keyframe,
            object_positions_local=objects.pos_local,
            object_positions_world=objects.pos_world,
            object_depths=objects.depths,
            object_localization_valid=objects.loc_valid,
            object_cluster_ids=objects.cluster_ids,
            object_detection_levels=objects.detection_levels,
            object_bbox_spherical=objects.bbox_spherical,
            cluster_centroids_world=clusters.centroids_world,
            cluster_centroids_geo=clusters.centroids_geo,
            cluster_observation_counts=clusters.obs_counts,
            cluster_confidence=clusters.conf,
            cluster_levels=clusters.levels,
            cluster_cutout_cluster_ids=clusters.cutout_cluster_ids,
            cluster_cutout_ids=clusters.cutout_ids,
            cluster_cutout_keyframe_ids=clusters.cutout_keyframe_ids,
            cluster_cutout_levels=clusters.cutout_levels,
            cluster_cutout_observation_counts=clusters.cutout_obs_counts,
            object_visual_similarity_scores=objects.visual_sim,
            object_visual_candidate_mask=objects.visual_cand,
            object_visual_assigned_mask=objects.visual_assign,
            object_textness_scores=objects.textness,
            object_ocr_texts=objects.ocr_texts,
            object_ocr_tokens=objects.ocr_tokens,
            object_ocr_keys=objects.ocr_keys,
            object_ocr_candidate_mask=objects.ocr_cand,
            object_ocr_assigned_mask=objects.ocr_assign,
            object_ocr_source=objects.ocr_source,
            cluster_ocr_texts=clusters.ocr_texts,
            cluster_ocr_tokens=clusters.ocr_tokens,
            cluster_ocr_keys=clusters.ocr_keys,
            cluster_ocr_observation_counts=clusters.ocr_obs_counts,
            cluster_ocr_source=clusters.ocr_sources,
            object_labels=objects.labels,
            object_sources=objects.sources,
        )
    finally:
        conn.close()


def has_cutouts(index: LoadedIndex) -> bool:
    return index.cutout_embeddings.size > 0 and index.cutout_ids.size > 0


def has_objects(index: LoadedIndex) -> bool:
    # Object presence is determined by row metadata, not embeddings: in pgvector
    # mode the embedding matrix is intentionally empty (vectors live in Postgres).
    return index.object_cutout_ids.size > 0


def has_localizations(index: LoadedIndex) -> bool:
    """Check if 3D localization data is available in the index."""
    return (
        index.object_cluster_ids is not None
        and index.cluster_centroids_geo is not None
        and index.object_localization_valid is not None
    )
