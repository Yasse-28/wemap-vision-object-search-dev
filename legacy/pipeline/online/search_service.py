from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from PIL import Image

from pipeline.core import pgvector_store
from pipeline.core.io import (
    LoadedIndex,
    has_cutouts,
    has_localizations,
    has_objects,
    load_index,
)
from pipeline.core.logging import logger
from pipeline.core.models.metaclip import get_shared_metaclip
from pipeline.core.search_numpy import (
    compute_cosine_similarities,
    top_k_cutout,
    top_k_object_by_cutout,
)
from pipeline.core.types import UNRESOLVED_LEVEL_SENTINEL, ObjectSearchResult
from pipeline.online.localize_3d import (
    ClusterRanking,
    OnlineLocalizationError,
    OnlineLocalizationParams,
    cluster_ocr_scores_from_summaries,
    localize_online_matches,
    rank_localization_clusters,
)
from pipeline.online.observation_coordinates import stored_observation_coordinates
from pipeline.online.observation_orientation import observation_orientation_quaternions
from pipeline.online.postprocess import postprocess_results
from pipeline.online.request_models import (
    ObjectLocalizationResponse,
    ObjectLocation,
    ObjectObservation,
)
from pipeline.online.router import Router, RouteResult
from pipeline.online.ui_helpers import keyframe_metadata

if TYPE_CHECKING:
    from pipeline.core.models.metaclip import MetaCLIP


class IndexNotReadyError(RuntimeError):
    """Raised when the requested search mode has no embeddings in the index."""


def _prenormalize_embeddings(index: LoadedIndex) -> None:
    """L2-normalise object and cutout embedding matrices in place (once at load time).

    After this, cosine similarity reduces to a single dot product with no per-query
    norm computation or matrix copy.
    """
    for attr in ("object_embeddings", "cutout_embeddings"):
        E = getattr(index, attr, None)
        if E is None or E.size == 0:
            continue
        E = np.asarray(E, dtype=np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        setattr(index, attr, (E / norms).astype(np.float32))


def _merge_cutout_results_by_keyframe(
    results: list[tuple[str, float]],
    index: LoadedIndex,
    *,
    limit: int,
) -> list[tuple[str, float]]:
    """Map cutout-level ids to keyframe ids and keep the best score per keyframe."""
    if not results or limit <= 0:
        return []

    cutout_to_keyframe = {
        str(int(cutout_id)): str(int(keyframe_id))
        for cutout_id, keyframe_id in zip(
            index.cutout_ids.tolist(),
            index.cutout_keyframe_ids.tolist(),
        )
    }

    best_by_keyframe: dict[str, float] = {}
    for cutout_id, score in results:
        keyframe_id = cutout_to_keyframe.get(str(cutout_id))
        if keyframe_id is None:
            continue
        previous = best_by_keyframe.get(keyframe_id)
        if previous is None or score > previous:
            best_by_keyframe[keyframe_id] = score

    merged = sorted(
        best_by_keyframe.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return merged[:limit]


def _cluster_level(index: LoadedIndex, cluster_id: int) -> int | None:
    if index.cluster_levels is None or cluster_id >= len(index.cluster_levels):
        return None
    level = int(index.cluster_levels[cluster_id])
    return None if level == UNRESOLVED_LEVEL_SENTINEL else level


def _aggregate_localized_clusters(
    idx: LoadedIndex,
    sims: np.ndarray,
) -> tuple[dict[int, float], dict[int, set[str]], dict[int, list[tuple[int, float]]]]:
    """Group valid localized objects by cluster from a precomputed sim vector.

    Returns (best similarity, keyframe-id set, observations) per cluster id.
    """
    assert idx.object_cluster_ids is not None
    assert idx.object_localization_valid is not None
    cluster_best_sim: dict[int, float] = {}
    cluster_keyframes: dict[int, set[str]] = {}
    cluster_observations: dict[int, list[tuple[int, float]]] = {}
    for obj_idx in range(len(sims)):
        if not idx.object_localization_valid[obj_idx]:
            continue
        cluster_id = int(idx.object_cluster_ids[obj_idx])
        if cluster_id < 0:
            continue
        sim = float(sims[obj_idx])
        if cluster_id not in cluster_best_sim or sim > cluster_best_sim[cluster_id]:
            cluster_best_sim[cluster_id] = sim
        cluster_keyframes.setdefault(cluster_id, set()).add(
            str(int(idx.object_keyframe_ids[obj_idx]))
        )
        cluster_observations.setdefault(cluster_id, []).append((obj_idx, sim))
    return cluster_best_sim, cluster_keyframes, cluster_observations


def _build_localized_object_locations(
    *,
    ranked_clusters: list[ClusterRanking],
    idx: LoadedIndex,
    cluster_observations: dict[int, list[tuple[int, float]]],
    cluster_keyframes: dict[int, set[str]],
    coordinates_by_object_idx: dict[int, tuple[float, float, float]],
    orientation_by_object_idx: dict[int, list[float]],
    max_observations_per_cluster: int,
) -> list[ObjectLocation]:
    """Assemble an ObjectLocation response for each ranked, pre-localized cluster."""
    assert idx.cluster_centroids_geo is not None
    localizations: list[ObjectLocation] = []
    for ranked in ranked_clusters:
        cluster_id = ranked.cluster_id
        geo = idx.cluster_centroids_geo[cluster_id]
        coordinates = (float(geo[0]), float(geo[1]), float(geo[2]))
        confidence = (
            float(idx.cluster_confidence[cluster_id])
            if idx.cluster_confidence is not None
            else 0.0
        )
        obs_count = (
            int(idx.cluster_observation_counts[cluster_id])
            if idx.cluster_observation_counts is not None
            else len(cluster_observations.get(cluster_id, []))
        )
        keyframe_ids = sorted(cluster_keyframes.get(cluster_id, set()))

        raw_observations = cluster_observations.get(cluster_id, [])
        raw_observations.sort(key=lambda x: x[1], reverse=True)
        observations: list[ObjectObservation] = []
        for obj_idx, obj_sim in raw_observations[:max_observations_per_cluster]:
            observations.append(
                ObjectObservation(
                    object_idx=obj_idx,
                    cutout_id=str(int(idx.object_cutout_ids[obj_idx])),
                    keyframe_id=str(int(idx.object_keyframe_ids[obj_idx])),
                    coordinates=coordinates_by_object_idx.get(obj_idx),
                    bbox=tuple(float(v) for v in idx.object_bboxes[obj_idx].tolist()),
                    similarity_score=obj_sim,
                    quaternion=orientation_by_object_idx.get(obj_idx),
                )
            )

        localizations.append(
            ObjectLocation(
                coordinates=coordinates,
                level=_cluster_level(idx, cluster_id),
                confidence=confidence,
                observation_count=obs_count,
                similarity_score=ranked.similarity_score,
                match_score=ranked.match_score,
                keyframe_ids=keyframe_ids,
                observations=observations,
            )
        )
    return localizations


def _debug_keyframe_metadata(
    map_path: Path,
    localizations: list[ObjectLocation],
) -> dict[str, dict[str, float | str | None]]:
    keyframe_ids: set[str] = set()
    for loc in localizations:
        keyframe_ids.update(loc.keyframe_ids)
        keyframe_ids.update(obs.keyframe_id for obs in loc.observations)

    debug_keyframes: dict[str, dict[str, float | str | None]] = {}
    for keyframe_id in sorted(keyframe_ids):
        try:
            meta = keyframe_metadata(map_path, keyframe_id)
        except Exception as exc:
            logger.warning(
                "Could not resolve debug metadata for keyframe %s: %s", keyframe_id, exc
            )
            continue
        if meta is None:
            continue
        debug_keyframes[keyframe_id] = {
            "lat": meta["lat"],
            "lon": meta["lon"],
            "alt": meta["alt"],
            "level": meta["level"],
        }
    return debug_keyframes


def _heading_to_north_deg(
    *,
    origin_lat: float,
    origin_lon: float,
    target_lat: float,
    target_lon: float,
) -> float:
    """Initial bearing from origin to target, with north=0 and east=90."""
    lat1 = math.radians(float(origin_lat))
    lat2 = math.radians(float(target_lat))
    dlon = math.radians(float(target_lon) - float(origin_lon))
    east = math.sin(dlon) * math.cos(lat2)
    north = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(dlon)
    return float(math.degrees(math.atan2(east, north)))


def _attach_observation_headings(
    map_path: Path,
    localizations: list[ObjectLocation],
) -> None:
    metadata_cache: dict[str, dict | None] = {}
    for loc in localizations:
        for obs in loc.observations:
            if obs.coordinates is None:
                continue
            if obs.keyframe_id not in metadata_cache:
                try:
                    metadata_cache[obs.keyframe_id] = keyframe_metadata(
                        map_path,
                        obs.keyframe_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not resolve heading metadata for keyframe %s: %s",
                        obs.keyframe_id,
                        exc,
                    )
                    metadata_cache[obs.keyframe_id] = None
            meta = metadata_cache[obs.keyframe_id]
            if meta is None or meta.get("lat") is None or meta.get("lon") is None:
                continue
            obs.heading = _heading_to_north_deg(
                origin_lat=float(meta["lat"]),
                origin_lon=float(meta["lon"]),
                target_lat=float(obs.coordinates[0]),
                target_lon=float(obs.coordinates[1]),
            )


class FileBackedObjectSearchService:
    def __init__(
        self,
        index_path: Path,
        device: Optional[str] = None,
        map_id: Optional[str] = None,
        *,
        use_pgvector: bool = False,
        pgvector_db_location: str = "aws",
        ef_search: int = 1000,
    ):
        """Load search state from a map directory or a direct path to object-search.db.

        When ``use_pgvector`` (from config.json ``objectSearch``), the per-object
        embedding matrix is NOT loaded into RAM; object retrieval is served from
        the per-map pgvector table (``map_<id>_objects``, L2/halfvec) and only the
        light columns needed for clustering stay in memory. ``ef_search`` is the
        HNSW search-breadth knob (``hnsw.ef_search``).
        """
        self.index_path = Path(index_path)
        self.map_path = (
            self.index_path.parent if self.index_path.is_file() else self.index_path
        )
        self._device = device
        self._map_id = map_id or self.map_path.name

        self._pg_conn = None
        self._pg_table: Optional[str] = None
        self._pg_ef_search = max(1, int(ef_search))
        use_pg = bool(use_pgvector)

        self._index: LoadedIndex = load_index(
            self.index_path, load_object_embeddings=not use_pg
        )
        _prenormalize_embeddings(
            self._index
        )  # normalizes cutouts; object embeddings empty in pg mode
        if use_pg:
            self._pg_table = pgvector_store.table_name(self._map_id)
            self._pg_conn = pgvector_store.connect(
                pgvector_store.db_state_from_location(pgvector_db_location)
            )
            logger.info(
                "Object retrieval: pgvector table=%s db=%s ef_search=%d"
                " (embeddings not in RAM)",
                self._pg_table,
                pgvector_db_location,
                self._pg_ef_search,
            )

        self._metaclip = cast("MetaCLIP", get_shared_metaclip(self._device))
        self._router: Optional[Router] = None

    def reload_index(self) -> None:
        self._index = load_index(
            self.index_path, load_object_embeddings=self._pg_conn is None
        )
        _prenormalize_embeddings(self._index)

    def _get_metaclip(self) -> "MetaCLIP":
        return self._metaclip

    def _pgvector_similarities(
        self, query_embedding: np.ndarray, candidate_count: int
    ) -> np.ndarray:
        """ANN top-K from pgvector → full-length sparse similarity vector.

        K = the localization ``candidate_count``; ``hnsw.ef_search`` is the
        configured ``EFSearch`` (clamped to >= K, since pgvector requires
        ef_search >= LIMIT). Entries are -inf except at the retrieved candidates
        (object_idx mapped to row position via the sorted ``object_ids``), so
        ``localize_online_matches`` selects exactly the ANN candidate set.
        """
        object_ids = self._index.object_ids
        assert object_ids is not None, "loaded index must have object_ids"
        assert self._pg_table is not None, "pgvector mode requires _pg_table"
        k = max(1, int(candidate_count))
        pgvector_store.set_ef_search(self._pg_conn, max(self._pg_ef_search, k))
        ids, sims = pgvector_store.query_topk(
            self._pg_conn, self._pg_table, query_embedding, k
        )
        full = np.full(object_ids.shape[0], -1.0e9, dtype=np.float32)
        if ids:
            ids_arr = np.asarray(ids, dtype=np.int64)
            pos = np.searchsorted(object_ids, ids_arr)
            pos = np.clip(pos, 0, object_ids.size - 1)
            valid = object_ids[pos] == ids_arr
            full[pos[valid]] = sims[valid]
        return full

    @staticmethod
    def _pgvector_safe_params(
        params: OnlineLocalizationParams,
    ) -> OnlineLocalizationParams:
        """single_linkage clustering needs per-candidate embeddings, which are not
        in RAM in pgvector mode — fall back to leader_canopy."""
        if params.clustering_method == "single_linkage":
            logger.warning(
                "pgvector mode: single_linkage needs in-RAM embeddings; using"
                " leader_canopy",
            )
            return dataclasses.replace(params, clustering_method="leader_canopy")
        return params

    def _encode_text_query(self, text: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        model = self._get_metaclip()
        text_features = model.get_text_features(text)
        if text_features.dim() > 1:
            text_features = text_features[0]
        q = np.asarray(
            text_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        return q, time.perf_counter() - t0

    def _encode_image_query(self, image: Image.Image) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        model = self._get_metaclip()
        image_features = model.get_image_features([image])
        if image_features.dim() > 1:
            image_features = image_features[0]
        q = np.asarray(
            image_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        return q, time.perf_counter() - t0

    def _search_objects_localized_by_embedding(
        self,
        query_embedding: np.ndarray,
        *,
        num_results: int,
        max_observations_per_cluster: int,
        min_similarity: float,
        query_text_for_ocr: str | None,
        log_query: str,
        time_embedding: float,
    ) -> ObjectLocalizationResponse:
        idx = self._index
        if not has_objects(idx):
            raise IndexNotReadyError("Object embeddings missing or empty in index")
        if not has_localizations(idx):
            raise IndexNotReadyError("3D localization data missing in index")
        assert idx.object_cluster_ids is not None
        assert idx.object_localization_valid is not None
        assert idx.cluster_centroids_geo is not None

        t0 = time.perf_counter()
        sims = compute_cosine_similarities(query_embedding, idx.object_embeddings)

        cluster_best_sim, cluster_keyframes, cluster_observations = (
            _aggregate_localized_clusters(idx, sims)
        )

        cluster_confidence = {
            cluster_id: float(idx.cluster_confidence[cluster_id])
            for cluster_id in cluster_best_sim
            if idx.cluster_confidence is not None
            and cluster_id < idx.cluster_confidence.shape[0]
        }
        cluster_ocr_scores = (
            cluster_ocr_scores_from_summaries(
                query_text=query_text_for_ocr,
                cluster_ids=set(cluster_best_sim.keys()),
                cluster_ocr_keys=idx.cluster_ocr_keys,
                cluster_ocr_tokens=idx.cluster_ocr_tokens,
                cluster_ocr_texts=idx.cluster_ocr_texts,
            )
            if query_text_for_ocr
            else {}
        )
        ranked_clusters = rank_localization_clusters(
            cluster_best_sim=cluster_best_sim,
            cluster_confidence=cluster_confidence,
            cluster_keyframes=cluster_keyframes,
            cluster_ocr_scores=cluster_ocr_scores,
            min_similarity=min_similarity,
        )[:num_results]

        response_object_indices = [
            obj_idx
            for ranked in ranked_clusters
            for obj_idx, _ in sorted(
                cluster_observations.get(ranked.cluster_id, []),
                key=lambda x: x[1],
                reverse=True,
            )[:max_observations_per_cluster]
        ]
        orientation_by_object_idx = observation_orientation_quaternions(
            index=idx,
            map_path=self.map_path,
            object_indices=response_object_indices,
        )
        coordinates_by_object_idx = stored_observation_coordinates(
            index=idx,
            map_path=self.map_path,
            object_indices=response_object_indices,
        )
        localizations = _build_localized_object_locations(
            ranked_clusters=ranked_clusters,
            idx=idx,
            cluster_observations=cluster_observations,
            cluster_keyframes=cluster_keyframes,
            coordinates_by_object_idx=coordinates_by_object_idx,
            orientation_by_object_idx=orientation_by_object_idx,
            max_observations_per_cluster=max_observations_per_cluster,
        )
        time_retrieval = time.perf_counter() - t0
        _attach_observation_headings(self.map_path, localizations)

        logger.info(
            "object_search_localized query=%r clusters=%d",
            log_query,
            len(localizations),
        )
        return ObjectLocalizationResponse(
            localizations=localizations,
            time_embedding_ms=round(time_embedding * 1000),
            time_retrieval_ms=round(time_retrieval * 1000),
            debug={"keyframes": _debug_keyframe_metadata(self.map_path, localizations)},
        )

    def search(
        self, text: str, num_results: int, search_type: str = "cutout"
    ) -> ObjectSearchResult:
        t0 = time.perf_counter()
        model = self._get_metaclip()
        text_features = model.get_text_features(text)
        if text_features.dim() > 1:
            text_features = text_features[0]
        q = np.asarray(
            text_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        time_embedding = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = self._index
        if not has_objects(idx):
            raise IndexNotReadyError("Object embeddings missing or empty in index")
        sims_objects = compute_cosine_similarities(q, idx.object_embeddings)
        raw_objects = top_k_object_by_cutout(
            idx.object_cutout_ids, sims_objects, num_results
        )
        if not has_cutouts(idx):
            raise IndexNotReadyError("Cutout embeddings missing or empty in index")
        sims_cutouts = compute_cosine_similarities(q, idx.cutout_embeddings)
        raw_cutouts = top_k_cutout(idx.cutout_ids, sims_cutouts, num_results)
        time_retrieval = time.perf_counter() - t0
        results = _merge_cutout_results_by_keyframe(
            raw_objects + raw_cutouts,
            idx,
            limit=num_results,
        )

        return ObjectSearchResult(
            results=results,
            router_object_type=None,
            time_router_ms=0,
            time_embedding_ms=round(time_embedding * 1000),
            time_retrieval_ms=round(time_retrieval * 1000),
        )

    def search_with_router(
        self, text: str, num_results: int, search_type: str = "cutout"
    ) -> ObjectSearchResult:
        if self._router is None:
            self._router = Router()
        t0 = time.perf_counter()
        if search_type == "auto":
            route = self._router.route(text)
        else:
            route = RouteResult(text=text, type=search_type, canonical_object=text)
        time_router = time.perf_counter() - t0

        t0 = time.perf_counter()
        model = self._get_metaclip()
        text_features = model.get_text_features(route.text)
        if text_features.dim() > 1:
            text_features = text_features[0]
        q = np.asarray(
            text_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        time_embedding = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx = self._index
        if route.type == "object":
            if not has_objects(idx):
                raise IndexNotReadyError("Object embeddings missing or empty in index")
            sims = compute_cosine_similarities(q, idx.object_embeddings)
            raw = top_k_object_by_cutout(idx.object_cutout_ids, sims, num_results)
        else:
            if not has_cutouts(idx):
                raise IndexNotReadyError("Cutout embeddings missing or empty in index")
            sims = compute_cosine_similarities(q, idx.cutout_embeddings)
            raw = top_k_cutout(idx.cutout_ids, sims, num_results)
        time_retrieval = time.perf_counter() - t0

        results = postprocess_results(
            raw,
            index=idx,
            route_type=route.type,
            query_text=text,
        )
        router_object_type = route.type if search_type == "auto" else None
        logger.info(
            "object_search text=%r route=%s results=%d",
            text,
            route.type,
            len(results),
        )
        return ObjectSearchResult(
            results=results,
            router_object_type=router_object_type,
            time_router_ms=round(time_router * 1000),
            time_embedding_ms=round(time_embedding * 1000),
            time_retrieval_ms=round(time_retrieval * 1000),
        )

    def search_objects_localized(
        self,
        text: str,
        num_results: int,
        max_observations_per_cluster: int = 10,
        min_similarity: float = 0.2,
    ) -> ObjectLocalizationResponse:
        """
        Search objects by text and return their 3D geographic locations.

        Returns unique clusters rather than individual detections.
        Groups object detections by their cluster assignment and returns
        the geographic coordinates of each cluster's centroid, along with
        individual observation details for rendering crops.
        """
        q, time_embedding = self._encode_text_query(text)
        return self._search_objects_localized_by_embedding(
            q,
            num_results=num_results,
            max_observations_per_cluster=max_observations_per_cluster,
            min_similarity=min_similarity,
            query_text_for_ocr=text,
            log_query=text,
            time_embedding=time_embedding,
        )

    def search_objects_localized_by_image(
        self,
        image: Image.Image,
        num_results: int,
        max_observations_per_cluster: int = 10,
        min_similarity: float = 0.2,
        *,
        log_query: str = "[image]",
    ) -> ObjectLocalizationResponse:
        """
        Search objects by image and return their 3D geographic locations.

        Image queries use only embedding similarity and spatial confidence; OCR
        text boosting is intentionally skipped because there is no text query.
        """
        q, time_embedding = self._encode_image_query(image)
        return self._search_objects_localized_by_embedding(
            q,
            num_results=num_results,
            max_observations_per_cluster=max_observations_per_cluster,
            min_similarity=min_similarity,
            query_text_for_ocr=None,
            log_query=log_query,
            time_embedding=time_embedding,
        )

    def search_objects_localized_online(
        self,
        text: str,
        num_results: int,
        *,
        localization_params: OnlineLocalizationParams,
        include_debug: bool = True,
    ) -> ObjectLocalizationResponse:
        """Search object embeddings and localize the best matches at request time."""
        t0 = time.perf_counter()
        model = self._get_metaclip()
        text_features = model.get_text_features(text)
        if text_features.dim() > 1:
            text_features = text_features[0]
        q = np.asarray(
            text_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        time_embedding = time.perf_counter() - t0

        return self._search_objects_localized_online_by_embedding(
            query_embedding=q,
            query_text=text,
            log_query=text,
            num_results=num_results,
            localization_params=localization_params,
            time_embedding=time_embedding,
            include_debug=include_debug,
        )

    def search_objects_localized_online_by_image(
        self,
        image: "Image.Image",
        num_results: int,
        *,
        localization_params: OnlineLocalizationParams,
        log_query: str = "[image]",
        include_debug: bool = True,
    ) -> ObjectLocalizationResponse:
        """Same as `search_objects_localized_online` but with an image query."""
        t0 = time.perf_counter()
        model = self._get_metaclip()
        image_features = model.get_image_features([image])
        if image_features.dim() > 1:
            image_features = image_features[0]
        q = np.asarray(
            image_features.detach().float().cpu().numpy().ravel(), dtype=np.float32
        )
        time_embedding = time.perf_counter() - t0

        return self._search_objects_localized_online_by_embedding(
            query_embedding=q,
            query_text="",  # No usable text for OCR scoring on image queries.
            log_query=log_query,
            num_results=num_results,
            localization_params=localization_params,
            time_embedding=time_embedding,
            include_debug=include_debug,
        )

    def _search_objects_localized_online_by_embedding(
        self,
        *,
        query_embedding: np.ndarray,
        query_text: str,
        log_query: str,
        num_results: int,
        localization_params: OnlineLocalizationParams,
        time_embedding: float,
        include_debug: bool = True,
    ) -> ObjectLocalizationResponse:
        idx = self._index
        if not has_objects(idx):
            raise IndexNotReadyError("Object embeddings missing or empty in index")

        t0 = time.perf_counter()
        if self._pg_conn is not None:
            sims = self._pgvector_similarities(
                query_embedding, localization_params.candidate_count
            )
            localization_params = self._pgvector_safe_params(localization_params)
        else:
            sims = compute_cosine_similarities(query_embedding, idx.object_embeddings)
        t_sims = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            localizations = localize_online_matches(
                index=idx,
                map_path=self.map_path,
                query_text=query_text,
                similarities=sims,
                params=localization_params,
            )
        except OnlineLocalizationError as e:
            raise IndexNotReadyError(str(e)) from e
        localizations = localizations[:num_results]
        t_localize = time.perf_counter() - t1
        _attach_observation_headings(self.map_path, localizations)

        t2 = time.perf_counter()
        debug_keyframes = (
            _debug_keyframe_metadata(self.map_path, localizations)
            if include_debug
            else {}
        )
        t_debug = time.perf_counter() - t2

        time_retrieval = t_sims + t_localize + t_debug

        logger.info(
            "localize_online query=%r clusters=%d | sims=%.0fms"
            " localize=%.0fms debug=%.0fms total=%.0fms",
            log_query,
            len(localizations),
            t_sims * 1000,
            t_localize * 1000,
            t_debug * 1000,
            time_retrieval * 1000,
        )
        return ObjectLocalizationResponse(
            localizations=localizations,
            time_embedding_ms=round(time_embedding * 1000),
            time_retrieval_ms=round(time_retrieval * 1000),
            debug={"keyframes": debug_keyframes},
        )
