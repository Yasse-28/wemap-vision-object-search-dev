"""
3D localization for object detections from cubemap images.

Projects 2D bounding box detections to 3D using depth maps, transforms to
world coordinates, and clusters multi-view observations into unique object
instances.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pipeline.core.logging import logger
from pipeline.core.types import UNRESOLVED_LEVEL_SENTINEL
from pipeline.offline.localize.depth_io import load_depth_from_zarr
from pipeline.offline.localize.georef import (
    GeoRef,
    load_georef_from_db,
    load_image_filename_to_keyframe_id,
)
from pipeline.offline.shared.geometry import (
    CUBEMAP_FACE_ORDER,
    relative_pose_cubemap_face,
)


@dataclass
class Detection3D:
    """A single detection with 3D position information."""

    detection_idx: int
    keyframe_id: int
    cutout_id: int
    face_name: str
    bbox: np.ndarray  # [x0, y0, x1, y1]
    depth: float  # sampled depth in meters
    position_keyframe: np.ndarray  # [x, y, z] in keyframe local frame
    position_local: np.ndarray  # [x, y, z] in WDS local-map frame
    geo_coords: (
        Tuple[float, float, float] | None
    )  # (lat, lon, alt) or None if no georef


@dataclass
class LocalizationResult:
    """Results of 3D localization for all detections."""

    # Per-detection data (aligned with object detection arrays)
    positions_keyframe: np.ndarray  # [N, 3] XYZ in keyframe local frame
    positions_local: np.ndarray  # [N, 3] XYZ in WDS local-map frame
    depths: np.ndarray  # [N] sampled depth values
    valid_mask: np.ndarray  # [N] bool, True if localization succeeded

    # Clustering results
    cluster_ids: np.ndarray  # [N] int, cluster assignment (-1 = noise/invalid)
    cluster_centroids_world: np.ndarray  # [K, 3] centroid XYZ per cluster
    cluster_centroids_geo: np.ndarray  # [K, 3] [lat, lon, alt] per cluster
    cluster_observation_counts: np.ndarray  # [K] number of detections per cluster
    cluster_confidence: np.ndarray  # [K] confidence score per cluster
    object_visual_similarity_scores: np.ndarray | None = (
        None  # [N] optional DINO similarity score
    )
    object_visual_candidate_mask: np.ndarray | None = (
        None  # [N] visual refinement candidate mask
    )
    object_visual_assigned_mask: np.ndarray | None = (
        None  # [N] assigned by visual refinement
    )
    object_textness_scores: np.ndarray | None = None  # [N] optional OCR textness score
    object_ocr_texts: np.ndarray | None = None  # [N] optional OCR raw text
    object_ocr_tokens: np.ndarray | None = (
        None  # [N] normalized OCR tokens, space-separated
    )
    object_ocr_keys: np.ndarray | None = (
        None  # [N] optional normalized OCR identity key
    )
    object_ocr_candidate_mask: np.ndarray | None = None  # [N] OCR candidate mask
    object_ocr_assigned_mask: np.ndarray | None = None  # [N] assigned by OCR refinement
    object_ocr_source: np.ndarray | None = (
        None  # [N] OCR source enum (0 none, 1 lightweight, 2 PaddleOCR-VL)
    )
    cluster_ocr_texts: np.ndarray | None = (
        None  # [K] representative OCR text per cluster
    )
    cluster_ocr_tokens: np.ndarray | None = (
        None  # [K] representative normalized OCR tokens
    )
    cluster_ocr_keys: np.ndarray | None = (
        None  # [K] representative normalized OCR key per cluster
    )
    cluster_ocr_observation_counts: np.ndarray | None = (
        None  # [K] observations supporting cluster OCR key
    )
    cluster_ocr_source: np.ndarray | None = (
        None  # [K] OCR source enum/priority for representative key
    )
    cluster_levels: np.ndarray | None = None  # [K] int32 level id per cluster
    detection_levels: np.ndarray | None = None  # [N] int32 level id per detection


def cubemap_pixel_to_ray(
    u: float,
    v: float,
    face_name: str,
    face_size: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """
    Convert a cubemap face pixel to a 3D ray direction in panorama frame.

    Args:
        u, v: Pixel coordinates in cubemap face (can be float for bbox centers)
        face_name: One of 'front', 'back', 'left', 'right', 'top', 'bottom'
        face_size: Size of the cubemap face in pixels
        fov_deg: Field of view in degrees

    Returns:
        Unit ray direction [x, y, z] in panorama/keyframe local frame
    """
    # Compute focal length and principal point
    f = 0.5 * face_size / np.tan(0.5 * np.radians(fov_deg))
    cx = cy = (face_size - 1) / 2.0

    # Pixel to normalized camera coords (looking along +Z)
    x = (u - cx) / f
    y = (v - cy) / f
    z = 1.0
    ray_cam = np.array([x, y, z], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    # Get face rotation matrix
    # relative_pose_cubemap_face returns a 4x4 matrix where R[:3,:3] = R.T
    # (world-to-camera style), so we need to transpose to get camera-to-world
    R_face = relative_pose_cubemap_face(face_name)[:3, :3]
    # The rotation stored is R.T, so R_face is already camera-to-world
    ray_pano = R_face.T @ ray_cam

    return ray_pano


def cubemap_pixel_to_equirect_uv(
    u: float,
    v: float,
    face_name: str,
    face_size: int,
    fov_deg: float = 90.0,
) -> Tuple[float, float]:
    """
    Convert cubemap face pixel to equirectangular UV coordinates.

    Args:
        u, v: Pixel coordinates in cubemap face
        face_name: Cubemap face name
        face_size: Size of cubemap face in pixels
        fov_deg: Field of view in degrees

    Returns:
        (u_eq, v_eq): Normalized UV in [0, 1] for equirectangular image
    """
    ray = cubemap_pixel_to_ray(u, v, face_name, face_size, fov_deg)

    # Convert ray direction to spherical coordinates
    x, y, z = ray
    lon = np.arctan2(x, z)  # azimuth
    lat = np.arcsin(np.clip(y, -1.0, 1.0))  # elevation

    # Convert to UV (matching equirectangular_projector.py convention)
    u_eq = (lon / (2 * np.pi) + 0.5) % 1.0
    v_eq = lat / np.pi + 0.5

    return u_eq, v_eq


def sample_depth_at_pixel(
    depth_map: np.ndarray,
    u_eq: float,
    v_eq: float,
    sample_radius: int = 3,
) -> float:
    """
    Sample depth at equirectangular UV coordinates with robust median filtering.

    Args:
        depth_map: Equirectangular depth map [H, W] in meters
        u_eq, v_eq: Normalized UV coordinates in [0, 1]
        sample_radius: Radius for median sampling (pixels)

    Returns:
        Depth value in meters, or np.nan if invalid
    """
    h, w = depth_map.shape

    # Convert UV to pixel coordinates
    px = int(u_eq * (w - 1))
    py = int(v_eq * (h - 1))

    # Clamp to valid range
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    if sample_radius <= 0:
        # Single pixel sample
        d = depth_map[py, px]
        return d if np.isfinite(d) else np.nan

    # Sample a region and take median
    y0 = max(0, py - sample_radius)
    y1 = min(h, py + sample_radius + 1)
    x0 = max(0, px - sample_radius)
    x1 = min(w, px + sample_radius + 1)

    region = depth_map[y0:y1, x0:x1]
    valid = region[np.isfinite(region)]

    if len(valid) < 3:
        return np.nan

    return float(np.median(valid))


def detection_to_3d_local(
    bbox: np.ndarray,
    face_name: str,
    depth_map: np.ndarray,
    face_size: int,
    fov_deg: float = 90.0,
    min_depth_m: float = 0.5,
    max_depth_m: float = 50.0,
    sample_radius: int = 3,
) -> Tuple[np.ndarray | None, float]:
    """
    Project a 2D detection to 3D in keyframe local frame.

    Args:
        bbox: Bounding box [x0, y0, x1, y1] in cubemap face pixels
        face_name: Cubemap face name
        depth_map: Equirectangular depth map [H, W] in meters
        face_size: Size of cubemap face in pixels
        fov_deg: Field of view in degrees
        min_depth_m: Minimum valid depth in meters
        max_depth_m: Maximum valid depth in meters
        sample_radius: Radius for depth sampling

    Returns:
        (position_3d, depth): 3D position [x,y,z] and sampled depth, or
        (None, nan) if invalid
    """
    # Bbox center in cubemap face
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    # Get equirectangular UV
    u_eq, v_eq = cubemap_pixel_to_equirect_uv(cx, cy, face_name, face_size, fov_deg)

    # Sample depth
    depth = sample_depth_at_pixel(depth_map, u_eq, v_eq, sample_radius)

    # Validate depth
    if not np.isfinite(depth) or depth < min_depth_m or depth > max_depth_m:
        return None, np.nan

    # Compute 3D ray direction
    ray = cubemap_pixel_to_ray(cx, cy, face_name, face_size, fov_deg)

    # 3D position = depth * ray_direction
    position_3d = depth * ray

    return position_3d, depth


def extract_keyframe_ids_from_image_paths(
    image_paths: List[Path],
    *,
    image_filename_to_keyframe_id: dict[str, int] | None = None,
) -> List[int]:
    """
    Extract keyframe IDs from image paths.

    Uses ``image_filename`` mapping when provided; otherwise ``int(path.stem)``.
    """
    from pipeline.offline.ingest.keyframe_id import keyframe_id_from_image_path

    keyframe_ids: list[int] = []
    for image_path in image_paths:
        keyframe_id = keyframe_id_from_image_path(
            image_path,
            image_filename_to_keyframe_id=image_filename_to_keyframe_id,
        )
        if keyframe_id is not None:
            keyframe_ids.append(keyframe_id)

    return keyframe_ids


def load_keyframe_poses_from_georef(
    georef_db_path: Path,
    keyframe_ids: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Load keyframe poses from georef.db.

    Args:
        georef_db_path: Path to georef.db
        keyframe_ids: Optional list of keyframe IDs to load poses for
    Returns:
        Dictionary mapping keyframe_id to 4x4 pose matrix (camera_pose)
    """
    poses: dict[int, np.ndarray] = {}

    with sqlite3.connect(georef_db_path) as conn:
        cursor = conn.cursor()
        if keyframe_ids is None:
            stmt = "SELECT id, pose FROM GeoRefKeyframe ORDER BY id ASC"
            rows = cursor.execute(stmt)
        elif not keyframe_ids:
            return poses
        else:
            stmt = "SELECT id, pose FROM GeoRefKeyframe WHERE id IN ({seq})".format(
                seq=",".join(["?"] * len(keyframe_ids))
            )
            rows = cursor.execute(stmt, keyframe_ids)

        for row in rows:
            kf_id = row[0]
            pose = (
                np.frombuffer(row[1], dtype=np.dtype(float).newbyteorder("<"))
                .reshape([4, 4])
                .T
            )
            poses[kf_id] = pose

    return poses


def transform_to_world(
    position_keyframe: np.ndarray,
    keyframe_pose: np.ndarray,
) -> np.ndarray:
    """
    Transform a 3D point from keyframe local frame to world frame.

    The keyframe_pose is the camera_pose (world-to-camera transform).
    To get world position, we need the inverse (camera-to-world).

    Args:
        position_keyframe: [x, y, z] in keyframe/camera local frame
        keyframe_pose: 4x4 camera_pose matrix (world-to-camera)

    Returns:
        [x, y, z] in world frame
    """
    # camera_pose is world-to-camera, so inverse gives camera-to-world
    local_pose = np.linalg.inv(keyframe_pose)

    # Transform point
    p_homo = np.array([*position_keyframe, 1.0])
    p_world = local_pose @ p_homo

    return np.asarray(p_world[:3])


def world_to_geo(
    position_local: np.ndarray,
    georef: GeoRef,
) -> Tuple[float, float, float]:
    """
    Convert world frame position to geographic coordinates.

    Args:
        position_local: [x, y, z] in local world frame (map origin)
        georef: GeoRef object with origin coordinates

    Returns:
        (latitude_deg, longitude_deg, altitude_m)
    """
    geopose = georef.local_position_to_world(position_local)

    lat = geopose.position.get_latitude_deg()
    lon = geopose.position.get_longitude_deg()
    alt = geopose.position.get_altitude()
    return (lat, lon, alt)


def build_level_keyframe_allowlist(
    level_filter: int,
    keyframe_poses: Dict[int, np.ndarray],
    georef: GeoRef,
) -> "set[int] | None":
    """
    Return the set of keyframe IDs whose camera position falls within level_filter.

    Returns None if levels are unavailable or level_filter id is not found.
    """
    target = georef.get_level(level_filter)
    if target is None:
        return None

    allowed: set[int] = set()
    for kf_id, pose in keyframe_poses.items():
        pos_world = np.linalg.inv(pose)[:3, 3]
        try:
            geopose = georef.local_position_to_world(pos_world)
        except Exception:
            continue
        coords = geopose.position
        if coords.heightFromGround is not None and target.contains(
            coords.heightFromGround,
            coords.get_latitude_longitude_deg(),
        ):
            allowed.add(int(kf_id))

    return allowed


def keyframe_levels_from_poses(
    keyframe_poses: Dict[int, np.ndarray],
    georef: GeoRef,
) -> dict[int, int]:
    """Resolve one indoor level per keyframe from the camera pose."""
    levels: dict[int, int] = {}
    for kf_id, pose in keyframe_poses.items():
        pos_world = np.linalg.inv(pose)[:3, 3]
        try:
            _, _, _, level = world_to_geo_with_level(
                pos_world.astype(np.float64), georef
            )
        except Exception:
            continue
        if level is not None:
            levels[int(kf_id)] = int(level)
    return levels


def world_to_geo_with_level(
    position_local: np.ndarray,
    georef: GeoRef,
) -> Tuple[float, float, float, "int | None"]:
    """
    Convert world frame position to geographic coordinates, including level.

    Returns:
        (latitude_deg, longitude_deg, altitude_m, level_id)
        level_id is None if no level was matched.
    """
    geopose = georef.local_position_to_world(position_local)
    lat = geopose.position.get_latitude_deg()
    lon = geopose.position.get_longitude_deg()
    alt = geopose.position.get_altitude()
    level = geopose.position.get_level()
    return (lat, lon, alt, level)


def _format_level_resolution_debug(
    *,
    georef: GeoRef | None,
    position_local: np.ndarray,
) -> str:
    if georef is None:
        return "georef unavailable"
    try:
        geopose = georef.local_position_to_world(
            np.asarray(position_local, dtype=np.float64)
        )
    except Exception as exc:
        return f"failed to project world position to geopose: {exc}"

    coords = geopose.position
    lat, lon = coords.get_latitude_longitude_deg()
    alt = float(coords.get_altitude())
    height_from_ground = (
        float(coords.heightFromGround)
        if coords.heightFromGround is not None
        else float(alt - georef.origin_coordinates.altitude)
    )
    parts = [
        f"lat={lat:.7f}",
        f"lon={lon:.7f}",
        f"alt={alt:.3f}",
        f"height_from_ground={height_from_ground:.3f}",
        f"origin_alt={float(georef.origin_coordinates.altitude):.3f}",
    ]
    if not georef.levels:
        parts.append("levels=[]")
        return ", ".join(parts)

    lat_lon = (lat, lon)
    level_parts: list[str] = []
    for level in georef.levels:
        altitude_ok = (
            float(level.floorHeightFromGround)
            <= height_from_ground
            <= float(level.ceilingHeightFromGround)
        )
        polygon_ok = True
        if level.geometry is not None:
            polygon_ok = bool(level.contains(height_from_ground, lat_lon))
            if altitude_ok:
                polygon_ok = bool(level.contains(height_from_ground, lat_lon))
            else:
                polygon_ok = False
        level_parts.append(
            "level={id}[alt_range={lo:.3f}..{hi:.3f},alt_ok={alt_ok},polygon={polygon}]".format(
                id=int(level.id),
                lo=float(level.floorHeightFromGround),
                hi=float(level.ceilingHeightFromGround),
                alt_ok=altitude_ok,
                polygon=("n/a" if level.geometry is None else polygon_ok),
            )
        )
    parts.append("level_checks=" + "; ".join(level_parts))
    return ", ".join(parts)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.asarray(
        np.divide(
            embeddings,
            np.clip(norms, a_min=1e-12, a_max=None),
        ).astype(np.float32)
    )


def _connected_components_from_edges(
    num_nodes: int,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    labels = np.full(num_nodes, -1, dtype=np.int32)
    next_label = 0
    for node_idx in range(num_nodes):
        if labels[node_idx] != -1:
            continue
        stack = [node_idx]
        labels[node_idx] = next_label
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] != -1:
                    continue
                labels[neighbor] = next_label
                stack.append(neighbor)
        next_label += 1
    return labels


def _find_parent(parent: np.ndarray, idx: int) -> int:
    root = int(idx)
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[idx]) != root:
        next_idx = int(parent[idx])
        parent[idx] = root
        idx = next_idx
    return root


def _union_parent(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    left_root = _find_parent(parent, left)
    right_root = _find_parent(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        parent[left_root] = right_root
    elif rank[left_root] > rank[right_root]:
        parent[right_root] = left_root
    else:
        parent[right_root] = left_root
        rank[left_root] += 1


def _labels_from_union_find(parent: np.ndarray) -> np.ndarray:
    root_to_label: dict[int, int] = {}
    labels = np.full(parent.shape[0], -1, dtype=np.int32)
    for idx in range(parent.shape[0]):
        root = _find_parent(parent, idx)
        label = root_to_label.setdefault(root, len(root_to_label))
        labels[idx] = int(label)
    return labels


def _relabel_clusters_compact(cluster_ids: np.ndarray) -> np.ndarray:
    relabeled = np.full(cluster_ids.shape, -1, dtype=np.int32)
    unique_labels = np.unique(cluster_ids)
    unique_labels = unique_labels[unique_labels >= 0]
    for new_label, old_label in enumerate(unique_labels.tolist()):
        relabeled[cluster_ids == old_label] = new_label
    return relabeled


def filter_clusters_by_min_keyframes(
    cluster_ids: np.ndarray,
    keyframe_ids: np.ndarray,
    *,
    min_keyframes: int = 3,
) -> np.ndarray:
    if min_keyframes <= 1:
        return _relabel_clusters_compact(cluster_ids)

    filtered = cluster_ids.copy()
    for cluster_id in np.unique(cluster_ids):
        if cluster_id < 0:
            continue
        cluster_keyframes = np.unique(keyframe_ids[cluster_ids == cluster_id])
        if cluster_keyframes.size < min_keyframes:
            filtered[cluster_ids == cluster_id] = -1
    return _relabel_clusters_compact(filtered)


def _should_connect_detections(
    *,
    source_idx: int,
    target_idx: int,
    valid_indices: np.ndarray,
    detection_levels: np.ndarray | None,
    normalized_embeddings: np.ndarray | None,
    embedding_similarity_threshold: float,
) -> bool:
    """Whether two spatially-neighboring detections may join the same cluster.

    Rejects pairs on different resolved levels, and pairs whose embeddings fall
    below the semantic similarity threshold.
    """
    if detection_levels is not None:
        level_a = int(detection_levels[valid_indices[source_idx]])
        level_b = int(detection_levels[valid_indices[target_idx]])
        if (
            level_a != UNRESOLVED_LEVEL_SENTINEL
            and level_b != UNRESOLVED_LEVEL_SENTINEL
            and level_a != level_b
        ):
            return False
    if normalized_embeddings is not None:
        similarity = float(
            normalized_embeddings[source_idx] @ normalized_embeddings[target_idx]
        )
        if similarity < embedding_similarity_threshold:
            return False
    return True


def cluster_detections_3d(
    positions_local: np.ndarray,
    valid_mask: np.ndarray,
    object_keyframe_ids: np.ndarray,
    eps_meters: float = 1.5,
    embeddings: np.ndarray | None = None,
    embedding_similarity_threshold: float = 0.85,
    min_keyframes_per_cluster: int = 3,
    detection_levels: np.ndarray | None = None,
    neighbor_chunk_size: int = 4096,
) -> np.ndarray:
    """
    Cluster 3D detections using joint spatial and semantic connectivity.

    Args:
        positions_local: [N, 3] WDS local-map positions
        valid_mask: [N] bool mask of valid positions
        object_keyframe_ids: [N] keyframe ID for each detection
        eps_meters: Spatial radius in meters for candidate neighbors
        embeddings: Optional [N, D] object embeddings used for semantic gating
        embedding_similarity_threshold: Minimum cosine similarity to connect
            two spatial neighbors.
        min_keyframes_per_cluster: Minimum number of distinct keyframes required
            for a cluster to be kept.

    Returns:
        [N] compact cluster labels (-1 for noise/invalid)
    """
    labels = np.full(len(positions_local), -1, dtype=np.int32)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return labels

    valid_positions = positions_local[valid_indices]
    neighbor_search = NearestNeighbors(radius=eps_meters)
    neighbor_search.fit(valid_positions)
    normalized_embeddings = None
    if embeddings is not None:
        normalized_embeddings = _normalize_embeddings(
            np.asarray(embeddings[valid_indices], dtype=np.float32)
        )

    n_valid = int(len(valid_indices))
    parent = np.arange(n_valid, dtype=np.int32)
    rank = np.zeros(n_valid, dtype=np.int8)
    chunk_size = max(1, int(neighbor_chunk_size))

    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        neighbor_indices = neighbor_search.radius_neighbors(
            valid_positions[start:end],
            return_distance=False,
        )
        for local_source_idx, neighbors in enumerate(neighbor_indices):
            source_idx = start + local_source_idx
            for target_idx in neighbors.tolist():
                if target_idx <= source_idx:
                    continue
                if not _should_connect_detections(
                    source_idx=source_idx,
                    target_idx=int(target_idx),
                    valid_indices=valid_indices,
                    detection_levels=detection_levels,
                    normalized_embeddings=normalized_embeddings,
                    embedding_similarity_threshold=embedding_similarity_threshold,
                ):
                    continue
                _union_parent(parent, rank, source_idx, int(target_idx))

    valid_labels = _labels_from_union_find(parent)
    labels[valid_indices] = valid_labels
    labels = filter_clusters_by_min_keyframes(
        labels,
        object_keyframe_ids,
        min_keyframes=min_keyframes_per_cluster,
    )
    return labels


def _median_member_level(
    member_levels: np.ndarray,
    member_keyframe_ids: np.ndarray | None,
    level_height_rank: dict[int, int],
) -> int:
    """Median floor level across a cluster's distinct contributing keyframes.

    Each object's level is its keyframe-pose-derived floor (depth-independent),
    so we take one level per distinct keyframe — a keyframe contributing several
    observations to the cluster must not be over-counted — drop unresolved
    levels, and return the lower-median.

    The median is taken over *physical floor height* (``level_height_rank``,
    derived from each Level's ``floorHeightFromGround``) because ``Level.id`` is
    a database id and is not guaranteed to increase with height. When the height
    ordering is unavailable (no georef levels, or a level id not present in it)
    we fall back to sorting by id. Returns ``UNRESOLVED_LEVEL_SENTINEL`` if no
    contributing keyframe had a resolved level (the caller then tries the georef
    centroid lookup).
    """
    levels = np.asarray(member_levels, dtype=np.int64)
    if member_keyframe_ids is not None:
        seen: dict[int, int] = {}
        for kf, lv in zip(np.asarray(member_keyframe_ids).tolist(), levels.tolist()):
            if lv != UNRESOLVED_LEVEL_SENTINEL and int(kf) not in seen:
                seen[int(kf)] = int(lv)
        resolved = list(seen.values())
    else:
        resolved = [
            int(lv) for lv in levels.tolist() if lv != UNRESOLVED_LEVEL_SENTINEL
        ]

    if not resolved:
        return UNRESOLVED_LEVEL_SENTINEL

    if level_height_rank and all(lid in level_height_rank for lid in resolved):
        resolved.sort(key=lambda lid: level_height_rank[lid])
    else:
        resolved.sort()
    # Lower median → always an actual level present among the keyframes (no
    # interpolated half-floor for an even count).
    return int(resolved[(len(resolved) - 1) // 2])


def _cluster_confidence(cluster_positions: np.ndarray, observation_count: int) -> float:
    """Confidence from observation count and spatial spread (higher = better)."""
    spread = (
        np.std(cluster_positions, axis=0).mean()
        if len(cluster_positions) > 1
        else 0.0
    )
    count_factor = min(1.0, observation_count / 5.0)  # Saturates at 5 observations
    spread_factor = max(0.0, 1.0 - spread / 2.0)  # Penalize spread > 2m
    return float(count_factor * (0.5 + 0.5 * spread_factor))


def _cluster_level(
    *,
    mask: np.ndarray,
    detection_levels: np.ndarray,
    detection_confidences: np.ndarray | None,
    detection_keyframe_ids: np.ndarray | None,
    level_height_rank: dict[int, int],
) -> int:
    """Pick a cluster's level: the seed (highest-confidence) detection's level,
    falling back to the per-keyframe median when no confidences are given."""
    if detection_confidences is not None:
        seed_local = int(np.argmax(detection_confidences[mask]))
        return int(detection_levels[mask][seed_local])
    member_keyframe_ids = (
        detection_keyframe_ids[mask] if detection_keyframe_ids is not None else None
    )
    return int(
        _median_member_level(
            detection_levels[mask],
            member_keyframe_ids,
            level_height_rank,
        )
    )


def _cluster_centroid_to_geo(
    *,
    label: int,
    centroid: np.ndarray,
    cluster_level: int,
    georef: GeoRef | None,
) -> Tuple[np.ndarray, int]:
    """Convert a centroid to (lat, lon, alt) and resolve the cluster level.

    When the level is already known (from a keyframe), altitude is clamped to
    that level's bounds; otherwise the level is taken from the geo lookup.

    Returns:
        (geo_xyz float32[3], resolved_level)
    """
    if georef is None:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32), cluster_level
    try:
        lat, lon, alt, level = world_to_geo_with_level(
            centroid.astype(np.float64), georef
        )
    except Exception as e:
        logger.warning(f"Failed to convert cluster {label} to geo: {e}")
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32), cluster_level
    geo = np.array([lat, lon, alt], dtype=np.float32)
    if cluster_level != UNRESOLVED_LEVEL_SENTINEL:
        # Level came from a keyframe — keep altitude consistent with it.
        lv_obj = georef.get_level(cluster_level)
        if lv_obj is not None:
            origin_alt = georef.origin_coordinates.altitude
            geo[2] = float(
                np.clip(
                    geo[2],
                    origin_alt + lv_obj.floorHeightFromGround,
                    origin_alt + lv_obj.ceilingHeightFromGround,
                )
            )
        return geo, cluster_level
    if level is not None:
        cluster_level = int(level)
    return geo, cluster_level


def compute_cluster_statistics(
    positions_local: np.ndarray,
    cluster_ids: np.ndarray,
    georef: GeoRef | None,
    detection_confidences: np.ndarray | None = None,
    detection_levels: np.ndarray | None = None,
    detection_keyframe_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cluster centroids and statistics.

    Args:
        positions_local: [N, 3] WDS local-map positions
        cluster_ids: [N] cluster assignments
        georef: GeoRef for geographic conversion (optional)
        detection_confidences: [N] detection confidence scores (optional)

    Returns:
        (centroids_world, centroids_geo, observation_counts, confidence_scores)
    """
    unique_labels = np.unique(cluster_ids)
    unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)

    n_clusters = len(unique_labels)

    centroids_world = np.zeros((n_clusters, 3), dtype=np.float32)
    centroids_geo = np.zeros((n_clusters, 3), dtype=np.float32)
    observation_counts = np.zeros(n_clusters, dtype=np.int32)
    confidence_scores = np.zeros(n_clusters, dtype=np.float32)
    cluster_levels = np.full(n_clusters, UNRESOLVED_LEVEL_SENTINEL, dtype=np.int32)

    # Order levels by physical floor height so the per-cluster median level is
    # geometrically central even when Level.id is a non-ordinal database id.
    level_height_rank: dict[int, int] = {}
    if georef is not None and getattr(georef, "levels", None):
        ordered_ids = [
            int(lv.id)
            for lv in sorted(georef.levels, key=lambda lvl: lvl.floorHeightFromGround)
        ]
        level_height_rank = {lid: rank for rank, lid in enumerate(ordered_ids)}

    for i, label in enumerate(unique_labels):
        mask = cluster_ids == label
        cluster_positions = positions_local[mask]

        # Weighted centroid if confidences provided
        if detection_confidences is not None:
            weights = detection_confidences[mask]
            weights = weights / weights.sum()
            centroid = np.average(cluster_positions, axis=0, weights=weights)
        else:
            centroid = cluster_positions.mean(axis=0)

        centroids_world[i] = centroid
        observation_counts[i] = mask.sum()
        confidence_scores[i] = _cluster_confidence(
            cluster_positions, int(observation_counts[i])
        )

        if detection_levels is not None:
            cluster_levels[i] = _cluster_level(
                mask=mask,
                detection_levels=detection_levels,
                detection_confidences=detection_confidences,
                detection_keyframe_ids=detection_keyframe_ids,
                level_height_rank=level_height_rank,
            )

        centroids_geo[i], cluster_levels[i] = _cluster_centroid_to_geo(
            label=int(label),
            centroid=centroid,
            cluster_level=int(cluster_levels[i]),
            georef=georef,
        )

    return (
        centroids_world,
        centroids_geo,
        observation_counts,
        confidence_scores,
        cluster_levels,
    )


class _LocalizationContext(NamedTuple):
    keyframe_poses: dict
    georef: Optional[GeoRef]
    keyframe_levels: dict
    kf_id_to_depth_stem: Dict[int, str]


def _load_localization_context(georef_db_path: Path) -> _LocalizationContext:
    """Load poses, GeoRef, levels, and depth stem map from georef.db."""
    logger.info("Loading keyframe poses from georef.db...")
    keyframe_poses = load_keyframe_poses_from_georef(georef_db_path)
    logger.info("Loaded %d keyframe poses", len(keyframe_poses))

    georef = load_georef_from_db(georef_db_path)
    if georef is None:
        logger.warning(
            "Could not load GeoRef - geographic coordinates will be unavailable"
        )
    keyframe_levels = (
        keyframe_levels_from_poses(keyframe_poses, georef) if georef is not None else {}
    )

    image_filename_to_keyframe_id = load_image_filename_to_keyframe_id(georef_db_path)
    kf_id_to_depth_stem: Dict[int, str] = {}
    if image_filename_to_keyframe_id:
        for image_filename, image_keyframe_id in image_filename_to_keyframe_id.items():
            kf_id_to_depth_stem[int(image_keyframe_id)] = Path(image_filename).stem

    return _LocalizationContext(
        keyframe_poses=keyframe_poses,
        georef=georef,
        keyframe_levels=keyframe_levels,
        kf_id_to_depth_stem=kf_id_to_depth_stem,
    )


def _localize_per_keyframe(
    *,
    ctx: _LocalizationContext,
    object_keyframe_ids: np.ndarray,
    object_cutout_ids: np.ndarray,
    object_bboxes: np.ndarray,
    depth_dir: Path,
    face_size: int,
    fov_deg: float,
    id_stride: int,
    min_depth_m: float,
    max_depth_m: float,
    positions_keyframe: np.ndarray,
    positions_local: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Project each detection to 3D; fills output arrays in-place."""
    unique_keyframes = np.unique(object_keyframe_ids)
    for kf_id in unique_keyframes:
        depth_stem = ctx.kf_id_to_depth_stem.get(int(kf_id))
        depth_candidates: List[Path] = []
        if depth_stem:
            depth_candidates.append(depth_dir / f"{depth_stem}.zarr")
        depth_candidates.append(depth_dir / f"{kf_id}.zarr")
        depth_path = next((p for p in depth_candidates if p.exists()), None)
        if depth_path is None:
            logger.debug("Depth map not found for keyframe %s", kf_id)
            continue
        try:
            depth_map, _ = load_depth_from_zarr(str(depth_path))
        except Exception as exc:
            logger.warning("Failed to load depth for keyframe %s: %s", kf_id, exc)
            continue
        if kf_id not in ctx.keyframe_poses:
            logger.debug("Pose not found for keyframe %s", kf_id)
            continue
        keyframe_pose = ctx.keyframe_poses[kf_id]
        kf_mask = object_keyframe_ids == kf_id
        for idx in np.where(kf_mask)[0]:
            cutout_id = object_cutout_ids[idx]
            bbox = object_bboxes[idx]
            local_index = cutout_id % id_stride
            if local_index >= len(CUBEMAP_FACE_ORDER):
                logger.debug(
                    "Invalid face index %d for detection %d", local_index, idx
                )
                continue
            face_name = CUBEMAP_FACE_ORDER[local_index]
            pos_local, depth = detection_to_3d_local(
                bbox, face_name, depth_map, face_size, fov_deg, min_depth_m, max_depth_m
            )
            if pos_local is None:
                continue
            pos_world = transform_to_world(pos_local, keyframe_pose)
            positions_keyframe[idx] = pos_local
            positions_local[idx] = pos_world
            depths[idx] = depth
            valid_mask[idx] = True


def _raise_for_unresolved_levels(
    *,
    still_unresolved: np.ndarray,
    object_keyframe_ids: np.ndarray,
    positions_local: np.ndarray,
    keyframe_poses: dict,
    georef: Optional[GeoRef],
) -> None:
    """Log per-keyframe level-resolution debug info, then raise.

    Raises:
        ValueError: Always — this is only called when unresolved levels remain.
    """
    unresolved_by_keyframe: Dict[int, int] = {}
    for idx in still_unresolved.tolist():
        kf_id = int(object_keyframe_ids[idx])
        unresolved_by_keyframe.setdefault(kf_id, idx)
    for kf_id, sample_idx in sorted(unresolved_by_keyframe.items())[:10]:
        unresolved_keyframe_pose = keyframe_poses.get(kf_id)
        keyframe_position_local = (
            np.linalg.inv(unresolved_keyframe_pose)[:3, 3].astype(np.float64)
            if unresolved_keyframe_pose is not None
            else None
        )
        detection_position_local = positions_local[sample_idx].astype(np.float64)
        logger.warning(
            "Unresolved level for keyframe %d: keyframe_pose={%s};"
            " detection_sample={%s}",
            kf_id,
            (
                _format_level_resolution_debug(
                    georef=georef,
                    position_local=keyframe_position_local,
                )
                if keyframe_position_local is not None
                else "pose unavailable"
            ),
            _format_level_resolution_debug(
                georef=georef,
                position_local=detection_position_local,
            ),
        )
    sample_keyframes = sorted(
        {int(object_keyframe_ids[idx]) for idx in still_unresolved[:10]}
    )
    raise ValueError(
        "Localized detections are missing resolved levels after keyframe and "
        f"detection fallback for keyframes {sample_keyframes}; clustering "
        "requires a resolved level per keyframe/cutout"
    )


def _resolve_detection_levels(
    *,
    valid_mask: np.ndarray,
    object_keyframe_ids: np.ndarray,
    positions_local: np.ndarray,
    keyframe_levels: dict,
    keyframe_poses: dict,
    georef: Optional[GeoRef],
    n_detections: int,
) -> np.ndarray:
    """Assign an indoor-level id to each valid detection.

    Levels come from the detection's keyframe when known, otherwise from a
    georef lookup at the detection's localized 3D position.

    Returns:
        [N] int32 level ids (UNRESOLVED_LEVEL_SENTINEL where not applicable).

    Raises:
        ValueError: If any valid detection still lacks a resolved level after
            both the keyframe and 3D-position fallbacks (clustering requires a
            resolved level per keyframe/cutout).
    """
    detection_levels = np.full(
        n_detections, UNRESOLVED_LEVEL_SENTINEL, dtype=np.int32
    )
    for idx in np.where(valid_mask)[0]:
        kf_id = int(object_keyframe_ids[idx])
        if kf_id in keyframe_levels:
            detection_levels[idx] = int(keyframe_levels[kf_id])

    # Fallback: resolve via 3D position in georef
    unresolved = np.where(
        valid_mask & (detection_levels == UNRESOLVED_LEVEL_SENTINEL)
    )[0]
    if unresolved.size and georef is not None:
        for idx in unresolved.tolist():
            try:
                _, _, _, level = world_to_geo_with_level(
                    positions_local[idx].astype(np.float64), georef
                )
            except Exception:
                continue
            if level is not None:
                detection_levels[idx] = int(level)

    # Clustering requires a resolved level per keyframe/cutout — fail loudly.
    still_unresolved = np.where(
        valid_mask & (detection_levels == UNRESOLVED_LEVEL_SENTINEL)
    )[0]
    if still_unresolved.size:
        _raise_for_unresolved_levels(
            still_unresolved=still_unresolved,
            object_keyframe_ids=object_keyframe_ids,
            positions_local=positions_local,
            keyframe_poses=keyframe_poses,
            georef=georef,
        )
    return detection_levels


def localize_detections(
    object_keyframe_ids: np.ndarray,
    object_cutout_ids: np.ndarray,
    object_bboxes: np.ndarray,
    depth_dir: Path,
    georef_db_path: Path,
    face_size: int,
    fov_deg: float = 90.0,
    id_stride: int = 1024,
    min_depth_m: float = 0.5,
    max_depth_m: float = 50.0,
    clustering_eps_m: float = 1.5,
    clustering_min_samples: int = 1,
    detection_confidences: np.ndarray | None = None,
    object_embeddings: np.ndarray | None = None,
    embedding_similarity_threshold: float = 0.85,
    min_keyframes_per_cluster: int = 3,
    skip_clustering: bool = False,
) -> LocalizationResult:
    """
    Main entry point: localize all detections to 3D and cluster.

    Args:
        object_keyframe_ids: [N] keyframe ID for each detection
        object_cutout_ids: [N] cutout ID for each detection
        object_bboxes: [N, 4] bounding boxes [x0, y0, x1, y1]
        depth_dir: Directory containing depth zarr files
        georef_db_path: Path to georef.db
        face_size: Cubemap face size in pixels
        fov_deg: Cubemap face FOV in degrees
        id_stride: Cutout ID stride for decoding face index
        min_depth_m: Minimum valid depth
        max_depth_m: Maximum valid depth
        clustering_eps_m: Spatial radius for candidate cluster neighbors
        clustering_min_samples: Unused legacy parameter kept for CLI parity
        detection_confidences: Optional [N] confidence scores
        object_embeddings: Optional [N, D] embeddings used for semantic gating
        embedding_similarity_threshold: Minimum cosine similarity to connect
            two spatial neighbors.
        min_keyframes_per_cluster: Minimum distinct keyframes required to keep
            a cluster.

    Returns:
        LocalizationResult with all 3D positions and clustering
    """
    n_detections = len(object_keyframe_ids)
    ctx = _load_localization_context(georef_db_path)

    positions_keyframe = np.full((n_detections, 3), np.nan, dtype=np.float32)
    positions_local = np.full((n_detections, 3), np.nan, dtype=np.float32)
    depths = np.full(n_detections, np.nan, dtype=np.float32)
    valid_mask = np.zeros(n_detections, dtype=bool)

    logger.info("Localizing %d detections...", n_detections)
    _localize_per_keyframe(
        ctx=ctx,
        object_keyframe_ids=object_keyframe_ids,
        object_cutout_ids=object_cutout_ids,
        object_bboxes=object_bboxes,
        depth_dir=depth_dir,
        face_size=face_size,
        fov_deg=fov_deg,
        id_stride=id_stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        positions_keyframe=positions_keyframe,
        positions_local=positions_local,
        depths=depths,
        valid_mask=valid_mask,
    )
    logger.info(
        "Successfully localized %d/%d detections", int(valid_mask.sum()), n_detections
    )

    # Assign keyframe/cutout level to each valid detection. Prefer the keyframe's
    # camera pose, but fall back to the localized detection position for poses that
    # sit on a level boundary or slightly outside polygonized indoor geometry.
    detection_levels = _resolve_detection_levels(
        valid_mask=valid_mask,
        object_keyframe_ids=object_keyframe_ids,
        positions_local=positions_local,
        keyframe_levels=ctx.keyframe_levels,
        keyframe_poses=ctx.keyframe_poses,
        georef=ctx.georef,
        n_detections=n_detections,
    )

    if skip_clustering:
        logger.info("Skipping offline clustering (skip_clustering=True)")
        return LocalizationResult(
            positions_keyframe=positions_keyframe,
            positions_local=positions_local,
            depths=depths,
            valid_mask=valid_mask,
            cluster_ids=np.full(n_detections, -1, dtype=np.int32),
            cluster_centroids_world=np.zeros((0, 3), dtype=np.float32),
            cluster_centroids_geo=np.zeros((0, 3), dtype=np.float32),
            cluster_observation_counts=np.zeros(0, dtype=np.int32),
            cluster_confidence=np.zeros(0, dtype=np.float32),
            cluster_levels=np.zeros(0, dtype=np.int32),
            detection_levels=detection_levels,
        )

    logger.info("Clustering detections with joint spatial+semantic graph...")
    cluster_ids = cluster_detections_3d(
        positions_local,
        valid_mask,
        object_keyframe_ids,
        eps_meters=clustering_eps_m,
        embeddings=object_embeddings,
        embedding_similarity_threshold=embedding_similarity_threshold,
        min_keyframes_per_cluster=min_keyframes_per_cluster,
        detection_levels=detection_levels,
    )
    n_clusters = len(np.unique(cluster_ids[cluster_ids >= 0]))
    logger.info("Found %d object clusters", n_clusters)

    centroids_world, centroids_geo, obs_counts, confidence, cluster_levels = (
        compute_cluster_statistics(
            positions_local,
            cluster_ids,
            ctx.georef,
            detection_confidences,
            detection_levels,
        )
    )
    return LocalizationResult(
        positions_keyframe=positions_keyframe,
        positions_local=positions_local,
        depths=depths,
        valid_mask=valid_mask,
        cluster_ids=cluster_ids,
        cluster_centroids_world=centroids_world,
        cluster_centroids_geo=centroids_geo,
        cluster_observation_counts=obs_counts,
        cluster_confidence=confidence,
        cluster_levels=cluster_levels,
        detection_levels=detection_levels,
    )


def can_localize_3d(map_path: Path, depth_dir: str = "depths") -> Tuple[bool, str]:
    """
    Check if 3D localization prerequisites are available.

    Args:
        map_path: Path to map directory
        depth_dir: Name of depth subdirectory

    Returns:
        (can_localize, reason_message)
    """
    depth_path = map_path / depth_dir
    georef_path = map_path / "georef.db"

    if not georef_path.exists():
        return False, f"georef.db not found at {georef_path}"

    if not depth_path.exists():
        return False, f"Depth directory not found at {depth_path}"

    # Check for at least one depth file
    depth_files = list(depth_path.glob("*.zarr"))
    if not depth_files:
        return False, f"No .zarr depth files in {depth_path}"

    return True, f"OK ({len(depth_files)} depth files found)"
