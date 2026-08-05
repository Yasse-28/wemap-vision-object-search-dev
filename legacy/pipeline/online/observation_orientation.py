from __future__ import annotations

import functools
import sqlite3
from pathlib import Path

import numpy as np

from pipeline.core.logging import logger
from pipeline.core.types import LoadedIndex

ROT_WDS_TO_EUS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
ROT_OPENGL_TO_OPENCV = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    u, _, vh = np.linalg.svd(r)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vh

    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.clip(np.linalg.norm(q), 1e-12, None)
    if q[0] < 0.0:
        q *= -1.0
    return [float(v) for v in q.tolist()]


def _cutout_rotation_by_id(index: LoadedIndex) -> dict[int, np.ndarray]:
    return {
        int(cutout_id): np.asarray(rotation, dtype=np.float64)[:3, :3]
        for cutout_id, rotation in zip(
            index.cutout_ids.tolist(),
            index.cutout_rotation_cutout_to_equirect,
        )
    }


@functools.lru_cache(maxsize=None)
def _load_all_keyframe_poses(georef_db_path: Path) -> dict[int, np.ndarray]:
    """Load all keyframe poses from georef.db once per path (cached)."""
    poses: dict[int, np.ndarray] = {}
    with sqlite3.connect(georef_db_path) as conn:
        for keyframe_id, pose_blob in conn.execute(
            "SELECT id, pose FROM GeoRefKeyframe"
        ):
            pose = (
                np.frombuffer(
                    pose_blob,
                    dtype=np.dtype(float).newbyteorder("<"),
                )
                .reshape([4, 4])
                .T
            )
            poses[int(keyframe_id)] = pose
    return poses


def observation_orientation_quaternions(
    *,
    index: LoadedIndex,
    map_path: Path,
    object_indices: list[int],
) -> dict[int, list[float]]:
    """Return wemap-vision-tools-style EUS observation view quaternions keyed
    by object index."""
    if not object_indices:
        return {}

    georef_db_path = Path(map_path) / "georef.db"
    if not georef_db_path.exists():
        logger.warning(
            "Observation orientation unavailable: %s not found", georef_db_path
        )
        return {}

    try:
        keyframe_poses = _load_all_keyframe_poses(georef_db_path)
    except Exception as exc:
        logger.warning(
            "Observation orientation unavailable: failed to load keyframe poses: %s",
            exc,
        )
        return {}

    cutout_rotations = _cutout_rotation_by_id(index)
    quaternions: dict[int, list[float]] = {}
    for obj_idx in object_indices:
        if obj_idx < 0 or obj_idx >= len(index.object_cutout_ids):
            continue
        keyframe_id = int(index.object_keyframe_ids[obj_idx])
        cutout_id = int(index.object_cutout_ids[obj_idx])
        keyframe_pose = keyframe_poses.get(keyframe_id)
        cutout_to_keyframe = cutout_rotations.get(cutout_id)
        if keyframe_pose is None or cutout_to_keyframe is None:
            continue

        keyframe_to_wds = np.linalg.inv(np.asarray(keyframe_pose, dtype=np.float64))[
            :3, :3
        ]
        cutout_to_wds = keyframe_to_wds @ cutout_to_keyframe
        cutout_opengl_to_eus = ROT_WDS_TO_EUS @ cutout_to_wds @ ROT_OPENGL_TO_OPENCV
        quaternions[obj_idx] = _rotation_to_quaternion_wxyz(cutout_opengl_to_eus)

    return quaternions
