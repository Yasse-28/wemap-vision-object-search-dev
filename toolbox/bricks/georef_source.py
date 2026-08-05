"""Keyframe poses — the stand-in for the Django ORM.

Production reads keyframe poses from `api_geokeyframe` (an EUS `PointField` plus a
`QuaternionField`, both written by the georef step). This repo has no such table, so
they come from the map directory instead. **Two formats exist, and they are not
equally good:**

| Map generation | File | Poses |
|---|---|---|
| **v2 (current)** | `{map_id}_{version}_{date}_{time}.json` | already EUS |
| v1 (legacy) | `georef.db` | transposed, world-to-camera, WDS/OpenCV |

`load_pose_source` prefers the v2 manifest and falls back to `georef.db`, so v1 maps
still work. Everything below the `PoseSource` section is the v1 path; see
`map_manifest` for v2.

## The v1 frame conversion, and why it is the sharpest edge in the port

`GeoRefKeyframe.pose` is a float64 4x4 blob that is, all at once:

1. **stored transposed** — `frombuffer(...).reshape(4, 4).T`,
2. **world-to-camera**, so the camera's world pose is `inv(pose)`,
3. **in WDS/OpenCV conventions**, whereas production stores EUS/OpenGL.

Three independent flips compose here. Drop any one and you get a result that is
wrong but entirely plausible — objects land mirrored, or rotated 180°, with no
error anywhere. `toolbox/tests/test_frame_conventions.py` pins this down; if you
touch this module, run it.

    position_eus    = ROT_WDS_TO_EUS @ inv(pose)[:3, 3]
    orientation_eus = quat(ROT_WDS_TO_EUS @ inv(pose)[:3, :3] @ ROT_OPENGL_TO_OPENCV)

`ROT_WDS_TO_EUS = diag(-1, -1, 1)`: East = -West, Up = -Down, South = South.

## Identity note

Production distinguishes `GeoKeyframe` (per-georef) from `VideoKeyframe` (per
capture). Here one id plays both roles — the `GeoRefKeyframe.id` for v1, the
`geo_keyframes` index for v2 — and it is also what `prepare` writes into
`metadata.parquet` as `video_keyframe_id`. So `geokeyframe_id == video_keyframe_id`
throughout.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolbox.bricks.map_manifest import find_manifest, load_manifest
from toolbox.bricks.vendored.geo_transform import GeoTransform
from toolbox.bricks.vendored.maths import matrix3, quaternion
from toolbox.logging import logger

# Both matrices are copied from the standalone lineage's
# `online/observation_orientation.py`, which is the code that made the georef.db
# poses agree with wemap-vision-tools. Now archived under legacy/.
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


@dataclass(frozen=True)
class KeyframePose:
    """One keyframe, in the frames production stores."""

    keyframe_id: int
    position_eus: np.ndarray  # (3,) float64, metres
    orientation_wxyz: np.ndarray  # (4,) float64, CameraFrame(OpenGL) → LocalFrame(EUS)


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Nearest true rotation matrix, via SVD.

    Poses round-tripped through float64 on disk are not exactly orthonormal, and
    `quaternion.from_matrix3` assumes they are (it reads matrix entries directly).
    Production never needs this because it stores the quaternion, not the matrix.
    """
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    r = u @ vh
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vh
    return np.asarray(r, dtype=np.float64)


def _decode_pose_blob(pose_blob: bytes) -> np.ndarray:
    """`GeoRefKeyframe.pose` blob → 4x4 world-to-camera matrix (WDS/OpenCV)."""
    return (
        np.frombuffer(pose_blob, dtype=np.dtype(float).newbyteorder("<"))
        .reshape([4, 4])
        .T
    )


def load_keyframe_poses(georef_db_path: str | Path) -> dict[int, KeyframePose]:
    """Every keyframe in `georef.db`, as EUS position + OpenGL→EUS quaternion."""
    georef_db_path = Path(georef_db_path)
    if not georef_db_path.is_file():
        raise FileNotFoundError(f"No georef.db at '{georef_db_path}'.")

    poses: dict[int, KeyframePose] = {}
    skipped = 0
    with sqlite3.connect(georef_db_path) as conn:
        for keyframe_id, pose_blob in conn.execute(
            "SELECT id, pose FROM GeoRefKeyframe ORDER BY id"
        ):
            if pose_blob is None:
                skipped += 1
                continue
            try:
                pose = _decode_pose_blob(pose_blob)
                camera_to_world = np.linalg.inv(pose)
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Skipping keyframe %s: bad pose (%s)", keyframe_id, exc)
                skipped += 1
                continue

            position_eus = ROT_WDS_TO_EUS @ camera_to_world[:3, 3]
            rotation_eus = matrix3.cast(
                _orthonormalize(
                    ROT_WDS_TO_EUS @ camera_to_world[:3, :3] @ ROT_OPENGL_TO_OPENCV
                )
            )
            poses[int(keyframe_id)] = KeyframePose(
                keyframe_id=int(keyframe_id),
                position_eus=np.asarray(position_eus, dtype=np.float64),
                orientation_wxyz=np.asarray(
                    quaternion.from_matrix3(matrix3.cast(rotation_eus)),
                    dtype=np.float64,
                ),
            )

    if skipped:
        logger.warning("Skipped %d keyframe(s) with no usable pose.", skipped)
    if not poses:
        raise ValueError(f"No usable keyframe poses in '{georef_db_path}'.")
    logger.info("Loaded %d keyframe poses from %s", len(poses), georef_db_path)
    return poses


def load_geo_transform(georef_db_path: str | Path) -> GeoTransform:
    """The map's EUS↔WGS84 transform plus its level bands, from `georef.db`."""
    return GeoTransform.from_georef_db(Path(georef_db_path))


def georef_db_path_for_map(map_path: str | Path) -> Path:
    """`georef.db` inside a map directory — the v1 filesystem convention."""
    return Path(map_path) / "georef.db"


# ---------------------------------------------------------------- unified entry point


@dataclass(frozen=True)
class PoseSource:
    """Everything the bricks need about a map's geometry, from either format.

    `geo_ref_id` and `venue_type` are populated for v2 only — `georef.db` records
    neither, which is why v1 needs them passed on the command line.
    """

    kind: str  # "manifest" | "georef_db"
    path: Path
    geo_transform: GeoTransform
    poses: dict[int, KeyframePose]
    image_filename_to_keyframe_id: dict[str, int] | None
    depth_filename_by_keyframe_id: dict[int, str] | None
    venue_type: str | None
    geo_ref_id: int | None


def load_pose_source(map_path: str | Path) -> PoseSource:
    """Load a map's geometry, preferring the v2 manifest over `georef.db`.

    Raises `FileNotFoundError` when neither is present — proceeding without poses
    would build an index whose candidates can never be positioned.
    """
    map_path = Path(map_path)

    manifest_path = find_manifest(map_path)
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        return PoseSource(
            kind="manifest",
            path=manifest_path,
            geo_transform=manifest.geo_transform(),
            # ManifestKeyframe carries the same three fields KeyframePose does, plus
            # filenames; narrow it here so downstream code sees one type.
            poses={
                kf.keyframe_id: KeyframePose(
                    keyframe_id=kf.keyframe_id,
                    position_eus=kf.position_eus,
                    orientation_wxyz=kf.orientation_wxyz,
                )
                for kf in manifest.keyframes
            },
            image_filename_to_keyframe_id=manifest.image_filename_to_keyframe_id(),
            depth_filename_by_keyframe_id=manifest.depth_filename_by_keyframe_id(),
            venue_type=manifest.venue_type,
            geo_ref_id=manifest.geo_ref_id,
        )

    georef_db = georef_db_path_for_map(map_path)
    if georef_db.is_file():
        logger.warning(
            "No v2 manifest in %s; falling back to the legacy georef.db. Its poses "
            "need "
            "three frame conversions (see this module's docstring), so prefer a v2 "
            "manifest when one can be exported.",
            map_path,
        )
        from toolbox.georef.georef import load_image_filename_to_keyframe_id

        return PoseSource(
            kind="georef_db",
            path=georef_db,
            geo_transform=load_geo_transform(georef_db),
            poses=load_keyframe_poses(georef_db),
            image_filename_to_keyframe_id=load_image_filename_to_keyframe_id(georef_db),
            depth_filename_by_keyframe_id=None,
            venue_type=None,
            geo_ref_id=None,
        )

    raise FileNotFoundError(
        f"'{map_path}' has neither a v2 manifest "
        "('{map_id}_{version}_{date}_{time}.json') nor a legacy georef.db. One of them "
        "is required: keyframe poses come from it."
    )
