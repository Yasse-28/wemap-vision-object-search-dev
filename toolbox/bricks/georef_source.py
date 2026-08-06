"""Keyframe poses — the stand-in for the Django ORM.

Production reads keyframe poses from `api_geokeyframe` (an EUS `PointField` plus a
`QuaternionField`, both written by the georef step). This repo has no such table, so
they come from the map's v2 manifest instead — see `map_manifest` for its shape.

**The poses need no conversion.** The manifest stores exactly what `api_geokeyframe`
stores: an EUS position in metres and a `[w, x, y, z]` quaternion taking
CameraFrame(OpenGL) to LocalFrame(EUS). It also carries `venue_type` and the real
`geo_ref_id`, so nothing about a map has to be passed on the command line.

`PoseSource` exists as a façade rather than handing `MapManifest` around directly:
`prepare_runner`, `prepare_postprocess`, `ingest_cli` and `service` all consume this
narrow shape, and keeping it lets the manifest reader stay a pure parser.

## Identity note

Production distinguishes `GeoKeyframe` (per-georef) from `VideoKeyframe` (per
capture). Here one id plays both roles — the `geo_keyframes` index — and it is also
what `prepare` writes into `metadata.parquet` as `video_keyframe_id`. So
`geokeyframe_id == video_keyframe_id` throughout.

Because that id is an **array index**, re-exporting a manifest renumbers every
keyframe: prepare and ingest must then be re-run together. Filenames, not ids, are
what tie an image or a depth map back to a pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolbox.bricks.map_manifest import load_map_manifest
from toolbox.bricks.vendored.geo_transform import GeoTransform


@dataclass(frozen=True)
class KeyframePose:
    """One keyframe, in the frames production stores."""

    keyframe_id: int
    position_eus: np.ndarray  # (3,) float64, metres
    orientation_wxyz: np.ndarray  # (4,) float64, CameraFrame(OpenGL) → LocalFrame(EUS)


@dataclass(frozen=True)
class PoseSource:
    """Everything the bricks need about a map's geometry."""

    path: Path
    geo_transform: GeoTransform
    poses: dict[int, KeyframePose]
    image_filename_to_keyframe_id: dict[str, int]
    depth_filename_by_keyframe_id: dict[int, str]
    venue_type: str | None
    geo_ref_id: int | None


def load_pose_source(map_path: str | Path) -> PoseSource:
    """Load a map's geometry from its v2 manifest.

    Raises `FileNotFoundError` when the map directory holds none — proceeding
    without poses would build an index whose candidates can never be positioned.
    """
    manifest = load_map_manifest(Path(map_path))
    return PoseSource(
        path=manifest.path,
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
