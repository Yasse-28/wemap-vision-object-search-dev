"""Read a v2 map manifest — the JSON that replaced `georef.db`.

v2 maps ship a single JSON in the map directory:

    {map_path}/{map_id}_{map_version}_{date}_{time}.json

It is a direct dump of the production objects, which makes it a far better pose
source than the v1 `georef.db`:

| Manifest | Production equivalent |
|---|---|
| `local_origin` — `[lng, lat, alt]` | `GeoRef.local_origin` (same axis order) |
| `geo_levels[]` — `value`, `min_altitude`, `max_altitude` | `GeoLevel` rows |
| `geo_levels[].geo_ref` | the real `geo_ref_id` |
| `geo_keyframes[]` — `x`/`y`/`z`, `orientation` | `GeoKeyframe.position` + ori. |
| `map.venue_type` | `Map.venue_type`, which selects the detection prompts |

**The poses need no conversion.** `georef.db` stores them transposed,
world-to-camera and in WDS/OpenCV, so the v1 path composes three flips to reach
EUS/OpenGL (see `georef_source`). The manifest already stores exactly what
`api_geokeyframe` stores, so that whole class of silent error does not apply here.

## Keyframe identity

The manifest carries **no integer keyframe id** — keyframes are identified by their
image filename (a UUID). We therefore use the **index in `geo_keyframes`** as the
local `video_keyframe_id`, exactly as `api_geokeyframe` ids are opaque integers.

That is safe because *both* halves of the build read the same manifest:
`prepare_runner` maps image filename → index, and `ingest_cli` maps index → pose. They
cannot disagree. The one rule it imposes: **re-exporting the manifest means re-running
prepare and ingest together**, since a different ordering renumbers everything. The
image filename is stored on each `geokeyframe` row so a mismatch is at least visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from toolbox.bricks.vendored.geo_transform import Coordinates, GeoTransform, Level
from toolbox.logging import logger

# {map_id}_{map_version}_{date}_{time}.json — e.g. bbhotel-choisy_2_20260805_083206.json
MANIFEST_PATTERN = re.compile(
    r"^(?P<map_id>.+)_(?P<version>\d+)_(?P<stamp>\d{8}_\d{6})\.json$"
)


@dataclass(frozen=True)
class ManifestKeyframe:
    """One keyframe, already in the frames production stores."""

    keyframe_id: int  # index in geo_keyframes
    image_filename: str
    depth_filename: str
    position_eus: np.ndarray  # (3,) float64, metres
    orientation_wxyz: np.ndarray  # (4,) float64, CameraFrame(OpenGL) → LocalFrame(EUS)


@dataclass(frozen=True)
class MapManifest:
    path: Path
    map_id: str
    map_version: int
    map_uuid: str
    venue_type: str | None
    geo_ref_id: int | None
    origin: Coordinates
    levels: tuple[Level, ...]
    keyframes: tuple[ManifestKeyframe, ...]

    def geo_transform(self) -> GeoTransform:
        """EUS↔WGS84 plus the level bands, built exactly as production does."""
        return GeoTransform(origin=self.origin, levels=self.levels)

    def keyframe_poses(self) -> dict[int, ManifestKeyframe]:
        return {kf.keyframe_id: kf for kf in self.keyframes}

    def image_filename_to_keyframe_id(self) -> dict[str, int]:
        return {kf.image_filename: kf.keyframe_id for kf in self.keyframes}

    def depth_filename_by_keyframe_id(self) -> dict[int, str]:
        return {kf.keyframe_id: kf.depth_filename for kf in self.keyframes}


def find_manifest(map_path: Path) -> Path | None:
    """The v2 manifest in a map directory, or None.

    When several exist, the newest wins by the `{date}_{time}` stamp in the name —
    that stamp is why the filename carries it. Ties fall back to mtime.
    """
    map_path = Path(map_path)
    if not map_path.is_dir():
        return None
    candidates = [
        (MANIFEST_PATTERN.match(path.name), path)
        for path in map_path.iterdir()
        if path.is_file() and path.suffix == ".json"
    ]
    matches = [(match, path) for match, path in candidates if match is not None]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "%d v2 manifests in %s; using the newest by timestamp. Stale ones renumber "
            "keyframes — remove them to be safe.",
            len(matches),
            map_path,
        )
    matches.sort(key=lambda item: (item[0].group("stamp"), item[1].stat().st_mtime))
    return matches[-1][1]


def _basename(url: str) -> str:
    """Filename an asset URL points at, with any query string discarded.

    Today's manifests point at a public bucket with no query string, so splitting
    on "/" would also work. It is parsed properly because the day a URL arrives
    presigned, the filename would become `abc.jpg?X-Amz-Signature=…` and match
    nothing on disk: no image would resolve to a keyframe id and no depth map
    would be found — both silent. The two other v2 manifest readers
    (`wemap-vision-python`'s `read_v2_map_data`, `wemap-vision-vps-offline`'s
    `download_v2_map_data`) already use `urlparse`, as does `retrieve-map-data`,
    which writes these files; disagreeing with them is the actual risk.
    """
    return Path(urlparse(str(url)).path).name


def load_manifest(path: Path) -> MapManifest:
    """Parse a v2 manifest. Raises `ValueError` on anything structurally missing."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    origin_raw = data.get("local_origin")
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError(f"{path}: 'local_origin' must be [lng, lat, alt].")
    # Axis order matches Django's PointField: longitude first. Getting this backwards
    # puts the whole map on the wrong continent, so it is spelled out rather than
    # unpacked positionally.
    origin = Coordinates(
        lng=float(origin_raw[0]), lat=float(origin_raw[1]), alt=float(origin_raw[2])
    )

    levels: list[Level] = []
    for entry in data.get("geo_levels") or []:
        levels.append(
            Level(
                value=float(entry["value"]),
                min_altitude=(
                    None
                    if entry.get("min_altitude") is None
                    else float(entry["min_altitude"])
                ),
                max_altitude=(
                    None
                    if entry.get("max_altitude") is None
                    else float(entry["max_altitude"])
                ),
                # Production stores a GEOS geometry; the manifest carries GeoJSON (or
                # null). The vendored GeoTransform tests it with a pure-Python
                # ray-cast, so a dict is what it wants.
                geometry=entry.get("geometry"),
            )
        )
    levels.sort(key=lambda level: level.value)

    geo_ref_ids = {
        int(entry["geo_ref"])
        for entry in data.get("geo_levels") or []
        if entry.get("geo_ref") is not None
    }
    if len(geo_ref_ids) > 1:
        raise ValueError(
            f"{path}: geo_levels reference several geo_refs: {geo_ref_ids}."
        )
    geo_ref_id = next(iter(geo_ref_ids), None)

    keyframes: list[ManifestKeyframe] = []
    for index, entry in enumerate(data.get("geo_keyframes") or []):
        image_filename = _basename(entry.get("image_url", ""))
        if not image_filename:
            raise ValueError(f"{path}: geo_keyframes[{index}] has no image_url.")
        orientation = entry.get("orientation")
        if not isinstance(orientation, list) or len(orientation) != 4:
            raise ValueError(
                f"{path}: geo_keyframes[{index}].orientation must be [w, x, y, z]."
            )
        keyframes.append(
            ManifestKeyframe(
                keyframe_id=index,
                image_filename=image_filename,
                depth_filename=_basename(entry.get("depth_url", "")),
                position_eus=np.array(
                    [float(entry["x"]), float(entry["y"]), float(entry["z"])],
                    dtype=np.float64,
                ),
                orientation_wxyz=np.array(
                    [float(value) for value in orientation], dtype=np.float64
                ),
            )
        )
    if not keyframes:
        raise ValueError(f"{path}: no geo_keyframes.")

    duplicates = len(keyframes) - len({kf.image_filename for kf in keyframes})
    if duplicates:
        raise ValueError(
            f"{path}: {duplicates} keyframe(s) share an image filename, so an image "
            "cannot be mapped to one pose."
        )

    match = MANIFEST_PATTERN.match(path.name)
    map_block = data.get("map") or {}
    manifest = MapManifest(
        path=path,
        map_id=str(
            map_block.get("name") or (match.group("map_id") if match else path.stem)
        ),
        map_version=int(match.group("version")) if match else 0,
        map_uuid=str(map_block.get("uuid") or ""),
        venue_type=map_block.get("venue_type") or None,
        geo_ref_id=geo_ref_id,
        origin=origin,
        levels=tuple(levels),
        keyframes=tuple(keyframes),
    )
    logger.info(
        "Loaded manifest %s: %d keyframes, %d level(s), venue=%s, geo_ref_id=%s",
        path.name,
        len(manifest.keyframes),
        len(manifest.levels),
        manifest.venue_type,
        manifest.geo_ref_id,
    )
    return manifest


def load_map_manifest(map_path: Path) -> MapManifest:
    """Find and load the manifest for a map directory. Raises if there is none."""
    path = find_manifest(Path(map_path))
    if path is None:
        raise FileNotFoundError(
            f"No v2 map manifest in '{map_path}'. Expected a file named "
            "'{map_id}_{version}_{date}_{time}.json'."
        )
    return load_manifest(path)
