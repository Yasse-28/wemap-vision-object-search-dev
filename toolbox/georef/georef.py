from __future__ import annotations

import functools
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import Transformer

from toolbox.logging import logger


class Coordinates:
    transformer_from_ecef = Transformer.from_crs(
        "EPSG:4978", "EPSG:4326", always_xy=True
    )
    transformer_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        level: int | None = None,
        heightFromGround: float | None = None,
        heightFromFloor: float | None = None,
        accuracy: float | None = None,
    ):
        self.latitude = latitude / 180 * np.pi
        self.longitude = longitude / 180 * np.pi
        self.altitude = altitude
        self.level = level
        self.heightFromGround = heightFromGround
        self.heightFromFloor = heightFromFloor
        self.accuracy = accuracy

    def get_latitude_deg(self) -> float:
        return self.latitude * 180 / np.pi

    def get_longitude_deg(self) -> float:
        return self.longitude * 180 / np.pi

    def get_latitude_longitude_deg(self) -> tuple[float, float]:
        return self.get_latitude_deg(), self.get_longitude_deg()

    def get_altitude(self) -> float:
        return self.altitude

    def get_level(self) -> int | None:
        return self.level

    def wds_to_ecef_rotation(self) -> np.ndarray:
        return np.asarray(_rz(-np.pi / 2 + self.longitude) @ _rx(np.pi + self.latitude))

    def ecef_to_wds_rotation(self) -> np.ndarray:
        return np.asarray(_rx(np.pi - self.latitude) @ _rz(np.pi / 2 - self.longitude))

    def to_ecef(self) -> np.ndarray:
        x, y, z = self.transformer_to_ecef.transform(
            xx=self.longitude,
            yy=self.latitude,
            zz=self.altitude,
            radians=True,
        )
        return np.asarray([x, y, z], dtype=np.float64)

    @classmethod
    def from_ecef(cls, ecef_coords: np.ndarray) -> "Coordinates":
        lng, lat, alt = cls.transformer_from_ecef.transform(
            xx=ecef_coords[0],
            yy=ecef_coords[1],
            zz=ecef_coords[2],
        )
        return cls(lat, lng, alt)


class GeoPose:
    def __init__(
        self,
        position: Coordinates,
        rot_opencv_to_wds: np.ndarray | None = None,
        tr_opencv_to_ecef: np.ndarray | None = None,
    ):
        self.position = position
        self.rot_opencv_to_wds = (
            np.eye(3, dtype=np.float64)
            if rot_opencv_to_wds is None
            else np.asarray(rot_opencv_to_wds, dtype=np.float64)
        )
        self.tr_opencv_to_ecef = (
            np.eye(4, dtype=np.float64)
            if tr_opencv_to_ecef is None
            else np.asarray(tr_opencv_to_ecef, dtype=np.float64)
        )

    @classmethod
    def from_tr_opencv_to_ecef(cls, tr_opencv_to_ecef: np.ndarray) -> "GeoPose":
        tr = np.asarray(tr_opencv_to_ecef, dtype=np.float64)
        position = Coordinates.from_ecef(tr[:3, 3])
        rot_opencv_to_ecef = tr[:3, :3]
        rot_opencv_to_wds = position.ecef_to_wds_rotation() @ rot_opencv_to_ecef
        return cls(position, rot_opencv_to_wds, tr)

    @classmethod
    def from_coordinates_and_rot_opencv_to_wds(
        cls,
        position: Coordinates,
        rot_opencv_to_wds: np.ndarray,
    ) -> "GeoPose":
        rot_opencv_to_wds = np.asarray(rot_opencv_to_wds, dtype=np.float64)
        rot_wds_to_ecef = position.wds_to_ecef_rotation()
        rot_opencv_to_ecef = rot_wds_to_ecef @ rot_opencv_to_wds
        tr = np.eye(4, dtype=np.float64)
        tr[:3, :3] = rot_opencv_to_ecef
        tr[:3, 3] = position.to_ecef()
        return cls(position, rot_opencv_to_wds, tr)


@dataclass(frozen=True)
class Level:
    id: int
    floorHeightFromGround: float
    ceilingHeightFromGround: float
    geometry: dict | None = None

    def contains(
        self, height_from_ground: float, lat_lon: tuple[float, float] | None = None
    ) -> bool:
        if not (
            self.floorHeightFromGround
            <= float(height_from_ground)
            <= self.ceilingHeightFromGround
        ):
            return False
        if self.geometry is None or lat_lon is None:
            return True

        return geojson_geometry_contains_lat_lon(self.geometry, lat_lon)


class GeoRef:
    def __init__(self, origin: Coordinates, levels: list[Level] | None = None):
        self.levels = list(levels or [])
        self.origin_coordinates = origin
        self.origin = GeoPose.from_coordinates_and_rot_opencv_to_wds(
            origin,
            np.eye(3, dtype=np.float64),
        )

    def local_to_world(self, image_opencv_to_origin: "np.ndarray | Any") -> GeoPose:
        local_tr = _se3_to_matrix(image_opencv_to_origin)
        tr_opencv_to_ecef = self.origin.tr_opencv_to_ecef @ local_tr
        geopose = GeoPose.from_tr_opencv_to_ecef(tr_opencv_to_ecef)
        self.apply_indoor_metadata(geopose.position)
        return geopose

    def local_position_to_world(self, position_world: np.ndarray) -> GeoPose:
        tr = np.eye(4, dtype=np.float64)
        tr[:3, 3] = np.asarray(position_world, dtype=np.float64)
        return self.local_to_world(tr)

    def apply_indoor_metadata(self, coords: Coordinates) -> Coordinates:
        coords.heightFromGround = coords.altitude - self.origin_coordinates.altitude
        coords.level = None
        coords.heightFromFloor = None
        level = self.level_for_coordinates(coords)
        if level is not None:
            coords.level = level.id
            coords.heightFromFloor = (
                coords.heightFromGround - level.floorHeightFromGround
            )
        return coords

    def level_for_coordinates(self, coords: Coordinates) -> Level | None:
        if coords.heightFromGround is None:
            coords.heightFromGround = coords.altitude - self.origin_coordinates.altitude
        lat_lon = coords.get_latitude_longitude_deg()
        return next(
            (
                level
                for level in self.levels
                if level.contains(float(coords.heightFromGround), lat_lon)
            ),
            None,
        )

    def get_level(self, level_id: int) -> Level | None:
        return next((level for level in self.levels if level.id == int(level_id)), None)

    def world_to_local(self, geopose: GeoPose) -> np.ndarray:
        return np.linalg.inv(self.origin.tr_opencv_to_ecef) @ geopose.tr_opencv_to_ecef


@functools.lru_cache(maxsize=None)
def load_georef_from_db(georef_db_path: Path) -> GeoRef | None:
    try:
        with sqlite3.connect(georef_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='GeoRef'"
            )
            if not cursor.fetchone():
                logger.warning("GeoRef table not found in georef.db")
                return None

            data = cursor.execute(
                "SELECT latitude, longitude, altitude FROM Georef"
            ).fetchone()
            if data is None:
                logger.warning("No origin data in GeoRef table")
                return None

            alt = data[2] if data[2] is not None else 0.0
            origin = Coordinates(latitude=data[0], longitude=data[1], altitude=alt)
            levels = _load_levels(cursor)
            return GeoRef(origin=origin, levels=levels)
    except Exception as exc:
        logger.error("Failed to load GeoRef: %s", exc)
        return None


def _load_levels(cursor: sqlite3.Cursor) -> list[Level]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Level'")
    if not cursor.fetchone():
        return []

    try:
        rows = cursor.execute(
            "SELECT id, min_altitude, max_altitude, geometry FROM Level"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = cursor.execute(
            "SELECT id, min_altitude, max_altitude FROM Level"
        ).fetchall()

    levels: list[Level] = []
    for row in rows:
        geometry = None
        if len(row) >= 4 and row[3] is not None:
            try:
                geometry = json.loads(row[3])
            except (TypeError, json.JSONDecodeError):
                logger.warning("Ignoring invalid geometry for level %s", row[0])
        levels.append(
            Level(
                id=int(row[0]),
                floorHeightFromGround=float(row[1]),
                ceilingHeightFromGround=float(row[2]),
                geometry=geometry,
            )
        )
    return levels


def georef_table_has_column(
    georef_db_path: Path,
    table: str,
    column: str,
) -> bool:
    with sqlite3.connect(georef_db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def georef_keyframe_filename_column(columns: set[str]) -> str | None:
    """Column on GeoRefKeyframe that stores the on-disk equirect image basename."""
    if "image_filename" in columns:
        return "image_filename"
    if "filename" in columns:
        return "filename"
    return None


@dataclass(frozen=True)
class GeorefKeyframeSource:
    """Equirect source paths stored for a GeoRefKeyframe row."""

    image_filename: str | None
    video_path: str | None


def _georef_keyframe_columns(georef_db_path: Path) -> set[str] | None:
    if not georef_db_path.is_file():
        return None
    try:
        with sqlite3.connect(georef_db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='GeoRefKeyframe'"
            ).fetchone()
            if row is None:
                return None
            return {
                str(item[1])
                for item in conn.execute("PRAGMA table_info(GeoRefKeyframe)").fetchall()
            }
    except sqlite3.Error:
        return None


def lookup_georef_keyframe_source(
    georef_db_path: Path,
    keyframe_id: str,
) -> GeorefKeyframeSource | None:
    """Read GeoRefKeyframe image basename and optional relative video_path."""
    columns = _georef_keyframe_columns(georef_db_path)
    if columns is None:
        return None

    select_columns: list[str] = []
    filename_column = georef_keyframe_filename_column(columns)
    if filename_column is not None:
        select_columns.append(filename_column)
    if "video_path" in columns:
        select_columns.append("video_path")
    if not select_columns:
        return None

    quoted = ", ".join(f'"{column}"' for column in select_columns)
    query = f"SELECT {quoted} FROM GeoRefKeyframe WHERE id = ?"
    try:
        with sqlite3.connect(georef_db_path) as conn:
            row = conn.execute(query, (keyframe_id,)).fetchone()
    except sqlite3.Error as exc:
        logger.warning(
            "Failed to resolve keyframe source from %s: %s",
            georef_db_path,
            exc,
        )
        return None
    if row is None:
        return None

    image_filename: str | None = None
    video_path: str | None = None
    index = 0
    if filename_column is not None:
        value = row[index]
        index += 1
        if value is not None:
            text = str(value).strip()
            image_filename = text or None
    if "video_path" in columns:
        value = row[index]
        if value is not None:
            text = str(value).strip()
            video_path = text or None

    if image_filename is None and video_path is None:
        return None
    return GeorefKeyframeSource(
        image_filename=image_filename,
        video_path=video_path,
    )


def lookup_georef_keyframe_image_filename(
    georef_db_path: Path,
    keyframe_id: str,
) -> str | None:
    """Resolve ``GeoRefKeyframe.id`` to an image basename (e.g. ``uuid.jpg``)."""
    columns = _georef_keyframe_columns(georef_db_path)
    if columns is None:
        return None
    filename_column = georef_keyframe_filename_column(columns)
    if filename_column is None:
        return None
    return _lookup_georef_image_filename_column_value(
        georef_db_path,
        keyframe_id,
        column_name=filename_column,
    )


def _lookup_georef_image_filename_column_value(
    georef_db_path: Path,
    keyframe_id: str,
    *,
    column_name: str,
) -> str | None:
    columns = _georef_keyframe_columns(georef_db_path)
    if columns is None or column_name not in columns:
        return None
    query = f'SELECT "{column_name}" FROM GeoRefKeyframe WHERE id = ?'
    try:
        with sqlite3.connect(georef_db_path) as conn:
            row = conn.execute(query, (keyframe_id,)).fetchone()
    except sqlite3.Error as exc:
        logger.warning(
            "Failed to read %s for keyframe %s from %s: %s",
            column_name,
            keyframe_id,
            georef_db_path,
            exc,
        )
        return None
    if row is None or row[0] is None:
        return None
    filename = str(row[0]).strip()
    return filename or None


def resolve_keyframe_equirect_image_path(
    map_path: Path,
    keyframe_id: str,
) -> tuple[Path | None, list[str]]:
    """
    Resolve the on-disk equirectangular image for a keyframe under ``images_360``.

    Resolution order:
    1. ``{map_path}/images_360/{image_filename}`` when ``GeoRefKeyframe.image_filename``
       exists and the row has a value.
    2. ``{map_path}/images_360/{keyframe_id}.jpg`` as fallback.
    """
    map_path = map_path.resolve()
    images_360_dir = map_path / "images_360"
    tried: list[Path] = []

    georef_db_path = map_path / "georef.db"
    columns = _georef_keyframe_columns(georef_db_path)
    if columns is not None and "image_filename" in columns:
        image_filename = _lookup_georef_image_filename_column_value(
            georef_db_path,
            keyframe_id,
            column_name="image_filename",
        )
        if image_filename:
            candidate = images_360_dir / image_filename
            tried.append(candidate)
            if candidate.is_file():
                return candidate, [str(path) for path in tried]

    fallback = images_360_dir / f"{keyframe_id}.jpg"
    tried.append(fallback)
    if fallback.is_file():
        return fallback, [str(path) for path in tried]

    return None, [str(path) for path in tried]


def load_image_filename_to_keyframe_id(georef_db_path: Path) -> dict[str, int] | None:
    """
    Load ``image filename -> GeoRefKeyframe.id`` when a filename column exists.

    Supports ``filename`` (UUID.jpg) and legacy ``image_filename`` columns.

    Returns ``None`` when georef.db is missing or has no usable filename column.
    """
    if not georef_db_path.is_file():
        return None
    try:
        with sqlite3.connect(georef_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='GeoRefKeyframe'"
            )
            if not cursor.fetchone():
                return None
            columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(GeoRefKeyframe)"
                ).fetchall()
            }
            filename_column = georef_keyframe_filename_column(columns)
            if filename_column is None:
                return None
            query = f'SELECT id, "{filename_column}" FROM GeoRefKeyframe'
            rows = cursor.execute(query).fetchall()
    except sqlite3.Error as exc:
        logger.warning(
            "Failed to load image filename mapping from %s: %s", georef_db_path, exc
        )
        return None

    mapping: dict[str, int] = {}
    for keyframe_id, image_filename in rows:
        if image_filename is None:
            continue
        filename = str(image_filename).strip()
        if not filename:
            continue
        mapping[filename] = int(keyframe_id)
    return mapping


def _se3_to_matrix(transform: "np.ndarray | Any") -> np.ndarray:
    if isinstance(transform, np.ndarray):
        tr = np.asarray(transform, dtype=np.float64)
        if tr.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform, got {tr.shape}")
        return tr

    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = np.asarray(transform.R, dtype=np.float64)
    tr[:3, 3] = np.asarray(transform.t, dtype=np.float64).reshape(3)
    return tr


def _rx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def geojson_has_polygon_geometry(geometry: dict) -> bool:
    return any(True for _ in _polygon_coordinates(geometry))


def geojson_geometry_contains_lat_lon(
    geometry: dict, lat_lon: tuple[float, float]
) -> bool:
    lat, lon = lat_lon
    return any(
        _point_in_polygon(float(lon), float(lat), polygon)
        for polygon in _polygon_coordinates(geometry)
    )


def _polygon_coordinates(geometry: dict) -> Iterable[list]:
    if geometry.get("type") == "FeatureCollection":
        for feature in geometry.get("features", []):
            yield from _polygon_coordinates(feature)
        return
    if geometry.get("type") == "Feature":
        yield from _polygon_coordinates(geometry.get("geometry") or {})
        return
    if geometry.get("type") == "Polygon":
        coordinates = geometry.get("coordinates") or []
        if coordinates:
            yield coordinates
        return
    if geometry.get("type") == "MultiPolygon":
        for polygon in geometry.get("coordinates") or []:
            if polygon:
                yield polygon


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    if not polygon or not _point_in_ring(x, y, polygon[0]):
        return False
    return not any(_point_in_ring(x, y, hole) for hole in polygon[1:])


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Coordinate frame conversion utilities
# ---------------------------------------------------------------------------

# Rotation from WDS (West, Down, South) to ENU (East, North, Up).
# WDS is the internal local map frame stored in GeoRefKeyframe.pose.
# ENU is the geospatial standard used for position_local in object-search.db.
#
# Derivation:
#   WDS basis vectors expressed in ENU coords:
#     West  → -East  → first  column [-1,  0,  0]
#     Down  → -Up    → second column [ 0,  0, -1]
#     South → -North → third  column [ 0, -1,  0]
#   det = -1 × (0×0 - (-1)×(-1)) × (-1) = 1  ✓ (proper rotation)
R_WDS_TO_ENU: np.ndarray = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

# Inverse: ENU → WDS  (= R_WDS_TO_ENU.T since it is orthogonal)
R_ENU_TO_WDS: np.ndarray = R_WDS_TO_ENU.T

# ENU → EUS (East, Up, South): 90° rotation around East (x) axis.
# Aligns with ARCore's Earth coordinate system and is Y-up for renderers.
R_ENU_TO_EUS: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def wds_to_enu(pos_wds: np.ndarray) -> np.ndarray:
    """Convert a 3D position vector from WDS local frame to ENU local frame.

    Both frames share the same metric origin (the GeoRef anchor point).
    The output is suitable for storage as ``position_local`` in object-search.db.
    """
    return R_WDS_TO_ENU @ np.asarray(pos_wds, dtype=np.float64)
