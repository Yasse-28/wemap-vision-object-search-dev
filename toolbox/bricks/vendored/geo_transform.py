"""Coordinate frames used by the v2 backend.

# Frames

- **CameraFrame** (OpenGL convention)
  - +X right, +Y up, -Z forward (camera looks at -Z).
  - The body frame of every keyframe. ``GeoKeyframe.orientation`` rotates a
    vector from this frame to LocalFrame.

- **LocalFrame** (EUS = East-Up-South)
  - +X East, +Y Up, +Z South.
  - Origin-centred LTP at the GeoRef ``local_origin``. One LocalFrame per map.
  - ``GeoKeyframe.position`` is in this frame.

- **WorldFrame** (ECEF = Earth-Centered Earth-Fixed)
  - Earth-centred Cartesian. Translation in metres from Earth's centre.
  - Intermediate between LocalFrame and WGS84.

- **WGS84**
  - ``Coordinates(lng, lat, alt)`` — longitude/latitude in degrees, altitude in
    metres. Used to draw points on the 2D map.

- **PerVideoFrame** (raw SFM, pre-georef)
  - Same OpenGL camera-frame convention as CameraFrame, but the world frame is
    the SFM reconstruction's local frame (one per capture, arbitrary
    origin/orientation/scale). ``VideoKeyframe.position`` and
    ``VideoKeyframe.orientation`` live here. The frontend ConstraintsSfmHandler
    maps PerVideoFrame to LocalFrame.

- **ERP frame** (depth / object_search)
  - See ``backend/utils/erp.py``: ``theta = atan2(x, z)``, ``phi = asin(y)``,
    theta=0 at ERP horizontal centre. Left-handed relative to CameraFrame
    (theta=0 maps to +Z, whereas CameraFrame forward is -Z). The ``erp``
    helpers return rays already converted to CameraFrame.

# Transitions

    CameraFrame
        │ Pose (device→EUS, = GeoKeyframe.pose)
        ▼
    LocalFrame (EUS)
        │ GeoTransform.local_to_world / world_to_local
        ▼
    WorldFrame (ECEF)
        │ GeoTransform.{local|wgs84}_position(...) skip ECEF in practice
        ▼
    WGS84 (Coordinates)

    PerVideoFrame  ──(frontend ConstraintsSfmHandler)──►  LocalFrame
    ERP frame      ──(erp.theta_phi_to_opengl_ray)──►     CameraFrame

# Conventions
- Quaternions are stored and exposed as ``[w, x, y, z]`` (matches SFM
  ``output.json`` and ``QuaternionField``).
- Compass heading: NORTH = 0, EAST = π/2 (clockwise from above).
  See ``eus_compass_heading_radians``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from toolbox.bricks.vendored.maths import (
    Matrix4,
    Quaternion,
    Vector3,
    Vector3Batch,
    matrix4,
    quaternion,
    vector3,
)

# WGS84 ellipsoid parameters.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

# Body -Z is the camera's forward direction (OpenGL convention).
OPENGL_FORWARD_VECTOR: Vector3 = vector3.cast(np.array([0.0, 0.0, -1.0]))


@dataclass(frozen=True)
class Coordinates:
    lng: float
    lat: float
    alt: float = 0.0


@dataclass(frozen=True)
class Level:
    value: float
    min_altitude: float | None
    max_altitude: float | None
    # Optional WGS84 footprint. Maps with stacked buildings can share an altitude
    # band across distinct floors (e.g. two malls), so the altitude alone is
    # ambiguous; the polygon disambiguates by horizontal position, like the v1
    # server did.
    #
    # PORT NOTE: production holds a GEOS geometry here and calls `.intersects()`.
    # georef.db stores GeoJSON, and this repo has no GEOS/GDAL, so we keep the
    # parsed GeoJSON dict and test containment with the pure-Python ray-cast in
    # toolbox.georef.georef. Same semantics for the polygons we actually get.
    geometry: dict | None = None


@dataclass(frozen=True)
class Pose:
    """A rigid body pose, stored as a 4x4 homogeneous matrix.

    For a GeoKeyframe pose, ``matrix`` maps a vector from CameraFrame (OpenGL,
    -Z = forward) to LocalFrame (EUS).
    """

    matrix: Matrix4

    @property
    def position(self) -> Vector3:
        return matrix4.extract_translation(self.matrix)

    @property
    def orientation_wxyz(self) -> Quaternion:
        return quaternion.from_matrix3(matrix4.extract_rotation(self.matrix))

    def compose(self, other: "Pose") -> "Pose":
        return Pose(matrix4.multiply(self.matrix, other.matrix))

    def invert(self) -> "Pose":
        return Pose(matrix4.invert(self.matrix))

    def rotate(self, v: Vector3) -> Vector3:
        return Vector3(matrix4.extract_rotation(self.matrix) @ v)

    def apply(self, v: Vector3) -> Vector3:
        return matrix4.multiply_vector3(self.matrix, v)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(matrix4.identity())

    @classmethod
    def from_position_orientation(
        cls, position: Vector3, orientation_wxyz: Quaternion
    ) -> "Pose":
        rotation = quaternion.to_matrix3(orientation_wxyz)
        return cls(matrix4.compose(position, rotation))


def _wgs84_point(lat: float | None, lng: float | None) -> tuple[float, float] | None:
    """A ``(lat, lng)`` pair, or ``None`` if either coord is missing.

    PORT NOTE: production returns a GEOS ``Point(lng, lat, srid=4326)``. We return
    a plain tuple and let ``_geometry_contains`` do the containment test, so this
    module needs no GEOS/GDAL.
    """
    if lat is None or lng is None:
        return None
    return (float(lat), float(lng))


def _geometry_contains(geometry: dict, point: tuple[float, float]) -> bool:
    """Whether a GeoJSON footprint contains a ``(lat, lng)`` point.

    Stands in for GEOS ``geometry.intersects(point)``. Imported lazily to keep
    this module free of a hard dependency on the georef reader.
    """
    from toolbox.georef.georef import geojson_geometry_contains_lat_lon

    return bool(geojson_geometry_contains_lat_lon(geometry, point))


def eus_compass_heading_radians(v: Vector3) -> float:
    """Compass heading of an EUS vector: NORTH = 0, EAST = π/2 (clockwise from above)."""
    return float(np.arctan2(v[0], -v[2]))


def eus_compass_heading_radians_batch(v: Vector3Batch) -> np.ndarray:
    return np.arctan2(v[:, 0], -v[:, 2])


def _wgs84_to_ecef(lng_deg: float, lat_deg: float, alt: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lng = np.radians(lng_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lng = np.sin(lng)
    cos_lng = np.cos(lng)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt) * cos_lat * cos_lng,
            (n + alt) * cos_lat * sin_lng,
            (n * (1.0 - _WGS84_E2) + alt) * sin_lat,
        ],
        dtype=np.float64,
    )


def _eus_to_ecef_rotation(lng_deg: float, lat_deg: float) -> np.ndarray:
    """3x3 rotation mapping EUS (East-Up-South) → ECEF, at the given geodetic origin."""
    lat = np.radians(lat_deg)
    lng = np.radians(lng_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lng, cos_lng = np.sin(lng), np.cos(lng)
    return np.array(
        [
            [-sin_lng, cos_lat * cos_lng, sin_lat * cos_lng],
            [cos_lng, cos_lat * sin_lng, sin_lat * sin_lng],
            [0.0, sin_lat, -cos_lat],
        ],
        dtype=np.float64,
    )


def _ecef_to_wgs84_batch(ecef: np.ndarray) -> np.ndarray:
    """ECEF (N, 3) → WGS84 (N, 3) as [lng, lat, alt] degrees/metres. Bowring iterative method."""
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    lng = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    sin_lat = np.sin(lat)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - n
    return np.column_stack((np.degrees(lng), np.degrees(lat), alt))


def _origin_from_point_or_tuple(obj) -> Coordinates:
    """Accept a Django Point (3D), a Coordinates, or a (lng, lat, alt) iterable."""
    if isinstance(obj, Coordinates):
        return obj
    x = getattr(obj, "x", None)
    if x is not None:
        return Coordinates(float(obj.x), float(obj.y), float(getattr(obj, "z", 0.0)))
    lng, lat, alt = obj
    return Coordinates(float(lng), float(lat), float(alt))


@dataclass(frozen=True)
class GeoTransform:
    """EUS↔ECEF↔WGS84 anchored at a GeoRef origin, plus altitude→level resolution.

    Pure math at runtime: no Django dependency once constructed. Use the
    factories ``from_geo_ref`` / ``from_local_origin`` to build from a Django
    object; or pass a ``Coordinates`` directly.
    """

    origin: Coordinates
    levels: tuple[Level, ...] = field(default_factory=tuple)
    eus_to_ecef: Matrix4 = field(init=False)
    ecef_to_eus: Matrix4 = field(init=False)

    def __post_init__(self) -> None:
        rotation_eus_to_ecef = _eus_to_ecef_rotation(self.origin.lng, self.origin.lat)
        origin_ecef = _wgs84_to_ecef(self.origin.lng, self.origin.lat, self.origin.alt)
        eus_to_ecef = matrix4.compose(vector3.cast(origin_ecef), rotation_eus_to_ecef)
        # __post_init__ on a frozen dataclass requires object.__setattr__.
        object.__setattr__(self, "eus_to_ecef", eus_to_ecef)
        object.__setattr__(self, "ecef_to_eus", matrix4.invert(eus_to_ecef))

    # ---- Factories ----------------------------------------------------------

    @classmethod
    def from_georef_db(cls, georef_db_path: Path) -> "GeoTransform":
        """Build from an on-disk ``georef.db`` instead of a Django ``GeoRef`` row.

        PORT NOTE: replaces production's ``from_geo_ref(geo_ref)``. Two things to
        keep straight, because getting either wrong silently yields ``level=None``
        for every candidate:

        1. **Axis order.** ``GeoRef.local_origin`` is a Django ``PointField`` i.e.
           ``(lng, lat, alt)``; georef.db stores named ``latitude``/``longitude``
           columns. We read the names, so no ordering ambiguity here.
        2. **Altitude datum.** georef.db's ``Level.min_altitude``/``max_altitude``
           are *height above the georef origin*, not absolute altitude (see
           ``GeoRef.apply_indoor_metadata``, which derives
           ``heightFromGround = altitude - origin.altitude``). That matches what
           ``levels_for_altitudes`` is fed everywhere in this codebase — the EUS
           local-up coordinate — so the bands transfer across unchanged. Do not
           "fix" this by adding the origin altitude.
        """
        from toolbox.georef.georef import load_georef_from_db

        georef = load_georef_from_db(Path(georef_db_path))
        if georef is None:
            raise ValueError(f"Could not load a GeoRef from '{georef_db_path}'.")

        origin = Coordinates(
            lng=georef.origin_coordinates.get_longitude_deg(),
            lat=georef.origin_coordinates.get_latitude_deg(),
            alt=float(georef.origin_coordinates.altitude),
        )
        levels = tuple(
            Level(
                value=float(level.id),
                min_altitude=level.floorHeightFromGround,
                max_altitude=level.ceilingHeightFromGround,
                geometry=level.geometry,
            )
            for level in georef.levels
        )
        return cls(origin=origin, levels=levels)

    @classmethod
    def from_local_origin(cls, local_origin) -> "GeoTransform":
        return cls(origin=_origin_from_point_or_tuple(local_origin))

    # ---- Frame transforms ---------------------------------------------------

    def local_to_world(self, pose: Pose) -> Pose:
        return Pose(matrix4.multiply(self.eus_to_ecef, pose.matrix))

    def world_to_local(self, pose: Pose) -> Pose:
        return Pose(matrix4.multiply(self.ecef_to_eus, pose.matrix))

    def local_position_to_wgs84(self, eus_xyz: Vector3) -> Coordinates:
        out = self.local_positions_to_wgs84(Vector3Batch(eus_xyz[None, :]))
        return Coordinates(float(out[0, 0]), float(out[0, 1]), float(out[0, 2]))

    def wgs84_to_local_position(self, coord: Coordinates) -> Vector3:
        ecef = _wgs84_to_ecef(coord.lng, coord.lat, coord.alt)
        rotation = matrix4.extract_rotation(self.eus_to_ecef)
        origin_ecef = matrix4.extract_translation(self.eus_to_ecef)
        return Vector3(rotation.T @ (ecef - origin_ecef))

    def local_positions_to_wgs84(self, eus_xyz_batch: Vector3Batch) -> Vector3Batch:
        rotation = matrix4.extract_rotation(self.eus_to_ecef)
        origin_ecef = matrix4.extract_translation(self.eus_to_ecef)
        ecef = eus_xyz_batch @ rotation.T + origin_ecef
        return Vector3Batch(_ecef_to_wgs84_batch(ecef))

    def wgs84_to_local_positions(self, wgs84_batch: Vector3Batch) -> Vector3Batch:
        lat = np.radians(wgs84_batch[:, 1])
        lng = np.radians(wgs84_batch[:, 0])
        alt = wgs84_batch[:, 2]
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lng, cos_lng = np.sin(lng), np.cos(lng)
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        ecef = np.column_stack(
            [
                (n + alt) * cos_lat * cos_lng,
                (n + alt) * cos_lat * sin_lng,
                (n * (1.0 - _WGS84_E2) + alt) * sin_lat,
            ]
        )
        rotation = matrix4.extract_rotation(self.eus_to_ecef)
        origin_ecef = matrix4.extract_translation(self.eus_to_ecef)
        return Vector3Batch((ecef - origin_ecef) @ rotation)

    # ---- Level resolution ---------------------------------------------------

    def level_for_altitude(
        self, altitude: float, lat: float | None = None, lng: float | None = None
    ) -> float | None:
        point = _wgs84_point(lat, lng)
        for level in self.levels:
            if level.min_altitude is not None and altitude < level.min_altitude:
                continue
            if level.max_altitude is not None and altitude > level.max_altitude:
                continue
            if (
                point is not None
                and level.geometry is not None
                and not _geometry_contains(level.geometry, point)
            ):
                continue
            return level.value
        return None

    def levels_for_altitudes(
        self,
        altitudes: np.ndarray,
        lats: np.ndarray | None = None,
        lngs: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.full(altitudes.shape, np.nan, dtype=np.float64)
        points = None
        if lats is not None and lngs is not None:
            points = [_wgs84_point(la, ln) for la, ln in zip(lats, lngs)]
        # Scan levels in order; first match wins. Vectorised mask.
        for level in self.levels:
            mask = np.isnan(out)
            if level.min_altitude is not None:
                mask &= altitudes >= level.min_altitude
            if level.max_altitude is not None:
                mask &= altitudes <= level.max_altitude
            if points is not None and level.geometry is not None and mask.any():
                # Drop band-matching points that fall outside this floor's footprint.
                # ponytail: per-point GEOS test, O(points × levels); fine at our
                # candidate counts, switch to prepared/STRtree if it ever bites.
                idx = np.nonzero(mask)[0]
                inside = np.array(
                    [
                        points[i] is not None
                        and _geometry_contains(level.geometry, points[i])
                        for i in idx
                    ],
                    dtype=bool,
                )
                mask[idx[~inside]] = False
            out[mask] = level.value
        return out
