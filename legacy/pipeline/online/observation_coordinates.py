from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.core.logging import logger
from pipeline.core.types import LoadedIndex

if TYPE_CHECKING:
    from pipeline.offline.localize.georef import GeoRef


def _load_georef_from_db(georef_db_path: Path) -> "GeoRef | None":
    # Lazy import keeps the online API importable in environments without pyproj.
    from pipeline.offline.localize.georef import load_georef_from_db

    return load_georef_from_db(georef_db_path)


def stored_observation_coordinates(
    *,
    index: LoadedIndex,
    map_path: Path,
    object_indices: list[int],
) -> dict[int, tuple[float, float, float]]:
    """Return per-object observation coordinates from stored localization arrays."""
    # object_positions_world now stores geographic [lat, lon, alt] float32 directly
    if index.object_positions_world is None or index.object_localization_valid is None:
        return {}

    coordinates: dict[int, tuple[float, float, float]] = {}
    for obj_idx in object_indices:
        if obj_idx < 0 or obj_idx >= len(index.object_positions_world):
            continue
        if not bool(index.object_localization_valid[obj_idx]):
            continue
        position_world = index.object_positions_world[obj_idx]  # [lat, lon, alt]
        if not np.all(np.isfinite(position_world)):
            continue
        coordinates[obj_idx] = (
            float(position_world[0]),
            float(position_world[1]),
            float(position_world[2]),
        )
    return coordinates


def online_observation_coordinates(
    *,
    map_path: Path,
    object_position_wds: dict[int, np.ndarray],
) -> dict[int, tuple[float, float, float]]:
    """Return per-object observation coordinates from request-time localized
    positions."""
    if not object_position_wds:
        return {}

    coordinates: dict[int, tuple[float, float, float]] = {}
    georef_db_path = Path(map_path) / "georef.db"
    if not georef_db_path.exists():
        logger.warning(
            "Observation coordinates unavailable: %s not found", georef_db_path
        )
        return {}
    try:
        georef = _load_georef_from_db(georef_db_path)
    except ModuleNotFoundError as exc:
        logger.warning(
            "Observation coordinates unavailable: missing dependency: %s", exc
        )
        return {}
    if georef is None:
        logger.warning(
            "Observation coordinates unavailable: failed to load %s", georef_db_path
        )
        return {}

    for obj_idx, position_wds in object_position_wds.items():
        if not np.all(np.isfinite(position_wds)):
            continue
        try:
            geopose = georef.local_position_to_world(
                np.asarray(position_wds, dtype=np.float64)
            )
        except Exception as exc:
            logger.warning(
                "Observation coordinates unavailable for object %d: %s", obj_idx, exc
            )
            continue
        coordinates[obj_idx] = (
            float(geopose.position.get_latitude_deg()),
            float(geopose.position.get_longitude_deg()),
            float(geopose.position.get_altitude()),
        )
    return coordinates
