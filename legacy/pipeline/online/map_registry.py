"""Per-map object-search service + history (loaded from config)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pipeline.core.database import HISTORY_DATABASE_NAME
from pipeline.core.logging import logger
from pipeline.core.types import OBJECT_SEARCH_INDEX_DB_FILENAME
from pipeline.online.config_loader import MapEntry, load_map_entries
from pipeline.online.history import ObjectSearchHistoryService
from pipeline.online.search_service import FileBackedObjectSearchService

MapLoadState = Literal["loading", "ready", "error"]


class ObjectSearchMapRegistry:
    """One ``FileBackedObjectSearchService`` + ``ObjectSearchHistoryService``
    per configured map.

    Maps are loaded concurrently in background threads so the server becomes
    available immediately.  ``get_service()`` returns ``None`` while a map is
    still loading; callers should check ``get_map_state()`` to distinguish
    "unknown map" (not in config) from "map still loading".
    """

    def __init__(
        self,
        entries: List[MapEntry],
        device: Optional[str] = None,
    ):
        self._device = device
        self._map_ids: List[str] = []
        self._entries: Dict[str, MapEntry] = {}
        self._services: Dict[str, FileBackedObjectSearchService] = {}
        self._histories: Dict[str, ObjectSearchHistoryService] = {}
        self._states: Dict[str, MapLoadState] = {}
        self._errors: Dict[str, Exception] = {}
        self._lock = threading.Lock()

        for e in entries:
            index_db = e.map_path / OBJECT_SEARCH_INDEX_DB_FILENAME
            if not index_db.is_file():
                raise FileNotFoundError(
                    f"Map {e.map_id!r}: missing object-search index {index_db} "
                    f"(run offline build_index with --map_path {e.map_path})"
                )
            hist_path = e.map_path / HISTORY_DATABASE_NAME
            self._histories[e.map_id] = ObjectSearchHistoryService(hist_path)
            self._states[e.map_id] = "loading"
            self._entries[e.map_id] = e
            self._map_ids.append(e.map_id)

    @classmethod
    def from_config_file(
        cls, config_path: Path, device: Optional[str] = None
    ) -> ObjectSearchMapRegistry:
        _, entries = load_map_entries(config_path)
        if not entries:
            raise ValueError("Config 'maps' array is empty")
        return cls(entries, device=device)

    def start_background_loading(self) -> None:
        """Fire off one background task per map; returns immediately."""
        for entry in self._entries.values():
            asyncio.create_task(self._load_map(entry))

    async def _load_map(self, entry: MapEntry) -> None:
        logger.info("Loading map %s from %s", entry.map_id, entry.map_path)
        try:
            os_cfg = entry.object_search
            service = await asyncio.to_thread(
                lambda: FileBackedObjectSearchService(
                    entry.map_path,
                    device=self._device,
                    map_id=entry.map_id,
                    use_pgvector=os_cfg.use_pgvector,
                    pgvector_db_location=os_cfg.pgvector_db_location,
                    ef_search=os_cfg.ef_search,
                )
            )
            with self._lock:
                self._services[entry.map_id] = service
                self._states[entry.map_id] = "ready"
            logger.info("Map %s ready", entry.map_id)
        except Exception as exc:
            with self._lock:
                self._states[entry.map_id] = "error"
                self._errors[entry.map_id] = exc
            logger.error("Map %s failed to load: %s", entry.map_id, exc)

    def map_ids(self) -> List[str]:
        return list(self._map_ids)

    def get_map_state(self, map_id: str) -> Optional[MapLoadState]:
        """Return the load state for a map, or ``None`` if the map is not configured."""
        return self._states.get(map_id)

    def get_service(self, map_id: str) -> Optional[FileBackedObjectSearchService]:
        return self._services.get(map_id)

    def get_history(self, map_id: str) -> Optional[ObjectSearchHistoryService]:
        return self._histories.get(map_id)

    def close(self) -> None:
        for h in self._histories.values():
            h.close()
        self._histories.clear()
        self._services.clear()
        self._states.clear()
        self._errors.clear()
        self._map_ids.clear()
