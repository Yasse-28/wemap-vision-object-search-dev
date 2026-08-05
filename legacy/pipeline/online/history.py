"""SQLite request history (async queue), compatible with legacy Request schema."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Event, Thread

from pipeline.core.database import ensure_history_schema
from pipeline.core.logging import logger

_STOP_SENTINEL = object()


class ObjectSearchHistoryService:
    """Background writer; `time_db_ms` column stores retrieval time for API parity."""

    def __init__(self, history_path: Path):
        ensure_history_schema(history_path)
        self.database_path = history_path
        self._enabled = Event()
        self._enabled.set()
        self._queue: Queue = Queue()
        self._thread = Thread(target=self._run, daemon=False)
        self._thread.start()

    def _run(self) -> None:
        try:
            conn = sqlite3.connect(str(self.database_path))
        except Exception:
            logger.exception("History DB connect failed: %s", self.database_path)
            self._enabled.clear()
            return

        while True:
            item = self._queue.get()
            if item is _STOP_SENTINEL:
                conn.commit()
                conn.close()
                break

            (
                prompt,
                search_type,
                enforced,
                timestamp,
                time_router_ms,
                time_embedding_ms,
                time_db_ms,
            ) = item
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            try:
                conn.execute(
                    "INSERT INTO Request(prompt, search_type, enforced,"
                    " timestamp, datetime, "
                    "time_router_ms, time_embedding_ms, time_db_ms)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (
                        prompt,
                        search_type,
                        int(enforced),
                        timestamp,
                        dt,
                        time_router_ms,
                        time_embedding_ms,
                        time_db_ms,
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception("History insert failed for prompt=%r", prompt)
                conn.rollback()

    def store_request(
        self,
        prompt: str,
        search_type: str,
        enforced: bool,
        timestamp: int,
        time_router_ms: int,
        time_embedding_ms: int,
        time_db_ms: int,
    ) -> None:
        if self._enabled.is_set():
            self._queue.put(
                (
                    prompt,
                    search_type,
                    enforced,
                    timestamp,
                    time_router_ms,
                    time_embedding_ms,
                    time_db_ms,
                )
            )

    def close(self, timeout: float = 5.0) -> None:
        if not self._enabled.is_set():
            return
        self._enabled.clear()
        self._queue.put(_STOP_SENTINEL)
        self._thread.join(timeout=timeout)
