import sqlite3
from pathlib import Path

from pipeline.core.database import HISTORY_DATABASE_NAME
from pipeline.online.history import ObjectSearchHistoryService


def test_history_service_writes_request_rows_under_map_path(tmp_path: Path):
    history_path = tmp_path / HISTORY_DATABASE_NAME
    history = ObjectSearchHistoryService(history_path)

    history.store_request(
        "ticket machine",
        "object",
        True,
        1_713_456_789,
        1,
        23,
        45,
    )
    history.close()

    conn = sqlite3.connect(str(history_path))
    try:
        row = conn.execute(
            "SELECT prompt, search_type, enforced, timestamp, "
            "time_router_ms, time_embedding_ms, time_db_ms FROM Request"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("ticket machine", "object", 1, 1_713_456_789, 1, 23, 45)
