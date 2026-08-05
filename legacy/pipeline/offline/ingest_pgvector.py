"""Populate a per-map pgvector table from an object-search.db.

Streams ``(object_idx, embedding)`` from the SQLite ``object`` table in keyset
chunks (so a 15 M-row map never materializes all blobs at once), binary-COPYs
each chunk into ``map_<id>_objects`` as ``halfvec``, then builds the L2 HNSW
index (same params as the production backend). Re-running rebuilds the table.

Usage (from the object-search/ repo root):

    python -m pipeline.offline.ingest_pgvector \\
        --map_path /path/to/<map> --map-id <map_id>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.core import pgvector_store  # noqa: E402
from pipeline.core.logging import logger  # noqa: E402
from pipeline.core.types import OBJECT_SEARCH_INDEX_DB_FILENAME  # noqa: E402


def _iter_embedding_chunks(
    db_path: str, chunk_size: int
) -> "Iterator[tuple[np.ndarray, np.ndarray]]":
    """Yield ``(ids, embeddings)`` chunks ordered by object_idx (keyset paging)."""
    conn = sqlite3.connect(db_path)
    try:
        last = -1
        while True:
            rows = conn.execute(
                "SELECT object_idx, embedding FROM object "
                "WHERE length(embedding) > 4 AND object_idx > ? "
                "ORDER BY object_idx LIMIT ?",
                (last, chunk_size),
            ).fetchall()
            if not rows:
                return
            last = int(rows[-1][0])
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            embs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            yield ids, embs
    finally:
        conn.close()


def _embedding_dim(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT embedding FROM object WHERE length(embedding) > 4 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"No object embeddings in {db_path}")
    return int(np.frombuffer(row[0], dtype=np.float32).shape[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate per-map pgvector table from object-search.db"
    )
    p.add_argument("--map_path", type=Path, required=True)
    p.add_argument(
        "--map-id",
        type=str,
        default=None,
        help="table key; defaults to the map folder name "
        "(e.g. $VPS_DATA/maps/sample-map -> 'sample-map'). "
        "Must match the map's 'id' in the serving config.json.",
    )
    p.add_argument(
        "--db-location",
        choices=("local", "aws"),
        default="local",
        help="which pgvector database to write to; must match the map's "
        "objectSearch.pgvectorDbLocation in the serving config.json.",
    )
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument(
        "--maintenance-work-mem",
        type=str,
        default="2GB",
        help="HNSW build memory; large enough to hold the graph in RAM avoids a "
        "slow on-disk build (bound by the postgres container's RAM).",
    )
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="parallel HNSW build workers (0 = serial, the safe default). "
        "Parallel needs /dev/shm >= maintenance_work_mem on the server "
        "(Docker: set shm_size); otherwise the build fails with DiskFull "
        "on a shared-memory segment.",
    )
    p.add_argument(
        "--unlogged",
        action="store_true",
        help="create an UNLOGGED table (faster COPY + build, no WAL); rebuildable "
        "from object-search.db so the lost-on-crash tradeoff is safe here.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    map_path = args.map_path.resolve()
    db_path = str(map_path / OBJECT_SEARCH_INDEX_DB_FILENAME)
    if not Path(db_path).is_file():
        raise SystemExit(f"Missing index db: {db_path}")
    map_id = args.map_id or map_path.name
    table = pgvector_store.table_name(map_id)
    dim = _embedding_dim(db_path)

    logger.info(
        "pgvector ingest: map_id=%s table=%s dim=%d db=%s",
        map_id,
        table,
        dim,
        args.db_location,
    )
    conn = pgvector_store.connect(
        pgvector_store.db_state_from_location(args.db_location, write=True)
    )
    try:
        pgvector_store.create_table(conn, table, dim, unlogged=args.unlogged)
        t0 = time.perf_counter()
        total = 0
        for ids, embs in _iter_embedding_chunks(db_path, args.chunk_size):
            total += pgvector_store.copy_embeddings(conn, table, ids, embs)
            logger.info("  copied %d rows", total)
        logger.info("COPY done: %d rows in %.1fs", total, time.perf_counter() - t0)

        t1 = time.perf_counter()
        pgvector_store.create_l2_hnsw_index(
            conn,
            table,
            maintenance_work_mem=args.maintenance_work_mem,
            parallel_workers=args.parallel_workers,
        )
        logger.info(
            "HNSW (l2, m=%d, ef_c=%d, maint_mem=%s, workers=%d) built in %.1fs",
            pgvector_store.HNSW_M,
            pgvector_store.HNSW_EF_CONSTRUCTION,
            args.maintenance_work_mem,
            args.parallel_workers,
            time.perf_counter() - t1,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
