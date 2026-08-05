"""pgvector store for object embeddings — per-map table, L2 / halfvec.

Mirrors the production backend index parameters: ``halfvec`` storage, an HNSW
index with ``halfvec_l2_ops`` (queried with ``<->``), ``m=16``,
``ef_construction=64``. MetaCLIP embeddings are unit-norm, so L2 ranking is
identical to cosine but cheaper (no per-comparison normalization).

Topology is one lean table per map — ``(id BIGINT PK = object_idx,
embedding halfvec(dim))`` — so queries need no map filter (hence no partial
index / iterative_scan), and a rebuild is just DROP + CREATE. Everything else
needed for clustering stays in object-search.db and is fetched by id.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from pipeline.core.pgvector import DatabaseState, create_connection_metaclip

if TYPE_CHECKING:
    import psycopg

_MEM_RE = re.compile(r"^\d+\s*(kB|MB|GB|TB)?$", re.IGNORECASE)

# Same as the production backend (backend/object_search/db/ingest.py).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def table_name(map_id: str) -> str:
    """Sanitized per-map table name, e.g. ``map_vinci_st_domingue_objects``."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(map_id)).strip("_").lower()
    return f"map_{slug}_objects"


def db_state_from_location(location: str, *, write: bool = False) -> DatabaseState:
    """Map a ``pgvectorDbLocation`` ('local'/'aws') to a DatabaseState.

    ``write`` selects read/write access: ingest needs write, serving is
    read-only. 'aws' has separate writer/reader endpoints, so it resolves to
    ``AWS_READ_WRITE`` (write) or ``AWS_READ_ONLY`` (read). 'local' is always
    read-write (single endpoint).
    """
    loc = str(location).lower()
    if loc == "local":
        return DatabaseState.LOCAL_READ_WRITE
    if loc == "aws":
        return DatabaseState.AWS_READ_WRITE if write else DatabaseState.AWS_READ_ONLY
    raise ValueError(
        f"Unknown pgvector db location: {location!r} (expected 'local' or 'aws')"
    )


def connect(
    state: DatabaseState = DatabaseState.LOCAL_READ_WRITE,
) -> psycopg.Connection:
    """Open a connection, ensure the vector extension, register adapters.

    Autocommit is on: the long-lived serving connection runs read-only SELECTs,
    and leaving them in an open transaction would sit "idle in transaction",
    pinning the xmin horizon and blocking VACUUM from reclaiming dead tuples
    DB-wide. Autocommit also lets COPY / CREATE INDEX commit per statement.
    """
    from pgvector.psycopg import register_vector

    conn = create_connection_metaclip(state)
    conn.autocommit = True
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_table(
    conn: psycopg.Connection,
    table: str,
    dim: int,
    *,
    unlogged: bool = False,
) -> None:
    """(Re)create an empty lean table ``(id BIGINT PK, embedding halfvec(dim))``.

    ``unlogged=True`` skips WAL for faster COPY + index build; the table is
    truncated on an unclean shutdown, which is fine here because it is fully
    rebuildable from object-search.db (just re-run the ingest).
    """
    kind = "UNLOGGED TABLE" if unlogged else "TABLE"
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(
        f'CREATE {kind} "{table}" (id BIGINT PRIMARY KEY, embedding halfvec({dim}))'
    )
    conn.commit()


def copy_embeddings(
    conn: psycopg.Connection,
    table: str,
    ids: np.ndarray,
    embeddings: np.ndarray,
) -> int:
    """Binary-COPY ``(id, embedding)`` rows. Returns the number written."""
    from pgvector import HalfVector

    with conn.cursor() as cur:
        with cur.copy(
            f'COPY "{table}" (id, embedding) FROM STDIN WITH (FORMAT BINARY)'
        ) as copy:
            copy.set_types(["int8", "halfvec"])
            for row_id, emb in zip(ids.tolist(), embeddings):
                copy.write_row([int(row_id), HalfVector(emb)])
    conn.commit()
    return len(ids)


def create_l2_hnsw_index(
    conn: psycopg.Connection,
    table: str,
    *,
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
    maintenance_work_mem: str = "2GB",
    parallel_workers: int = 0,
) -> None:
    """Build the HNSW index over ``embedding`` with ``halfvec_l2_ops``.

    Build accelerators (the build, not the COPY, dominates ingest time):
    - ``maintenance_work_mem``: must be big enough to hold the whole HNSW graph
      in RAM, else pgvector builds it on disk (orders of magnitude slower). For
      ~N rows of D-dim halfvec, budget roughly ``N * (D*2 bytes + ~150)``.
    - ``parallel_workers`` (default 0 = serial): pgvector can parallelize the
      build, but parallel workers coordinate through a shared-memory segment
      sized to ``maintenance_work_mem`` in /dev/shm. Enable it ONLY where
      /dev/shm >= maintenance_work_mem (Docker default is 64 MB → set shm_size;
      managed RDS is instance-dependent), else the build fails with DiskFull on
      the shared segment. Serial uses private backend RAM and is always safe.
    """
    if maintenance_work_mem:
        if not _MEM_RE.match(maintenance_work_mem.strip()):
            raise ValueError(f"Invalid maintenance_work_mem: {maintenance_work_mem!r}")
        conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem.strip()}'")
    # Always set explicitly: the server default is 2, so parallel_workers=0 must
    # set it to 0 to TRULY disable parallelism — otherwise a parallel build still
    # runs and allocates a /dev/shm segment (~maintenance_work_mem), which fails
    # with DiskFull where /dev/shm is small (Docker default, managed RDS).
    conn.execute(
        f"SET max_parallel_maintenance_workers = {max(0, int(parallel_workers))}"
    )
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS "{table}_emb_l2_idx" ON "{table}" '
        f"USING hnsw (embedding halfvec_l2_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
    )
    conn.commit()


def set_ef_search(conn: psycopg.Connection, ef_search: int) -> None:
    """Set HNSW ef_search for subsequent queries (should be >= the LIMIT)."""
    conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")


def query_topk(
    conn: psycopg.Connection,
    table: str,
    query: np.ndarray,
    k: int,
) -> tuple[list[int], np.ndarray]:
    """Top-``k`` by L2 (``<->``). Returns ``(ids, cosine_similarities)``.

    For unit-norm vectors ``||a-b||^2 = 2 - 2 cos``, so cosine = 1 - dist^2 / 2;
    we return that so downstream ranking/thresholds match the in-RAM cosine path.
    """
    from pgvector import HalfVector

    qv = HalfVector(np.asarray(query, dtype=np.float32))
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT id, embedding <-> %s AS dist FROM "{table}" '
            f"ORDER BY embedding <-> %s LIMIT %s",
            (qv, qv, int(k)),
        )
        rows = cur.fetchall()
    ids = [int(r[0]) for r in rows]
    dist = np.asarray([float(r[1]) for r in rows], dtype=np.float32)
    sims = 1.0 - (dist * dist) / 2.0
    return ids, sims
