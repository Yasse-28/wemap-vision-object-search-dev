"""Bulk ingest of object-search candidates into ``object_search_candidate``.

Ported from `backend/object_search/db/ingest.py`. **The only change is where the
connection comes from**: production grabs Django's `connection`, we take a
psycopg2 connection as the first argument. Every byte of the COPY encoding, the
advisory-lock protocol and the HNSW parameters are unchanged — this is the part of
the port where a "small improvement" is a bug, because the encoding is validated
only by the server accepting it.

Per-row ``INSERT`` is the bottleneck of the index step (measured ~720s for a
mid-size capture). Binary ``COPY ... FROM STDIN`` brings it down to ~6s for
200 K rows — see third_party/object_search/BENCHMARK.md for measurements.
"""

from __future__ import annotations

import struct
import time

import numpy as np

INDEX_NAME_TEMPLATE = "idx_object_search_candidate_hnsw_georef_{geo_ref_id}"

# Serializes CREATE INDEX CONCURRENTLY across writers: two CICs on the *same*
# table (object_search_candidate) deadlock, which parallel ingest / v1-migration
# jobs hit. Arbitrary fixed key — every writer takes the same one, so only one
# builds at a time.
_HNSW_INDEX_LOCK_KEY = 0x4F53_4348_4E53_5758  # "OSCHNSWX"

_COLUMNS = (
    "geo_ref_id",
    "geokeyframe_id",
    "theta_center",
    "phi_center",
    "angular_width",
    "angular_height",
    "embedding",
    "thumbnail",
    "depth",
    "object_position",
    "detector_source",
    "label",
    "detection_score",
)

_N_FIELDS = len(_COLUMNS)

# PostgreSQL binary COPY envelope (see https://www.postgresql.org/docs/current/sql-copy.html)
_PGCOPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack(">II", 0, 0)  # 19 bytes
_PGCOPY_TRAILER = struct.pack(">h", -1)

# PostGIS EWKB prefix for a little-endian POINTZ with srid=0:
#   0x01                — byte-order marker (little-endian)
#   0xA0000001 (LE)     — type: Point(1) | HasZ(0x80000000) | HasSRID(0x20000000)
#   0x00000000 (LE)     — SRID = 0
# Followed by three float64 (LE): x, y, z  → total 33 bytes.
_EWKB_POINTZ_PREFIX = (
    b"\x01" + struct.pack("<I", 0xA0000001) + struct.pack("<i", 0)
)  # 9 bytes; append struct.pack('<ddd', x, y, z) to complete the 33-byte EWKB


class _ChunkReader:
    """Expose a bytes-chunk iterator as the ``read(size)`` file object that
    psycopg2's ``copy_expert`` consumes.

    Lets the COPY payload stream to the server one row at a time instead of
    being materialised in a single in-memory buffer, so ingest memory stays
    flat regardless of candidate count.
    """

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self._buf = bytearray()
        self._done = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            self._buf.extend(b"".join(self._chunks))
            self._done = True
            data = bytes(self._buf)
            self._buf.clear()
            return data
        while len(self._buf) < size and not self._done:
            try:
                self._buf.extend(next(self._chunks))
            except StopIteration:
                self._done = True
        data = bytes(self._buf[:size])
        del self._buf[:size]
        return data


def encode_copy_stream(
    *,
    geo_ref_id: int,
    geokeyframe_ids: np.ndarray,
    theta_center: np.ndarray,
    phi_center: np.ndarray,
    angular_width: np.ndarray,
    angular_height: np.ndarray,
    embeddings: np.ndarray,
    thumbnail_keys: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    detector_sources: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    detection_scores: np.ndarray | None = None,
):
    """Yield the binary COPY stream: header, one bytes blob per row, trailer.

    Split out of ``bulk_copy`` (which is otherwise a verbatim port) so the
    encoding can be asserted byte-for-byte in tests without a live Postgres. The
    bytes are identical to what production streams.
    """
    n, dim = embeddings.shape

    # Pre-compute row-invariant byte sequences.
    pack = struct.pack
    n_fields_hdr = pack(">h", _N_FIELDS)
    geo_ref_id_bytes = pack(">iq", 8, geo_ref_id)  # int32 length + int64 value
    halfvec_field_len = pack(">i", 4 + dim * 2)  # int32 length: hdr(4) + data(dim*2)
    halfvec_hdr = pack(">HH", dim, 0)  # uint16 dim + uint16 unused

    yield _PGCOPY_HEADER
    for i in range(n):
        row = [
            n_fields_hdr,
            # geo_ref_id: bigint
            geo_ref_id_bytes,
            # geokeyframe_id: bigint
            pack(">iq", 8, int(geokeyframe_ids[i])),
            # theta_center, phi_center, angular_width, angular_height: float64
            pack(">id", 8, float(theta_center[i])),
            pack(">id", 8, float(phi_center[i])),
            pack(">id", 8, float(angular_width[i])),
            pack(">id", 8, float(angular_height[i])),
            # embedding: halfvec — uint16 dim, uint16 unused, dim×float16 (BE).
            # '>f2' byte-swaps this row's slice on little-endian hosts.
            halfvec_field_len,
            halfvec_hdr,
            embeddings[i].astype(">f2").tobytes(),
        ]
        # thumbnail: text (UTF-8) or NULL
        if thumbnail_keys is not None and thumbnail_keys[i]:
            tk = str(thumbnail_keys[i]).encode()
            row.append(pack(">i", len(tk)))
            row.append(tk)
        else:
            row.append(pack(">i", -1))
        # depth: float8 or NULL (NaN/None → NULL = no depth available)
        if depths is None:
            row.append(pack(">i", -1))
        else:
            d = depths[i]
            is_null = d is None
            df = 0.0
            if not is_null:
                try:
                    df = float(d)
                    is_null = np.isnan(df)
                except (TypeError, ValueError):
                    is_null = True
            if is_null:
                row.append(pack(">i", -1))
            else:
                row.append(pack(">id", 8, df))
        # object_position: PostGIS EWKB POINTZ (srid=0) or NULL
        # NaN in any coordinate → NULL (depth unavailable for this candidate)
        if object_positions is None:
            row.append(pack(">i", -1))
        else:
            px, py, pz = (
                float(object_positions[i, 0]),
                float(object_positions[i, 1]),
                float(object_positions[i, 2]),
            )
            if np.isnan(px) or np.isnan(py) or np.isnan(pz):
                row.append(pack(">i", -1))
            else:
                ewkb = _EWKB_POINTZ_PREFIX + pack("<ddd", px, py, pz)
                row.append(pack(">i", len(ewkb)))
                row.append(ewkb)
        # detector_source, label: text or NULL. NULL when the column is
        # absent (arg is None), the value is missing (None or a float NaN —
        # how pandas materialises parquet string nulls, e.g. gdino rows carry
        # no label), or the string is empty.
        for col in (detector_sources, labels):
            v = None if col is None else col[i]
            if v is not None and not (isinstance(v, float) and v != v) and str(v):
                s = str(v).encode()
                row.append(pack(">i", len(s)))
                row.append(s)
            else:
                row.append(pack(">i", -1))
        # detection_score: float8 or NULL (NaN/None → NULL)
        if detection_scores is None:
            row.append(pack(">i", -1))
        else:
            sc = detection_scores[i]
            is_null = sc is None
            scf = 0.0
            if not is_null:
                try:
                    scf = float(sc)
                    is_null = np.isnan(scf)
                except (TypeError, ValueError):
                    is_null = True
            if is_null:
                row.append(pack(">i", -1))
            else:
                row.append(pack(">id", 8, scf))
        yield b"".join(row)
    yield _PGCOPY_TRAILER


def bulk_copy(
    conn,
    *,
    geo_ref_id: int,
    geokeyframe_ids: np.ndarray,
    theta_center: np.ndarray,
    phi_center: np.ndarray,
    angular_width: np.ndarray,
    angular_height: np.ndarray,
    embeddings: np.ndarray,
    thumbnail_keys: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    object_positions: np.ndarray | None = None,
    detector_sources: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    detection_scores: np.ndarray | None = None,
) -> int:
    """Bulk-insert candidate rows for one georef using binary ``COPY FROM STDIN``.

    All array arguments must share the same length ``n``. ``embeddings`` has
    shape ``(n, dim)``, dtype castable to float16 (pgvector halfvec).
    ``object_positions`` has shape ``(n, 3)`` (x, y, z in EUS/LocalFrame) or
    ``None`` to write NULL for every row. Rows where any coordinate is NaN are
    also written as NULL. Returns the number of rows written.

    Binary COPY avoids float→text serialization overhead. The payload is
    streamed to the server row by row (see ``_ChunkReader``), so peak memory is
    one row rather than the whole batch — the ingest CLI calls this once per
    video capture to keep memory flat regardless of map size.

    PORT NOTE: ``conn`` replaces Django's global ``connection``.
    """
    n, _dim = embeddings.shape
    if not (
        len(geokeyframe_ids)
        == len(theta_center)
        == len(phi_center)
        == len(angular_width)
        == len(angular_height)
        == n
    ):
        raise ValueError("All arrays must share the same length.")
    if n == 0:
        return 0

    chunks = encode_copy_stream(
        geo_ref_id=geo_ref_id,
        geokeyframe_ids=geokeyframe_ids,
        theta_center=theta_center,
        phi_center=phi_center,
        angular_width=angular_width,
        angular_height=angular_height,
        embeddings=embeddings,
        thumbnail_keys=thumbnail_keys,
        depths=depths,
        object_positions=object_positions,
        detector_sources=detector_sources,
        labels=labels,
        detection_scores=detection_scores,
    )

    columns_sql = ", ".join(_COLUMNS)
    with conn.cursor() as cursor:
        cursor.copy_expert(
            f"COPY object_search_candidate ({columns_sql}) FROM STDIN WITH BINARY",
            _ChunkReader(chunks),
        )
    return n


def drop_partial_hnsw_index(conn, geo_ref_id: int) -> None:
    """Drop this georef's partial HNSW index if it exists. **Dev-only addition.**

    Call it before re-ingesting a georef. With the index still in place every
    COPY'd row becomes an incremental HNSW insert: measured here at ~1 000 rows per
    minute against a 1 M-row re-ingest, i.e. hours instead of minutes, and pgvector
    documents that an incrementally built graph also has lower recall than a bulk
    build. `create_partial_hnsw_index` cannot undo it afterwards — its
    `IF NOT EXISTS` skips a *valid* index, so it rebuilds nothing.

    Plain `DROP INDEX` (not `CONCURRENTLY`) so it participates in the ingest
    transaction: if the COPY fails, the index is still there.
    """
    index_name = INDEX_NAME_TEMPLATE.format(geo_ref_id=geo_ref_id)
    with conn.cursor() as cursor:
        cursor.execute(f"DROP INDEX IF EXISTS {index_name}")


def create_partial_hnsw_index(conn, geo_ref_id: int) -> None:
    """Create the per-georef partial HNSW index over ``embedding``.

    Partial indexes per georef keep query latency low while letting older
    versions live alongside new ones. ``halfvec_l2_ops`` matches the field
    type and the cosine pre-norm done by MetaCLIP2 (L2 on unit vectors is
    equivalent to cosine).

    ``CREATE INDEX CONCURRENTLY`` cannot run in a transaction, so ``conn`` must
    be in autocommit. An advisory lock serializes concurrent builders (see
    ``_HNSW_INDEX_LOCK_KEY``): two CICs on the shared table deadlock.
    """
    index_name = INDEX_NAME_TEMPLATE.format(geo_ref_id=geo_ref_id)
    with conn.cursor() as cursor:
        # One builder at a time on the shared table: concurrent CICs deadlock.
        # Acquire by POLLING pg_try_advisory_lock, never a blocking
        # pg_advisory_lock: a builder blocked *inside* pg_advisory_lock() holds an
        # open transaction, and the CIC below waits for every concurrent
        # transaction — so the lock holder's CIC would wait on the waiter's txn
        # while the waiter waits on the lock == deadlock. Polling holds no txn
        # between tries (autocommit), so waiters never block the running CIC.
        # Held across the DROP + CIC, released in finally.
        while True:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", [_HNSW_INDEX_LOCK_KEY])
            if cursor.fetchone()[0]:
                break
            time.sleep(1)
        try:
            # A CIC killed mid-build (e.g. the deadlock above) leaves an INVALID
            # index behind; IF NOT EXISTS would then skip the rebuild forever, so
            # drop it first if present-and-invalid.
            cursor.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)",
                [index_name],
            )
            row = cursor.fetchone()
            if row is not None and not row[0]:
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            cursor.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} "
                f"ON object_search_candidate USING hnsw (embedding halfvec_l2_ops) "
                f"WITH (m = 16, ef_construction = 64) "
                f"WHERE geo_ref_id = %s",
                [geo_ref_id],
            )
        finally:
            cursor.execute("SELECT pg_advisory_unlock(%s)", [_HNSW_INDEX_LOCK_KEY])
