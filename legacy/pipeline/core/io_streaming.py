"""Streaming (incremental) SQLite write helpers for the hybrid offline pipeline.

Unlike the batch-write path in ``io.py`` which accumulates the full dataset in
memory and writes it atomically at the end of each stage, these helpers write
each mini-batch directly to the DB so that:

- The DB is always consistent and resumable (crash loses at most one batch).
- RAM footprint is bounded to the current mini-batch.

The connection is opened in WAL journal mode so the online service can read
while the build writes concurrently.

Typical usage in ``build_index.py``::

    conn = open_build_db(path)
    # ... in the GPU loop (one atomic transaction per mini-batch) ...
    write_cutout_batch(conn, cutout_rows, commit=False)
    write_object_batch(conn, object_rows, commit=False)  # complete rows
    mark_keyframes_processed(conn, keyframe_ids, commit=False)
    conn.commit()  # single commit → batch is all-or-nothing on crash
    # ... end of run ...
    conn.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional

import numpy as np

from pipeline.core.database import (
    INDEX_METADATA_PARAM_KEY,
    ensure_object_search_index_schema,
)
from pipeline.core.types import ObjectSearchIndexMetadata

# ---------------------------------------------------------------------------
# Typed row containers
# ---------------------------------------------------------------------------


class CutoutRow(NamedTuple):
    cutout_id: int
    keyframe_id: int
    center_x: Optional[float]
    center_y: Optional[float]
    rotation: bytes  # 4×4 float32 .tobytes()
    embedding: bytes  # [dim] float32 .tobytes()


class ObjectRow(NamedTuple):
    object_idx: int
    keyframe_id: int
    cutout_id: int
    bbox_coordinates: bytes  # [x1, y1, x2, y2] 4×float32 .tobytes()
    bbox_spherical_coordinates: Optional[
        bytes
    ]  # [theta, phi, fov_x, fov_y] 4×float32 .tobytes()
    embedding: bytes  # [dim] float32 .tobytes()
    position_keyframe: Optional[
        bytes
    ]  # [3] float32, XYZ in keyframe camera frame (OpenCV)
    position_local: Optional[bytes]  # [3] float32, XYZ in local metric ENU frame
    position_world: Optional[bytes]  # [3] float32, geographic [lat, lon, alt]
    depth: Optional[float]
    localization_valid: Optional[int]  # 0 or 1
    label: Optional[str]
    detection_source: Optional[str]  # "yolo" or "gdino"
    level: Optional[int]  # indoor level id from GeoRef Level table, or None
    textness_score: Optional[float] = None  # MetaCLIP-based text likelihood score
    ocr_text: Optional[str] = None  # raw OCR text from PP-OCR
    ocr_tokens: Optional[str] = None  # normalized space-separated tokens
    ocr_key: Optional[str] = None  # normalized OCR identity key
    ocr_source: Optional[int] = None  # 0=none, 1=lightweight PP-OCR


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def open_build_db(path: Path) -> sqlite3.Connection:
    """Open (or create) an object-search.db at *path* ready for streaming writes.

    - Creates the full index schema if the file does not exist yet.
    - Enables WAL journal mode for concurrent read access.
    - Sets PRAGMA synchronous=NORMAL (safe with WAL, faster than FULL).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    ensure_object_search_index_schema(conn)
    return conn


def write_index_metadata(
    conn: sqlite3.Connection, metadata: ObjectSearchIndexMetadata
) -> None:
    """Write (or overwrite) the index metadata JSON in the params table."""
    conn.execute(
        "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
        (INDEX_METADATA_PARAM_KEY, metadata.to_json().encode("utf-8")),
    )
    conn.commit()


def load_index_metadata(
    conn: sqlite3.Connection,
) -> Optional[ObjectSearchIndexMetadata]:
    """Load index metadata from the params table, or return None if absent."""
    row = conn.execute(
        "SELECT value FROM params WHERE key = ?", (INDEX_METADATA_PARAM_KEY,)
    ).fetchone()
    if row is None:
        return None
    return ObjectSearchIndexMetadata.from_json(row[0].decode("utf-8"))


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------


def load_processed_cutout_ids(conn: sqlite3.Connection) -> np.ndarray:
    """Return the set of cutout IDs that have already been through detection.

    Legacy helper kept for resuming DBs written before the ``processed_keyframe``
    table existed.  New builds use :func:`load_processed_keyframe_ids`.
    """
    row = conn.execute(
        "SELECT value FROM params WHERE key = 'processed_cutout_ids'"
    ).fetchone()
    if row is None or row[0] is None:
        return np.array([], dtype=np.int64)
    return np.frombuffer(row[0], dtype=np.int64)


def load_processed_keyframe_ids(
    conn: sqlite3.Connection,
    *,
    id_stride: Optional[int] = None,
) -> np.ndarray:
    """Return keyframe IDs whose detection batch has fully committed.

    Reads the ``processed_keyframe`` table.  For backward compatibility it also
    folds in any keyframes recorded under the legacy ``processed_cutout_ids``
    blob (so a build started before this change still resumes correctly); pass
    ``id_stride`` to decode ``cutout_id // id_stride`` from that blob.
    """
    rows = conn.execute("SELECT keyframe_id FROM processed_keyframe").fetchall()
    ids = {int(r[0]) for r in rows}

    legacy = load_processed_cutout_ids(conn)
    if legacy.size and id_stride:
        ids.update(int(cid) // int(id_stride) for cid in legacy.tolist())

    if not ids:
        return np.array([], dtype=np.int64)
    return np.array(sorted(ids), dtype=np.int64)


def mark_keyframes_processed(
    conn: sqlite3.Connection,
    keyframe_ids: Iterable[int],
    *,
    commit: bool = True,
) -> None:
    """Record *keyframe_ids* as fully processed (INSERT OR IGNORE, O(1) each)."""
    ids = [(int(kf),) for kf in keyframe_ids]
    if not ids:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO processed_keyframe (keyframe_id) VALUES (?)",
        ids,
    )
    if commit:
        conn.commit()


def load_embedded_cutout_ids(conn: sqlite3.Connection) -> np.ndarray:
    """Return cutout IDs that have a real (non-empty) embedding in the cutout table."""
    rows = conn.execute(
        "SELECT cutout_id FROM cutout WHERE length(embedding) > 4"
    ).fetchall()
    if not rows:
        return np.array([], dtype=np.int64)
    return np.array([r[0] for r in rows], dtype=np.int64)


def count_objects_missing_positions(conn: sqlite3.Connection) -> int:
    """Number of object rows that still have NULL ENU position_local."""
    row = conn.execute(
        "SELECT count(*) FROM object WHERE position_local IS NULL"
    ).fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Batch writers
# ---------------------------------------------------------------------------


def write_cutout_batch(
    conn: sqlite3.Connection,
    rows: List[CutoutRow],
    *,
    commit: bool = True,
) -> None:
    """INSERT OR REPLACE a batch of cutout rows.

    Pass ``commit=False`` to defer the commit so several writes can share a
    single per-batch transaction.
    """
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO cutout
            (cutout_id, keyframe_id, center_x, center_y, rotation, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()


def write_object_batch(
    conn: sqlite3.Connection,
    rows: List[ObjectRow],
    *,
    commit: bool = True,
) -> None:
    """INSERT a batch of complete object rows (including positions and labels).

    Uses INSERT OR REPLACE so re-running on the same keyframe after a crash
    is idempotent.  Pass ``commit=False`` to defer the commit so several writes
    can share a single per-batch transaction.
    """
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO object
            (object_idx, keyframe_id, cutout_id,
             bbox_coordinates, bbox_spherical_coordinates,
             embedding, position_keyframe, position_local, position_world,
             depth, localization_valid,
             label, detection_source, level,
             textness_score, ocr_text, ocr_tokens, ocr_key, ocr_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()


def update_object_ocr_batch(
    conn: sqlite3.Connection,
    rows: list,
) -> None:
    """UPDATE ocr fields for objects that have been through lightweight PP-OCR.

    Each entry in *rows* must be a tuple:
        (ocr_text, ocr_tokens, ocr_key, ocr_source, object_idx)
    Only rows where the OCR produced a non-empty accepted read are passed in.
    """
    if not rows:
        return
    conn.executemany(
        "UPDATE object SET ocr_text=?, ocr_tokens=?, ocr_key=?, ocr_source=?"
        " WHERE object_idx=?",
        rows,
    )
    conn.commit()


def update_processed_cutout_ids(
    conn: sqlite3.Connection,
    new_ids: np.ndarray,
) -> None:
    """Merge *new_ids* into the ``processed_cutout_ids`` blob in the params table."""
    existing = load_processed_cutout_ids(conn)
    merged = np.asarray(
        sorted(set(existing.tolist()).union(new_ids.tolist())),
        dtype=np.int64,
    )
    conn.execute(
        "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
        ("processed_cutout_ids", merged.tobytes()),
    )
    conn.commit()
