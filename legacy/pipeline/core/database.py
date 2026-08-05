"""SQLite schema helpers: object-search request history and object-search index DDL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

HISTORY_DATABASE_NAME = "object-search-history.db"

# params.key for JSON-encoded ObjectSearchIndexMetadata (replaces legacy manifest_json).
INDEX_METADATA_PARAM_KEY = "index_metadata_json"
LEGACY_MANIFEST_PARAM_KEY = "manifest_json"


def ensure_history_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS Request (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                search_type TEXT,
                enforced INTEGER,
                timestamp REAL,
                datetime TEXT,
                time_router_ms INTEGER,
                time_embedding_ms INTEGER,
                time_db_ms INTEGER
            );
            """)
        conn.commit()
    finally:
        conn.close()


def create_object_search_index_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS params (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cutout (
            cutout_id INTEGER PRIMARY KEY,
            keyframe_id INTEGER NOT NULL,
            center_x REAL,
            center_y REAL,
            rotation BLOB NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    # Resume bookkeeping for the hybrid builder: one row per keyframe whose
    # detection/embedding/write batch has fully committed. Replaces the legacy
    # monolithic ``processed_cutout_ids`` blob in ``params`` (which was rewritten
    # in full every mini-batch, an O(N^2) cost over a large build).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_keyframe (
            keyframe_id INTEGER PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster (
            cluster_id INTEGER PRIMARY KEY,
            centroid_world BLOB NOT NULL,
            centroid_geo BLOB NOT NULL,
            observation_count INTEGER NOT NULL,
            confidence REAL NOT NULL,
            level INTEGER NOT NULL,
            ocr_text TEXT,
            ocr_tokens TEXT,
            ocr_key TEXT,
            ocr_observation_count INTEGER,
            ocr_source INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_cutout (
            cluster_id INTEGER NOT NULL REFERENCES cluster(cluster_id),
            cutout_id INTEGER NOT NULL REFERENCES cutout(cutout_id),
            keyframe_id INTEGER NOT NULL,
            level INTEGER NOT NULL,
            observation_count INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, cutout_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object (
            object_idx INTEGER PRIMARY KEY,
            keyframe_id INTEGER NOT NULL,
            cutout_id INTEGER NOT NULL,
            bbox_coordinates BLOB,
            bbox_spherical_coordinates BLOB,
            embedding BLOB NOT NULL,
            position_keyframe BLOB,
            position_local BLOB,
            position_world BLOB,
            depth REAL,
            localization_valid INTEGER,
            cluster_id INTEGER REFERENCES cluster(cluster_id),
            level INTEGER,
            visual_similarity_score REAL,
            visual_candidate INTEGER,
            visual_assigned INTEGER,
            textness_score REAL,
            ocr_text TEXT,
            ocr_tokens TEXT,
            ocr_key TEXT,
            ocr_candidate INTEGER,
            ocr_assigned INTEGER,
            ocr_source INTEGER,
            label TEXT,
            detection_source TEXT
        )
    """)
    conn.commit()


def ensure_object_search_index_schema(conn: sqlite3.Connection) -> None:
    """Create missing tables and add additive columns for newer index versions."""
    create_object_search_index_tables(conn)
    cursor = conn.cursor()
    object_columns = {row[1] for row in cursor.execute("PRAGMA table_info(object)")}
    if "level" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN level INTEGER")
    if "label" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN label TEXT")
    if "detection_source" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN detection_source TEXT")
    # Position field renames (additive: keep old columns as aliases, add new ones)
    if "position_keyframe" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN position_keyframe BLOB")
    # position_local and position_world already exist (possibly with old semantics);
    # the new position_world column (lat/lon/alt) is added if missing.
    # Old DBs: position_local = camera frame, position_world = WDS local frame.
    # New DBs: position_keyframe = camera frame, position_local = ENU local frame,
    #          position_world = geographic [lat, lon, alt].
    if "position_world" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN position_world BLOB")
    if "bbox_coordinates" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN bbox_coordinates BLOB")
    if "bbox_spherical_coordinates" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN bbox_spherical_coordinates BLOB")
    # Lightweight OCR fields (added by hybrid pipeline)
    if "textness_score" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN textness_score REAL")
    if "ocr_text" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_text TEXT")
    if "ocr_tokens" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_tokens TEXT")
    if "ocr_key" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_key TEXT")
    if "ocr_candidate" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_candidate INTEGER")
    if "ocr_assigned" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_assigned INTEGER")
    if "ocr_source" not in object_columns:
        cursor.execute("ALTER TABLE object ADD COLUMN ocr_source INTEGER")
    conn.commit()


def migrate_legacy_manifest_param_to_index_metadata(conn: sqlite3.Connection) -> None:
    """If only legacy manifest_json exists, copy it to index_metadata_json and
    remove the old row."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT value FROM params WHERE key = ?", (INDEX_METADATA_PARAM_KEY,)
    )
    if cursor.fetchone() is not None:
        return
    cursor.execute(
        "SELECT value FROM params WHERE key = ?", (LEGACY_MANIFEST_PARAM_KEY,)
    )
    row = cursor.fetchone()
    if row is None:
        return
    cursor.execute(
        "INSERT OR REPLACE INTO params (key, value) VALUES (?, ?)",
        (INDEX_METADATA_PARAM_KEY, row[0]),
    )
    cursor.execute("DELETE FROM params WHERE key = ?", (LEGACY_MANIFEST_PARAM_KEY,))
    conn.commit()
