"""Local DDL for the tables production creates via Django migrations.

Two tables:

- `object_search_candidate` — faithful to `api.models.ObjectSearchCandidate`
  (migrations 0089/0090/0095/0098/0114/0115). The mirrored online service queries
  this table directly, so its name, its `geo_ref_id` column and the `halfvec(1024)`
  embedding are **not ours to rename**.
- `geokeyframe` — a minimal local stand-in for `api.models.GeoKeyframe`, holding
  only what candidate enrichment joins on: the EUS position and the orientation
  quaternion. Populated from the v2 map manifest by `georef_source`.

## Types are load-bearing here

`ingest.bulk_copy` writes these rows with PostgreSQL's **binary** COPY format,
which encodes each value at a fixed width with no server-side coercion. So:

- the four angular columns must be `DOUBLE PRECISION` — declare `REAL` and every
  COPY fails at runtime with a length error;
- `object_position` must be `geometry(PointZ, 0)` — **SRID 0, not 4326**. The EWKB
  prefix hardcodes `srid=0`; declaring 4326 rejects every row.

## Extensions

Needs both `vector` (for `halfvec`) and `postgis` (for `geometry` + `ST_X/Y/Z`).
The stock `pgvector/pgvector:pg17` image ships only the former, which is why
`infra/postgres/` builds its own image.
"""

from __future__ import annotations

from toolbox.bricks import Connection
from toolbox.logging import logger

CREATE_EXTENSIONS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS postgis",
)

CREATE_GEOKEYFRAME = """
CREATE TABLE IF NOT EXISTS geokeyframe (
    id                BIGINT NOT NULL,
    geo_ref_id        BIGINT NOT NULL,
    video_keyframe_id BIGINT NOT NULL,
    orientation       DOUBLE PRECISION[4] NOT NULL,
    position          geometry(PointZ, 0) NOT NULL,
    image             VARCHAR(512) NOT NULL DEFAULT '',
    depth_map         VARCHAR(512) NOT NULL DEFAULT '',
    PRIMARY KEY (geo_ref_id, id),
    UNIQUE (geo_ref_id, video_keyframe_id)
)
"""

CREATE_CANDIDATE = """
CREATE TABLE IF NOT EXISTS object_search_candidate (
    id              BIGSERIAL PRIMARY KEY,
    geo_ref_id      BIGINT NOT NULL,
    geokeyframe_id  BIGINT NOT NULL,
    theta_center    DOUBLE PRECISION NOT NULL,
    phi_center      DOUBLE PRECISION NOT NULL,
    angular_width   DOUBLE PRECISION NOT NULL,
    angular_height  DOUBLE PRECISION NOT NULL,
    embedding       halfvec(1024) NOT NULL,
    thumbnail       VARCHAR(255),
    depth           DOUBLE PRECISION,
    object_position geometry(PointZ, 0),
    detector_source VARCHAR(16),
    label           VARCHAR(128),
    detection_score DOUBLE PRECISION,
    FOREIGN KEY (geo_ref_id, geokeyframe_id)
        REFERENCES geokeyframe (geo_ref_id, id) ON DELETE CASCADE
)
"""

#: Per-map embedding centroid, written by ingest when it centres the vectors it
#: stores. The online service subtracts it from every query embedding for that
#: georef, so **the presence of the row is the switch**: an index built centred is
#: always queried centred, and there is no flag the two sides can disagree about.
#: Absent row = untouched vectors = the behaviour production has today.
#:
#: Dev-only for now. Production owns this table through a Django migration that does
#: not exist yet — see `docs/adr/` before promoting it.
CREATE_EMBEDDING_CENTROID = """
CREATE TABLE IF NOT EXISTS object_search_embedding_centroid (
    geo_ref_id BIGINT PRIMARY KEY,
    centroid   halfvec(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS object_search_candidate_geo_ref_id_idx "
    "ON object_search_candidate (geo_ref_id)",
    "CREATE INDEX IF NOT EXISTS object_search_candidate_geokeyframe_id_idx "
    "ON object_search_candidate (geokeyframe_id)",
    "CREATE INDEX IF NOT EXISTS geokeyframe_geo_ref_id_idx "
    "ON geokeyframe (geo_ref_id)",
)

LEGACY_SCHEMA_ERROR = """\
geokeyframe still uses the legacy single-column primary key, which lets one map
overwrite another's poses. Drop both tables and re-ingest each map:
  psql <your database DSN> -c 'DROP TABLE object_search_candidate, geokeyframe'
  python -m toolbox.bricks.ingest_cli <each map dir>"""


def _has_legacy_geokeyframe_primary_key(conn: Connection) -> bool:
    """Return whether the visible geokeyframe table is keyed only by ``id``."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT array_agg(attribute.attname ORDER BY pk_column.ordinality)
            FROM pg_constraint AS con
            JOIN LATERAL unnest(con.conkey) WITH ORDINALITY
                AS pk_column(attnum, ordinality) ON TRUE
            JOIN pg_attribute AS attribute
              ON attribute.attrelid = con.conrelid
             AND attribute.attnum = pk_column.attnum
            WHERE con.conrelid = to_regclass('geokeyframe')
              AND con.contype = 'p'
              AND con.conname = 'geokeyframe_pkey'
            GROUP BY con.oid
            """
        )
        row = cursor.fetchone()
    return row is not None and list(row[0]) == ["id"]


def ensure_schema(conn: Connection) -> None:
    """Create the extensions, tables and plain indexes if they are absent.

    The per-georef partial HNSW index is *not* created here — it is built after
    ingest by `ingest.create_partial_hnsw_index`, because building it before the
    rows land would be wasted work.
    """
    if _has_legacy_geokeyframe_primary_key(conn):
        raise RuntimeError(LEGACY_SCHEMA_ERROR)

    with conn.cursor() as cursor:
        for statement in CREATE_EXTENSIONS:
            cursor.execute(statement)
        cursor.execute(CREATE_GEOKEYFRAME)
        cursor.execute(CREATE_CANDIDATE)
        cursor.execute(CREATE_EMBEDDING_CENTROID)
        for statement in CREATE_INDEXES:
            cursor.execute(statement)
    conn.commit()
    logger.info("object-search schema ready.")
