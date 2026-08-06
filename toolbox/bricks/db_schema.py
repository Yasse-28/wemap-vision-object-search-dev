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
    id                BIGINT PRIMARY KEY,
    geo_ref_id        BIGINT NOT NULL,
    video_keyframe_id BIGINT NOT NULL,
    orientation       DOUBLE PRECISION[4] NOT NULL,
    position          geometry(PointZ, 0) NOT NULL,
    image             VARCHAR(512) NOT NULL DEFAULT '',
    depth_map         VARCHAR(512) NOT NULL DEFAULT '',
    UNIQUE (geo_ref_id, video_keyframe_id)
)
"""

CREATE_CANDIDATE = """
CREATE TABLE IF NOT EXISTS object_search_candidate (
    id              BIGSERIAL PRIMARY KEY,
    geo_ref_id      BIGINT NOT NULL,
    geokeyframe_id  BIGINT NOT NULL REFERENCES geokeyframe(id) ON DELETE CASCADE,
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
    detection_score DOUBLE PRECISION
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


def ensure_schema(conn: Connection) -> None:
    """Create the extensions, tables and plain indexes if they are absent.

    The per-georef partial HNSW index is *not* created here — it is built after
    ingest by `ingest.create_partial_hnsw_index`, because building it before the
    rows land would be wasted work.
    """
    with conn.cursor() as cursor:
        for statement in CREATE_EXTENSIONS:
            cursor.execute(statement)
        cursor.execute(CREATE_GEOKEYFRAME)
        cursor.execute(CREATE_CANDIDATE)
        for statement in CREATE_INDEXES:
            cursor.execute(statement)
    conn.commit()
    logger.info("object-search schema ready.")
