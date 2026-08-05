"""Integration tests for the bricks against a real Postgres.

`test_copy_encoding.py` asserts the COPY *bytes*; this file checks that a real server
accepts them, which is the only validation those bytes get in production. It also
covers the things a unit test cannot: that both extensions load, that the declared
column types survive binary COPY, that the EWKB really round-trips at SRID 0, and
that the partial HNSW index builds and answers a query.

**Skipped unless a database is reachable**, so the default `pytest` run stays
hermetic. To run it:

    docker compose -f infra/postgres/compose.yml up -d
    DATABASE_HOST=localhost DATABASE_USER=postgres DATABASE_PASSWORD=… \\
      pytest toolbox/tests/test_integration_db.py

Every test works inside a throwaway database, created and dropped per module, so
nothing existing is touched.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import numpy as np
import pytest

from toolbox.bricks import db_schema
from toolbox.bricks.candidates import load_enriched_candidates
from toolbox.bricks.ingest import bulk_copy, create_partial_hnsw_index
from toolbox.bricks.localize import LocalizationParams, build_localize_response
from toolbox.bricks.vendored.geo_transform import Coordinates, GeoTransform, Level

psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")

GEO_REF_ID = 42
DIM = 1024
TEST_DB = "object_search_pytest"

# Three keyframes on an east-west line, 5 m apart, at eye height.
KEYFRAMES = [(1, 0.0, 1.6, 0.0), (2, 5.0, 1.6, 0.0), (3, 10.0, 1.6, 0.0)]


def _admin_dsn() -> dict[str, Any] | None:
    """Connection kwargs for the maintenance database, or None if unconfigured."""
    host = os.environ.get("DATABASE_HOST")
    password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
        "PGVECTOR_PASSWORD"
    )
    if not host or not password:
        return None
    return {
        "host": host,
        "port": int(os.environ.get("DATABASE_PORT", "5432")),
        "dbname": "postgres",
        "user": os.environ.get("DATABASE_USER", "postgres"),
        "password": password,
    }


@pytest.fixture(scope="module")
def conn() -> Iterator[Any]:
    """A connection to a freshly created, schema-ready throwaway database."""
    admin_kwargs = _admin_dsn()
    if admin_kwargs is None:
        pytest.skip("DATABASE_HOST / DATABASE_PASSWORD not set")
    try:
        admin = psycopg2.connect(connect_timeout=3, **admin_kwargs)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"no database reachable: {exc}")

    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()

    connection = psycopg2.connect(**{**admin_kwargs, "dbname": TEST_DB})
    try:
        try:
            db_schema.ensure_schema(connection)
        except psycopg2.errors.UndefinedFile as exc:  # pragma: no cover
            pytest.skip(
                "the database lacks pgvector and/or PostGIS — build the image in "
                f"infra/postgres/ rather than using stock pgvector: {exc}"
            )
        with connection.cursor() as cur:
            for kf_id, x, y, z in KEYFRAMES:
                cur.execute(
                    "INSERT INTO geokeyframe (id, geo_ref_id, video_keyframe_id, "
                    "orientation, position, image, depth_map) VALUES "
                    "(%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s,%s,%s), 0), %s, %s)",
                    [
                        kf_id,
                        GEO_REF_ID,
                        kf_id,
                        [1.0, 0.0, 0.0, 0.0],
                        x,
                        y,
                        z,
                        f"{kf_id}.jpg",
                        f"{kf_id}.tif",
                    ],
                )
        connection.commit()
        yield connection
    finally:
        connection.close()
        admin = psycopg2.connect(**admin_kwargs)
        admin.autocommit = True
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        admin.close()


def _unit_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float16)


def _copy_fixture_rows(connection: Any) -> np.ndarray:
    """Seven candidates: two spatial groups 7 m apart, plus one with no depth."""
    embeddings = _unit_embeddings(7)
    positions = np.array(
        [
            [2.0, 1.6, -3.0],
            [2.2, 1.6, -3.1],
            [1.9, 1.6, -2.9],
            [9.0, 1.6, -3.0],
            [9.1, 1.6, -3.2],
            [8.9, 1.6, -2.8],
            [np.nan, np.nan, np.nan],
        ]
    )
    with connection.cursor() as cur:
        cur.execute(
            "DELETE FROM object_search_candidate WHERE geo_ref_id = %s", [GEO_REF_ID]
        )
    bulk_copy(
        connection,
        geo_ref_id=GEO_REF_ID,
        geokeyframe_ids=np.array([1, 2, 3, 1, 2, 3, 1], dtype=np.int64),
        theta_center=np.linspace(-0.5, 0.5, 7),
        phi_center=np.full(7, -0.1),
        angular_width=np.full(7, 1.0),
        angular_height=np.full(7, 0.8),
        embeddings=embeddings,
        thumbnail_keys=np.array([f"thumbs/{i:06d}.jpg" for i in range(7)]),
        depths=np.array([3.4, 3.5, 3.3, 3.4, 3.6, 3.2, np.nan]),
        object_positions=positions,
        # The last row exercises how pandas materialises parquet string nulls.
        detector_sources=np.array(["yolo"] * 6 + [np.nan], dtype=object),
        labels=np.array(["bench"] * 6 + [None], dtype=object),
        detection_scores=np.array([0.9] * 6 + [np.nan]),
    )
    connection.commit()
    return embeddings


def _geo_transform() -> GeoTransform:
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(
            Level(value=0.0, min_altitude=-2.0, max_altitude=4.0),
            Level(value=1.0, min_altitude=4.0, max_altitude=10.0),
        ),
    )


def test_schema_declares_the_types_binary_copy_requires(conn: Any) -> None:
    """Widths are not negotiable: binary COPY does no server-side coercion."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, udt_name FROM information_schema.columns "
            "WHERE table_name = 'object_search_candidate'"
        )
        types = dict(cur.fetchall())
    for column in (
        "theta_center",
        "phi_center",
        "angular_width",
        "angular_height",
        "depth",
        "detection_score",
    ):
        assert types[column] == "float8", f"{column} must be DOUBLE PRECISION"
    assert types["embedding"] == "halfvec"
    assert types["object_position"] == "geometry"
    assert types["geo_ref_id"] == "int8"


def test_both_extensions_are_installed(conn: Any) -> None:
    """Stock pgvector has no PostGIS — hence infra/postgres/Dockerfile."""
    with conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension")
        installed = {row[0] for row in cur.fetchall()}
    assert {"vector", "postgis"} <= installed


def test_bulk_copy_is_accepted_and_nulls_land_as_null(conn: Any) -> None:
    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(object_position), count(depth), count(label), "
            "count(detector_source), count(detection_score) "
            "FROM object_search_candidate WHERE geo_ref_id = %s",
            [GEO_REF_ID],
        )
        total, positions, depths, labels, sources, scores = cur.fetchone()
    assert total == 7
    # The NaN/None row must be NULL in every nullable column, not 0.0 or "nan".
    assert (positions, depths, labels, sources, scores) == (6, 6, 6, 6, 6)


def test_object_position_round_trips_through_postgis(conn: Any) -> None:
    """Proves the hand-rolled EWKB is what PostGIS thinks it is — including SRID 0."""
    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ST_X(object_position), ST_Y(object_position), "
            "ST_Z(object_position), ST_SRID(object_position) "
            "FROM object_search_candidate WHERE geo_ref_id = %s "
            "AND object_position IS NOT NULL ORDER BY id LIMIT 1",
            [GEO_REF_ID],
        )
        x, y, z, srid = cur.fetchone()
    assert (round(x, 6), round(y, 6), round(z, 6)) == (2.0, 1.6, -3.0)
    assert srid == 0, "declaring the column as 4326 would reject every row"


def test_embedding_keeps_its_dimension(conn: Any) -> None:
    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT vector_dims(embedding::vector) FROM object_search_candidate LIMIT 1"
        )
        assert cur.fetchone()[0] == DIM


def test_partial_hnsw_index_builds_valid_and_is_idempotent(conn: Any) -> None:
    _copy_fixture_rows(conn)
    index_name = f"idx_object_search_candidate_hnsw_georef_{GEO_REF_ID}"
    conn.commit()  # CIC cannot run in a transaction
    conn.autocommit = True
    try:
        create_partial_hnsw_index(conn, GEO_REF_ID)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)",
                [index_name],
            )
            row = cur.fetchone()
            assert row is not None and row[0], "index must exist and be valid"
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE indexname = %s", [index_name]
            )
            definition = cur.fetchone()[0]
        assert "halfvec_l2_ops" in definition
        assert "m='16'" in definition and "ef_construction='64'" in definition
        assert f"(geo_ref_id = {GEO_REF_ID})" in definition, "must be partial"
        # Re-running must not fail or invalidate the index.
        create_partial_hnsw_index(conn, GEO_REF_ID)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass(%s)",
                [index_name],
            )
            assert cur.fetchone()[0]
    finally:
        conn.autocommit = False


def test_ann_query_retrieves_the_query_vector_first(conn: Any) -> None:
    embeddings = _copy_fixture_rows(conn)
    conn.commit()
    conn.autocommit = True
    try:
        create_partial_hnsw_index(conn, GEO_REF_ID)
    finally:
        conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = 1000")
        literal = "[" + ",".join(str(float(v)) for v in embeddings[0]) + "]"
        cur.execute(
            "SELECT id, embedding <-> %s::halfvec AS d FROM object_search_candidate "
            "WHERE geo_ref_id = %s ORDER BY d LIMIT 3",
            [literal, GEO_REF_ID],
        )
        hits = cur.fetchall()
    assert len(hits) == 3
    # Distance ~0 to itself, and cosine recovered as 1 - d²/2 must be ~1.
    assert hits[0][1] < 1e-2
    assert 1.0 - hits[0][1] ** 2 / 2 > 0.99
    conn.rollback()


def test_enrichment_excludes_rows_without_a_position_and_resolves_levels(
    conn: Any,
) -> None:
    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM object_search_candidate WHERE geo_ref_id = %s ORDER BY id",
            [GEO_REF_ID],
        )
        ids = [row[0] for row in cur.fetchall()]
    hnsw_results = [
        {"id": cid, "similarity": sim}
        for cid, sim in zip(ids, [0.95, 0.93, 0.91, 0.88, 0.86, 0.84, 0.80])
    ]

    enriched = load_enriched_candidates(
        conn, GEO_REF_ID, hnsw_results, _geo_transform()
    )

    assert len(enriched) == 6, "the NULL-position row must be filtered out"
    assert [c.similarity for c in enriched] == sorted(
        [c.similarity for c in enriched], reverse=True
    )
    top = enriched[0]
    # EUS up = 1.6 falls in the [-2, 4] band, so level 0 — from the *keyframe* pose.
    assert top.level == 0
    assert top.vkf_level == 0
    assert 48.85 < top.lat < 48.86
    assert 2.35 < top.lng < 2.36
    assert top.thumbnail == "thumbs/000000.jpg"
    assert 0.0 <= top.video_keyframe_heading < 360.0


def test_localize_clusters_two_groups_seven_metres_apart(conn: Any) -> None:
    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM object_search_candidate WHERE geo_ref_id = %s ORDER BY id",
            [GEO_REF_ID],
        )
        ids = [row[0] for row in cur.fetchall()]
    hnsw_results = [
        {"id": cid, "similarity": sim}
        for cid, sim in zip(ids, [0.95, 0.93, 0.91, 0.88, 0.86, 0.84, 0.80])
    ]
    geo_transform = _geo_transform()
    enriched = load_enriched_candidates(conn, GEO_REF_ID, hnsw_results, geo_transform)

    response = build_localize_response(
        enriched,
        geo_transform,
        params=LocalizationParams(min_keyframes_per_cluster=2, clustering_eps_m=2.0),
        time_embedding_ms=11,
        time_retrieval_ms=22,
    )

    localizations = response["localizations"]
    assert len(localizations) == 2, "two groups 7 m apart at eps=2 m"
    for localization in localizations:
        assert localization["observation_count"] == 3
        assert len(localization["keyframe_ids"]) == 3
        assert localization["level"] == 0
        assert 0.0 <= localization["match_score"] <= 1.0
        lat, lng, _alt = localization["coordinates"]
        assert 48.85 < lat < 48.86 and 2.35 < lng < 2.36
    assert response["time_embedding_ms"] == 11
    assert response["time_retrieval_ms"] == 22


# ------------------------------------------------------- the HTTP wire contract


def _fake_map_dir(tmp_path: Any) -> Any:
    """A v2 map directory whose manifest matches `_geo_transform()`.

    The manifest is the current pose format, so this is also the path a real request
    takes. Its keyframes mirror `KEYFRAMES`, since the candidate rows reference them.
    """
    import json

    map_dir = tmp_path / "wire-map"
    map_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "local_origin": [2.3522, 48.8566, 35.0],
        "map": {"name": "wire-map", "uuid": "u", "venue_type": "hotel"},
        "geo_levels": [
            {
                "value": 0.0,
                "min_altitude": -2.0,
                "max_altitude": 4.0,
                "geometry": None,
                "geo_ref": GEO_REF_ID,
            },
            {
                "value": 1.0,
                "min_altitude": 4.0,
                "max_altitude": 10.0,
                "geometry": None,
                "geo_ref": GEO_REF_ID,
            },
        ],
        "geo_keyframes": [
            {
                "x": x,
                "y": y,
                "z": z,
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "image_url": f"https://example/u/images/{kf_id}.jpg",
                "depth_url": f"https://example/u/depths/{kf_id}.tif",
            }
            for kf_id, x, y, z in KEYFRAMES
        ],
    }
    (map_dir / "wire-map_2_20260805_083206.json").write_text(json.dumps(manifest))
    return map_dir


def _configure_service(monkeypatch: pytest.MonkeyPatch, map_dir: Any) -> Any:
    """Point the bricks service at one map, restoring its module state after.

    `state` is a module global; using monkeypatch rather than assigning to it keeps
    these tests from leaking configuration into anything that runs later.
    """
    from toolbox.bricks import service as bricks_service

    monkeypatch.setattr(
        bricks_service.state,
        "maps",
        {
            "wire-map": bricks_service.MapEntry(
                id="wire-map", path=map_dir, geo_ref_id=GEO_REF_ID
            )
        },
    )
    # A fresh cache, so a GeoTransform built by an earlier test is not reused.
    monkeypatch.setattr(bricks_service.state, "_geo_transforms", {})
    return bricks_service


def test_text_endpoint_returns_the_keys_the_toolbox_frontend_reads(
    conn: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the `/text` wire contract.

    The toolbox frontend builds its `EnrichedResult` rows straight from
    `candidates` (see `object-search/api.ts::enrichedFromCandidates`) — it used to
    re-enrich through the retired SQLite index. A renamed or dropped key here breaks
    the panel with no error on this side, so the exact key set is asserted.

    `results` is kept alongside for `parseTextPairs`, which still reads the
    standalone service's `[[id, score], …]` shape.
    """
    from fastapi.testclient import TestClient

    _copy_fixture_rows(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM object_search_candidate WHERE geo_ref_id = %s ORDER BY id",
            [GEO_REF_ID],
        )
        ids = [row[0] for row in cur.fetchall()]

    bricks_service = _configure_service(monkeypatch, _fake_map_dir(tmp_path))

    # Stand in for the mirrored online service (it would need a GPU).
    monkeypatch.setattr(
        bricks_service,
        "query_by_text",
        lambda *args, **kwargs: [
            {"id": cid, "similarity": sim}
            for cid, sim in zip(ids, [0.95, 0.93, 0.91, 0.88, 0.86, 0.84, 0.80])
        ],
    )
    # The service opens its own connection; point it at the test database.
    # Built from the same env as the fixture: psycopg2's `conn.dsn` strips the
    # password, so reusing it would fail authentication.
    kwargs = _admin_dsn()
    assert kwargs is not None
    test_dsn = " ".join(
        [
            f"host={kwargs['host']}",
            f"port={kwargs['port']}",
            f"dbname={TEST_DB}",
            f"user={kwargs['user']}",
            f"password={kwargs['password']}",
        ]
    )
    monkeypatch.setattr(bricks_service.db, "build_dsn", lambda: test_dsn)

    client = TestClient(bricks_service.create_app())
    response = client.post(
        "/wire-map/object-search/text", json={"text": "bench", "num_results": 50}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) >= {"candidates", "results", "router_object_type"}
    assert payload["router_object_type"] is None

    candidates = payload["candidates"]
    assert len(candidates) == 6, "the NULL-position row is excluded"
    # Exactly the keys enrichedFromCandidates reads.
    for key in (
        "id",
        "similarity",
        "lat",
        "lng",
        "alt",
        "level",
        "thumbnail_key",
        "video_keyframe_id",
    ):
        assert key in candidates[0], f"frontend reads '{key}'"
    assert candidates[0]["level"] == 0
    assert candidates[0]["thumbnail_key"] == "thumbs/000000.jpg"

    # `results` mirrors `candidates`, as [[id, score], …] with a *string* id.
    pairs = payload["results"]
    assert len(pairs) == len(candidates)
    assert pairs[0] == [str(candidates[0]["id"]), candidates[0]["similarity"]]


def test_localize_offline_is_retired_with_an_explanation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """501, not 404: the endpoint is gone on purpose and the message must say so."""
    from fastapi.testclient import TestClient

    bricks_service = _configure_service(monkeypatch, _fake_map_dir(tmp_path))
    client = TestClient(bricks_service.create_app(), raise_server_exceptions=False)
    response = client.post("/wire-map/object-search/localize-offline", json={})
    assert response.status_code == 501
    assert "pgvector" in response.json()["detail"]
