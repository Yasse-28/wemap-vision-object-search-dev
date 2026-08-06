"""The Django-shaped hole, filled in pure Python.

In production (`wemap-vision-backend`) the object-search pipeline is three layers:

    third_party/object_search/prepare
      →  backend/object_search/ (Django)
      →  services/object_search_online/

The first and third are mirrored verbatim in this repo. The middle one is Django,
so it is not — but it is not merely plumbing: it owns the four things that turn a
bag of embeddings into map positions. Those are ported here, from the backend:

| Brick | Ported from |
|---|---|
| `prepare_postprocess` | `object_search_prepare.py::_sample_depths` |
| `ingest` / `ingest_cli` | `db/ingest.py`, `object_search_ingest.py` |
| `candidates` | `object_search/candidates.py` |
| `localize` | `object_search/v1_5_logic.py` |

Two substitutions run through all of them:

- **Poses** come from the map's v2 manifest (`map_manifest`) instead of the
  `api_geokeyframe` / `GeoRef` tables. See `georef_source`.
- **Files** come from a map directory instead of S3.

Everything else — the 3D lifting formula, the binary COPY encoding, the HNSW
parameters, the clustering and the ranking weights — is kept byte-identical in
behaviour to production on purpose. When they diverge, production wins.
"""

from typing import Any

# A psycopg2 connection. psycopg2 ships no type stubs, so this resolves to `Any`
# either way; naming it documents intent at every call site without pretending to a
# precision we do not have. Lives here rather than in `db.py` so modules that only
# *use* a connection (`db_schema`, `ingest`) need no psycopg2 import.
Connection = Any

__all__ = ["Connection"]
