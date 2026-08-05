"""Postgres connection for the bricks.

Deliberately reads the **same environment variables** as the mirrored online
service (`services/object_search_online/app.py::_build_dsn`), so one `.env`
configures both and they can never end up pointing at different databases.

The mirror only takes the env-var branch when `ENVIRONMENT_NAME=onprem` —
otherwise it goes to AWS Secrets Manager. Local development therefore sets
`ENVIRONMENT_NAME=onprem`, and so do we. `PGVECTOR_*` is a leftover of the
standalone lineage: it now configures only the dev container in
`infra/postgres/compose.yml`, and nothing in the pipeline reads it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2

from toolbox.bricks import Connection

_REQUIRED = ("DATABASE_NAME", "DATABASE_USER", "DATABASE_PASSWORD")


def build_dsn() -> str:
    """A libpq DSN from `DATABASE_*`.

    Raises with the full list of what is missing, rather than the mirror's bare
    `KeyError` on the first absent name — this is the developer-facing path.
    """
    host = os.environ.get("DATABASE_HOST")
    missing = [name for name in _REQUIRED if not os.environ.get(name)]
    if not host:
        missing.insert(0, "DATABASE_HOST")
    if missing:
        raise RuntimeError(
            "Missing database environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill it in."
        )
    return (
        f"host={host} "
        f"port={os.environ.get('DATABASE_PORT', '5432')} "
        f"dbname={os.environ['DATABASE_NAME']} "
        f"user={os.environ['DATABASE_USER']} "
        f"password={os.environ['DATABASE_PASSWORD']}"
    )


@contextmanager
def connect(dsn: str | None = None) -> Iterator[Connection]:
    """Open a connection, closing it on the way out."""
    conn = psycopg2.connect(dsn or build_dsn())
    try:
        yield conn
    finally:
        conn.close()
