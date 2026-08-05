from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg


class DatabaseState(Enum):
    LOCAL_READ_WRITE = "local_read_write"
    AWS_READ_ONLY = "aws_read_only"
    AWS_READ_WRITE = "aws_read_write"


def create_connection_metaclip(database_state: DatabaseState) -> psycopg.Connection:
    """Open a pgvector connection for the requested database state.

    Credentials are read from the environment so that no secret is stored in
    source. The following variables are used:

    - ``PGVECTOR_USER`` (optional, default ``postgres``)
    - ``PGVECTOR_PASSWORD`` (required)
    - ``PGVECTOR_DBNAME`` (optional, default ``object_search_db``; local state only)
    - ``PGVECTOR_HOST`` (optional, default ``localhost``; local state only)
    - ``PGVECTOR_HOST_RO`` (required for ``AWS_READ_ONLY``)
    - ``PGVECTOR_HOST_RW`` (required for ``AWS_READ_WRITE``)
    - ``PGVECTOR_PORT`` (optional, default ``5432``)

    Args:
        database_state: Which database endpoint to connect to.

    Returns:
        An open psycopg connection.

    Raises:
        KeyError: If a required environment variable is not set.
        ValueError: If ``database_state`` is not a recognised value.
    """
    import psycopg  # optional backend driver, imported lazily

    user = os.environ.get("PGVECTOR_USER", "postgres")
    password = os.environ["PGVECTOR_PASSWORD"]
    port = os.environ.get("PGVECTOR_PORT", "5432")

    if database_state == DatabaseState.LOCAL_READ_WRITE:
        dbname = os.environ.get("PGVECTOR_DBNAME", "object_search_db")
        host = os.environ.get("PGVECTOR_HOST", "localhost")
        return psycopg.connect(
            f"dbname={dbname} user={user} password={password} "
            f"host={host} port={port}"
        )
    if database_state == DatabaseState.AWS_READ_ONLY:
        host = os.environ["PGVECTOR_HOST_RO"]
        return psycopg.connect(
            f"user={user} password={password} host={host} port={port}"
        )
    if database_state == DatabaseState.AWS_READ_WRITE:
        host = os.environ["PGVECTOR_HOST_RW"]
        return psycopg.connect(
            f"user={user} password={password} host={host} port={port}"
        )

    raise ValueError(f"Unknown database state: {database_state!r}")
