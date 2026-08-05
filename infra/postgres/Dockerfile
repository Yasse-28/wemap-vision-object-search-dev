# pgvector + PostGIS in one image.
#
# The stock `pgvector/pgvector:pg17` ships only the `vector` extension, but the
# object-search schema needs both:
#   - `halfvec(1024)` for embeddings                     → vector
#   - `geometry(PointZ, 0)` + ST_X/ST_Y/ST_Z for positions → postgis
#
# Without PostGIS, `CREATE EXTENSION postgis` fails, `object_search_candidate`
# cannot be created, and candidate enrichment cannot run. Building on top of the
# pgvector image (rather than on postgis/postgis) keeps `vector` at its exact
# upstream version, which is the one production runs.
FROM pgvector/pgvector:pg17

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-postgis-3 \
    && rm -rf /var/lib/apt/lists/*
