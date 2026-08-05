"""Dev-only tooling for the object-search platform.

Everything under this package is *absent from production* (`wemap-vision-backend`).
It exists so the mirrored pipeline can be run, inspected and benchmarked locally:

- `toolbox.bricks`    — the pure-Python stand-in for the Django layer that owns
                        3D lifting, pgvector ingest, candidate enrichment and
                        clustering/ranking in production.
- `toolbox.georef`    — reads keyframe poses from an on-disk `georef.db`, in place
                        of the production `api_geokeyframe` / `GeoRef` tables.
- `toolbox.benchmark` — HTTP benchmark scoring the service against ground truth.

This is a Python package (rather than a bare directory on `sys.path`) on purpose:
the mirrored trees expose flat top-level modules named `db`, `app`, `models`,
`lib`, `query` and `build_index`, and a package namespace keeps us from colliding
with them.
"""
