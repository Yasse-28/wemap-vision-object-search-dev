# wemap-vision-object-search — Object Search Platform

This repository is the **platform hub** for Wemap's text-to-object search in
geolocated 360° panoramas: the place to run, inspect and iterate on the pipeline
that ships in production.

> **The pipeline here is a mirror of production.**
> `third_party/object_search/` and `services/object_search_online/` are
> byte-for-byte copies of `wemap-vision-backend`. Do not edit them here — fix the
> backend and re-sync. `scripts/check-mirror.sh` enforces it.

Architecture & rationale:
[ADR 0001](docs/adr/0001-object-search-platform-structure.md) (the platform role),
[ADR 0002](docs/adr/0002-align-on-backend-pipeline.md) (the alignment on
production, and what it retired).

## What's here

| Path | Role |
|------|------|
| [`third_party/object_search/`](third_party/object_search/) | **Mirror.** The offline `prepare` job (detection + MetaCLIP2 embeddings), the shared inference/indexing helpers, the pgvector benchmarks, and `annotation_service`. Copies of production — see [`PROVENANCE.md`](third_party/PROVENANCE.md). |
| [`services/object_search_online/`](services/object_search_online/) | **Mirror.** Production's GPU service: embed + HNSW → a flat `[{id, similarity}]` list. |
| [`toolbox/`](toolbox/) | **Owned dev tooling.** The Python `bricks` (what Django owns in production), the pose readers (v2 manifest + legacy `georef.db`), the HTTP benchmark, and the TypeScript UI to analyse, annotate and benchmark. |
| [`legacy/`](legacy/) | The retired standalone lineage. Reference only — unmaintained, and excluded from packaging, tests and lint. |
| [`docs/`](docs/) | Architecture decision records. |
| `infra/postgres/` | Local pgvector **+ PostGIS** dev database. |
| `scripts/` | `check-mirror.sh`, `build-index.sh`, and the two service launchers. |

## How the pieces run

```
                        toolbox UI (:45700)
                        proxies /{map_id}/object-search/…
                                  │
                                  ▼
                        bricks service (:45678)          ← the toolbox spawns this
                        enrichment, clustering, ranking
                                  │ POST /object-search/by-text
                                  ▼
                     mirrored online service (:8000)     ← you start this
                     MetaCLIP2 embed + pgvector HNSW
                                  │
                                  ▼
                      Postgres (pgvector + PostGIS)
```

The **bricks** are the local stand-in for the Django layer that, in production, owns
3D lifting, pgvector ingest, candidate enrichment and clustering/ranking. They are
ported from the backend and kept behaviourally identical. The toolbox starts the
bricks service for you; it deliberately does **not** start the mirrored online
service, which loads MetaCLIP on the GPU.

## Quickstart

```bash
conda create --name=wemap-vision python=3.11 && conda activate wemap-vision
pip install -e ".[dev,toolbox,prepare]"
pip install 'git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d'
cp .env.example .env      # fill in DATABASE_* and PGVECTOR_PASSWORD

docker compose -f infra/postgres/compose.yml up -d
```

### Build an index for a map

A map directory holds the ERP images (`images/`, or `images_360/` on v1 maps),
`depths/*.tif`, and a **pose source**:

| Map generation | Pose source |
|---|---|
| **v2 (current)** | `{map_id}_{version}_{date}_{time}.json` — poses already in EUS, plus the venue and the real `geo_ref_id` |
| v1 (legacy) | `georef.db` |

```bash
scripts/build-index.sh /path/to/map
```

For v2 maps the venue and georef id come from the manifest, so no flags are needed;
pass `--venue` / `--geo-ref-id` for v1 maps, which record neither.

That runs three steps, none of them optional:

1. `toolbox.bricks.prepare_runner` — resolves each image to its keyframe id from the
   pose source, then runs the mirrored `prepare` (detection + cutouts + embeddings).
   **Not `python -m prepare`:** that CLI numbers keyframes positionally and writes no
   thumbnails — see its module docstring.
2. `toolbox.bricks.prepare_postprocess` — adds `thumbnail_key` and `depth`.
   **Skipping this yields an index with no 3D positions, and a `localize` that
   silently returns nothing.**
3. `toolbox.bricks.ingest_cli` — 3D lifting, binary COPY into pgvector, partial
   HNSW index.

### Serve and query

```bash
scripts/run-online-service.sh                    # :8000, loads MetaCLIP (GPU)
scripts/run-bricks-service.sh /path/config.json  # :45678
```

```bash
curl -s localhost:45678/my-map/object-search/localize \
  -H 'content-type: application/json' \
  -d '{"text": "fire extinguisher", "num_results": 20}'
```

### Toolbox

See [`toolbox/README.md`](toolbox/README.md).

## Checks

```bash
scripts/check-mirror.sh /path/to/wemap-vision-backend   # the mirror has not drifted
ruff check . && black --check .
scripts/check-types.sh                                  # mypy, one pass per sys.path root
pytest
cd toolbox && npm run type-check
```

`pytest` is hermetic by default; the nine integration tests in
`toolbox/tests/test_integration_db.py` skip unless a database is reachable. They are
worth running when touching ingest, the schema or enrichment — they are what proves a
real Postgres accepts the hand-rolled binary COPY:

```bash
docker compose -f infra/postgres/compose.yml up -d
DATABASE_HOST=localhost DATABASE_USER=postgres DATABASE_PASSWORD=… pytest
```

## Where to read next

`AI_CONTEXT.md` is the navigation entry point: a routing table from "my task is
about X" to the one or two files worth opening. It exists so that neither a person
nor an agent has to read this tree to find anything.
