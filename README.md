# wemap-vision-object-search-dev — Object Search Dev Platform

This repository is the **dev platform** for Wemap's text-to-object search in
geolocated 360° panoramas: the place to run, inspect and iterate on the pipeline
that ships in production.

The pipeline itself is **not here**. It is `third_party/object_search/`, a
submodule of [`wemap-vision-object-search`](https://github.com/wemap/wemap-vision-object-search),
which is in turn a byte-for-byte mirror of `wemap-vision-backend`. Clone
accordingly:

```bash
git clone --recurse-submodules <this repo>
# already cloned?  git submodule update --init
```

> **Never edit anything under `third_party/object_search/`.** It is production
> code; fix the backend and re-sync. `scripts/check-mirror.sh` enforces it.
> Anything dev-only belongs in this repo instead.

Architecture & rationale:
[ADR 0001](docs/adr/0001-object-search-platform-structure.md) (the platform role),
[ADR 0002](docs/adr/0002-align-on-backend-pipeline.md) (the alignment on
production, and what it retired),
[ADR 0003](docs/adr/0003-split-pipeline-into-a-submodule.md) (this split),
[ADR 0004](docs/adr/0004-v2-map-data-only.md) (v2 map data only),
[ADR 0005](docs/adr/0005-explorer-reads-the-parquet.md) (the explorer reads
`metadata.parquet`), and
[ADR 0006](docs/adr/0006-integrate-annotations-into-toolbox.md) (the Toolbox owns
its local annotation SQLite store).

## What's here

| Path | Role |
|------|------|
| [`third_party/object_search/`](third_party/object_search/) | **Mirror.** The offline `prepare` job (detection + MetaCLIP2 embeddings), the shared inference/indexing helpers, the pgvector benchmarks, and `annotation_service`. Copies of production — see [`PROVENANCE.md`](third_party/object_search/PROVENANCE.md). |
| [`third_party/object_search/services/object_search_online/`](third_party/object_search/services/object_search_online/) | **Mirror.** Production's GPU service: embed + HNSW → a flat `[{id, similarity}]` list. |
| [`toolbox/`](toolbox/) | **Owned dev tooling.** The Python `bricks` (what Django owns in production), the v2 manifest pose reader, the HTTP benchmark, and the TypeScript UI to analyse, annotate and benchmark. |
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

# Two installs: the pipeline declares its own dependencies, this repo adds the
# dev-only ones on top. Order matters only in that both must be present.
pip install -e './third_party/object_search[prepare]'
pip install -e '.[dev]'
pip install 'git+https://github.com/ultralytics/CLIP.git@81ff68ed7ffcac3b40484c914f104f816757308d'
cp .env.example .env      # fill in DATABASE_* and PGVECTOR_PASSWORD

docker compose -f infra/postgres/compose.yml up -d
```

### Build an index for a map

A map directory mirrors the S3 layout — `images/*.jpg`, `depths/*.tif` — plus the
**v2 manifest** `{map_id}_{version}_{date}_{time}.json`, which carries the poses
(already in EUS), the venue and the real `geo_ref_id`.

```bash
scripts/build-index.sh /path/to/map
```

Everything the build needs about the map comes from that manifest, so there are no
`--venue` / `--geo-ref-id` flags to get wrong. The legacy `georef.db` format is no
longer read; see `docs/adr/0004-v2-map-data-only.md`.

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
scripts/check-all.sh                                    # everything below, in order
```

Or one at a time:

```bash
scripts/check-mirror.sh /path/to/wemap-vision-backend   # the mirror has not drifted
ruff check . && black --check .
scripts/check-types.sh                                  # mypy, toolbox/ only
pytest                                                  # 136 tests: toolbox/tests
(cd third_party/object_search && pytest)                # 20 tests: the mirror's own
cd toolbox && npm run type-check && npm test -w backend
```

**Two `pytest` runs, on purpose.** The submodule owns its 20 mirrored tests and runs
them with its own config; recursing into it from here would drag in the
`annotation_service` and `object_search_online` sys.path roots, whose flat `app.py`
modules collide by name. `scripts/check-all.sh` runs both.

`pytest` is hermetic by default; the nine integration tests in
`toolbox/tests/test_integration_db.py` skip unless a database is reachable. They are
worth running when touching ingest, the schema or enrichment — they are what proves a
real Postgres accepts the hand-rolled binary COPY:

```bash
docker compose -f infra/postgres/compose.yml up -d
DATABASE_HOST=localhost DATABASE_USER=postgres DATABASE_PASSWORD=… pytest
```

## Versioning

This repo is where ideas get validated before being ported to the real backend
(`wemap-vision-backend`) or its mirror. There is no environment-branch model
(`dev`/`staging`/`prod`) — this repo has no production counterpart to mirror,
and promotion means re-implementing the validated logic in the backend, not
merging a branch.

Work happens on short-lived `feat/*`/`fix/*` branches merged into `main` once
done, same as today. When a piece of work on `main` is validated and ready to
be ported (or has been ported), mark it with an annotated tag:

```bash
git tag -a validated/<slug> -m "<one line: what was validated, link to ADR/plan if any>"
```

`<slug>` matches the feature, e.g. `validated/review-feedback-boost`. The tag
is a marker, not a release — it does not imply the repo is installable or
stable at that point, only that whatever it names was deliberately checked
before moving on.

## Where to read next

`AI_CONTEXT.md` is the navigation entry point: a routing table from "my task is
about X" to the one or two files worth opening. It exists so that neither a person
nor an agent has to read this tree to find anything.
