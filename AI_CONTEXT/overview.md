# Overview — architecture map

**Purpose:** the big picture — what each component is, how data flows, and the
cross-component contracts (ports, paths, DB) — so you can pick the right area.

**Read this when:** starting any task, or when you need "where does X live / how
do the pieces talk".

## The one thing to understand first

Since [ADR 0003](../docs/adr/0003-split-pipeline-into-a-submodule.md) this repo is the
**dev platform** around the production object-search pipeline, which it consumes
as the `third_party/object_search/` submodule (itself a mirror of `wemap-vision-backend`).

> `third_party/object_search/` and `third_party/object_search/services/object_search_online/` are
> **byte-for-byte copies of the backend**. Do not edit them. Fix the backend
> and re-sync; `scripts/check-mirror.sh` will otherwise fail.

Anything that is *not* in production lives in `toolbox/` (dev-only, maintained) or
`legacy/` (the retired standalone lineage, unmaintained).

## Components

| Component | Path | Language | Role |
|---|---|---|---|
| **Mirror: prepare** | `third_party/object_search/prepare/` | Python | Offline job: ERP images → detector proposals → MetaCLIP2 embeddings. **Copy of production.** |
| **Mirror: inference/indexing** | `third_party/object_search/{inference,indexing}/` | Python | Shared MetaCLIP2 embedder, ERP re-crop, Numba spatial pre-filter. **Copy of production.** |
| **Mirror: online service** | `third_party/object_search/services/object_search_online/` | Python | GPU FastAPI: embed + HNSW → flat `[{id, similarity}]`. **Copy of production.** |
| **Mirror: annotation service** | `third_party/object_search/annotation_service/` | Python | Production annotation CRUD + ground-truth export mirror. The dev Toolbox uses its own compatible integrated SQLite store. |
| **Bricks** | `toolbox/bricks/` | Python | Dev-only port of the four things Django owns in production: 3D lifting + pgvector ingest, candidate enrichment, clustering/ranking, and the depth bridge. |
| **Pose reader** | `toolbox/bricks/map_manifest.py` | Python | Dev-only, replacing the Django ORM: reads the v2 map manifest. Mirrored in TS by `toolbox/backend/src/map-manifest.ts`. |
| **Benchmark** | `toolbox/benchmark/` | Python | Dev-only: HTTP benchmark scoring localize against ground truth, for full runs or an explicitly filtered single prompt. |
| **Toolbox UI** | `toolbox/{backend,frontend}/` | TypeScript | Dev tool: inspect, search, annotate, benchmark. Proxies the Python services; does no ML. |
| **Legacy** | `legacy/` | Python | The retired standalone lineage. Reference only — not maintained, not linted, not tested. |

## What object search does

Text query → embed with MetaCLIP2 → retrieve matching detected-object crops in
geolocated 360° panoramas → cluster their 3D positions → return ranked,
geolocated object instances (lat/lon/alt, level, heading, confidence).

## Data flow

**Build (offline)** — `scripts/build-index.sh` runs all three steps. A v2 map
directory ships only its manifest, so fetch the pixels first with the sibling repo
`../retrieve-map-data` (`retrieve_map_data.py <map_dir>` → `images/` + `depths/`).

```
images/*.jpg   (alongside depths/*.tif)
  → toolbox.bricks.prepare_runner               ← NOT `python -m prepare`, see Gotchas
      pose source: image filename → keyframe id (+ venue, + geo_ref_id)
      then calls the mirrored prepare(), which does:
        multi-pass YOLO-World + GroundingDINO, pooled by class-agnostic NMS
        gnomonic ERP→rectilinear cutout per proposal
        MetaCLIP2 → 1024-d float16, L2-normalised
        GDINO labels filled by zero-shot MetaCLIP2 argmax
    ⇒ metadata.parquet + embeddings.npy + thumbnails/

  → toolbox.bricks.prepare_postprocess          ← NOT OPTIONAL, see Gotchas
      thumbnail_file → thumbnail_key
      sample depths/*.tif at each (theta, phi)  ⇒ adds the `depth` column

  → toolbox.bricks.ingest_cli
      pose source → keyframe poses (EUS position + OpenGL→EUS quaternion)
      spatial thinning (indexing.grid.filter_by_distance, 1.5 m)
      3D lift: object_pos = camera_pos + depth · rotate(orientation, ray(θ, φ))
      binary COPY → object_search_candidate
      CREATE INDEX CONCURRENTLY … hnsw (embedding halfvec_l2_ops) WHERE geo_ref_id = …
```

**Serve (online)** — two processes:

```
toolbox UI (:45700)
  └─ proxies /{map_id}/object-search/… →  bricks service (:45678)
                                            │  enrichment + clustering + ranking
                                            └─ POST /object-search/by-text
                                                 →  mirrored online service (:45677)
                                                      MetaCLIP2 embed + pgvector HNSW
                                                      ⇒ [{id, similarity}]
```

The bricks service is the local stand-in for Django's object-search API, and the
toolbox spawns it on demand. The mirrored online service you start yourself
(`scripts/run-online-service.sh`) — it loads MetaCLIP on the GPU, which is not
something to trigger implicitly.

## Ports and paths

| Component | Listens | Calls | Path shape |
|---|---|---|---|
| Mirrored online service | 45677 | Postgres | `/object-search/by-text`, `/by-image`, `/health` — **no `{map_id}` segment**; scoping is the `geo_ref_id` body field |
| Bricks service | 45678 | online service + Postgres | `/{map_id}/object-search/{localize,text}`, `/health` |
| Toolbox backend | 45700 | bricks service | `/ui/api/…` owned; `/{map_id}/object-search/…` proxied |
| annotation_service (production mirror only) | 8001 | its own SQLite | `/{slug}/object-search/…` |

45678 is kept deliberately: the bricks service is a drop-in replacement for the
standalone service it supersedes, so the toolbox, the benchmark and
wemap-vision-tools keep the port and path shape they already agree on. This closes
gap 2 of ADR 0001; **gap 1 remains** — livemap calls
`…/geopose/object-search/text`, which nothing here serves.

## Important constants and contracts

- Embedding: **1024-d float16**, L2-normalised, `facebook/metaclip-2-worldwide-huge-quickgelu`.
- `theta_center = atan2(x, z)` — **0 at the ERP horizontal centre**; `phi = asin(y)`.
  One convention across the whole pipeline (`prepare/convention.py`).
- Table `object_search_candidate`; index `idx_object_search_candidate_hnsw_georef_{id}`,
  `halfvec_l2_ops`, `m=16`, `ef_construction=64`, partial per georef.
- Query: `ef_search = max(k, 1000)`; cosine recovered as `1 − d²/2`.
- `object_position`: `geometry(PointZ, 0)` — **SRID 0, not 4326**.
- Ranking: `match_score = cluster_best_sim / best_cluster_of_the_query`. One term, no
  free parameter. **Dev-only divergence** — production still ships
  `0.50·normalised_similarity + 0.15·confidence + 0.35·min(1, n_keyframes/3)`; see
  [`bricks.md`](bricks.md) for the measurement that replaced it.
- Geometric support is **filtered, not scored**: `min_keyframes_per_cluster = 2`,
  `min_observations_per_cluster = 1` (off), `max_cluster_spread_m = None` (off).
  `max_depth_m = None` (off) caps a *detection*'s own depth before association.
- Clustering: leader-canopy, `eps = 2.0 m`. `semantic_gate_threshold` (`None` = off)
  adds ConceptGraphs' second, semantic gate — see [`bricks.md`](bricks.md).
- Local DB needs **both** `vector` and `postgis` — see `infra/postgres/Dockerfile`.

## Pose source and coordinate frames

One format: the **v2 manifest** (`{map_id}_{version}_{date}_{time}.json`), a dump of
the production objects. `x/y/z` are already EUS and `orientation` is already the
`[w, x, y, z]` OpenGL→EUS quaternion, so **no conversion applies**. It also carries
`venue_type` and the real `geo_ref_id`. See `toolbox/bricks/map_manifest.py`.

Keyframe ids are **indices into `geo_keyframes`** — the manifest has no integer id.
Re-exporting it renumbers everything, so prepare and ingest must be re-run together.

The one place a frame conversion survives is the toolbox's TS backend, whose routes
speak WDS world-to-camera on the wire (`point_world_wds`). `map-manifest.ts` adapts
EUS→WDS at the boundary; nothing else in the system uses WDS.

EUS is East-Up-South: `+X` East, `+Y` Up, `+Z` South, so North is `-Z` and the
OpenGL camera forward (`-Z`) means "looking North".

## Infrastructure

| Path | Purpose |
|---|---|
| `infra/postgres/Dockerfile` | `pgvector/pgvector:pg17` + PostGIS. The stock image has no PostGIS, and the schema needs both extensions. |
| `infra/postgres/compose.yml` | Local dev instance, built from that Dockerfile. `PGVECTOR_PASSWORD` required; `pgdata/` gitignored. Run: `docker compose -f infra/postgres/compose.yml up -d`. Sets `shm_size: 2gb` — Docker's 64 MB default kills a parallel HNSW build on a million-row georef. |

Note `PGVECTOR_*` configures only the container. The pipeline itself reads
`DATABASE_*` (plus `ENVIRONMENT_NAME=onprem`), because that is what the mirrored
service reads — see `.env.example`.

## Gotchas

**Three silent failure modes.** Each produces plausible wrong output, not an error.
All three are covered by tests; read ADR 0002 §Traps before touching them.

1. **Skipping `prepare_postprocess`** → no `depth` → every `object_position` NULL →
   `localize` returns `[]`, indistinguishable from "found nothing".
2. **Frame conventions** — the manifest needs no conversion, but the TS backend's
   EUS→WDS adapter does; get it wrong and objects land mirrored or 180° off, with no
   error. Pinned by `toolbox/backend/src/map-manifest.test.ts`.
3. **Level datum** — level bands are heights above the origin; feed
   `levels_for_altitudes` the EUS up coordinate, never the WGS84 altitude. Null
   bounds mean unbounded, not zero.
4. **`python -m prepare` numbers keyframes positionally** (`enumerate`), so its
   `video_keyframe_id` is not the pose source's keyframe id. Feed that to ingest and
   candidates attach to whichever keyframes share those ids. Use
   `toolbox.bricks.prepare_runner`, which resolves real ids first.
5. **Re-exporting a v2 manifest renumbers keyframes** — ids are `geo_keyframes`
   indices. Re-run prepare *and* ingest together after a new export.

**Toolbox state.** `localize-offline` is gone for good. Text search and
`localize-online` both work — text search reads the rows the bricks `text` endpoint
already enriched instead of re-enriching through an index. The object-search explorer
browses `{map}/object-search/metadata.parquet` directly (read in TypeScript, see ADR
0005); OCR and cluster ids have no source in v2 and that surface is gone.

The Annotation tab can explicitly score its current prompt through the same Python
benchmark evaluator used by full runs. Single-prompt artifacts live under
`benchmark/prompt-scores/<slug>/`, below the run-list scan depth, and therefore never
appear in `/benchmark/status`.

**Three keyframe sets, deliberately not merged.** The manifest's keyframes (have a
pose), the parquet's (prepare ran on them) and pgvector's (survived ingest's 1.5 m
pruning). `GET /{map_id}/object-search/index-coverage` on the bricks service reports
the third; it is allowed to fail, and then keyframes read "unknown".

## Deep references

- Sync point and what was deliberately left out: `../third_party/object_search/PROVENANCE.md`
- Vendored production helpers and their intended deltas: `../toolbox/bricks/vendored/PROVENANCE.md`
- Why each retired area was retired: `../legacy/README.md`
- Architecture decisions: `../docs/adr/`
- The mirror's own spec (prepare contract, parquet schema, pgvector design):
  `../third_party/object_search/AI_CONTEXT.md`
