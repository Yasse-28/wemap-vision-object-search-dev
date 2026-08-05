# Overview — architecture map

**Purpose:** the big picture — what each component is, how data flows, and the
cross-component contracts (ports, paths, DB) — so you can pick the right area.

**Read this when:** starting any task, or when you need "where does X live / how
do the pieces talk".

## The one thing to understand first

Since [ADR 0002](../docs/adr/0002-align-on-backend-pipeline.md) this repo is a
**mirror of the production object-search pipeline** in `wemap-vision-backend`,
plus the dev tooling production does not need.

> `third_party/object_search/` and `services/object_search_online/` are
> **byte-for-byte copies of the backend**. Do not edit them here. Fix the backend
> and re-sync; `scripts/check-mirror.sh` will otherwise fail.

Anything that is *not* in production lives in `toolbox/` (dev-only, maintained) or
`legacy/` (the retired standalone lineage, unmaintained).

## Components

| Component | Path | Language | Role |
|---|---|---|---|
| **Mirror: prepare** | `third_party/object_search/prepare/` | Python | Offline job: ERP images → detector proposals → MetaCLIP2 embeddings. **Copy of production.** |
| **Mirror: inference/indexing** | `third_party/object_search/{inference,indexing}/` | Python | Shared MetaCLIP2 embedder, ERP re-crop, Numba spatial pre-filter. **Copy of production.** |
| **Mirror: online service** | `services/object_search_online/` | Python | GPU FastAPI: embed + HNSW → flat `[{id, similarity}]`. **Copy of production.** |
| **Mirror: annotation service** | `third_party/object_search/annotation_service/` | Python | Annotation CRUD + ground-truth export. **Copy of production** (minus its Terraform). |
| **Bricks** | `toolbox/bricks/` | Python | Dev-only port of the four things Django owns in production: 3D lifting + pgvector ingest, candidate enrichment, clustering/ranking, and the depth bridge. |
| **Pose readers** | `toolbox/bricks/map_manifest.py`, `toolbox/georef/` | Python | Dev-only, replacing the Django ORM: the v2 map manifest, and the legacy `georef.db`. |
| **Benchmark** | `toolbox/benchmark/` | Python | Dev-only: HTTP benchmark scoring localize against ground truth. |
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
images_360/*.jpg   (v2: images/*.jpg, alongside depths/*.tif)
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
                                                 →  mirrored online service (:8000)
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
| Mirrored online service | 8000 | Postgres | `/object-search/by-text`, `/by-image`, `/health` — **no `{map_id}` segment**; scoping is the `geo_ref_id` body field |
| Bricks service | 45678 | online service + Postgres | `/{map_id}/object-search/{localize,text}`, `/health` |
| Toolbox backend | 45700 | bricks service | `/ui/api/…` owned; `/{map_id}/object-search/…` proxied |
| annotation_service | 8001 | its own SQLite | `/{slug}/object-search/…` |

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
- Ranking: `0.50·normalised_similarity + 0.15·confidence + 0.35·keyframe_score`,
  with `keyframe_score = min(1, n_keyframes/3)`.
- Clustering: leader-canopy, `eps = 2.0 m`, `min_keyframes_per_cluster = 2`.
- Local DB needs **both** `vector` and `postgis` — see `infra/postgres/Dockerfile`.

## Pose sources and coordinate frames

Two formats, and they differ in how much work they need:

- **v2 manifest** (`{map_id}_{version}_{date}_{time}.json`) — a dump of the production
  objects. `x/y/z` are already EUS and `orientation` is already the `[w, x, y, z]`
  OpenGL→EUS quaternion, so **no conversion applies**. It also carries `venue_type`
  and the real `geo_ref_id`. See `toolbox/bricks/map_manifest.py`.
- **v1 `georef.db`** — poses stored transposed, world-to-camera, in WDS/OpenCV. Three
  flips compose to reach production's convention, in `toolbox/bricks/georef_source.py`:

```
position_eus    = ROT_WDS_TO_EUS @ inv(pose)[:3, 3]
orientation_eus = quat(ROT_WDS_TO_EUS @ inv(pose)[:3, :3] @ ROT_OPENGL_TO_OPENCV)
```

`load_pose_source` prefers the manifest and falls back to `georef.db`.

EUS is East-Up-South: `+X` East, `+Y` Up, `+Z` South, so North is `-Z` and the
OpenGL camera forward (`-Z`) means "looking North".

## Infrastructure

| Path | Purpose |
|---|---|
| `infra/postgres/Dockerfile` | `pgvector/pgvector:pg17` + PostGIS. The stock image has no PostGIS, and the schema needs both extensions. |
| `infra/postgres/compose.yml` | Local dev instance, built from that Dockerfile. `PGVECTOR_PASSWORD` required; `pgdata/` gitignored. Run: `docker compose -f infra/postgres/compose.yml up -d`. |

Note `PGVECTOR_*` configures only the container. The pipeline itself reads
`DATABASE_*` (plus `ENVIRONMENT_NAME=onprem`), because that is what the mirrored
service reads — see `.env.example`.

## Gotchas

**Three silent failure modes.** Each produces plausible wrong output, not an error.
All three are covered by tests; read ADR 0002 §Traps before touching them.

1. **Skipping `prepare_postprocess`** → no `depth` → every `object_position` NULL →
   `localize` returns `[]`, indistinguishable from "found nothing".
2. **Frame conventions** — v1 only: three flips compose in `georef_source.py`; drop
   one and objects land mirrored or 180° off. v2 manifests need no conversion.
3. **Level datum** — level bands are heights above the origin in both formats; feed
   `levels_for_altitudes` the EUS up coordinate, never the WGS84 altitude.
4. **`python -m prepare` numbers keyframes positionally** (`enumerate`), so its
   `video_keyframe_id` is not the pose source's keyframe id. Feed that to ingest and
   candidates attach to whichever keyframes share those ids. Use
   `toolbox.bricks.prepare_runner`, which resolves real ids first.
5. **Re-exporting a v2 manifest renumbers keyframes** — ids are `geo_keyframes`
   indices. Re-run prepare *and* ingest together after a new export.

**Toolbox degradations.** The index-explorer panel still depends on the retired
SQLite index and answers `501`; `localize-offline` is gone for good. Text search and
`localize-online` both work — text search reads the rows the bricks `text` endpoint
already enriched instead of re-enriching through the index.

## Deep references

- Sync point and what was deliberately left out: `../third_party/PROVENANCE.md`
- Vendored production helpers and their intended deltas: `../toolbox/bricks/vendored/PROVENANCE.md`
- Why each retired area was retired: `../legacy/README.md`
- Architecture decisions: `../docs/adr/`
- The mirror's own spec (prepare contract, parquet schema, pgvector design):
  `../third_party/object_search/AI_CONTEXT.md`
