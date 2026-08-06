# The bricks — Django's job, in pure Python

**Purpose:** the four things Django owns in production, ported so the pipeline runs
end to end without it: 3D lifting + pgvector ingest, candidate enrichment,
clustering/ranking, and the depth bridge.

**Read this when:** the task concerns ingest, the local schema, candidate
enrichment, clustering, ranking, the `localize` endpoint, or keyframe poses.

## Why this exists

The mirror covers prepare (offline) and embed+HNSW (online). Between them,
production has `backend/object_search/` — Django. That layer is not plumbing: it owns
everything that turns a bag of embeddings into map positions. It is ported here
rather than mirrored, because Django cannot be, and it lives under `toolbox/` rather
than beside the mirror, because production does not need it.

**The algorithms are behaviourally identical to production on purpose.** When they
diverge, production wins. A "small improvement" here is a bug.

## Key files — `toolbox/bricks/`

| Path | Ported from (backend) | Key symbols |
|---|---|---|
| `ingest.py` | `db/ingest.py` | `encode_copy_stream`, `bulk_copy`, `create_partial_hnsw_index`, `_EWKB_POINTZ_PREFIX`, `_HNSW_INDEX_LOCK_KEY`, `INDEX_NAME_TEMPLATE` |
| `ingest_cli.py` | `object_search_ingest.py` | `run_ingest`, `_compute_object_positions`, `_ingest_capture`, `_upsert_geokeyframes`, `discover_capture_dirs`, `EMBEDDING_DIM=1024`, `DEFAULT_MIN_DISTANCE=1.5` |
| `candidates.py` | `candidates.py` | `EnrichedCandidate`, `load_enriched_candidates`, `_prefilter_hnsw_results`, `K_INTERNAL=1000`, `LOOSE_ALPHA=0.3` |
| `localize.py` | `v1_5_logic.py` | `cluster_detections_leader_canopy`, `compute_cluster_statistics`, `rank_localization_clusters`, `localize_from_enriched_candidates`, `build_localize_response`, `LocalizationParams`, `UNRESOLVED_LEVEL=-1`, `PLACEHOLDER_BBOX` |
| `prepare_runner.py` | `object_search_prepare.py` (its `image_entries` construction) | `collect_image_entries`, `run` |
| `prepare_postprocess.py` | `object_search_prepare.py::_sample_depths` | `postprocess_metadata`, `sample_depths` |
| `map_manifest.py` | *(no counterpart — replaces the ORM)* | `load_map_manifest`, `find_manifest`, `MapManifest`, `ManifestKeyframe` |
| `vendored/proposal_cutouts.py` | overrides the **mirror's** `prepare/proposal_cutouts.py` | `create_proposal_cutouts`, `install`, `DEFAULT_CUTOUT_BATCH=10` — memory-only delta, see gotcha 9 |
| `georef_source.py` | *(no counterpart — replaces the ORM)* | `load_pose_source`, `PoseSource`, `KeyframePose` — a thin façade over `map_manifest` |
| `db_schema.py` | `api/models.py` + migrations | `ensure_schema`, `CREATE_CANDIDATE`, `CREATE_GEOKEYFRAME` |
| `db.py` | — | `build_dsn`, `connect` (reads the mirror's `DATABASE_*`) |
| `service.py` | `v1_5_views.py` | `create_app`, `query_by_text`, `query_by_image`, `LocalizeRequest`, `load_map_entries`, `index_coverage` (**dev-only, no production counterpart**: per-keyframe `ingested`/`no_position` counts, so the toolbox can tell prepared-but-pruned keyframes from indexed ones without a `pg` client of its own) |
| `vendored/` | `utils/`, `depth/service/decode.py`, `viewer360/`, `v1_legacy.py` | Copies — see its `PROVENANCE.md` |

`localize.py` differs from production in **four import lines only**. Treat any
behavioural change to it as a bug.

## What replaced what

| Production | Here |
|---|---|
| `GeoKeyframe` / `GeoRef` / `GeoLevel` ORM | the v2 map manifest (`map_manifest`) |
| S3 `get_object` | files under the map directory |
| `django.db.connection` | an injected psycopg2 connection |
| `api.utils.spatial_sampling.select_keyframes_by_distance` | `indexing.grid.filter_by_distance` — the mirror's original, of which the backend helper is itself a vendored copy. This port *removes* a vendoring. |
| Slack `ProcessTracker` | `_step()`, a logging context manager |
| Django migrations | `db_schema.ensure_schema` |

## Local schema

Two tables. `object_search_candidate` is faithful to `api.models` — its name, its
`geo_ref_id` column and the `halfvec(1024)` embedding are **not ours to rename**,
because the mirrored service queries it directly. `geokeyframe` is a minimal local
stand-in holding only what enrichment joins on (EUS position, orientation quaternion),
populated from the pose source.

Needs both the `vector` and `postgis` extensions.

## Types are load-bearing

`bulk_copy` writes with PostgreSQL's **binary** COPY, which encodes each value at a
fixed width with no server-side coercion:

- the four angular columns must be `DOUBLE PRECISION` — `REAL` fails every row at
  runtime with a length error;
- `object_position` must be `geometry(PointZ, 0)` — **SRID 0, not 4326**; the EWKB
  prefix hardcodes it.

`toolbox/tests/test_copy_encoding.py` asserts the bytes field by field, because in
normal use the only thing validating this encoding is the server accepting it.

## Frames

**Manifests need no conversion** — they store what `api_geokeyframe` stores (EUS
position, `[w, x, y, z]` orientation), plus `venue_type` and the real `geo_ref_id`.
Nothing in the Python half flips an axis.

The one surviving conversion lives in the TS backend
(`toolbox/backend/src/map-manifest.ts`), whose routes speak WDS world-to-camera on
the wire. It composes `diag(-1,-1,1)` and `diag(1,-1,-1)`; see its docstring.

Also note: one id plays the role of **both** `geokeyframe_id` and
`video_keyframe_id` here — the `geo_keyframes` index. Production distinguishes them,
we do not.

## Gotchas

1. **Never chain `python -m prepare` into ingest.** Its CLI numbers keyframes with
   `enumerate`, so `video_keyframe_id` is a position, not the manifest's keyframe id;
   it also passes no `crops_output_dir`, so no thumbnails are written.
   `prepare_runner` resolves real ids first — the same thing the Django command does.
   Ids are `geo_keyframes` indices, so **re-exporting the manifest renumbers them**:
   re-run prepare and ingest together.
2. **`prepare_postprocess` is not optional, and skipping it is silent.** `prepare`
   emits no `depth` column; without it `bulk_copy` writes NULL, every
   `object_position` is NULL, enrichment filters every row, and `localize` returns
   `[]` — indistinguishable from "the model found nothing".
3. **The EUS axis convention is unchecked elsewhere.** `toolbox/tests/test_manifest_frames.py`
   pins it by making the geodesy path and the rotation path agree on a compass
   bearing; they share no code. Read x/y/z in the wrong order and objects land
   mirrored or 180° off, with no error.
4. **Levels are heights above the origin.** `levels_for_altitudes` must be fed the
   EUS *up* coordinate (`eus_xyz[:, 1]`), never the WGS84 altitude. Get it backwards
   and every level is `None`, which silently disables the level-compatibility guard
   and merges objects across floors.
5. **`geo_ref_id` comes from the manifest, and only from there.** It is the partition
   key of the table and of the partial HNSW index, so a mismatch between what ingest
   wrote and what the service queries returns zero hits with no error. That is why
   neither `ingest_cli` nor `service` accepts it as an override any more.
6. **`create_partial_hnsw_index` needs autocommit** (`CREATE INDEX CONCURRENTLY`
   cannot run in a transaction) and polls `pg_try_advisory_lock` rather than blocking
   — a blocking `pg_advisory_lock` would hold a transaction open and deadlock against
   the CIC. That comment in the source is load-bearing; do not "simplify" it.
7. **`vendored/` is copied production code.** Fix bugs in the backend and re-sync;
   see `../toolbox/bricks/vendored/PROVENANCE.md`.
8. **The on-disk filename is the basename of the URL *path*.**
   `map_manifest._basename` parses `image_url`/`depth_url` with `urlparse`, and so
   do the other three v2 readers. Current manifests point at a public bucket with
   no query string, so splitting on `/` happens to work — but a presigned URL would
   yield `abc.jpg?X-Amz-Signature=…`, matching nothing on disk, and both symptoms
   (no keyframe id resolved, no depth found) are silent.
9. **Cutout extraction is the GPU-memory bottleneck, and one mirrored function is
   overridden because of it.** `vendored/proposal_cutouts.py` + `install()` replace the
   mirror's `create_proposal_cutouts` at runtime: as mirrored it holds two replicated
   ERPs at once (1.85 GiB each at 5760×2880) and OOMs an 8 GB card. Output is bitwise
   identical — only peak memory changes. On 8 GB use
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or `--cutout-batch 4`. Note
   `--batch-size` is the *MetaCLIP* batch and has no effect on this.

## Where images and depths come from

A map directory holds only the manifest; the pixels are fetched separately into
`{map}/images/` and `{map}/depths/` — the names `prepare_runner.resolve_images_dir`
and `prepare_postprocess.sample_depths` look for. Both resolve files **only** by the
basename the manifest records; there is no id-derived fallback, because ids are array
indices and `2.tif` would silently match the wrong keyframe. The sibling repo
`../retrieve-map-data` does that (`retrieve_map_data.py <map_dir>`), reading the
same manifest and using the same basename rule. `scripts/build-index.sh` fails
early and by name when either directory is absent.

## Cross-refs

- Where the inputs come from: [`mirror-prepare.md`](mirror-prepare.md)
- Where the ANN hits come from: [`mirror-serving.md`](mirror-serving.md)
- The full rationale and trap list: `../docs/adr/0002-align-on-backend-pipeline.md`
