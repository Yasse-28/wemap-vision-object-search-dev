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
| `candidates.py` | `candidates.py` | `EnrichedCandidate`, `load_enriched_candidates`, `_prefilter_hnsw_results`, `apply_feedback_boost`, `normalize_prototype_similarities`, `_parse_embedding`, `FEEDBACK_NORMALIZATIONS`, `K_INTERNAL=1000`, `LOOSE_ALPHA=0.3` |
| `feedback.py` | *(dev-only — no production counterpart)* | `ReviewFeedback`, `load_review_feedback`, `normalize_query`, `DB_FILENAME` — reads the toolbox's `object-search-annotations.db` |
| `localize.py` | `v1_5_logic.py` | `cluster_detections_leader_canopy`, `compute_cluster_statistics`, `rank_localization_clusters`, `_similarity_ratio_scores`, `filter_clusters_by_geometry`, `localize_from_enriched_candidates`, `build_localize_response`, `LocalizationParams`, `UNRESOLVED_LEVEL=-9999`, `PLACEHOLDER_BBOX` |
| `prepare_runner.py` | `object_search_prepare.py` (its `image_entries` construction) | `collect_image_entries`, `run` |
| `prepare_postprocess.py` | `object_search_prepare.py::_sample_depths` | `postprocess_metadata`, `sample_depths` |
| `map_manifest.py` | *(no counterpart — replaces the ORM)* | `load_map_manifest`, `find_manifest`, `MapManifest`, `ManifestKeyframe` |
| `vendored/proposal_cutouts.py` | overrides the **mirror's** `prepare/proposal_cutouts.py` | `create_proposal_cutouts`, `install`, `DEFAULT_CUTOUT_BATCH=10` — memory-only delta, see gotcha 11 |
| `georef_source.py` | *(no counterpart — replaces the ORM)* | `load_pose_source`, `PoseSource`, `KeyframePose` — a thin façade over `map_manifest` |
| `db_schema.py` | `api/models.py` + migrations | `ensure_schema`, `CREATE_CANDIDATE`, `CREATE_GEOKEYFRAME` |
| `db.py` | — | `build_dsn`, `connect` (reads the mirror's `DATABASE_*`) |
| `service.py` | `v1_5_views.py` | `create_app`, `query_by_text`, `query_by_image`, `LocalizeParams` (query-less half, also validates the multipart form so both branches share one set of defaults), `LocalizeRequest`, `load_map_entries`, `index_coverage` (**dev-only, no production counterpart**: per-keyframe `ingested`/`no_position` counts, so the toolbox can tell prepared-but-pruned keyframes from indexed ones without a `pg` client of its own) |
| `vendored/` | `utils/`, `depth/service/decode.py`, `viewer360/`, `v1_legacy.py` | Copies — see its `PROVENANCE.md` |

### Review-feedback boost (dev-only)

`feedback.py` → `candidates.py` → `localize.py`. `LocalizeRequest.feedback_alpha`/
`feedback_beta` (both `0.0` = off, and off is exact: `_ranking_similarities` never reads
the boosted field) promote/demote a candidate by its similarity to the cutouts a user
marked `true_positive`/`false_positive` for the *same normalised query*.
`feedback_normalization` (`"none"` default, `"center"`, `"standardize"`) rescales those
prototype columns across the retrieved set first — they are image↔image similarities
(~0.7–0.9) against a text↔image base (~0.15–0.30), so raw they are mostly a constant
offset, and a constant offset *flattens* `rank_localization_clusters`' relative
normalisation instead of sharpening it.

Only `cluster_best_sim` is boosted. The prefilter, the sort, the `select_top_candidates`
truncation, the clustering seed order, the centroid weights and the observation order
all stay on raw similarity — the boost changes cluster *ranking*, never cluster
*geometry*. `detection_review.target_id` is a BIGSERIAL that does not survive a
reingest; `_log_prototype_resolution` warns when the prototypes resolve to nothing,
which is the only way to tell an inert boost from an unhelpful one.

### Scoring: one term, and geometry moved to filters (dev-only divergence)

`match_score = cluster_best_sim / best_cluster_of_the_query` — "this cluster reaches
X% of the quality of this query's best match". It **replaces** production's
`0.50·norm_sim + 0.15·confidence + 0.35·min(1, n_keyframes/3)`, and the replacement is
deliberate, not a port slip. Measured on `bbhotel-choisy` (12 prompts, 674 annotations,
1287 clusters — the only benchmark ground truth to trust, see the memory note):

| | mAP | macro F1 @ best shared threshold | leave-one-prompt-out |
|---|---|---|---|
| weighted mixture | 0.652 | 0.598 (t=0.776) | 0.533 |
| **ratio** | 0.653 | **0.632** (t=0.905) | **0.611** |
| raw similarity | 0.653 | 0.502 (t=0.224) | 0.496 |

Why the mixture had to go:

- **the size terms carry no ranking signal.** `similarity_score` alone scores mAP
  0.653 against the mixture's 0.652; `min(1, kf/3)` alone scores 0.318;
- **they were saturated** — 65% of clusters have ≥ 3 keyframes, 53% have ≥ 5
  observations, and both terms cap there, so for two thirds of clusters they were a
  constant;
- **size was counted three times** — `kf/3`, `min(1, n_obs/5)` inside `confidence`,
  and `max` over N detections (which grows with N);
- **the old normalisation depended on a filter parameter.** The denominator was
  `best − min_similarity`, so moving `min_similarity` moved every score with no change
  in evidence. The ratio has no free parameter: `min_similarity` is now purely the
  absolute floor, the ratio is purely the relative gate, and the two can be swept
  independently.

Known cost, stated plainly: the per-prompt *ceiling* is slightly lower (mean best F1
0.693 vs 0.712). The ratio trades unreachable ceiling for reachable, transferable
performance — the mixture's optimum does not survive being applied to a prompt it was
not fitted on, and the ratio's nearly does (−2.1 points vs −6.5).

Geometry did not disappear, it became **filters** (`filter_clusters_by_geometry`, run
*before* ranking so the ratio's denominator is a cluster we would actually return):
`min_keyframes_per_cluster` (2), `min_observations_per_cluster` (1 = off),
`max_cluster_spread_m` (`None` = off). Both new knobs default to off because no
threshold paid for itself when swept — `kf ≥ 3` as a hard filter *cost* 4.7 points of
mAP. `confidence`, `observation_count` and `spread_m` are all on the response so a
caller can gate on them itself instead of receiving them diluted into one number.

Not established: that one threshold transfers across **maps**. The LOO above covers
prompts on one map only.

### Two-gate association (ConceptGraphs) — opt-in experiment

`semantic_gate_threshold` (`None` = off = production's rule) makes
`cluster_detections_leader_canopy` require **both** gates before a detection joins a
cluster: the 2 m spatial radius *and* a cutout↔cutout cosine against the seed. It is
ConceptGraphs' conjunctive association rule (geometry AND semantics) in place of
geometry alone. Needs `load_enriched_candidates(..., with_embeddings=True)`, which the
service turns on exactly when the threshold is set — the embeddings are a few MB per
query, so the default path never fetches them.

Swept on `bbhotel-choisy`, acceptance threshold refit for every row:

| association | mAP | macro F1 | LOO | mAP groupé | LOO groupé |
|---|---|---|---|---|---|
| geometry only | 0.653 | 0.632 | 0.611 | 0.713 | 0.627 |
| + gate 0.70 | 0.652 | 0.625 | 0.600 | 0.706 | 0.604 |
| **+ gate 0.80** | 0.694 | 0.649 | 0.628 | 0.692 | 0.610 |
| **+ gate 0.85** | 0.698 | 0.654 | 0.629 | 0.639 | 0.565 |
| + gate 0.90 | 0.630 | 0.579 | 0.556 | 0.559 | 0.457 |

The optimum is a flat band at **0.80–0.85**; 0.70 does not bite (any two cutouts of one
venue already sit near 0.7) and 0.90 over-splits.

Per class, read on **AP** — a per-class F1 table read at each column's own globally
refit threshold is confounded, since that threshold maximises the macro and penalises
any class whose optimum is elsewhere. At the 0.80 gate, versus no gate:

- extended / over-merged objects gain a lot — `lampe` 0.552 → 0.748, `plante`
  0.461 → 0.681, `table` 0.246 → 0.375 (0.799 at 0.90), `chaise` 0.138 → 0.221
  (0.690 at 0.90);
- compact objects are **near-untouched**, not degraded: `extincteur` 0.999 → 0.996,
  `cctv` 0.908 → 0.909, and `ascenseur` improves (0.929 → 0.970). The only real losses
  are `detecteur de fumée` (−0.054) and `TV` (−0.089).

**Why it stays off by default — two reasons, and the second is the stronger one.**

The two ground-truth views disagree structurally: the strict view (one target per
annotation) rewards splitting, the grouped view (single linkage at 2 m, 674 annotations
collapsed to 118 targets) rewards merging, so the gate wins on one and loses on the
other. That is the annotation-grouping defect (single linkage chains: 213 chairs → 5
targets), not a tuning question.

And **the gate is redundant with a smaller `clustering_eps_m`.** Every number above was
measured at the 2 m default. Sweeping the radius: strict mAP is 0.788 at `eps` = 0.5
against 0.653 at 2 m — a bigger gain than the gate's, from one parameter — and at
`eps` = 0.5 the gate *degrades* it (0.788 → 0.752 → 0.703). The gate was undoing an
over-merge the radius had created; two splitting mechanisms for one need, and the radius
is the cheaper (no per-query embeddings). The one place they still compound is `chaise`
(0.620 → 0.831).

Do not read that as "set `eps` to 0.5" either: strict and grouped mAP move in **opposite,
monotone** directions across the whole range, so the bench cannot rank two granularities
at all — it reports which ground truth you picked. See
`docs/plans/2026-08-11-clustering-radius-and-a-degenerate-metric.md`. Interventions that
change *ranking* at fixed granularity are measurable here; interventions that change
granularity are not.

### Other divergences from production

`localize.py` also differs in four import lines, one dev-only opt-in, and one bug fix.
`LocalizationParams.level_strategy` (`"seed"`, the default and production's behaviour,
or `"median"`) selects how `_cluster_level_from_detections` picks the floor a cluster
claims. `UNRESOLVED_LEVEL=-9999` avoids colliding with the real basement level;
production still uses `-1` and must be fixed before the next re-sync. Treat any other
behavioural change as a bug, and any change of the strategy default as one too.

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
populated from the pose source. Its primary key is `(geo_ref_id, id)`, and candidates
reference that composite key, because `id` is only a per-manifest array index.

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
we do not. The index restarts at zero in every manifest, so only the composite
`(geo_ref_id, id)` key keeps maps with overlapping indices isolated.

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
5. **The unresolved-level collision is fixed here, but not yet in production.** This
   port uses `UNRESOLVED_LEVEL=-9999`, so a real basement level `-1` participates in the
   merge veto, altitude clamp, and response like every other floor. The backend's
   `object_search/v1_5_logic.py` still uses `-1`; fix it before the next re-sync or that
   sync will reintroduce cross-floor merges and serialize basement levels as `null`.
6. **Local geokeyframe ids are per georef.** They are manifest array indices, not
   globally unique production primary keys. The composite `(geo_ref_id, id)` key and
   the extra georef predicate in candidate joins are therefore a deliberate dev-only
   divergence from the ported production query. `ensure_schema` refuses a legacy
   single-column-key database rather than migrating it, because rebuilding requires
   deleting embeddings and re-ingesting maps.
7. **`geo_ref_id` comes from the manifest, and only from there.** It is the partition
   key of the table and of the partial HNSW index, so a mismatch between what ingest
   wrote and what the service queries returns zero hits with no error. That is why
   neither `ingest_cli` nor `service` accepts it as an override any more.
8. **`create_partial_hnsw_index` needs autocommit** (`CREATE INDEX CONCURRENTLY`
   cannot run in a transaction) and polls `pg_try_advisory_lock` rather than blocking
   — a blocking `pg_advisory_lock` would hold a transaction open and deadlock against
   the CIC. That comment in the source is load-bearing; do not "simplify" it.
9. **`vendored/` is copied production code.** Fix bugs in the backend and re-sync;
   see `../toolbox/bricks/vendored/PROVENANCE.md`.
10. **The on-disk filename is the basename of the URL *path*.**
   `map_manifest._basename` parses `image_url`/`depth_url` with `urlparse`, and so
   do the other three v2 readers. Current manifests point at a public bucket with
   no query string, so splitting on `/` happens to work — but a presigned URL would
   yield `abc.jpg?X-Amz-Signature=…`, matching nothing on disk, and both symptoms
   (no keyframe id resolved, no depth found) are silent.
11. **Cutout extraction is the GPU-memory bottleneck, and one mirrored function is
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
