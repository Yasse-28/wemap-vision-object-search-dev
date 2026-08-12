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
| `ingest.py` | `db/ingest.py` | `encode_copy_stream`, `bulk_copy`, `create_partial_hnsw_index`, `drop_partial_hnsw_index` (**dev-only**), `_EWKB_POINTZ_PREFIX`, `_HNSW_INDEX_LOCK_KEY`, `INDEX_NAME_TEMPLATE` |
| `ingest_cli.py` | `object_search_ingest.py` | `run_ingest`, `_compute_object_positions`, `_ingest_capture`, `_upsert_geokeyframes`, `discover_capture_dirs`, `EMBEDDING_DIM=1024`, `DEFAULT_MIN_DISTANCE=1.5` |
| `candidates.py` | `candidates.py` | `EnrichedCandidate`, `load_enriched_candidates`, `_prefilter_hnsw_results`, `apply_feedback_boost`, `normalize_prototype_similarities`, `_parse_embedding`, `FEEDBACK_NORMALIZATIONS`, `K_INTERNAL=1000`, `LOOSE_ALPHA=0.3` |
| `feedback.py` | *(dev-only — no production counterpart)* | `ReviewFeedback`, `load_review_feedback`, `normalize_query`, `DB_FILENAME` — reads the toolbox's `object-search-annotations.db` |
| `localize.py` | `v1_5_logic.py` | `cluster_detections_leader_canopy`, `compute_cluster_statistics`, `rank_localization_clusters`, `_similarity_ratio_scores`, `filter_clusters_by_geometry`, `localize_from_enriched_candidates`, `build_localize_response`, `LocalizationParams`, `UNRESOLVED_LEVEL=-9999`, `PLACEHOLDER_BBOX` |
| `prepare_runner.py` | `object_search_prepare.py` (its `image_entries` construction) | `collect_image_entries`, `run` |
| `prepare_postprocess.py` | `object_search_prepare.py::_sample_depths` | `postprocess_metadata`, `sample_depths` |
| `map_manifest.py` | *(no counterpart — replaces the ORM)* | `load_map_manifest`, `find_manifest`, `MapManifest`, `ManifestKeyframe` |
| `v1_index_convert.py` | *(dev-only — no production counterpart)* | `convert`, `load_keyframe_map`, `ConversionStats`, `SCHEMA` — re-shapes a `legacy/` SQLite index into `metadata.parquet` + `embeddings.npy` |
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

### Two-gate association — legacy opt-in experiment

`semantic_gate_threshold` (`None` = off = production's rule) makes
`cluster_detections_leader_canopy` require **both** gates before a detection joins a
cluster: the 2 m spatial radius *and* a cutout↔cutout cosine against the seed. This was
previously attributed to ConceptGraphs, incorrectly: it is our conjunctive seed rule.
ConceptGraphs uses an accumulated descriptor, greedy-best assignment, and a sum of
semantic and geometric scores. The opt-in `incremental` association implements those
choices, while substituting a distance falloff between single depth-projected points
for the paper's point-cloud nearest-neighbour ratio. Both modes need
`load_enriched_candidates(..., with_embeddings=True)` when they use semantics; the
default geometry-only leader/canopy path still does not fetch embeddings.

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

### ConceptGraphs' own rule, measured — and negative

`association="incremental"` (`"leader_canopy"` = the default = production) implements
what the paper actually does, after we read it and found our attribution wrong: an
**accumulated descriptor** `f_obj ← (n·f_obj + f_new)/(n+1)` instead of the seed cutout,
**greedy-best** assignment instead of first-catch, and a **sum** `φ_sem + φ_geo ≥ δ_sim`
instead of a conjunction. `descriptor` and `combination` exist so the three can be
attributed separately. Our `φ_geo = max(0, 1 − d/eps)` on the nearest member is a
substitution: they use a nearest-neighbour ratio between point clouds, and we have one
point per detection.

Measured on `bbhotel-choisy`, and the answer is no — **read the last column, not the
first**:

| association | median spread | mAP strict | vs the eps curve at that spread | mAP groupé |
|---|---|---|---|---|
| leader/canopy, eps 2 m | 0.374 | 0.672 | — | 0.713 |
| + seed gate 0.80 | 0.267 | 0.703 | **−0.001** | 0.694 |
| greedy-best, no semantics | 0.382 | 0.649 | −0.021 | 0.726 |
| + gate 0.80, seed descriptor | 0.248 | 0.686 | −0.022 | 0.711 |
| + gate 0.80, running mean | 0.271 | 0.652 | −0.051 | 0.715 |
| sum, δ_sim 1.2, running mean | 0.263 | 0.690 | −0.015 | 0.715 |
| sum, δ_sim 1.4, running mean | 0.184 | 0.699 | −0.033 | 0.698 |

Every raw gain in column three is bought by cutting clusters smaller, and `eps` alone
buys the same cut more cheaply: sweeping it from 3 m to 0.3 m traces a strict-mAP curve
from 0.630 to 0.792 that **no association variant beats at its own granularity**. The
accumulated descriptor is the worst of the three ideas here (−0.051 against −0.022 for
the same rule on the seed), and best-match assignment on its own costs 0.021.

The one thing that is not noise: all the semantic variants gain +0.005 to +0.023 on the
**grouped** view at matched granularity, where the geometric radius gains nothing. Weak
evidence, on the ground truth with the single-linkage defect, and not a reason to ship.

Method, and it is the reusable part: compare on the granularity curve, not on the
metric. `median_spread_m` is the confounder for every intervention that splits or
merges, and `toolbox/benchmark/association_sweep.py` reports it on every row for that
reason.

### C-DOG ray association — opt-in experiment

`association="cdog"` builds a sparse graph between detections from different
keyframes. Candidate pairs are generated from depth-projected points within
`cdog_pair_radius_m` (5 m by default), but an edge is decided from the normalized EUS
rays: metric ray-to-ray distance at a positive closest approach within
`cdog_range_m`, the usual level veto, and an optional cutout-cosine gate. This is a
transfer, not a literal reproduction: C-DOG uses 2D epipolar distance in pixels; this
implementation substitutes 3D metric ray distance. C-DOG also assumes one detection
per object per view, while our class-agnostic proposals can contain several boxes for
one object in one keyframe.

Edges are filtered by normalized open-neighbourhood overlap (`cdog_delta`) before
connected components are taken. This is the anti-chaining step; components are not
taken on the raw graph. Because the pair-radius shortcut reads the projected points,
association is not wholly depth-independent even though its edge criterion is.

`centroid_from="rays"` is deliberately independent of `association`: it reports the
least-squares intersection of each cluster's member rays, falling back to the existing
depth centroid for ill-conditioned near-parallel systems and logging the fallback
count. `spread_m`, levels, keyframe support, and all other statistics keep their depth
definitions. Judge association sweeps against the strict-mAP/`median_spread_m`
reference curve, not by raw mAP alone; ray centroiding can be evaluated separately
because it moves positions without changing cluster membership.

### C-DOG's ray consistency — negative, and we know why

`association="cdog"` builds edges from **ray-to-ray distance** instead of point
proximity (a detection's ray is `origin = keyframe position`, `direction =
normalize(depth point − origin)`, so normalising divides the depth back out), then
filters them by δ-neighbourhood overlap instead of taking connected components. It is
the one association here that does not consume the noisy depth *magnitude*.

Measured against the eps curve at equal spread: **−0.061** at δ = 0 (edges only) and
worse with the overlap filter on (−0.13 to −0.175). δ > 0 hurts because our
neighbourhoods are open sets, so a two-detection object has overlap 0 and is always cut.

The diagnostic that explains it, and it outranks the sweep. Taking as proxy ground truth
"two detections within 1 m of the *same* annotation" versus "of two different annotations
of the same class", over 16 740 same-object and 719 375 different-object pairs on six
prompts:

| pairwise cue | AUC | unusable same-object pairs |
|---|---|---|
| distance between depth-projected points | **0.879** | 0 % |
| ray-to-ray distance at closest approach | 0.768 | **21.9 %** (closest approach behind a camera) |
| cutout↔cutout cosine | **0.529** | 0 % |

Two conclusions, both load-bearing for anything that comes next:

- **the ray criterion is worse than the depth point it was meant to replace.** Our
  keyframes are 1.5 m apart, so pairs observing one object often have a tiny baseline;
  the common perpendicular is then ill-conditioned and its parameters go negative. The
  depth magnitude is noisy, but short-baseline ray geometry is noisier;
- **the cutout cosine is a coin flip for association.** 0.529 within one prompt's
  candidate set — which is the only setting association ever runs in. Two different
  chairs look exactly as alike as two views of one chair. That single number explains
  the semantic gate, ConceptGraphs' accumulated descriptor, and the sum rule all coming
  back neutral-to-negative: they were adding a feature that carries no information about
  the question being asked.

What survives: `centroid_from="rays"` (default `"depth"`, unchanged). Least-squares
triangulation of a cluster's member rays is well-conditioned even when individual pairs
are not, and it is worth **+0.010 of strict mAP at eps 2 m** — small, but the only
intervention measured today that moves the metric *without* moving the granularity,
because it changes where a cluster is, not which detections are in it. It fades as `eps`
tightens (+0.001 at 1 m, −0.002 at 0.5 m), which is consistent: it helps most where
clusters are largest.

### Minimum-cost multicut association — opt-in experiment

`association="multicut"` builds a signed graph and partitions it with greedy additive
edge contraction (GAEC). Positive costs attract and negative costs repel, so the
partition chooses its own number of clusters instead of applying a local edge cutoff.
Pairs are generated only when their depth-projected points are within
`multicut_pair_radius_m` (6 m by default). That radius is a **graph sparsification**,
not the association rule: every retained pair contributes its signed evidence. Unlike
C-DOG, same-keyframe pairs are retained so duplicate proposals around one object can
merge.

The cost is the linear log-odds model
`geo_weight * (1 - distance / geo_pivot) + sem_weight * (cosine - sem_pivot)`.
Defaults are `(1.0, 1 m, 0.0, 0.8)`: depth geometry carries the measured signal and
semantics remains explicitly sweepable while defaulting to exact zero. Setting
`multicut_geo_source="ray"` substitutes closest-approach ray distance; behind-camera
or out-of-range approaches create no edge. Level incompatibility is a hard cannot-link
constraint outside the weighted graph, never a large negative cost that summed positive
parallel edges could overwhelm during contraction.

GAEC repeatedly contracts the maximum positive edge and sums parallel costs after each
merge. It uses a lazy heap with node-index tie-breaking and has no Kernighan–Lin pass.
Run `python -m toolbox.benchmark.pair_cue_separability --map-path ...
--ann-base-url ... --prompts ...` to reproduce the depth/ray/cosine percentile and AUC
diagnostic before adding another pairwise term.

### Minimum-cost multicut — neutral, and that is the informative part

`association="multicut"` drops the threshold-then-greedy shape entirely: a sparse signed
graph (`w = geo_weight·(1 − d/geo_pivot) + sem_weight·(cos − sem_pivot)`, a linear form
that *is* the log-odds of a logistic model), partitioned by greedy additive edge
contraction. The number of clusters is an output, not a parameter, and unlike every
other mode here it lets two proposals from the **same keyframe** merge. Level
incompatibility is a hard cut on the edge, not a large negative cost, because summed
parallel edges could otherwise outvote one.

Swept against the `eps` curve at equal median spread, over the whole useful range of
`geo_pivot` (0.15 m to 3 m): **−0.017 to +0.013, and ±0.005 almost everywhere.** Sorted
by granularity, the multicut points and the `eps` points interleave. Global
correlation-clustering buys nothing over greedy leader/canopy on this data — which is
worth knowing, because it says the greedy pass was never the limitation.

Two negatives that were *predicted* by the pair-cue AUC table above, which is the best
argument for running that diagnostic before implementing anything:

- **the semantic weight is monotonically harmful**: −0.006, −0.014, −0.017 as
  `sem_weight` goes 0.5 → 1 → 2. Exactly what an AUC of 0.529 implies;
- **ray geometry is much worse than depth geometry** here too: −0.069 and −0.145 at
  `geo_pivot` 1 m and 2 m.

### The frontier, and what it means

Four association rules, one curve. Strict mAP against median cluster spread:

| median spread | leader/canopy `eps` | multicut `geo_pivot` |
|---|---|---|
| 0.046–0.049 | 0.719 | 0.725 |
| 0.055–0.056 | 0.735 | 0.748 |
| 0.069–0.073 | 0.787 | 0.788 |
| 0.089–0.101 | 0.792 | 0.779 |
| 0.277 | 0.702 | 0.702 |
| 0.374 | 0.672 | 0.672 |

The best strict point on the whole surface is `eps` ≈ 0.55 (mAP 0.802, macro F1 0.691,
LOO 0.653 — against 0.672 / 0.632 / 0.611 at the 2 m default), and grouped mAP is 0.680
there against 0.738 at `eps` 3 m. So the two views still disagree monotonically and the
bench still cannot say which granularity is right. **No association rule tested today
moves that frontier; they only move along it.** The thing to fix is the ground truth, not
the algorithm — see the annotation-grouping defect.

### Scoring the partition instead of the cluster list — the metric that ranks

Every number above had to be read against a granularity curve, because strict and
grouped mAP move in opposite monotone directions. `association_sweep.py` now also scores
the **partition of the detections** directly, which is free of that: label each detection
with the annotation nearest its depth-projected point within `--near-m` (1.0 m), call two
labelled detections a positive pair when they share an annotation, and measure the
partition the association induces.

    pair_precision — over-merging costs it       pair_recall — over-splitting costs it

Neither is free, so unlike mAP this has an **interior optimum**:

| configuration | median spread | pair P | pair R | **pair F1** | mAP strict | mAP groupé |
|---|---|---|---|---|---|---|
| leader/canopy `eps` 0.55 *(best strict mAP)* | 0.097 | 0.765 | 0.265 | 0.375 | **0.802** | 0.680 |
| leader/canopy `eps` 3.0 *(best grouped mAP)* | 0.582 | 0.458 | 0.538 | 0.437 | 0.630 | **0.738** |
| leader/canopy `eps` 2.0 *(today's default)* | 0.374 | 0.531 | 0.512 | 0.480 | 0.672 | 0.713 |
| leader/canopy `eps` 1.25 | 0.231 | 0.629 | 0.484 | 0.512 | 0.712 | 0.692 |
| incremental, sum δ 1.35 | 0.205 | 0.609 | 0.507 | 0.512 | 0.702 | 0.702 |
| **multicut `geo_pivot` 1.5** | 0.270 | 0.596 | 0.533 | **0.523** | 0.702 | 0.710 |

Three things follow.

**The strict-mAP winner is shattering objects.** `eps` 0.55 scores 0.802 there and pair
recall 0.265: it splits one annotation across many small clusters, and the strict view
pays it for the fragment that lands closest. It is the worst configuration in the table
on the metric that asks whether detections of one object ended up together.

**The right granularity is `eps` ≈ 1.25–1.5, not 2.0 and not 0.55.** Stable under the
proxy radius: the optimum sits in the same band at `--near-m` 0.5, 1.0 and 1.5, and both
extremes lose in all three. It also lands where strict and grouped mAP **cross**, which
is an independent check that the pairwise metric is measuring the thing the two views
disagree about.

**At fixed granularity the semantic rules do gain a little** — multicut `pivot` 1.5
+0.013, incremental sum δ 1.35–1.4 +0.015 to +0.021 against the `eps` curve interpolated
at their own spread. Small, but it is the only consistently positive column produced
today, and it qualifies the AUC result above: a cue with AUC 0.529 is nearly worthless on
its own, yet still breaks ties usefully once geometry has narrowed the candidates to near
neighbours. The same rules measured on strict mAP looked negative, because there they
were only moving granularity.

The control ablation says the gain is semantic and not the linkage: incremental with
best-match assignment and nearest-member distance but **no** semantic term scores −0.003
at `eps` 1.5, i.e. nothing.

**Recommended, not applied:** `clustering_eps_m` 2.0 → 1.5 improves pair F1 (0.480 →
0.510), strict mAP (0.672 → 0.702) and macro F1 (0.632 → 0.654, LOO 0.611 → 0.640), and
costs 0.011 of grouped mAP. It is a production default, so it belongs in
`wemap-vision-backend` first, and the grouped ground truth should be repaired before
anyone leans on the last digit.

Caveat on the proxy: only a few hundred detections per prompt fall within `--near-m` of
an annotation, so this metric sees the well-covered objects, not the missed ones. It
ranks associations; it does not measure recall of the map.

### Other divergences from production

`localize.py` also differs in four import lines, one dev-only opt-in, and one bug fix.
`LocalizationParams.level_strategy` (`"seed"`, the default and production's behaviour,
or `"median"`) selects how `_cluster_level_from_detections` picks the floor a cluster
claims. `UNRESOLVED_LEVEL=-9999` avoids colliding with the real basement level;
production still uses `-1` and must be fixed before the next re-sync. Treat any other
behavioural change as a bug, and any change of the strategy default as one too.

### Converting a v1 SQLite index (dev-only)

`v1_index_convert.py` re-shapes a `legacy/` index (`object-search.db`) into the v2
prepare outputs, so a map whose v1 index covers more keyframes than the v2 one can be
compared without re-running detection. It is a re-shaping, not a re-computation: both
lineages embed proposal cutouts with the same MetaCLIP2 checkpoint (v1 float32, v2
float16) and v1's `bbox_spherical_coordinates` is `[theta, phi, fov_x, fov_y]` in the
*same* convention as `theta_center`/`phi_center`/`angular_*`.

Three things are load-bearing and all three are silent when wrong. **`phi` is
negated**: v1 builds its ray in OpenCV (`y` down) and stores `phi = asin(y)`, while
v2's `phi` is positive up — keep the sign and every object mirrors about the horizon
(median |Δ| against the depth TIFF: 0.09 m flipped, 1.31 m not). v1 `keyframe_id` →
manifest index is resolved **by image filename**, since v1 ids are `georef.db` rows
and do not equal manifest indices. And parquet row `i` must stay embedding row `i`.
Pinned by `toolbox/tests/test_v1_index_convert.py`.

**Re-ingest drops the partial HNSW index first** (`drop_partial_hnsw_index`, called by
`run_ingest`). `create_partial_hnsw_index` alone does not rebuild one: its
`IF NOT EXISTS` skips a *valid* index, so every COPY'd row became an incremental
insert into the old graph — measured at ~1 000 rows/min against 1 046 404 rows copied
in 109 s index-free. Dev-only addition; production has no counterpart.

**Ingest re-thins, and that throws most of a converted index away.**
`filter_by_distance` selects keyframes from *all* the manifest's poses, not from the
ones the index covers, so a v1 index already thinned at 1.5 m keeps only 20 % of its
rows (2 538 of 11 340 keyframes) when ingested at the 1.5 m default — the prod-dump
index loses the same way (14 %). Ingest a converted index with a small
`--min-distance` (0.05 m → 98 % of rows) so the thinning it already had is respected
instead of re-applied.

What does not survive: `detection_score` (v1 never stored it → NULL, debug-only) and
thumbnails (v1 rendered previews on the fly, so `thumbnail_key` is empty and search
results carry no cutout image; the explorer's own ERP re-render still works). `depth`
is carried over as v1 sampled it, rather than re-sampled by `prepare_postprocess`.

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
