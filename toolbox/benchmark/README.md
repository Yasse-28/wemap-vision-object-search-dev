# Object Search HTTP Benchmark

Compares localized object-search clusters returned by the online service against
annotator ground truth in GeoJSON:

`{map_path}/benchmark/annotations.geojson`

Each annotation's query prompt is taken from `feature.properties.prompt` when
present, otherwise from `feature.properties.class`. Annotations are grouped by
their resolved prompt, and one query is issued per prompt.

## Run

Start the local standalone object-search service, then run from the repo root:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id>
```

By default it posts to:

`http://localhost:45678/<map-id>/object-search/localize`

GeoPose service example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --api-style geopose \
  --base-url http://localhost:45677
```

Standalone online localization example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --online \
  --candidate-count 1000 \
  --clustering-eps-m 1.5
```

Remote localization example:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --map-id <map-id> \
  --localize-url 'https://vps-api.wemap-vision-computing-1.getwemap.com/{map_id}/object-search/localize'
```

The toolbox benchmark runner spawns this script automatically; the invocations
above are for running it standalone.

## Scoring

The script accepts clusters whose selected score is strictly greater than `0.9`
by default:

```bash
python -m toolbox.benchmark.object_search_http_benchmark \
  --map-path /path/to/vps-data/maps/<map-id> \
  --acceptance-threshold 0.9
```

The default score field is `match_score`. If that field is missing, the script
falls back to `similarity_score`, then `confidence`.

Each accepted cluster is a true positive when it can be greedily matched to one
unmatched annotation of the same class within the annotation `accuracy` radius in
meters. Unmatched accepted clusters are false positives. Unmatched annotations
are false negatives.

Passing `--group-annotation-radius-m > 0` adds a second "grouped" result block
where nearby annotations of the same class within that radius count as a single
ground-truth target.

## Outputs

By default (or under `--output-dir`):

- `results.json` / `metrics.json`: config, summary, per-class rows, per-prompt rows, matches
- `results.csv`: per-prompt metrics (plus `grouped_results.csv` when grouping is enabled)
- `raw_results.json`: raw predictions/annotations per prompt (only with `--output-dir`)
- `prompt_geojson/*.geojson`: one GeoJSON per evaluated prompt

Per-prompt GeoJSON colors:

- TP prediction clusters: green (`#22c55e`)
- FP prediction clusters: red (`#ef4444`)
- FN reference annotations: grey (`#9ca3af`)
- Matched reference annotations: blue (`#2563eb`)

## Offline association sweeps

`association_sweep.py` scores many localization configurations without re-running
retrieval. Retrieval is identical across association/ranking configs, so it fetches the
ANN hits and enriches them **once per prompt**, caches that, and then calls
`localize_from_enriched_candidates` in process — about 6 s per configuration on
`bbhotel-choisy` instead of a full HTTP run.

It reuses this module's own matching, curve and threshold code, so it cannot disagree
with the HTTP benchmark about what a true positive is. `--verify` proves that on the
default configuration by comparing every cluster against the live bricks service
(measured deviation: exactly 0).

```bash
python -m toolbox.benchmark.association_sweep \
  --map-path /path/to/maps/bbhotel-choisy \
  --ann-base-url http://127.0.0.1:45677 \
  --cache-dir .cache/assoc --grid grid.json --out-dir out/ --verify
```

**HOTA is the row to read when a configuration changes the granularity.** Every row also
carries `det_a`, `ass_a` and `hota`, the higher-order tracking accuracy transposed to
this problem: `det_a` is `TP/(TP+FP+FN)` over detections against annotations, `ass_a` is
the mean Jaccard overlap between the cluster a detection landed in and the set of
detections belonging to its annotation, and `hota` is the geometric mean of the two,
averaged over five localisation radii. Splitting an object and merging two cost `ass_a`
symmetrically and leave `det_a` untouched, which is precisely what strict and grouped mAP
cannot do — on vinci, three `clustering_eps_m` values give an identical `det_a` of 0.306
while `ass_a` ranks them 0.492 / 0.474 / 0.313. Prefer it over `pair_f1`, which measures
the same partition but weights objects by the square of their observation count.

**Calibration, and what a shared threshold costs.** `ece`, `mce` and `overconfidence`
compare each cluster's score against the rate at which clusters of that score are right,
over quantile bins (fixed-width bins would put nearly every prediction in one).
`threshold_spread_strict` is the p90−p10 of the thresholds each prompt would pick for
itself — the dispersion a single fitted threshold has to paper over. Read the numbers
against `accuracy_ceiling`: the match is one-to-one, so returning four clusters per
object makes three of them wrong whatever the score says, and an error beyond that gap
is granularity, which is `ass_a`'s business rather than calibration's. The reliability
table (`reliability`, JSON only) is where the useful shape is — on vinci the score is
monotone up to about 0.95 and then **inverts**, which is why 0.9 has to be re-fitted per
map.

**Where an object was lost, in three numbers.** `r_obj` is the share of annotations some
*retrieved candidate* came within `--near-m` of, measured before association and before
any filter — an annotation missing there is out of reach of everything downstream.
`recall_at_all` is the share a returned cluster matched, and `recall_at_1` / `recall_at`
(JSON) the share a caller sees at each cutoff. The gaps are the diagnosis, and today's
mAP sums all three: on vinci, `r_obj` 0.759 → returned 0.739 → **R@1 0.044**, R@20 0.378.
Retrieval loses 24 points, association loses 2, and the ranking loses 36. `r_obj` is
identical across association configurations by construction, which is also a useful
sanity check on a grid.

**Split and merge, counted outright.** `fragmentation_rate` is the share of well-covered
annotations whose detections span more than one cluster, `merge_rate` the share of
clusters holding more than one annotation's. Unlike `mean_clusters_per_annotation` these
are bounded rates, so the two failure directions are directly comparable: at
`clustering_eps_m` 0.5 vinci fragments 84 % of annotations and merges 3 % of clusters; at
3.0 it fragments 15 % and merges 30 %. `homogeneity` and `completeness` are the entropy
pair over the same partition — the first cannot see splitting and the second cannot see
merging, which is why neither is reported alone. **Do not fit a radius on `v_measure`**:
it is biased towards more clusters and reads 0.81–0.91 across configurations that HOTA
separates cleanly.

**`panoptic_quality` = `segmentation_quality` x `recognition_quality`.** A cluster matches
an annotation when their detection sets overlap by more than half, which makes the match
unique with no greedy pass. `recognition_quality` is the half worth having on its own —
`TP/(TP + FP/2 + FN/2)`, how many objects came out as one whole cluster each, with no
sensitivity to how precisely they are placed. Clusters the association *dropped* (a
negative cluster label) are not predictions and are excluded from the false positives:
counting them would make a filter that removes an outlier score worse than one that keeps
it. The rest is largely redundant with `hota` by construction, and is kept for that one
half.

Each grid entry is a `LocalizationParams` override plus a `label`; an unknown key is an
error, not a silent no-op. Every row carries both ground-truth views (strict and
grouped, each with its own fitted shared threshold and leave-one-prompt-out estimate)
plus granularity controls — cluster counts, median observation count, median spread.
Those controls are not decoration: the bench cannot rank two granularities, so a change
in split/merge behaviour has to be read off them rather than off the metric.

Ready-made grids live in `grids/`; `depth-cap-boost-{vinci,bbhotel}.json` are the
12/08 depth-cap and review-boost runs, and `rescorer-comparison-{vinci,bbhotel}.json`
compare the five review-rescoring methods. Each is centred on its own map's granularity
optimum (they differ, see `AI_CONTEXT/bricks.md`).

A grid entry may also name a `rescorer` and its `rescorer_params`; the driver then loads
each prompt's reviewed embeddings and runs it over the retrieved set. Always keep the
`identity` row: it is the no-review baseline seen through the rescoring path, so a
difference between the two would be an artefact rather than a review signal.

## The VLM validation gate

`--with-vlm` scores every cached candidate with Qwen3-VL once per prompt
(`toolbox/bricks/vlm_gate.py`), caches the table under `<cache-dir>/vlm/`, and reuses it
for every grid row that names a `vlm_gate`. Scoring runs at roughly 12 cutouts/s on the
dev card, so a 12-prompt map costs ~15 minutes **once**; the sweep itself then runs at
its usual speed.

Two levels, both applying the same scores:

- `"detection"` folds `p(yes)` into each candidate's ranking similarity before
  association, weighted by `vlm_alpha` and rescaled by `feedback_normalization`;
- `"cluster"` aggregates (`vlm_aggregate`: `mean`, `max`, `min`) the scores of a
  returned cluster's own observations and re-ranks — the multi-view verification shape
  the 3D-grounding literature uses.

Neither drops anything: the gate is a score, so cluster membership and coordinates are
identical with and without it, and every row stays comparable at fixed granularity.

`python -m toolbox.benchmark.vlm_cue_separability --map-path …` is the diagnostic to run
first: it scores the cutouts a human already reviewed and reports the AUC of `p(yes)`
between their `true_positive` and `false_positive` classes. Run it before wiring a gate
into anything — a cue that cannot separate labelled cutouts will not rank unlabelled
ones.

**Converted v1 indexes cannot be gated as they stand.** Their `thumbnail_key` is a
virtual path that the toolbox re-renders from the ERP on demand, so there is no file to
show the model; `vlm_scores.load_or_score` logs the unreadable count and leaves those
candidates unscored rather than rejected.

`--with-feedback` additionally resolves the map's review prototypes, which is what makes
`feedback_alpha`, `feedback_beta` and `feedback_normalization` do anything offline —
without it those keys are accepted and silently inert. It fetches the raw prototype
columns once (both gains at zero) into its own cache entries, and the gains are then
applied per configuration. Each prompt's prototype coverage is logged before the sweep
starts: a prompt with none reproduces the baseline bit for bit, which is the failure
mode to rule out before concluding the boost did not help.

## Analysing one map, before proposing anything

`map_analysis.py` describes a prepared map rather than a query's results: it reads the
**whole parquet**, re-places every row in EUS from `depth` and the two ERP angles the
way ingestion does, and needs no database, no ANN service and no candidate cache. One
map in, a text report plus a JSON payload out, with `--layers-dir` adding GeoJSON
layers a livemap can render directly (`map_layers.py`).

```bash
PYTHONPATH=.:third_party/object_search python -m toolbox.benchmark.map_analysis \
  --map-path /path/to/map --json-out analysis.json --layers-dir layers/
```

| section | what it answers |
|---|---|
| `s0` | what the map holds, and **what it is missing** — a prepare run that wrote no label or no score makes several sections below silently empty, and a table of zeros looks the same as a measured zero |
| `s1` | per-detection distributions split by detector: range, angular size and aspect, `phi`/`theta`, score, implied physical size, proposals per panorama |
| `s2` | the free labels against the ground truth — `P(attached \| label, source)` at several radii **next to the base rate**, `P(label \| class)` and its inverse, normalised mutual information, score calibration, and the embedding neighbourhood purity |
| `s3` | annotations against detections, **depth-free measurement first**, then observability and what `min_keyframes_per_cluster` costs — a retention curve, not a single number: it prices the threshold against the ground truth *before* any ranking runs, and says whether the annotations it keeps are the ones the detector actually found |
| `s4` | pairwise cues over three deliberately drawn populations, raw and conditional AUC, with the share of pairs each cue applies to |
| `s5` | intra-view duplicates split by detector and settled by embedding, and the co-visible lower bound on the number of objects |
| `s6` | hubness of the embedding space, with **no ground truth at all** — how unevenly the index shares out the role of nearest neighbour, and what a plain recentring would change |
| `s7` | detector labels against the annotation's *set* of acceptable labels (ADR 0009); reports that no annotation carries sets rather than a table of zeros |

### The layers (`map_layers.py`)

A distribution says how much; these say **where**, which for depth is the whole point —
it degrades in particular parts of a building, and a percentile hides that.

| layer | geometry | what it shows |
|---|---|---|
| `depth-range` | 2 m cells | mean distance from keyframe to detection, against the 15 m trusted range. `beyond_trusted_share` is the property to check: a moderate mean held down by near rows can hide a far minority |
| `depth-blowups` | points | every row placed past 30 m, worst first and capped. Deliberately not aggregated — they line up along windows, mirrors and glass, and that alignment is the diagnostic |
| `depth-scatter` | 2 m cells | how far apart the observations of one annotated object land. **The fragmentation field**: read it against the clustering radius, since a spread at or past it cannot survive one cluster. Needs annotations |
| `detection-coverage` | 2 m cells | the depth-free measurement, spatialised — a red cell is the detector missing that part of the building, owing nothing to depth. Needs annotations |
| `parallax` | 2 m cells | the widest baseline any two keyframes within trusted range ever offered that cell, plus the trajectory anisotropy. A capture ceiling: red here cannot be fixed by any algorithm |
| `ground-truth`, `capture-distance`, `embedding-agreement` | points | one per annotation — reached or not and why, how close the capture came, how alike the cutouts look |
| `keyframes`, `detection-grid` | points | the capture itself, and detection density per cell |

**Only `detection-grid` is a density.** A viewer's own heat-map style weights how many
features are in a place, so pointing it at a *mean* draws a count instead: every cell
layer above therefore carries its value and an already-resolved `marker-color`, and is
meant to be drawn as a flat filled square. Cells carry an `altitude_m` but **no indoor
level**, so on a multi-storey map read one floor at a time.

**Two readings this tool exists to protect.** A label or a cue is only informative
above the **base rate** — on a densely annotated map half of all detections sit within
2 m of some annotation, so an uninformative label scores 50 %. And an annotation whose
source panorama carries no detection at all is not a detector failure: `s3` separates
that case, the "detected in 2D but placed elsewhere" case and the genuine miss, because
they have unrelated remedies.

**The depth-free measurement.** An annotation records the panorama and the pixel it was
clicked in, so asking whether a detection box covers that pixel *in that same panorama*
is a comparison of two angles in one frame. It owes nothing to the depth map — unlike
the 3D attachment, which shares the annotation's own construction
(`ray(u, v) * depth_map(u, v)`) and therefore measures agreement, never accuracy.

**Observability, in two halves.** `s3` reports what the capture *achieved* next to what
it made *available*: parallax, occupied azimuth sectors out of twelve, the widest empty
sector, and the share of observation pairs wide enough to triangulate on
(`>= 5 deg`). A poor achieved figure on a rich available one is an algorithm problem; a
poor available one is a capture problem no association will ever fix, and the line
counting annotations seen under a degenerate baseline is that ceiling stated outright.
Only *positional* coverage is measured — every keyframe is a panorama and already looks
in every direction, so rotational coverage carries no information here.

**Hubness (`s6`) is the one section that needs nothing but the parquet.** In a
high-dimensional space the nearest-neighbour relation stops being symmetric: a few
cutouts sit near the centre of mass and surface for prompts they have nothing to do
with, while others are never anyone's neighbour and no threshold can reach them. The
report prints the raw figures next to the same figures after subtracting the mean
embedding, because that subtraction is a one-line change in the scoring path and the
gap between the two columns is what it would buy. On vinci (20 000 rows, k=10) the
skewness of the k-occurrence count falls 3.96 → 1.95 and the antihub share 8.4 % → 2.7 %.
Hub labels are printed against their base rate, per the reading rule above.

**Known limit.** `detection_review` verdicts cannot be joined here: their `target_id` is
a pgvector candidate id, not a parquet `row_index`. `s0` says so rather than printing an
empty table.

## Checking the annotations before trusting a measurement

`validate_annotations.py` judges the ground truth, not the pipeline. It reads the map's
`object-search-annotations.db` **directly**, not `benchmark/annotations.geojson`: that
export is only rewritten when a benchmark run starts, so it describes the map as the last
run saw it. On vinci-st-domingue-zone-1 it sat a day behind — 9 annotations of 6 classes
where the store held 12, every ADR 0009 field reported absent because the export predated
them, and a separability alarm about clicks that had already been replaced. The command
prints the path it read on the first line.

```bash
python -m toolbox.benchmark.validate_annotations --map-path /path/to/map
```

Pass `--geojson <path>` to read an export instead — the one case that wants it is asking
what a past benchmark actually scored, rather than what is annotated now.

Three blocks, in the order they should be acted on: which contract fields are present
(a field at 40 % is reported as **worse than absent** — a metric reading it would score a
biased slice), what contradicts itself (one click recorded twice, one `object_id` under
two classes or two extents, one class under two synonym sets, an extent that is a typo),
and what the map has earned — whether the separability gate opens the classification
columns, and whether the label-set metrics have anything to measure. Exit status is
non-zero only for contradictions: an unannotated map is incomplete, not wrong.

The annotator's side is
[docs/plans/2026-08-18-cahier-des-charges-annotation.md](../../docs/plans/2026-08-18-cahier-des-charges-annotation.md);
the contract itself is [ADR 0009](../../docs/adr/0009-ground-truth-annotation-contract.md).

## Typing the errors, and pricing each one

`error_decomposition.py` types every returned cluster — *correct*, *duplicate*,
*classification*, *localisation*, *classification+localisation*, *background* — plus the
objects no cluster reached, *missed*, and then measures what recall would be if that one
type of mistake had not happened. Run it from the sweep:

```bash
python -m toolbox.benchmark.association_sweep --map-path ... --decompose
```

**The base is recall, not mAP, and that is deliberate.** Strict mAP is paid for
fragmentation here, so "suppress the duplicates" would come out near zero — the tool
would mislead exactly where it is most useful. Two axes are reported instead: `dR@10`,
which is rank-aware (needed for *duplicate* to be definable at all) and is where the
pipeline loses, and `dR tous`, the same fix with every cluster allowed. **The pair is the
diagnosis**: a type that costs `dR@10` and nothing overall is purely a ranking problem, a
type that costs both loses the object. On vinci's baseline:

| type | count | dR@10 | dR all |
|---|---|---|---|
| missed | 108 | +0.252 | +0.252 |
| background | 161 | +0.095 | +0.000 |
| duplicate | 96 | +0.037 | +0.000 |
| localisation | 44 | +0.029 | +0.063 |

So 13 points of the ranking loss are junk and duplicates occupying top-10 slots — objects
that *are* returned — and 25 points are objects never found at all.

**Two columns are withheld by measurement, not by map name.** Telling a cluster on the
wrong class from one on nothing needs classes further apart than the radius that matches
them. `separability` measures the share of annotations with another class inside their own
radius and withholds *classification* and *classification+localisation* above a third,
printing the number it refused on. With today's ground truth — no `extent_m`, so a flat
5 m radius — that is 60.5 % on vinci and both columns are withheld on both maps. They
become available as `extent_m` lands, which is why that field is the one to annotate
first: see [ADR 0009](../../docs/adr/0009-ground-truth-annotation-contract.md).

Matching here is greedy **by rank**, not by distance as `match_predictions` does, because
a duplicate is defined by a better-scoring cluster having already taken the object.
Recall figures from this module are therefore close to but not identical with the sweep's
`recall_at`.

## Scoring a label against a set, not a string

`label_set_metrics.py` implements OpenLex3D's two open-set metrics — **top-N frequency**
(does any of the N proposed labels fall in category C) and **set ranking** (does the
proposed order match the ideal order of the categories, as nDCG against the object's own
best possible ordering). Both read the four ranked label sets ADR 0009 puts on an
annotation: synonyms, depictions, visually similar, clutter.

The module never produces labels — it takes a ranked list per object, because that list
can come from the detector's own `label` column, from `gdino_labels.encode_classes`
scored against a cluster embedding, or from a VLM, and choosing between those is a
separate question. `rank_labels` covers the embedding case and is pure numpy.

`map_analysis.py` section `s7` wires it to the map's own detector labels, ordered by
detector score — deliberately not a MetaCLIP ranking, because that tool loads no model
and keeping that property is worth more than a sharper ranking. Until the annotations
carry label sets, `s7` says `0/258 annotations portent des ensembles` rather than printing
a table of zeros.

## Depth quality where it places a detection

`depth_boundary_quality.py` measures the depth maps **under the detector's boxes**,
which is the only place a detection's position comes from. It needs no ground truth and
no pose.

```bash
PYTHONPATH=.:third_party/object_search python -m toolbox.benchmark.depth_boundary_quality \
  --map-path /path/to/map --sample 150 --workers 6
```

The standard monocular-depth figures (AbsRel, `delta1`) cannot answer this: they are
dominated by large flat surfaces, which is exactly the part of the panorama no detection
is placed from. Three quantities come out instead, per detection and pooled by range
band — whether the sampled pixel is within two pixels of a depth discontinuity, whether
it *is* a flying pixel (a value stranded between two surfaces, corresponding to empty
space), and the p90/p10 depth ratio inside the box.

Read every one of them against the `scene` block, which is the same rate over the whole
map. On vinci (40 keyframes, 3 871 boxes): borders cover **1.0 %** of the map's pixels
but **5.3 %** of sampled pixels sit on one — detections land on discontinuities five
times more often than chance — rising to 16.6 % beyond 15 m. Flying pixels are rare
everywhere (0.06 % of the map, 0.10 % at the sample), so they are *not* the mechanism
here; the in-box depth ratio of 1.68 (2.45 beyond 15 m) is.

The scene pass runs at full resolution in row bands on purpose: subsampling the map
first would compare pixels several apart and report a larger, different quantity than
the per-detection figures it is the baseline for.

## Putting a name back on a G-DINO box

A prepare run can leave every GroundingDINO row carrying one placeholder label —
`gdino_venue` on 813 467 of vinci's 1 063 142 rows. The information is not lost, only
unwritten: the venue prompt lists the exact phrases the detector was asked for
(`prepare.prompts.gdino_classes`), the cutouts already carry MetaCLIP image
embeddings, and MetaCLIP text embeddings live in the same space.

```bash
PYTHONPATH=.:third_party/object_search python -m toolbox.benchmark.gdino_labels \
  --map-path /path/to/map --validate --write
```

`--write` produces `object-search/gdino_labels.parquet`, a **sidecar**: it sits beside
`metadata.parquet` and is joined back on `row_index`, so what prepare wrote stays
byte-for-byte what prepare wrote. A re-run of prepare cannot silently swallow the
estimate, and deleting the file undoes it.

**This is an estimate, and the margin says how much of one.** The stored labels of a
healthy map come from GroundingDINO; these come from MetaCLIP's opinion of the same
crop. Validated against bbhotel, whose 44 labels are real: 64.4 % top-1 agreement
overall, but the top1-minus-top2 margin separates the two regimes almost perfectly —
49.9 % below 0.01, 84.5 % from 0.01 to 0.03, **99.9 %** from 0.03 to 0.06, 100 % above.
Filter on the margin rather than trusting the label.

Most of the disagreement is vocabulary redundancy rather than error: the hotel prompt
holds three phrases for one camera and five for one table, so `dining table` ->
`rectangular table` and `surveillance camera` -> `security camera` dominate the
confusions. Collapsing synonym groups would raise the agreement mechanically.
Every row also carries `mean_clusters_per_annotation`: over the annotations with at
least two labelled detections, how many distinct clusters hold them. 1.0 means every
well-covered object came out whole. The JSON adds `fragmentation_by_class`, and with
it `mean_detections_per_annotation` — **the number is not readable without that
control**, since an annotation seen forty times has more chances of being cut than one
seen twice. Like `pair_precision` it only sees well-covered objects; it ranks
associations, it does not measure recall of the map.

## Two diagnostics that run before an association is written

`pair_cue_separability.py` reports rank AUC for every pairwise cue (depth-point
distance, ray distance, cutout cosine, `|Δ video_keyframe_id|`, same-keyframe angular
gap) both raw and **conditionally on the geometry** — restricted to pairs already
within `--conditional-m` of each other. The conditional column is the one that
decides: any cue correlated with spatial proximity scores well raw while adding
nothing to an association that starts from depth distance. It also reports median
metric distance by `|Δ id|` band, which is how you check a map was captured in one
pass before reading anything temporal.

`extent_baskets.py` is the gate for changing the merge criterion itself. It builds
*solo* baskets (one annotation's detections; right answer: one cluster) and *pair*
baskets (two annotations within 4 m; right answer: two clusters) from the benchmark
annotations, then partitions them with `gasp1v2`. Two families, because one number
cannot tell a fix from a coarser granularity.

Both read the same candidate cache as the sweep (`--cache-dir`), so a session that has
run one sweep can run every diagnostic offline — no postgres, no ANN service.

Results of the 2026-08-15 fragmentation study, including why the "estimate the object
extent" diagnosis was refuted, are in
`docs/plans/2026-08-15-fragmentation-resultats-E0-a-E4.md`.

## The hand-labelled baskets, and the depth-point refinements

`extent_baskets.py` attaches detections to the nearest annotation, which conditions on
the result: the detections that fail to reach their annotation are exactly the ones a
fragmentation study is about. The five modules below take their ground truth from
`detection_group_label` instead — a human grouping boxes in the panorama — and share
one harness. All of them take the same `--map-path` / `--cache-dir` pair and run
offline.

| module | what it answers |
|---|---|
| `matching_baskets.py` | T0/T1. Resolves the hand labels against the cache, builds solo/pair/mixed baskets, replays the matching tab's seven methods, and splits each group's spread into **radial** (wrong depth), **tangential** (box centre sliding) and per-keyframe **bias** — the last one against a shuffled null, since a keyframe holding `n` residuals scores about `1/sqrt(n)` on pure noise. Also emits the co-visibility matrix (`--covisibility-out`). |
| `covisibility_cue.py` | T1b. AUC of "no view saw both fragments although the views of one covered the other", with the depth maps used as visibility oracles rather than as position estimates. Reports a bootstrap interval next to the point estimate. |
| `depth_policies.py` | T2. Re-reads each box's depth as `center` / `median` / `nearest_mode` / `trimmed` and re-scores. |
| `ray_refinement.py` | T4. Associates, triangulates each cluster from its members' rays, slides every member to the foot of that point, re-associates. One pass. |
| `pose_offsets.py` | T8. One translation per keyframe, ridge-regularised, **cross-validated by group** — the training-set number is meaningless here and is printed only to show the gap. |
| `sigma_calibration.py` | T3. Fits `merge_score._sigmas` on the measured radial residuals, one point per detection. `--refine` fits on the positions T4 leaves. |

A basket set must be **fixed before a comparison**: every module that moves the
detections passes the baseline's `confusable_pairs`, so two policies are not scored on
two different populations.

Results, gates and verdicts: `docs/plans/2026-08-17-resultats-T0-a-T8.md`.
