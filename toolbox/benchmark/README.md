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
| `s3` | annotations against detections, **depth-free measurement first** |
| `s4` | pairwise cues over three deliberately drawn populations, raw and conditional AUC, with the share of pairs each cue applies to |
| `s5` | intra-view duplicates split by detector and settled by embedding, and the co-visible lower bound on the number of objects |

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

**Known limit.** `detection_review` verdicts cannot be joined here: their `target_id` is
a pgvector candidate id, not a parquet `row_index`. `s0` says so rather than printing an
empty table.

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
