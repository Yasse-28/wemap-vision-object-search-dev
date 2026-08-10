# Spec — score the current prompt from the Annotation tab

Status: **not started**. Written 2026-08-10, for a Codex coding session on branch
`feat/explorer-reads-parquet` (which carries the feedback boost, the α/β inputs and
the annotation undo/list work — commits `0142a43`, `0c07fd9`).

## Goal

While tuning the review-feedback boost, the user annotates a few detections, re-runs
the query, and wants to know whether precision/recall actually improved. Today the
only source of P/R/F1 is a full benchmark run over every prompt, and **no benchmark
run can ever reflect the boost**: the script builds its localize payload at
`toolbox/benchmark/object_search_http_benchmark.py:722-734` and never sends
`feedback_alpha`/`feedback_beta` (nor `min_keyframes_per_cluster`), so scores on disk
are a boost-off baseline measured with the script's own parameters.

Add: **a button in the Annotation tab that scores the current prompt alone, through
the existing Python evaluator, with the panel's current parameters** — and show the
newest full run's score for the same prompt next to it as a baseline.

## Hard constraint — reuse the evaluator, do not reimplement it

The matching logic (haversine, greedy prediction↔annotation matching, annotation
grouping, PRF) lives in `object_search_http_benchmark.py` (`match_predictions`,
`group_annotations`, `evaluate_prompt`, `compute_prf`). **Do not port any of it to
TypeScript.** A second evaluator would drift from the first and `AI_RULES.md`
forbids parallel systems. The single-prompt path spawns the same script with a
filter.

## What already exists (verified)

- `metrics.json` per run holds `config`, `summary`, `by_class`, `by_prompt` — one row
  per prompt with `class_name`, `prompt`, `precision`, `recall`, `f1`,
  `true_positives`, `false_positives`, `false_negatives`, `matches`, `elapsed_ms`
  (`run_benchmark`, and `main` writing it at `:1144`).
- `GET /ui/api/maps/{id}/benchmark/runs/{runId}` returns that file verbatim
  (`toolbox/backend/src/workbench-api.ts:171`, `benchmarkRunPayload` at
  `benchmark-runner.ts:428`), and `GET /benchmark/status` lists runs newest-first
  (`listRuns` at `:398`).
- `regenerateGroundTruth` (`benchmark-runner.ts:565`) rewrites
  `{map}/benchmark/annotations.geojson` from the SQLite store atomically before every
  run — the single-prompt path gets fresh ground truth by calling the same helper.
- Prompts come from each ground-truth feature's `properties.prompt`, falling back to
  `class` (`load_annotations` at `:139`, `buildGroundTruth` at
  `annotation-store.ts:543-556`).
- On `bbhotel-choisy`, run `2026-07-30_17-47-36` scores `extincteur` at
  P=0.857 R=0.931 F1=0.893 (TP=27 FP=6 FN=2) — use it as the manual-check reference.

## Change 1 — `toolbox/benchmark/object_search_http_benchmark.py`

### 1a. `--only-prompt` (repeatable)

Filter `prompt_to_annotations` right after it is built (`run_benchmark`, `:686-689`),
before `url`/progress/loop. Compare on a **casefold+strip normalisation of both
sides**, so `Extincteur` scores the prompt stored as `extincteur`. This is the same
convention as `toolbox/bricks/feedback.py::normalize_query`; add a two-line local
helper with a comment pointing at it rather than importing — the benchmark script is
standalone by design and must not grow a dependency on `toolbox.bricks`.

If the filter leaves **no** prompt, exit with a distinct, documented status
(`return 2` from `main`, message on stderr naming the prompt and the annotations
path). The caller has to be able to tell "this prompt has no ground truth" from
"this prompt scored zero" — conflating them is the one failure mode that would make
the whole feature misleading.

Everything downstream (`by_class`, `summary`, `prompt_geojson`) then covers the
single prompt; that is correct, do not special-case it.

### 1b. Parameters that decide comparability

Add, all optional and all defaulting to "not sent":

| Arg | Payload key | Default |
|---|---|---|
| `--feedback-alpha` | `feedback_alpha` | `0.0` |
| `--feedback-beta` | `feedback_beta` | `0.0` |
| `--min-keyframes-per-cluster` | `min_keyframes_per_cluster` | `None` |
| `--max-observations-per-cluster` | `max_observations_per_cluster` | `None` |

Add each to the `payload` dict at `:722` **only when it is set** (non-zero for the
gains, not `None` for the counts), so an existing invocation posts a byte-identical
body and old baselines stay comparable. `LocalizeRequest` already accepts all four
(`toolbox/bricks/service.py:268-286`), so no service change.

The last two are not scope creep: the Annotation tab sends
`min_keyframes_per_cluster: 3` while the service defaults to 2, so without them the
score would describe a clustering the user is not looking at.

Also record the four values in the `config` dict (`:897`), so a `metrics.json` states
what it measured. This is what makes the baseline-vs-current comparison honest.

### 1c. Tests

`toolbox/tests/test_object_search_http_benchmark.py` already fakes the HTTP layer
(`test_run_benchmark_collects_raw_records_and_progress_events`). Add:

- `--only-prompt` keeps exactly the matching prompt, case-insensitively;
- an unmatched `--only-prompt` exits 2 and writes no `metrics.json`;
- with the gains at their defaults the posted payload has no `feedback_*` keys, and
  with `--feedback-beta 0.4` it has `feedback_beta: 0.4`.

## Change 2 — `toolbox/backend/src/benchmark-runner.ts`

### 2a. `BenchmarkRunParams`

Add `feedback_alpha`, `feedback_beta`, `min_keyframes_per_cluster`,
`max_observations_per_cluster` (all optional numbers). In `benchmarkScriptArgs`
(`:512`) push each flag **only when the param is a finite number** — do not route
them through `numberParam`, whose job is to substitute a default. The full-run path
must stay unchanged when the UI sends nothing.

### 2b. `scorePromptPayload(options, map, prompt, params)`

A new exported async function, synchronous from the caller's point of view (the HTTP
request waits for the score). It reuses, in this order and for the reasons already
commented in `runBenchmarkJob` (`:587-599`):

1. `assertAnnServiceReachable(options)` then `ensurePythonService(options, map, …)`
   for a local target — same ordering rationale as the full run.
2. `regenerateGroundTruth(map)`.
3. spawn the script with `benchmarkScriptArgs(...)` plus
   `--only-prompt <prompt>`, `--no-prompt-geojson`, and
   `--output-dir {map}/benchmark/prompt-scores/<slug>`.

**Output location matters.** `listRuns` (`:398`) treats every child of
`{map}/benchmark/` whose name matches `/^[0-9A-Za-z._-]+$/` and which contains
`metrics.json` as a benchmark run — including a dot-prefixed one. Writing one level
deeper (`prompt-scores/<slug>/metrics.json`) keeps single-prompt scores out of the
run history, because `prompt-scores/metrics.json` does not exist. Do not invent a
new top-level directory and do not write into a run id.

Return shape:

```ts
{ prompt, row, config, scored_at }   // row = metrics.json's by_prompt[0]
```

Errors:
- exit status 2 → `WorkbenchRouteError(404, ...)` with a message saying the prompt has
  no ground truth in `annotations.geojson`. The UI depends on this being
  distinguishable.
- no `metrics.json` → `WorkbenchRouteError(500, ...)` with the stderr tail, as
  `runBenchmarkJob` already does at `:658`.

Concurrency: refuse with 409 when `activeJob` is running (`:675` already has the
message to copy), and keep a module-level in-flight flag so two score requests
cannot spawn two scripts against one GPU service. A single-prompt score is **not** a
job: do not touch `activeJob`, do not appear in `/benchmark/status`.

### 2c. Route

`POST /ui/api/maps/{id}/benchmark/score-prompt`, body `{ prompt: string, params?: BenchmarkRunParams }`,
next to the existing benchmark routes in `workbench-api.ts:162-183`. Reject an empty
prompt with 400.

## Change 3 — frontend

- `toolbox/frontend/src/benchmark/api.ts`: a `scorePrompt(mapId, prompt, params)`
  client and its result type in `benchmark/types.ts` (`PromptScore`), reusing the
  existing error-detail helper style in that file.
- `toolbox/frontend/src/object-search/ObjectSearchPanel.tsx`, in the review toolbar
  added by `0c07fd9` (around `:1440`):
  - a `Score this prompt` button, disabled while `reviews.isLoading`, while a score
    is in flight, or when there is no `resultQuery`;
  - the result rendered as `P 0.86 · R 0.93 · F1 0.89 · 27/6/2 (TP/FP/FN)`;
  - the 404 case rendered as a muted `no benchmark ground truth for "<prompt>"`, not
    as an error banner — it is a normal state for most queries;
  - **the baseline**: the matching `by_prompt` row from the newest run in
    `/benchmark/status` → `/benchmark/runs/{runId}`, matched on the same
    casefold+strip normalisation, shown as `baseline (run <id>): F1 0.89`. Fetch it
    once per map+query, not per render.
- Parameters sent = the panel's `onlineOverrides`, mapped the same way
  `onlineOverrideEntries` maps them (`merge_radius` → `clustering_eps_m`), plus
  `feedback_alpha`/`feedback_beta`. Show `min_similarity` / `acceptance_threshold`
  (the script's defaults, 0.15 / 0.9) next to the score, or in a `title`, so the user
  can see the score was not measured with the Sensitivity slider they are looking at.

Keep it inside the existing `object-search-review-*` CSS vocabulary; no new panel.

## Non-goals

- No TypeScript re-implementation of matching, grouping or PRF (see the hard
  constraint).
- No automatic scoring on every search: a score costs a full localize plus a
  MetaCLIP embed on the GPU. Explicit button only.
- No change to `min_similarity`, `acceptance_threshold`, `group_annotation_radius_m`
  semantics or defaults, and no attempt to reconcile the benchmark's acceptance
  threshold with the panel's Sensitivity slider. Display both; unifying them is a
  separate decision.
- No change to `toolbox/bricks/*` — `LocalizeRequest` already takes every field.
- Single-prompt scores must not appear in the run list or in `/benchmark/status`.
- No caching or history UI for scores beyond the file left on disk.

## Two things that will bite

1. **The two annotation sets are unrelated.** The Annotation tab's ✓/× are
   `detection_review` rows keyed on `object_search_candidate.id`, with no position;
   they feed the boost only. The benchmark's ground truth is `manual_detection` rows
   flagged `used_as_ground_truth`, with lat/lng (`annotation-store.ts:532-579`).
   Annotating in the tab therefore **never** adds a scorable prompt — it only changes
   the predictions. Say so in the UI copy for the 404 case.
2. **Candidate ids do not survive a reingest** (`toolbox/bricks/feedback.py`
   docstring). Any score measured with α/β ≠ 0 is meaningless if the index was
   rebuilt after the annotations were collected. Out of scope to detect here, but do
   not add anything that hides it.

## Verification

- `cd toolbox && npm run type-check && npm test -w backend`
- `pytest toolbox/tests/test_object_search_http_benchmark.py`
- Manual, on `bbhotel-choisy`, index fully ingested, bricks + online services up:
  1. Annotation tab, query `extincteur`, α=β=0 → `Score this prompt` returns numbers
     in the neighbourhood of the baseline P=0.857 R=0.931 F1=0.893. They will not be
     identical (the tab sends `min_keyframes_per_cluster: 3`, `clustering_eps_m: 2`,
     the run used 1.5) — confirm `config` in the response reports the values actually
     used.
  2. Query `pas-un-prompt-du-benchmark` → the muted no-ground-truth message, no
     error banner, exit path verified.
  3. Mark two true detections incorrect, set `feedback_beta = 0.5`, score again →
     recall drops. That movement is the whole point of the feature.
  4. `GET /benchmark/status` still lists exactly the runs it listed before, i.e. the
     score wrote nothing into the run history.
