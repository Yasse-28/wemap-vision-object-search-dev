# ADR 0005 — The explorer reads `metadata.parquet`, in TypeScript

- **Status:** Accepted
- **Date:** 2026-08-06
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0002](0002-align-on-backend-pipeline.md) — completes its
  retirement of the standalone lineage on the last axis still holding out.

## Context

ADR 0002 retired the standalone `build_index.py`, and with it `object-search.db`.
Nothing has written that file since. The toolbox's TypeScript backend nevertheless
kept reading it by shelling out to the `sqlite3` binary, and `requireIndex` answered
**501** in its absence. Nine routes depended on it, and
`ObjectSearchExplorerPanel` gated **its entire render** on `status.available`: on a
v2 map the panel was a single box reading "no object-search.db found".

Two of those routes did not need an index at all. `indexKeyframeEquirectPreviewPng`
called `requireIndex` *and* required `geometry === "cubemap"` before its
`drawBoxes === false` fast path, which only serves a file from disk; both mounted
panels used exactly that path and both swallowed the error, so no panorama appeared
and no message explained why. The same held for the depth pin's `projection: "erp"`
branch. That was the cheap half of this decision.

The expensive half is that v2 provides **no replacement index**. There is no index
file: the vector index lives in pgvector, and what is on disk is
`{map}/object-search/metadata.parquet` (one row per proposal),
`embeddings.npy`, and `thumbnails/{row_index:06d}.jpg`. Three differences from the
old model are structural, not cosmetic:

- **No cutout → objects hierarchy.** A parquet row *is* both the cutout and the
  detection. The triple walk `keyframe_ids → cutout_ids_by_keyframe →
  object_count_by_cutout` has nothing to stand on.
- **No OCR, no `cluster_id`.** OCR was never ported (ADR 0002), and clustering is
  now query-time only. A large part of the explorer's UI had no source.
- **No cubemap.** The geometry is a gnomonic projection of the ERP, so `id_stride`,
  `CUBEMAP_FACE_ORDER` and `cubemapPixelToEquirect` are meaningless.

## Decision

1. **Read the parquet natively in TypeScript** (`hyparquet`, pure JS, no runtime
   dependencies; Snappy is built in and is what pyarrow writes by default). The whole
   file is loaded once and cached by mtime; paging is a `slice`. Rejected: a Python
   bridge or a new bricks endpoint — either would put a process boundary between the
   explorer and a file it can just read, for a local dev tool.
2. **Delete the OCR/cluster surface** rather than stub it: types, UI, routes, and the
   two never-mounted panels (`IndexExplorerPanel`, `LatentScatter`).
3. **Rename the route prefix** `/object-search-index` → `/object-search-metadata`. A
   dead bookmark returning 404 beats a name that promises a database.
4. **The stored thumbnail is the default preview.** It is the only image certain to be
   what MetaCLIP2 embedded, and a route already serves it
   (`/preview.png?preview_path=<thumbnail_key>`). The ffmpeg re-render keeps an honest
   scope: maps prepared without crops, a widened context view, and the cutout↔ERP
   mapping the depth pin needs.
5. **Invert the panel's guard.** Photosphere, depth preview, depth pin, view cone,
   keyframe graph, livemap markers and annotations depend only on the manifest, so the
   panel renders as soon as the map is known and the metadata warning is scoped to the
   row table.
6. **pgvector coverage comes from the bricks service**
   (`GET /{map_id}/object-search/index-coverage`), as counts rather than ids, and is
   **nullable end to end**. The TS backend proxies to it with a timeout and treats any
   failure as "unknown". Rejected: a `pg` client in Node — it would duplicate DSN
   construction far from `bricks/db.py`, and make Postgres a hard dependency of the
   explorer, which is the failure mode this whole line of work is leaving behind.

## Consequences

- `MapEntry.object_search_index_path`, `MapSummary.legacy_index_available` and every
  `sqlite3` helper in the TS backend are gone. The `sqlite3` binary is no longer a
  dependency of the toolbox backend.
- Area thresholds in the bbox post-processing changed unit — **square degrees**, since
  a proposal has an angular extent and no fixed raster. `localStorage` key bumped
  accordingly; NMS now runs across a whole keyframe rather than per cutout.
- Annotation ground truth records `source_row_index`; `source_cutout_id` is still read
  so pre-migration files keep loading.
- Multiple capture directories under `{map}/object-search/` are **refused** with an
  explanation: `row_index` restarts at 0 per capture and `prepare_postprocess` writes
  the same `thumbnail_key` prefix for all of them, so their thumbnails overwrite each
  other on disk.
- Reconstructed boxes are float16-derived: accurate to a few ERP pixels. Anything
  user-facing says "reconstructed".

## Traps this closes, and how

Each of these produces plausible wrong output rather than an error.

| Trap | Guard |
|---|---|
| Geometry off by a sign, an axis or a unit | `erp-cutout-fidelity.test.ts`: re-renders real rows and cross-correlates (NCC) against the thumbnails on disk. One pass/fail covers float16 decode → radians → degrees → ffmpeg rotation order → resolution → mask. Opt-in (`OBJECT_SEARCH_TEST_MAP`); observed worst 0.93 over 39 rows on `bbhotel-choisy`. |
| INT64 ids arriving as `BigInt` | Converted at the reader boundary; a `Map<number, …>` would otherwise never match and the table would be empty with no error. Pinned by test. |
| Non-2:1 ERP | `assertEquirect2to1`. Off 2:1 the pipeline's aspect rescale stops being a no-op and the stored angles stop being the rendered FOV. |
| Parquet never post-processed | `postprocessed: false` in the status, and a banner naming the command — not 404 previews and "no depth" everywhere, which look like a bad capture. |
| A rejected read cached forever | The cache entry is deleted on failure (the inherited `loadIndex` bug). Pinned by test. |
| Rows of a keyframe not contiguous | Asserted at load: `row_index` is a global monotonic counter, so non-contiguity means the file is not from this pipeline. |
| A proposal spanning ≥180° | `isRenderableGnomonic` → 422. Clamping would show a different region; the stored thumbnail is degenerate for the same reason (`tan(3.12) < 0` upstream). |
| `coverage` becoming a hard dependency | Null on any failure, at every layer. |

## Known gaps

- A search result cannot be traced back to its parquet row: pgvector does not carry
  `row_index`. So a result with no `thumbnail_key` has no preview, and the "object
  overlay" select in the text inspector is gone rather than reimplemented.
- The status payload carries one entry per prepared keyframe (~440 kB for 3674). Fine
  for a local tool; if it stops being fine, page it rather than dropping the counts.
- OCR is not coming back through this route. If it returns to the pipeline, it will
  arrive as parquet columns, and the explorer should read them like any other column.
