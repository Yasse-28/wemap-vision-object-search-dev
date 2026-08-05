# AI_CONTEXT — how to navigate this repo efficiently

This repository is the **object-search platform** (see `README.md`,
`docs/adr/0001-object-search-platform-structure.md` and
`docs/adr/0002-align-on-backend-pipeline.md`). It holds a **mirror of the
production object-search pipeline** from `wemap-vision-backend`, the dev tooling
that production does not need, and an archive of the retired standalone lineage.

**Read this first:** `third_party/object_search/` and
`services/object_search_online/` are **byte-for-byte copies of the backend**. Do
not edit them here — fix the backend and re-sync. `scripts/check-mirror.sh`
enforces it. Dev-only code goes in `toolbox/`; retired code lives in `legacy/`,
which is unmaintained and excluded from packaging, tests and lint.

**This file and the `AI_CONTEXT/` directory exist so an agent can locate the
code relevant to a task without reading the whole tree.** Reading everything is
expensive and unnecessary; reading the right 1–3 source files is not.

## How to use this (the navigation protocol)

1. **Orient** — read this file's routing table below and
   [`AI_CONTEXT/overview.md`](AI_CONTEXT/overview.md) to identify which
   subsystem(s) your task touches.
2. **Narrow** — open only the matching `AI_CONTEXT/<area>.md` description
   file(s). Each one maps concepts → exact source paths → key symbols, plus the
   data flow and gotchas for that area.
3. **Dive** — open only the specific source files the description points you to.

Do **not** bulk-read directories, and do **not** read `legacy/` unless the task is
explicitly about what was retired — it is a dead archive and reading it is the
fastest way to implement the wrong thing.

## Routing table

| If the task concerns… | Read this description | Code lives in |
|---|---|---|
| Architecture, data flow, ports/paths, "where does X live" | [`AI_CONTEXT/overview.md`](AI_CONTEXT/overview.md) | (whole repo) |
| Detection, venue prompts, ERP geometry, cutouts, the parquet contract | [`AI_CONTEXT/mirror-prepare.md`](AI_CONTEXT/mirror-prepare.md) | `third_party/object_search/{prepare,inference,indexing}/` |
| ANN query path, DSN/env config, HNSW params, annotation ground truth | [`AI_CONTEXT/mirror-serving.md`](AI_CONTEXT/mirror-serving.md) | `services/object_search_online/`, `third_party/object_search/annotation_service/` |
| Ingest, local schema, 3D lifting, enrichment, clustering, ranking, `localize`, keyframe poses | [`AI_CONTEXT/bricks.md`](AI_CONTEXT/bricks.md) | `toolbox/{bricks,georef}/` |
| The dev/test/annotate/benchmark tool (TypeScript) | [`AI_CONTEXT/toolbox.md`](AI_CONTEXT/toolbox.md) | `toolbox/{backend,frontend}/`, `toolbox/benchmark/` |
| What was retired and why | `legacy/README.md` | `legacy/` |
| Why the repo is shaped this way | `docs/adr/` | — |

**Before changing anything under a mirrored path, stop.** The answer is almost
always "change it in `wemap-vision-backend` instead".

## Maintenance contract (REQUIRED)

**When you change code, update the matching `AI_CONTEXT/*.md` in the same
change.** These descriptions are only useful if they stay true. Specifically:

- Add/rename/remove a source file, function, endpoint, table column, config
  key, or model → reflect it in the relevant `AI_CONTEXT/<area>.md`.
- Change a data flow, default value, or cross-component contract (e.g. a port,
  path prefix, or DB field) → update `AI_CONTEXT/overview.md` too.
- A structural/architectural decision → add or amend an ADR in `docs/adr/`.
- **Re-sync a mirrored tree** → update the sync point in
  `third_party/PROVENANCE.md`, and re-run `scripts/check-mirror.sh`.
- **Touch a vendored helper** → update `toolbox/bricks/vendored/PROVENANCE.md` so
  the intended delta from production stays recorded.
- Keep the external `../object-search-description.md` in sync as well (it is the
  human-facing companion to these files).

Keep each description **concise and navigational** — paths and key symbols, not
copies of the code. If a description and the code disagree, the code wins;
correct the description.

## File-format convention for `AI_CONTEXT/*.md`

Each area file should contain, in order: a one-line **purpose**, a **"read this
when"** line, a **key-files table** (`path` → responsibility → key symbols), the
**data flow / important constants**, and **gotchas / cross-refs**.
