# Spec — port the livemap-tools review-annotation tool into `toolbox/frontend`

Status: **implemented, awaiting review**. Written 2026-08-07, on branch
`feat/port-review-annotation-tool` (based on `feat/explorer-reads-parquet`,
which already carries the [[review-feedback-boost]] backend feature).

## Goal

`toolbox/frontend` can browse object-search results but cannot judge them.
Port the "review" feature from livemap-tools' `object-search-annotate` tool:
letting a user mark a detection cutout (or a whole cluster) `true_positive` /
`false_positive`, with undo/redo, against the same `detection_review` table
that [[review-feedback-boost]] already reads for the similarity boost.

This spec covers **the review primitive only** — true_positive/false_positive
toggling on cutouts and clusters. Not in scope: missed-detection drawing
(bbox/polygon), detection-class management, the Ground-truth tab, or the
benchmark tab. Those exist in the source tool but are separate features with
separate value; porting them is future work, not blocking this one.

## Source

`~/Workspace/codes/wemap/frontend-related-projects/livemap-related-projects/livemap-tools/object-search-annotate/`:

- `object-search-annotate.tsx` (4297 lines) — monolith: search, clustering,
  keyframe graph, panorama viewing, review, missed-detection drawing,
  ground-truth, benchmark, all in one component. The review logic is
  interleaved with the rest, not a separable module — extraction is the main
  cost of this port, not raw volume.
- `Panorama360Viewer.tsx` (774 lines) — separate component, not review-specific.

### Review-relevant surface (verified by reading the file, line numbers may
drift if the source changes)

- **State**: `cutoutReviews: Map<string, ReviewStatus>` (line 913),
  `reviewUndoStack` / `reviewRedoStack: ReviewUndoAction[]` (923-924).
  `ReviewStatus = 'true_positive' | 'false_positive'` (line 118).
- **Action types**: `set-cluster-review`, `set-cutout-review`,
  `add-missed-detection`, `delete-missed-detection` — only the first two are
  in scope here.
- **Core functions** (~1528-1624): `applyReviewAction`,
  `applyInverseReviewAction`, `commitReviewAction`, `handleUndoReview`,
  `handleRedoReview`. Plain `useState` + `useCallback`, no external state
  library — straightforward to re-host.
- **Query key**: `annotationQueryRef.current = selectedImageFile ? selectedImageFile.name : prompt.trim()` —
  the review is tagged with the *raw* search text (or filename for an image
  search), not a normalised form. Matches what [[review-feedback-boost]]'s
  `feedback.py::normalize_query` casefold+strips on read.
- **Backend**: `ANNOTATION_BASE = 'https://object-search-annotations.maaap.it'`
  (line 20), `annotationEndpoint = ${ANNOTATION_BASE}/${props.slug}`. Calls
  `POST /{slug}/object-search/annotations/detection-review` and
  `DELETE /{slug}/object-search/annotations/detection-review` with
  `{target_type, target_id, query, status}` / `{target_type, target_id, query}`,
  and reads `GET /{slug}/object-search/annotations?query=` for existing state.
- **Keyboard**: global `keydown` handler (~2650) — Ctrl/Cmd+Z undo,
  Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y redo. No auth, no router, no external state
  management — nothing to strip out on that front.

### Backend counterpart already in this repo

`third_party/object_search/annotation_service/app.py` is the FastAPI
reimplementation of the same Node routes (see its docstring: "Faithful port
of the annotation routes of wemap-vision-tools `object-search-router.ts`").
Confirmed route parity for what this port needs:

| Route | Verified in `app.py` |
|---|---|
| `GET /{slug}/object-search/annotations?query=` | yes — returns `detection_reviews`, `missed_detections`, `detection_classes` |
| `POST /{slug}/object-search/annotations/detection-review` | yes — body `DetectionReviewUpsert`, returns `{detection_review_id}` |
| `DELETE /{slug}/object-search/annotations/detection-review` | yes — body `DetectionReviewDelete`, 204 |

The initial implementation proxied this service. Review feedback changed that
boundary: `toolbox/backend/src/annotation-store.ts` now owns the compatible
per-map SQLite file directly, so running the Toolbox does not require a separate
Python annotation process. The mirrored service remains the production contract.

## Target integration point

`toolbox/frontend/src/object-search-explorer/ObjectSearchExplorerPanel.tsx`
is the existing results panel (livemap + photosphere side-by-side, same
pairing as the source tool's `Panorama360Viewer.tsx`). This is where a cutout
is already rendered per search result — the natural place to add a
true_positive/false_positive control per cutout and per cluster.

`toolbox/frontend/src/annotations/` is a *different* feature (point/polygon
geojson annotations → `annotations/annotations.geojson`) and is not the right
home for this — naming collision aside, its data model has nothing to do with
`detection_review`. A new `toolbox/frontend/src/object-search-review/` (or
similar, name TBD) is more consistent with how `object-search-explorer/` is
scoped.

## What changes, concretely

1. **New review state + actions module** (plain hooks, ported near-verbatim
   from `applyReviewAction`/`applyInverseReviewAction`/undo-redo stacks) —
   this is the part that's safe to port mechanically since it has no
   dependency on the rest of the monolith.
2. **A per-cutout / per-cluster review control** wired into
   `ObjectSearchExplorerPanel.tsx`, replacing whatever UI affordance the
   monolith used inline (checkmark/cross buttons or similar — confirm exact
   UI on read-through, not guessed here).
3. **Fetch client** for the three annotation routes above, pointed at
   `annotationBaseUrl` (already plumbed through the toolbox backend/CLI
   options — `OBJECT_SEARCH_ANNOTATION_URL`).
4. **Keyboard shortcuts** for undo/redo, scoped to whatever the toolbox's
   existing shortcut conventions are (check for conflicts with
   `object-search-explorer`'s own shortcuts before reusing Ctrl/Cmd+Z/Y).

## Non-goals / explicitly out of scope

- Missed-detection drawing (bbox/polygon capture) and detection-class CRUD.
- The Ground-truth tab and benchmark tab.
- Any change to `annotation_service` — it already supports what's needed.
- Any change to [[review-feedback-boost]] — this port is what will let
  people *produce* the annotations that feature already knows how to
  consume; the two are independent deliverables that happen to share a
  table.

## Open questions (resolve before implementation, not guessed here)

- Exact per-cutout UI affordance in the source tool (icon/button placement)
  — read `object-search-annotate.tsx` around where `cutoutReviews` is
  rendered, not summarized in this spec.
- Whether cluster-level review in the target UI should fan out to all cutouts
  in the cluster the same way the source does (`set-cluster-review` action),
  or whether the toolbox's cluster model differs enough to need a different
  rule.
- Where undo/redo state should live if `ObjectSearchExplorerPanel.tsx` is
  already fairly large — new hook file vs. inline, follow whatever the
  toolbox's existing convention is for panel-scoped state.
