# Spec — make annotation iteration recoverable (persistent undo + per-query annotation list)

Status: **not started**. Written 2026-08-10, for a Codex coding session on branch
`feat/explorer-reads-parquet` (which already carries the review-feedback boost, the
review-annotation port, and the uncommitted `feedback_alpha`/`feedback_beta` inputs
from [`toolbox-feedback-alpha-beta-ui.md`](toolbox-feedback-alpha-beta-ui.md)).

**Frontend only. No change to `toolbox/bricks/*`, `toolbox/backend/*`, or the
ranking.**

## The problem, and its actual cause

Workflow being supported: tune the feedback boost by annotating a few detections
per iteration, re-running the same query each time.

What breaks it: a negative annotation on a well-centred prototype subtracts
`beta * neg_sim` from every candidate's score
(`apply_feedback_boost`, `toolbox/bricks/candidates.py:143`), and
`rank_localization_clusters` drops every cluster scoring below `min_similarity`
(`toolbox/bricks/localize.py:295`). One annotation can therefore empty the result
list. Once it is empty there is no way back inside the UI, for two independent
reasons:

1. **The undo history is wiped by the search itself.** The load effect in
   `toolbox/frontend/src/object-search-review/useObjectSearchReviews.ts:80` lists
   `targetIds` in its dependency array and calls `setUndoStack([])` /
   `setRedoStack([])`. `targetIds` is derived from the current results
   (`useObjectSearchReviews.ts:66-75`), so **every re-search resets the history** —
   including the re-search that produced the empty list. `canUndo` is false exactly
   when undo is needed.
2. **An annotation is only reachable through the result row that carries it.**
   `ReviewButtons` is rendered per observation/cluster
   (`ObjectSearchPanel.tsx:1879`, `:1931`), and the same effect discards every
   annotation whose `targetId` is not in the current results
   (`useObjectSearchReviews.ts:93-99`). No results → nothing rendered → nothing to
   click.

So the user has to set alpha/beta back to 0, re-run, hunt the offending row and
un-annotate it. That is the loop this spec removes.

## What is already there (do not rebuild it)

- `GET /ui/api/maps/{map_id}/review-annotations?query=` **already returns every
  `detection_review` row for the query**, unfiltered, ordered by
  `detection_review_id` (`toolbox/backend/src/annotation-store.ts:433-467`, route at
  `toolbox/backend/src/workbench-api.ts:188`). The list this spec asks for needs
  **no new backend route** — only the frontend's own filtering has to go.
- `DELETE /review-annotations/detection-review` already removes one row, and
  `setDetectionReview(..., null)` already calls it (`object-search-review/api.ts:75`).
- Undo/redo buttons and the `Ctrl/Cmd+Z` handler already exist
  (`ObjectSearchPanel.tsx:1444-1463`, `useObjectSearchReviews.ts:200-227`). Fix what
  resets them; do not add a second history.

## Change 1 — `object-search-review/useObjectSearchReviews.ts`

### 1a. Stop resetting on results, keep resetting on query

Remove `targetIds` from the load effect's dependency array; depend on
`[options.enabled, options.mapId, options.query]` only. Resetting the history when
**`query` or `mapId`** changes is required and must stay: `persistChanges` writes
with `options.query` (`:118`), so undoing an annotation belonging to another query
would write the wrong row.

Also drop the `targetIds.size === 0` early return that clears state — with no
results there are still annotations, and they are now the point.

`targetIds` is still needed (see 1c), so keep the memo.

### 1b. Keep every annotation for the query, not just the visible ones

Remove the `.filter((item) => targetIds.has(item.targetId))` at `:95`. Consequence
to accept deliberately: `reviewedCount` / `truePositiveCount` /
`falsePositiveCount` (rendered at `ObjectSearchPanel.tsx:1442`) now count all
annotations recorded for the query rather than only those visible in the current
result set. That is the number this workflow needs — the counter currently drops to
0 the moment the boost empties the list, which reads as "your annotations are gone".
Put a one-line comment saying so, because it is a visible semantic change.

### 1c. Order the annotations, and expose them

Store a per-target record instead of a bare status, so the list can show the newest
annotation first:

```ts
type ReviewRecord = { status: ReviewStatus; seq: number };
```

- `seq` is a local monotonic counter (a `useRef`), **not** `created_at`. Rows loaded
  from the server arrive already ordered by `detection_review_id`, so assigning
  `seq = index + 1` on load preserves the server's order; annotations made in the
  session get the next counter value and therefore sort as newest. `created_at` is
  returned by the route but is a second-resolution string and is not worth parsing
  for this.
- Keep the existing public `reviews: ReadonlyMap<number, ReviewStatus>` — derive it
  from the records with `useMemo` so `LocalizeInspector` and every existing caller
  are untouched.
- Assign `seq` **outside** the `setRecords` updater (compute the assignments from
  the `records` value in the callback's closure, then call `setRecords`). Bumping a
  ref inside a state updater is impure and double-invokes under StrictMode.
- When `undo` restores a previously deleted annotation it gets a fresh `seq`, i.e.
  it reappears as the newest entry. Fine — do not add machinery to preserve it.

Additions to the exported `ObjectSearchReviews` type:

```ts
annotations: ReviewAnnotation[];              // newest first
clearAnnotation: (targetId: number) => void;  // goes through `commit`, so it is undoable
```

with `ReviewAnnotation = { targetId: number; status: ReviewStatus; inResults: boolean }`.
`inResults` is `targetIds.has(targetId)` — that is what `targetIds` is for now, and
it is what tells the user "this annotation is why the list is empty".

`clearAnnotation` must route through the existing `commit` (`:152`) so it lands on
the undo stack like any other change. Do not call `setDetectionReview` directly.

## Change 2 — `object-search-review/ReviewControls.tsx`

Add a second exported component next to `ReviewButtons`; same file, same flat style
(a presentational component taking data + callbacks, no fetching):

```ts
export function ReviewAnnotationList(props: {
  annotations: ReviewAnnotation[];
  onClear: (targetId: number) => void;
}): JSX.Element | null
```

- Returns `null` when `annotations` is empty.
- Collapsible, collapsed by default, header `Annotations (n)` — it sits above the
  results and must not push them off-screen. Use `<details>`/`<summary>` unless a
  collapsible idiom already exists nearby in this panel; check
  `CollapsibleOnlineOverrides` in `ObjectSearchPanel.tsx` first and match it if it
  is a plain button + conditional render.
- One row per annotation, newest first: candidate id (`#{targetId}`), a
  correct/incorrect badge reusing the existing `.object-search-review-button.is-true`
  / `.is-false` colour vocabulary, a muted `hidden` marker when `!inResults`, and a
  `×` button calling `onClear(targetId)` with `aria-label={"Remove annotation for candidate " + targetId}`.
- No thumbnail and no label: the annotation store keys on
  `object_search_candidate.id` only, and resolving a label would mean a metadata
  round-trip per row. Out of scope.

## Change 3 — `object-search/ObjectSearchPanel.tsx`

- Render `<ReviewAnnotationList annotations={reviews.annotations} onClear={reviews.clearAnnotation} />`
  immediately after the existing `object-search-review-toolbar` block
  (`:1437-1465`), inside `.os-results-scroll` so it scrolls with the pane.
- **Relax the toolbar gate.** It is currently
  `props.reviewMode && result?.mode === "localize-online"` (`:1437`). It must also
  render when there are annotations but no usable result, otherwise the recovery
  path disappears in exactly the failure case this spec targets:
  `props.reviewMode && (result?.mode === "localize-online" || reviews.annotations.length > 0)`.
  Apply the same condition to the new list.
- Add the newest annotation to the Undo button's `title` (e.g.
  `Undo review of #1234 (Ctrl/Cmd+Z)`) using `reviews.annotations[0]`, falling back
  to the current text when there is none. Cheap, and it is what makes the button
  trustworthy when the list below is empty.

## Change 4 — CSS (`object-search/object-search-vision.css`)

New classes only, next to the existing review block (`.object-search-review-toolbar`
is at `:1850`). Match the surrounding palette (`#f8fafc` surface, `#465c74` text,
`rgba(201, 213, 228, 0.8)` borders, 12px/650 for headers) and keep rows compact
(28px controls, as `.object-search-review-history-actions .object-search-secondary-button`
does). Do not restyle anything that exists.

## Change 5 — docs (required by the maintenance contract)

Update the `object-search-review/` bullet in `AI_CONTEXT/toolbox.md` (the "Panels"
list): it currently says "per-query TP/FP state with undo/redo". It must say that
the history survives a re-search, that the annotation list is per-query and
independent of the displayed results, and that the counters cover the whole query.
No ADR — this changes no contract or boundary.

## Non-goals

- **No alpha/beta on/off toggle.** Considered and deliberately excluded from this
  change; the undo path is the fix.
- **No change to `localize.py` / `candidates.py`.** In particular do not make the
  boost fall back to raw ranking when it empties the list: `localize.py` is
  behaviourally identical to production by design (`AI_CONTEXT/bricks.md`), and that
  belongs in `wemap-vision-backend` first.
- No new backend route, no schema change, no persistence of the collapsed/expanded
  state.
- No thumbnails or labels in the annotation list.
- Do not touch the unrelated finding that `embedding_similarity_threshold`,
  `use_stored_positions` and `robust_centroid` are sent by the panel but are not
  fields of `LocalizeRequest` (`toolbox/bricks/service.py:268-286`) and so are
  silently ignored. Real, separate, backend-side.

## Verification

- `cd toolbox && npm run type-check`, plus `npm run lint` if the repo has one wired.
- Manual, on `bbhotel-choisy` with a **fully ingested** index (the DB currently holds
  a 30-keyframe partial ingest, all on level 0 — re-run
  `python -m toolbox.bricks.ingest_cli <map>` first, after sourcing this repo's
  `.env`, not the sibling repo's):
  1. Annotation tab, query `extincteur`, `feedback_beta = 0.5`.
  2. Mark a good detection incorrect, re-run → list empties.
  3. **The toolbar and the annotation list are still visible; `Undo` is enabled and
     names the annotation.** Undo, re-run → results come back.
  4. Repeat, this time recovering with the `×` in the annotation list instead.
  5. Change the query and back: history resets (expected), the list reloads from the
     store for the new query.
