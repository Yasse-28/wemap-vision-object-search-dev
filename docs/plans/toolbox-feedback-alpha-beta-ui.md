# Spec — expose feedback_alpha/feedback_beta in the toolbox UI

Status: **not started**. Written 2026-08-07, for a Codex coding session on branch
`feat/explorer-reads-parquet` (already carries [[review-feedback-boost]] and the
review-annotation port).

## Goal

The bricks `/object-search/localize` endpoint already accepts `feedback_alpha`
and `feedback_beta` (`toolbox/bricks/service.py:284-285`, both `Field(default=0.0,
ge=0.0, le=1.0)`) — verified working end-to-end via curl, boosting/penalizing
candidates near reviewed cutouts. Nothing in the toolbox frontend sets them; they
are always `0.0` today. Add two number inputs so a user can tune them from the
UI instead of curl.

Scope: **the two inputs and their wiring only.** No new visualization of
pos_sim/neg_sim beyond what already exists (see `feedback_delta` etc. already
in `EnrichedObservation`/API types if present — check before adding, don't
duplicate), no persistence of alpha/beta across sessions, no change to the
backend (already correct).

## Where this plugs in — verified by reading the code, not guessed

`toolbox/frontend/src/object-search/ObjectSearchPanel.tsx` already has an
"Online overrides" collapsible section (`CollapsibleOnlineOverrides`, ~line
1530) bound to a `OnlineLocalizeOverrides` object
(`toolbox/frontend/src/object-search/types.ts:69-88`) with fields like
`merge_radius`, `embedding_similarity_threshold`, `min_keyframes_per_cluster`,
`candidate_count`. This is the right home for `feedback_alpha`/`feedback_beta`
— same tuning-knob nature, same panel, and it's already rendered in **both**
the "Object Search" and "Annotation" tabs (`ObjectSearchPanel` is shared via the
`reviewMode` prop), so no tab-specific gating is needed.

**Why this is low-effort**: `onlineOverrideEntries()`
(`toolbox/frontend/src/object-search/api.ts:225-232`) turns every key of
`OnlineLocalizeOverrides` into a POST body field 1:1 (only `merge_radius` is
renamed, to `clustering_eps_m`). Adding `feedback_alpha`/`feedback_beta` to the
type and its default object means they flow to the backend automatically —
`runObjectSearch`'s JSON-body branch (text/localize) spreads
`onlineOverrideEntries(...)` straight into the payload
(`toolbox/frontend/src/object-search/api.ts:341-346`). **No change needed in
api.ts.**

The image-upload branch (`toolbox/frontend/src/object-search/api.ts:329-335`)
also appends every override key to the `FormData`, including these two — this
is harmless (the bricks service's image-query branch never constructs a
`LocalizeRequest`, so extra form fields are ignored; see
`toolbox/bricks/service.py`'s `if content_type.startswith("multipart/form-data")`
branch, which reads only `text`/`image`/`num_results`/`candidate_count`). Do
not special-case this away — it would be speculative work for a case that does
nothing today.

## What changes, concretely

1. **`toolbox/frontend/src/object-search/types.ts`**: add
   `feedback_alpha: number` and `feedback_beta: number` to
   `OnlineLocalizeOverrides`, both defaulted to `0.0` in
   `DEFAULT_ONLINE_OVERRIDES` — this is what makes "off by default" hold in the
   UI too, matching the backend default.
2. **`ObjectSearchPanel.tsx`'s `CollapsibleOnlineOverrides`**: add two
   `<label className="object-search-online-input">` number inputs, same pattern
   as `embedding_similarity_threshold` (step `0.01`, or a slider if you prefer —
   follow whatever the existing `Sensitivity`/`Merge radius` sliders elsewhere in
   this file use, for visual consistency; check both patterns before picking
   one). Clamp in the input (`min={0} max={1}`) to match the backend's
   `ge=0.0, le=1.0` — the backend still validates independently, this is just to
   avoid a round-trip 422 for an obviously out-of-range value.
3. No changes to `toolbox/bricks/*`, `toolbox/backend/*`, or `api.ts`.

## Non-goals

- No display of `pos_sim`/`neg_sim`/`similarity_boosted`/`feedback_delta` beyond
  what may already be rendered (check `LocalizeInspector` / observation card
  rendering before adding anything — if it's already there from the
  review-annotation port, this spec does not ask for more).
- No persistence (localStorage, URL params) of the alpha/beta values across
  reloads — same lifetime as the other overrides in this panel.
- No change to validation range or defaults on the backend.

## Verification

- `cd toolbox && npm run type-check && npm test -w backend` (frontend
  type-check; no backend behavior changed but the CI-equivalent check should
  still pass).
- Manual: run a localize search with alpha/beta at 0 (unchanged output, byte
  for byte, vs. before this change) then with alpha=0.3/beta=0.2 on a map with
  existing `detection_review` rows, confirm `similarity_boosted`/
  `feedback_delta` in the response move, same as the curl check already done
  for the backend.
