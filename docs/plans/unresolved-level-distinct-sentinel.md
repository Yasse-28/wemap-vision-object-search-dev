# Spec — give "no level resolved" a sentinel that cannot be a floor

Status: **not started**. Written 2026-08-10, branch `feat/explorer-reads-parquet`, on top
of the uncommitted `level_strategy` work.

## The bug

`UNRESOLVED_LEVEL = -1` (`toolbox/bricks/localize.py:24`). On any map with a basement,
`geo_levels` contains a level whose `value` is `-1.0`. "No level resolved" and "level -1"
are then the same integer, and three things go wrong in silence:

1. `_levels_compatible` (`:146`) returns `True` whenever either side is `-1`, so
   **basement detections merge into clusters on any floor**. Measured on
   `bbhotel-choisy`, query `lamp`, 78 clusters: 56 mix levels, every one of them only
   because of `-1`, and 4 chain `0—(-1)—1` with the basement acting as the bridge.
2. `build_localize_response` (`:548`) maps `-1` to `None`, so **a cluster genuinely in
   the basement is indistinguishable from one whose floor is unknown** — with
   `level_strategy="median"` this is 23 of the 26 clusters whose level changes.
3. The altitude clamp at `:288` is skipped for basement clusters (it guards on
   `!= UNRESOLVED_LEVEL`), so their centroid altitude is never brought into the band.

This is a **production** bug: `localize.py` is a faithful port of
`backend/object_search/v1_5_logic.py`, which has the same constant. The same fix must
land there — see "Divergence bookkeeping" below.

## The fix

Change the sentinel's *value* so no georef can produce it:

```python
# A level value no georef will ever declare. It was -1, which collides with the real
# basement level on every map that has one — see gotcha 5 in AI_CONTEXT/bricks.md.
UNRESOLVED_LEVEL = -9999
```

Nothing else about the sentinel's role changes. Every use is symbolic already — the
constant is referenced by name at all six sites in `localize.py` (`:146`, `:147`,
`:254`, `:288`, `:309`, `:471`, `:548`) and by name in the tests, so the value change is
the whole change. Verify that with a grep before and after; a bare `-1` compared against
a level anywhere is a bug this spec is meant to remove, not preserve.

Constraints on the value: it is stored in an `int32` array (`np.full(..., dtype=np.int32)`
at `:254`), so it must fit; and it must stay negative so the existing `level < 0` style
guards elsewhere keep rejecting it (see the audit item below).

**Do not refactor to a boolean "resolved" mask.** That is the cleaner shape in the
abstract, but this file is a port: the backend fix should be the same three-character
diff, and a structural change here makes the next re-sync a merge instead of a copy.

## What changes behaviourally — all of it intended, none of it silent

On a map **without** a basement: nothing at all. On a map with one:

- Basement detections stop merging with other floors → **more clusters, each smaller**.
  Some will then fall below `min_keyframes_per_cluster` and disappear entirely. That is
  the correct outcome, not a regression.
- The response reports `level: -1` for basement clusters instead of `null`. Any consumer
  that reads `level == null` as "unknown" now gets a usable floor.
- Basement centroids get their altitude clamped into the basement band, like every other
  floor.
- The `levels_for_altitudes` fallback at `:309` now fires only for genuinely unresolved
  clusters.

## Audit — one place to check, and report on

`toolbox/benchmark/object_search_http_benchmark.py:304` has `if level < 0: continue` in
`enrich_prediction_levels_from_artifact`, which would also drop a genuine level `-1`. It
reads `cluster_levels` from `object-search.npz`, the **retired** v1 artifact
(`load_cluster_levels_from_artifact` returns `None` when the bundle is absent, so on v2
maps this path is dead). **Check whether it can still be reached on any current map.**
If it cannot, leave it alone and say so in the summary; if it can, it is a second
instance of the same collision and needs the same treatment. Do not change it blind.

The benchmark's own matching is unaffected either way: `match_predictions` (`:325`) pairs
predictions to annotations on horizontal distance only and never looks at a level.

## Tests — `toolbox/tests/test_localize.py`

The existing `test_unresolved_level_stays_mergeable` (`:121`) uses the constant by name
and must keep passing unchanged. Add, using a `GeoTransform` whose `levels` include
`Level(value=-1.0, ...)` so the fixture has a real basement:

1. **A basement detection and a ground-floor detection at the same spot do not merge.**
   Two candidates within `eps`, `vkf_level` `-1` and `0`. Before this change they form
   one cluster; after it, two. This is the test that would have caught the bug.
2. **A basement cluster reports `level: -1`, not `null`,** through
   `build_localize_response`.
3. **A genuinely unresolved cluster still reports `null`** — no `vkf_level`, no `level`,
   and a centroid altitude outside every band so the fallback cannot rescue it. This
   pins that the sentinel still works as a sentinel.
4. **`-1` no longer disables the merge veto**: the equivalent of case 1 but asserting
   `_levels_compatible`-driven behaviour through `cluster_detections_leader_canopy`, so
   the guard is pinned at the clustering level and not only end to end.

## Divergence bookkeeping (required)

This is the first deliberate *behavioural* divergence of `localize.py` from production,
as opposed to the opt-in `level_strategy`. It must be recorded, or the next agent will
"restore" it:

- `AI_CONTEXT/bricks.md`: rewrite gotcha 5 — it currently says no level strategy can work
  around the collision and that the fix belongs in the backend. It must now say the fix
  is **applied here**, what changed, and that `wemap-vision-backend`'s `v1_5_logic.py`
  still carries the bug and must be fixed before the next re-sync, or the sync will
  reintroduce it.
- Same file, the `localize.py` line in the key-files table lists `UNRESOLVED_LEVEL=-1` —
  update the value.
- Same file, the paragraph after the table ("differs from production in four import
  lines only, plus one dev-only opt-in") must name this divergence too.
- `docs/plans/port-frontend-clustering-for-comparison.md` has a trap note saying
  production's level veto is inert on 56 of 78 clusters on `bbhotel-choisy`. Once this
  lands that is no longer true; update it.

## Verification

- `pytest toolbox/tests/` (157 passed / 10 skipped before this change), plus `black
  --check`, `ruff check`, `mypy toolbox/bricks/localize.py`.
- `cd toolbox && npm run type-check` — no frontend change is expected, this is a sanity
  check that nothing typed against `level` broke.
- **Measure and report**, on `bbhotel-choisy` with the fully-ingested index. Start a
  service on a **free port** (45679; do not kill the one on 45678, it is the user's), then
  for `level_strategy` in `seed`, `median`, POST `lamp` with `num_results=200,
  candidate_count=1000, clustering_eps_m=2, min_keyframes_per_cluster=3` and report:
  - the cluster count before and after this change (expect more clusters);
  - the level distribution for both strategies (before this change: seed
    `{0: 57, 1: 11, 3: 2, 4: 1, 5: 3, None: 4}`, median `{0: 46, 1: 7, None: 25}`);
  - how many clusters still mix two resolved camera levels (expect **zero**, and that is
    the headline number: it was 4 chaining through `-1`).

Stop the extra service when done.
