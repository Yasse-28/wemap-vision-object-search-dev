# Spec — compare the client-viewer's clustering with production's

Status: **not started**. Written 2026-08-10, rewritten 2026-08-11 after reading both
implementations side by side. Branch `feat/explorer-reads-parquet`.

The original version of this spec planned to port the client viewer's clustering into
this repo as a second strategy, to score the two against each other. **That measurement
would have returned nothing: the two are the same algorithm.** What follows is the
comparison that replaced it, and the two knobs that do carry a measurable difference.

## The reference implementation (do not modify that repo)

`wemap-vision-frontend`, `src/pages/client-viewer/useObjectSearch.ts`. Sibling checkout:
`../../frontend-related-projects/wemap-vision-frontend`.

```ts
// useObjectSearch.ts:90 — "representative-based radius clustering (NOT single-linkage)"
function clusterByDistance(candidates: Candidate[], epsilon: number): Candidate[][] {
    const clusters: Candidate[][] = [];
    for (const c of candidates) {
        const match = clusters.find(cl => distanceMeters3D(cl[0], c) < epsilon);
        if (match) match.push(c);
        else clusters.push([c]);
    }
    return clusters;
}
```

| Step | Where | Default |
|---|---|---|
| `filterBySimilarity` — keep `similarity >= alpha * candidates[0].similarity` | `useObjectSearch.ts:72` | `alpha = 0.85` |
| `clusterByDistance` | `:90` | `epsilon = 3.0` m |
| drop clusters with fewer than `minViews` **candidates** | `:192` | `minViews = 2` |
| pin position = `cl[0]` (the representative), cluster order = creation order | `:193` | — |

It runs **in the browser**, on the flat v2 candidate list from
`/object-search/by-text` — not on `/localize`. So none of the localize-side machinery
applies to it: no `min_similarity`, no `match_score`, no level clamp on the reported
position, and **no review-feedback boost** (that lives on the localize path only).

`distanceMeters3D` (`:66`) is `hypot(horizontal, altitudeDelta)`, the horizontal term
being an equirectangular approximation (`:56`) despite the comment calling it haversine.
At a 3 m radius the difference is noise.

## The algorithms are the same

Both maintain a greedy list of leaders and give each point to the first leader within
the radius. The two loops are written inside out from each other, which is what made
them look different:

| | Frontend | Production (`localize.py:161`) |
|---|---|---|
| Iterates | **points**, scanning clusters in creation order | **seeds**, in order of decreasing similarity |
| Rule | join the first cluster whose `cl[0]` is `< eps` away | the seed absorbs every *unassigned* point `<= eps` away |

They coincide because the candidate list **arrives sorted by descending similarity**
(`load_enriched_candidates` ends on `enriched.sort(key=similarity, reverse=True)`, and
the frontend's `filterBySimilarity` preserves order). Leaders are therefore created in
similarity order, so "first leader in creation order" and "most similar seed" name the
same leader. Verified: 300 random trials (15–90 points, ε = 3 m, level veto off,
`min_keyframes=1`) produce **0 differing partitions**.

Two footnotes. The boundary differs — `< epsilon` client-side, `<= eps` in
`localize.py` — which only matters on exact ties. And the frontend's comment claiming a
deliberate deviation from its own `AI_CONTEXT.md` (single-linkage → representative)
records the change that *aligned* it with production; the spec it deviates from is the
one that was wrong.

## What actually differs

Ordered by how much it moves the result.

1. **The level veto — the only thing that changes the partition.** `_levels_compatible`
   (`localize.py:143`) refuses to merge two *resolved, different* levels. The frontend
   has no veto; the floor only enters through the vertical term of the distance, so two
   objects separated by a slab thinner than `epsilon` (default 3 m!) merge. Measured on
   synthetic detections split over two slabs 2.5 m apart, ε = 3 m: the partition differs
   in **298/300** trials, production producing ~10 % more clusters.
2. **The candidate set.** Frontend: `alpha = 0.85` of the best similarity. Production:
   `LOOSE_ALPHA = 0.3` of it, then capped at twice the count above that bar
   (`candidates.py::_prefilter_hnsw_results`), then `candidate_count`, then
   `min_similarity` at ranking time. Production clusters a far wider set; in practice
   this is the most visible difference of all, and it is not a clustering difference.
3. **Minimum size.** `minViews` counts **candidates**; `min_keyframes_per_cluster`
   counts **distinct keyframes** (`filter_clusters_by_min_keyframes`). Ten detections
   from one keyframe pass the first and fail the second.
4. **Cluster position.** Frontend: the representative, an actual detection. Production:
   the similarity-weighted centroid, clamped into the claimed level's altitude band
   (`localize.py:288`).
5. **Output order.** Frontend: creation order, i.e. by representative similarity.
   Production: `match_score = 0.50·normalised_similarity + 0.15·confidence +
   0.35·keyframe_score`.

Not a difference: the distance. `hypot(horizontal, vertical)` and the Euclidean norm
over EUS `(x, up, z)` are the same quantity — **do not port the geodesy**, cluster on
`positions_eus` as `localize.py` already does.

## Implementation — two knobs, not a second algorithm

Follow the `level_strategy` / `feedback_normalization` precedent: opt-in parameters
whose defaults keep production's behaviour, so `localize.py`'s default path stays the
ported one (`AI_CONTEXT/bricks.md` makes any other behavioural change a bug). Each knob
isolates one of differences 1 and 2, so a run measures one thing.

1. **`toolbox/bricks/localize.py`** — `LocalizationParams.level_veto: bool = True`,
   forwarded to `cluster_detections_leader_canopy` as `detection_levels=None` when
   false. Passing `None` is the existing "no veto" path, already exercised by tests;
   do not add a second code path inside the clustering loop.
2. **`toolbox/bricks/candidates.py`** — `_prefilter_hnsw_results(hnsw_results, alpha)`,
   default `LOOSE_ALPHA`, plumbed through `load_enriched_candidates(prefilter_alpha=…)`.
   **The `n_above * 2` cap stays**: it is production's shape, and removing it in the
   same knob would conflate the bar with the cap. Note in the docstring that this makes
   the emulated frontend set a superset of the real one.
3. **`toolbox/bricks/service.py`** — both on `LocalizeParams` (so the image branch gets
   them too), forwarded by `to_params()` / passed to `load_enriched_candidates`.
4. **`toolbox/frontend/`** — a checkbox and a number input in the online overrides,
   beside `level_strategy`; `onlineOverrideEntries` already forwards them. Add both to
   `CONFIG_SUMMARY_KEYS` in `benchmark/config-summary.ts` so a stored run says which
   regime it ran in.
5. **`toolbox/benchmark/object_search_http_benchmark.py`** — `--level-veto` /
   `--prefilter-alpha`, added to the payload only when non-default and recorded in
   `config`, threaded through `BenchmarkRunParams` and `benchmarkScriptArgs`. **This is
   what makes the comparison a measurement**; without it the two regimes can only be
   eyeballed.
6. **`toolbox/tests/test_localize.py`** — pin that a cluster spanning two *resolved*
   levels forms with `level_veto=False` and does not with it on. That is difference 1,
   and it is the one most likely to be "fixed" by a later reader.
7. **`AI_CONTEXT/bricks.md`** — extend the `localize.py` note listing the dev-only
   opt-ins, and add the cross-repo pointer to `useObjectSearch.ts` so nobody has to
   rediscover where the other implementation lives.

## How to compare, once it is in

Same map, same prompt, one knob at a time — the benchmark's per-prompt P/R/F1 is the
verdict:

```bash
python -m toolbox.benchmark.object_search_http_benchmark --map-path <map> --online \
  --only-prompt extincteur --output-dir /tmp/baseline
# then --level-veto false --output-dir /tmp/no-veto
# then --prefilter-alpha 0.85 --output-dir /tmp/tight-prefilter
```

Hold `candidate_count`, `clustering_eps_m`, `min_keyframes_per_cluster` and
`min_similarity` fixed across runs. Two traps specific to this comparison:

- **`bbhotel-choisy` has a real level `-1`.** This repo uses the distinct
  `UNRESOLVED_LEVEL=-9999` sentinel (gotcha 5 in `AI_CONTEXT/bricks.md`), so its veto
  works on basement detections; production still has the collision. Do not read a delta
  against production as caused by the veto alone.
- The frontend's defaults (`alpha 0.85`, `epsilon 3.0`, `minViews 2`) are not this
  repo's (`LOOSE_ALPHA 0.3`, `clustering_eps_m 2.0`, `min_keyframes_per_cluster 2`).
  Comparing defaults compares tunings.

## Out of scope, deliberately

Differences 3, 4 and 5 are real but are not clustering: they are the size filter, the
reported position and the ranking. Each would need its own knob, and mixing two changes
into one measurement tells you nothing about either.
