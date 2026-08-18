# ADR 0007 — Export a drawn region as a new map, and own its lifecycle

- **Status:** Accepted
- **Date:** 2026-08-18
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0004](0004-v2-map-data-only.md) — the v2 manifest remains the
  only pose source, but this repo is no longer purely a consumer of it.

## Context

Until now this repo only ever *read* a map directory. The manifest is a dump of
production's Django objects; the pixels are fetched by the sibling
`../retrieve-map-data`; `prepare_runner` and `prepare_postprocess` write artifacts
*inside* an existing map, and nothing creates one.

That makes any experiment on a sub-area of a venue cost a full-map cycle. A hotel
lobby and a hotel's four floors are not the same object-search problem, and there
is no way to isolate the first without re-preparing and re-ingesting the second.
The Toolbox already has the gesture that would answer this — the ROI polygon in
the Explorer's livemap — but it only ever counted annotations inside it.

## Decision

The Explorer's ROI tool gains a **target**: count annotations (unchanged) or select
keyframes. In the second mode a *session* accumulates regions across floors, and
`toolbox/bricks/export_roi.py` writes the selected keyframes as a **new map
directory**: manifest, `images/`, `depths/`, and — optionally — a subset
`object-search/`. `toolbox/bricks/delete_map.py` removes such a map completely.

Four consequences were chosen deliberately.

### 1. Keyframes are renumbered, and the renumbering is recorded

Keyframe ids are indices into `geo_keyframes`; a subset renumbers by construction.
Leaving holes was rejected: `prepare_runner` and `ingest_cli` both assume
`id == index`, and breaking that assumption fails silently — candidates attach to
whichever keyframes happen to occupy those indices. The exported manifest is
therefore dense from 0, and `export-provenance.json` carries the old→new table so
the mapping is auditable rather than merely correct.

### 2. `geo_ref_id` is allocated, not inherited

There is no table per map. `object_search_candidate` and `geokeyframe` are single
tables partitioned by a column, and `ingest_cli` opens with
`DELETE ... WHERE geo_ref_id = %s`. An exported map that kept its source's id would
**erase the source** on its next ingest.

Nothing detected that before, because production manifests carry distinct ids and
this repo never minted one. Writing manifests here makes the collision reachable,
so `allocate_geo_ref_id` takes the next free value across every configured manifest
and, when the database answers, both tables — and
`bricks/service.py::load_map_entries` now refuses a config where two maps share one.

### 3. Sub-maps are first-class, and only they are deletable

A `parent_map` key in the config entry makes a map a sub-map: the home page nests
it under its parent, to any depth, and the delete control appears only on it. A map
received from production is never deletable here — this repo cannot rebuild its
pixels. A map that still has children is refused, so a subtree cannot be orphaned.
Deletion purges the database first and the directory second: an orphaned row cannot
be traced back to what it described, an orphaned directory can.

### 4. Two counts are shown, not one

The panel counts client-side, from the marker table, on the floor the Wemap SDK
reports. The save dialog shows python's count, which resolves the floor from the
manifest's altitude bands — the only definition the exported manifest can be
consistent with. These can disagree. The dialog displays the gap rather than
smoothing it over: a systematic difference means the SDK's floors and the
manifest's bands disagree, which is a finding, not a rendering detail.

## Alternatives considered

- **A dedicated tab.** Rejected: the livemap, the keyframe markers, the floor
  readback and the polygon tool are all already mounted in the Explorer, and
  `MatchingPanel.tsx` shows what a second copy of that plumbing costs.
- **Doing the export in TypeScript.** Rejected: the geometry (EUS→WGS84, the level
  bands, point-in-ring) already lives in `vendored/geo_transform.py`, the parquet
  subset wants pyarrow, and the deletion needs a Postgres client the TS backend does
  not have.
- **Copying the pixels.** Rejected as the default: hardlinks are instant and free on
  the same filesystem, with a copy fallback on `EXDEV`. An export that took minutes
  per attempt would not be used to iterate.
- **Re-serialising the config to add the entry.** Rejected: the reader strips
  comments before `JSON.parse`, so a round trip would silently delete every comment
  the maintainer wrote. The entry is spliced textually, after a `.bak`, and the
  route declines to guess when the `maps` array cannot be located unambiguously.

## Consequences

- The repo now writes and deletes map directories. `AI_CONTEXT/overview.md` and
  `AI_CONTEXT/bricks.md` say so.
- An exported map without its artifacts must go through `scripts/build-index.sh`
  before it answers a query; with them, it is explorable immediately.
- `MapEntry`/`MapSummary` carry `parent_map`; `MapSummary` also carries
  `child_map_ids`, resolved server-side so the deletion gate lives in one place.
- The directory browser is the first route that lists the dev machine's filesystem.
  It is confined to the source map's parent directory and the config directory, and
  pinned by `export-roi.test.ts`.
