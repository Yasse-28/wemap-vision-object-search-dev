# ADR 0004 — Read v2 map data exclusively

- **Status:** Accepted
- **Date:** 2026-08-06
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0002](0002-align-on-backend-pipeline.md) — its decision 3 ("two
  pose formats") falls; the v1 half is removed.

## Context

ADR 0002 decided that poses come from the map directory rather than the ORM, and
accepted **two** formats: the v2 manifest and the legacy `georef.db`, with
`load_pose_source` preferring the former. That was the right call at the time —
maps in the v1 format still existed, and dropping them would have stranded work in
progress.

Since then every map that matters ships a v2 manifest, and the cost of the second
format turned out to be concentrated in exactly the places this repo is least able
to absorb it. All three of its costs are *silent*:

- **Three composing frame flips.** `georef.db` stores poses transposed,
  world-to-camera, in WDS/OpenCV. Drop any one flip and objects land mirrored or
  180° off, with no exception anywhere. An entire test file existed to prove the
  composition was load-bearing.
- **Filename fallbacks.** Depth and image resolution tried `{keyframe_id}.tif` and
  `{keyframe_id}.jpg` after the manifest's own name. Under v2 the keyframe id is an
  **array index**, so a leftover `2.tif` matches a real request and serves an
  unrelated depth map.
- **`--venue` and `--geo-ref-id`.** They existed only because `georef.db` records
  neither. A wrong `--venue` indexes a different candidate set than production; a
  wrong `--geo-ref-id` writes under a partition key the online service never
  queries, returning zero hits with no error.

The v1 format bought nothing against those risks: it could not supply `venue_type`
or `geo_ref_id` at all.

There was also an asymmetry worth naming. The Python half had been v2-first since
ADR 0002, but the toolbox's TypeScript backend had **no manifest reader at all** —
its keyframe markers, depth pin, view cone and world-point projection parsed
`georef.db` directly by shelling out to `sqlite3`. Those routes therefore did not
work on any current map. Keeping two formats in the Python half while the TS half
supported only the dead one was the worst of both.

## Decision

1. **The v2 manifest is the only pose source.** `georef.db` and `reloc.db` are no
   longer read anywhere. `toolbox/georef/` is deleted;
   `georef_source.load_pose_source` becomes a thin façade over `map_manifest`.

2. **The TS backend gets a real manifest reader**, `toolbox/backend/src/map-manifest.ts`,
   mirroring `toolbox/bricks/map_manifest.py` — same discovery rule, same basename
   semantics, same errors. Divergence between the two would number keyframes
   differently in the index and in the workbench, silently.

3. **Its routes keep speaking WDS.** `point_world_wds` is a wire field the frontend
   round-trips, so rather than convert every payload to EUS, one adapter
   (`manifestKeyframeToWorldToCameraWds`) puts a manifest pose back into the WDS
   world-to-camera frame the existing math expects. This is the v1 conversion run
   backwards, and it is the single place that convention now survives.

   The cost is accepted knowingly: WDS exists nowhere else in the system. The
   alternative — changing the meaning of a field crossing the network in both
   directions — would make a stale browser tab send WDS to an EUS endpoint and get a
   plausible wrong answer back.

4. **Asset resolution is by manifest basename only.** No `{keyframe_id}.tif`, no
   `{keyframe_id}.jpg`, no `int(stem)` id fallback. Images live in `images/`;
   `images_360/` is still searched by the TS backend only because a wrong
   *directory* can only fail loudly, whereas a wrong *filename* fails silently.

5. **`--venue` and `--geo-ref-id` are removed**, not kept as overrides. The manifest
   is the single source, and it is also what `ingest_cli` and `service` read, so the
   two cannot disagree. A stale `geo_ref_id` in a toolbox config is ignored with a
   warning.

6. **Zarr depth support is removed** from the TS backend. The pipeline's depth
   format is the frozen sqrt-quantised uint16 TIFF; the Zarr reader belonged to the
   standalone lineage.

## Consequences

**A v1 map now fails loudly.** `load_pose_source` raises `FileNotFoundError` naming
the expected filename, and the workbench reports the parse error in
`unavailable_reason` instead of showing the map as available and then 500ing inside
a panel. That is the intended outcome: nothing degrades quietly.

**Two vendored-code deltas are recorded** in `toolbox/bricks/vendored/PROVENANCE.md`.
`GeoTransform.from_georef_db` is gone, which leaves the vendored class with no
factory corresponding to upstream's `from_geo_ref` — a new divergence that a
re-sync would otherwise read as a regression. The GeoJSON ray-cast moved from the
deleted `toolbox/georef/` into `geo_transform.py` itself.

**What this does *not* fix.** The retired `object-search.db` index-explorer routes
still return 501; that is the pgvector migration (ADR 0002), an orthogonal axis. Only
`/keyframes/{id}`, depth-preview, view-cone and project-world-point become v2-native.
The cutout and ERP-preview routes remain 501 because they depend on the retired
index, not on the map format.

**Keyframe ids stayed indices, and that is now the sharpest remaining edge.** A
bookmarked URL or saved annotation holding a v1 `GeoRefKeyframe.id` resolves to
whatever keyframe occupies that index — not a 404. `/keyframes/{id}` therefore
reports `image_filename`, which is the only thing that makes such a mismatch
checkable.

**Tests moved rather than shrank.** `test_frame_conventions.py` and
`test_keyframe_id.py` are gone; `test_manifest_frames.py` keeps the two assertions
that still bite (the geodesy-vs-rotation agreement on the EUS axes, and the level
datum), and `map-manifest.test.ts` carries the frame battery to the one place a
conversion still happens. `test_prepare_postprocess.py` is new — depth resolution
had no coverage at all before. The TS suite no longer needs a `sqlite3` binary.
