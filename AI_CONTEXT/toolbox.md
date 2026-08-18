# toolbox — dev/test/annotate/benchmark tool

**Purpose:** local TypeScript app (React frontend + Node backend) to inspect
indexes, run/annotate/benchmark search. It **proxies** the Python services and
implements no ML, ranking, embedding, or localization itself.

The Python half of the toolbox (`toolbox/bricks`, `toolbox/benchmark`) is
documented separately in [`bricks.md`](bricks.md).

**Read this when:** changing the workbench UI/API, the object-search explorer,
annotation, or the benchmark runner.

The annotation ownership boundary is recorded in
[ADR 0006](../docs/adr/0006-integrate-annotations-into-toolbox.md).

## Backend (`toolbox/backend/src/`, plain Node http, run with `tsx`)

| Path | Responsibility | Key symbols |
|---|---|---|
| `main.ts` | Entry: HTTP server; routes `/ui/api/...`, serves UI, proxies `/{map_id}/object-search/…` to the bricks service | `server.listen(options.port…)`, `isObjectSearchRoute`, `mapSummaries`, `proxyRequest` |
| `http-utils.ts` | Options/args, static serving, proxy | `parseArgs`, `WorkbenchOptions`; `pythonApiBaseUrl` = **bricks** service (`OBJECT_SEARCH_PYTHON_API`, :45678), `annApiBaseUrl` = **mirrored online** service (`OBJECT_SEARCH_ANN_URL`, :45677), `repoRoot`, `OBJECT_SEARCH_WORKBENCH_PORT=45700` |
| `config.ts` | Load map config, and add/remove one entry **textually** — the reader strips comments before `JSON.parse`, so re-serialising would delete every comment in the file. Keys are written quoted for the same reason: it is JSON-with-comments, not full JSON5. Backs up to `{config}.bak`, writes atomically, and refuses to guess when the `maps` array cannot be located. | `loadMapEntries`, `loadGlobalObjectSearch`, `appendMapEntry`, `removeMapEntry`, `MapEntry` (`id, path, emmid, geo_ref_id, object_search, parent_map`) |
| `annotation-store.ts` | Integrated per-map SQLite annotation CRUD, legacy migration, one-shot benchmark GeoJSON import, ground-truth assembly, and the reference detection partition | `annotationDatabasePath`, `listAnnotations`, `upsertDetectionReview`, `deleteDetectionReview`, `buildGroundTruth`, `listDetectionGroups`, `upsertDetectionGroupLabel`, `deleteDetectionGroupLabel` |
| `object-search-metadata.ts` | **Reads `{map}/object-search/metadata.parquet` natively** (`hyparquet`, pure JS) into typed numeric columns and dictionary-coded strings; rows are materialised only on demand. Replaces the retired `object-search.db` reader: no cutout→objects hierarchy, one row *is* one proposal and one detection. Caches two maps by mtime and **drops the entry when the read fails**. | `resolveMetadataPath`, `loadMetadata`, `requireMetadata`, `rowsForKeyframe`, `rowSlice`, `rowByIndex`, `MetadataRow`, `LoadedMetadata`, `MetadataError` |
| `erp-geometry.ts` | The only place the four stored angle columns become something drawable. Replaces the cubemap algebra. | `erpBboxRatios`, `erpRectsWrapped`, `cutoutRatioToErpUv`, `assertEquirect2to1`, `gnomonicFfmpegFilter`, `paddingMask`, `isRenderableGnomonic`, `angularAreaDeg2` |
| `workbench-index.ts` | Manifest-backed routes (keyframe markers, depth pin, view cone, world-point and proposal-neighbour projection, ERP + depth previews, `previewFromPathPng`) **plus** the parquet-backed row routes. The manifest half needs no metadata and must never be gated on it. | `objectSearchMetadataStatusPayload`, `metadataRowsPayload`, `metadataRowPayload`, `metadataRowRenderPng`, `proposalNeighborProjectionsPayload`, `proposalNeighborProjectionRenderPng`, `projectProposalIntoNearestKeyframes`, `featureMatchAssessment`, `appearanceSsimAssessment`, `indexKeyframeEquirectPreviewPng`, `indexDepthPinPayload`, `indexViewConePayload`, `indexProjectWorldPointPayload`, `keyframeMetadataPayload`, `previewFromPathPng`, `rowFilterParamsFromQuery`, `keyframeHeadingDegreesFromPose`; shells to a Python interp for uint16-TIFF depth decode and optional OpenCV SIFT/SSIM scoring (`OBJECT_SEARCH_WORKBENCH_PYTHON`), and to `ffmpeg`/`convert` for renders |
| `workbench-api.ts` | `/ui/api/...` route handling | `isWorkbenchUiMapRoute` (its trailing segment is optional, so `DELETE /ui/api/maps/{id}` reaches the same dispatcher), `handleWorkbenchUiMapRoute` |
| `python-process.ts` | One place for the interpreter search order and PYTHONPATH, which `benchmark-runner.ts` and `workbench-index.ts` each held a copy of | `pythonBinaryCandidates`, `pythonEnv`, `runPython`, `lastJsonLine` |
| `export-roi.ts` | ROI export + map deletion routes. Thin: the work is `toolbox/bricks/export_roi.py` and `delete_map.py` — the geometry lives beside the manifest reader, and this backend has no Postgres client. Job shape copied from `benchmark-runner.ts` (202 + poll), one export at a time so two runs cannot allocate the same `geo_ref_id`. The directory browser is **confined** to the source map's parent and the config directory. | `listDirectoriesPayload`, `createDirectoryPayload`, `exportPreviewPayload`, `startExportRun`, `exportStatusPayload`, `deletionPreviewPayload`, `deleteMapPayload` |
| `keyframe-graph.ts` | 360-viewer graph payload, read straight from `360-viewer/graph.geojson` — no pose source involved | `parseKeyframeGraph`, `keyframeGraphPayload` |
| `map-manifest.ts` | **The v2 manifest reader**, mirroring `toolbox/bricks/map_manifest.py`, plus the EUS→WDS pose adapter every route below depends on | `loadMapManifest`, `findManifest`, `parseManifest`, `MANIFEST_PATTERN`, `assetBasename`, `formatLevel`, `quaternionToMatrix3`, `manifestKeyframeToWorldToCameraWds` |
| `benchmark-runner.ts` | Run the HTTP benchmark and synchronously score one prompt | **Spawns the bricks service** (`python -m toolbox.bricks.service --config <the toolbox config>`) when unreachable, and keeps it alive across runs. **Only health-checks the mirrored online service** — never spawns it (it loads MetaCLIP on the GPU); `assertAnnServiceReachable` fails early with instructions. Before each run or prompt score exports the integrated annotation SQLite store to `{map}/benchmark/annotations.geojson` atomically (best-effort: falls back to the file on disk). Spawns `toolbox/benchmark/object_search_http_benchmark.py` with `cwd=repoRoot` and `PYTHONPATH` set by `pythonEnv()`. Full results live in `{map_path}/benchmark/{runId}/`; prompt scores live one level deeper under `benchmark/prompt-scores/<slug>/` so `listRuns` cannot expose them. | `startBenchmarkRun`, `scorePromptPayload`, `benchmark{Run,Status}Payload` |

## Frontend (`toolbox/frontend/src/`, React 19 + Vite + three.js)

### Route surface (`/ui/api/maps/{map_id}/…`)

| Route | Notes |
|---|---|
| `GET /object-search-metadata` | Eager status and aggregate counters only, plus `marker_count` and the numerically smallest `first_keyframe_id`. `summary` is null with no parquet; `coverage` is null when the bricks service does not answer. Never fails on either. |
| `GET /object-search-metadata/markers` | On-demand columnar manifest poses. Dense ids are omitted with `ids_are_dense: true`; levels use `levels` + `level_codes` (`-1` = null). |
| `GET /object-search-metadata/keyframes` | Server-paged columnar keyframe summaries with `offset`, `limit`, `sort`, `include_empty`, optional `keyframe_id`, and post-filter `total`. |
| `GET /object-search-metadata/rows` | Flat row table: `keyframe_ids`, `offset`, `limit`, `detector_source`, `label`, `with_depth`. One request per explorer page. |
| `GET /object-search-metadata/rows/{row_index}` | One row + `preview_path` (its `thumbnail_key`) + `preview_debug`. |
| `GET /object-search-metadata/rows/{row_index}/render.png` | Re-render from the ERP (`size`, `fov_scale`). **Not** the default preview — 422 for a proposal spanning ≥180°. |
| `GET /object-search-metadata/rows/{row_index}/neighbor-projections?count=N&diverse=true` | Lift the proposal centre through its stored depth and select N nearby manifest poses (1–12). `diverse=true` (default) applies the 0.5 m source/inter-result baseline when possible; `diverse=false` returns the strictly nearest poses without that baseline. Returns the applied `minimum_baseline_m`, projected centre/extent, explainable geometric confidence (source distance, viewpoint angle, apparent size), target-depth visibility confidence, COLMAP-style feature score, and appearance score. Features run SIFT on the centred proposal region, apply L2 2-NN ratio matching, verify a homography with RANSAC, and combine inlier ratio with verified support. Appearance runs SSIM on the exact proposal region after a bounded translation search, so textureless objects can still compare visually. Missing OpenCV/SIFT reports both scores `unavailable` without failing the projection. Target depth near the expected range is `clear`; nearer depth is `occluded`; farther depth is `depth_mismatch`; missing/undecodable depth stays `unknown` without failing the geometric result. |
| `GET /object-search-metadata/rows/{row_index}/neighbor-projections/{target_keyframe_id}.png` | Rectilinear context render (`size`, `fov_scale`) centred on one of those projected target views. |
| `GET /object-search-metadata/keyframes/{id}/equirect-preview.png` | Bare ERP (`draw_boxes=false`, JPEG) or with reconstructed boxes (PNG). |
| `GET /object-search-metadata/keyframes/{id}/depth-preview.png`, `POST …/depth-pin`, `…/view-cone`, `…/project-world-point` | Manifest-only, except `depth-pin` with `projection: "cutout"`, which needs `row_index`. |
| `GET /object-search-metadata/keyframe-graph` | From `360-viewer/graph.geojson`. |
| `GET /preview.png?preview_path=` | Serves a file from the map directory. **The default preview** for a proposal (`thumbnail_key`) and for a search result. `object-search/rows/{row_index}.png` is a **virtual** key with no file behind it: a v1-converted index has no crops, so the route re-renders that row from the ERP instead (`VIRTUAL_ROW_PREVIEW`). |
| `GET /review-annotations`, `POST\|DELETE /review-annotations/detection-review` | Integrated review annotations in `{map}/object-search-annotations.db`; no external service. |
| `GET /group-annotations`, `POST\|DELETE /group-annotations/label` | The **reference partition**: which detections are one physical object, in `detection_group_label` of the same DB. Query-independent (unlike a review), keyed by `(keyframe_id, theta_center, phi_center)` at 6 decimals, `group_name` NULL = "not an object". Exists to score an association algorithm's partition against a human's — pair F1, not mAP. Not part of the benchmark's ground truth. |
| `GET\|POST /export-roi/directories` | The save dialog's folder browser and its "New folder". Confined to the source map's parent directory and the config directory; a `..` that escapes is 403. |
| `POST /export-roi/preview` | Synchronous `--dry-run`: the **authoritative** keyframe count, the per-level split, and the allocated `geo_ref_id`. |
| `POST /export-roi`, `GET /export-roi/status` | 202 + poll, as the benchmark does. |
| `GET /deletion-preview`, `DELETE /ui/api/maps/{id}` | What deletion would remove, then the deletion. `DELETE` requires `{confirm_id}` equal to the map id — re-checked server-side, so the typed confirmation is not only cosmetic. 409 when the map is not a sub-map or still has children. |
| `POST /benchmark/score-prompt` | Runs the existing Python evaluator for one ground-truth prompt with the Annotation panel's current localization parameters; returns 404 when that prompt has no benchmark ground truth. |

The prefix was `/object-search-index`, which promised a database that no longer
exists; and `/cutouts/{id}/{preview.png,detections}`, `…/objects/{id}/crop.png`,
`…/cluster-ocr` and `…/prompt-latent` are gone.

Panels (each in its own dir with `api.ts` + `types.ts`):
- `object-search/` — **text search and online localization both work.** Text search
  builds its rows from the bricks `text` endpoint's already-enriched `candidates`
  (`enrichedFromCandidates`), so it no longer touches the retired SQLite index.
  `localize-offline` is gone: the mode toggle no longer offers it, and the bricks
  service answers 501 for anything still calling the path. `ObjectSearchPanel.tsx`.
  Its panorama pane has a "Bounding boxes" checkbox that overlays the keyframe's
  parquet detections (rows fetched only while it is on, raw — no post-process
  controls here); the ERP corner helper `bboxPolygonRatios` lives in
  `object-search-explorer/bboxPostProcess.ts` and is shared with the explorer.
  Hovering a box names its cluster and its rank inside it. The join is angular —
  `theta_center`/`phi_center` matched within 1e-3 rad against the observation, because
  pgvector carries no `row_index` and the observation `bbox` is a placeholder. It
  needs the two angles `toolbox/bricks/localize.py` adds to each observation, so it
  stays empty against a remote production endpoint.
- `object-search-explorer/` — proposals, keyframes, previews; livemap + photosphere
  side-by-side; bbox post-process controls; depth-based annotation. **"Show track"**
  (`keyframeTrack.ts`) joins consecutive keyframes in manifest order — the route the
  capture walked — for the maps with no `360-viewer/graph.geojson`, which is most of
  them. It is drawn as lines only: the keyframe under the cursor is revealed by the
  same hover mechanism the graph uses, so nothing renders ten thousand markers. Two
  cuts, both silent if wrong: a floor change (a segment carries one level and the
  livemap filters by floor) and a gap over 15 m (a break between captures, not a
  corridor). It relies on `/object-search-metadata/markers` staying in manifest
  order — sorted by id as a string, `"10"` precedes `"2"` and the track is noise. Reads the parquet
  rows through `/object-search-metadata/*`. Pagination is **keyframe-major** (one ERP
  and its proposals); its compact proposal grid can independently show the rows kept
  or discarded by the current bbox post-processing controls. Detector visibility
  (G-DINO / YOLO-W) sits beside those filters and remains active when geometric bbox
  post-processing is disabled; each detector's share of all raw proposals across the
  map appears below the proposal total in the top health strip. Cutout depth pins are placed only from the selected
  proposal's context preview with Ctrl+click; while Ctrl is held, the same red depth
  cursor as the panorama follows the pointer. Proposal cards only change selection.
  The selected-proposal inspector can project a depth-bearing proposal into 1–12
  nearest camera poses, preferring a 0.5 m baseline between views when the capture
  permits it: it lifts the source centre to 3D, transfers the proposal's approximate
  physical extent, and renders centred target cutouts. Clicking a target opens that
  keyframe in the panorama and faces the projected centre. Each card keeps two scores
  separate: geometry combines source distance, viewpoint angle and apparent size;
  visibility compares the projected distance with the target depth sample and names
  foreground occlusion separately from a farther depth mismatch. Hovering a score
  exposes its inputs; absent target depth is reported as such, never treated as zero.
  The panel renders as soon as the map is known — the
  metadata warning is scoped to the proposal grid, not the panel
  (`ObjectSearchExplorerPanel.tsx`, `EquirectPhotoSphereViewer.tsx`, `bboxPostProcess.ts`).
- `index-explorer/` — `api.ts` + `types.ts` only, shared by both panels. The
  directory name is a leftover of `IndexExplorerPanel.tsx`/`LatentScatter.tsx`, which
  were never mounted and were deleted with this migration.
- `matching/` — the **Matching tab** (`MatchingPanel.tsx`, route `/ui/maps/:mapId/matching`),
  the reference partition and the matching basket. The panel posts the basket to
  `/{map_id}/object-search/matching` and shows the two readings side by side: the
  cosine matrix and the triangulation's inlier/outlier split, with the parallax
  flagged below 10°, plus a livemap of the keyframes, their rays (fixed 30 m
  segments, coloured by inlier status), the depth-based positions and the
  triangulated point — the rays are what makes a weak parallax visible rather than
  merely reported. Its "Load group" picker pulls an annotated group's members into
  the basket — annotating and filling the basket are otherwise two unrelated
  gestures, and "does the group I just annotated hold up?" is the main question.
  The basket lives in `localStorage` (per map): `sessionStorage` is per browser tab
  and made it look empty on arrival. The panel scrolls as a whole — a fixed-height
  layout with only the basket scrolling was tried and reverted. `pairScore.ts` scores a
  partition against the annotated one (pair P/R/F1, over annotated detections only). It
  backs the Object Search cluster sidebar (clusters vs annotations). The Matching tab
  scores its partition with `partitionScore.ts` instead — **object level** (an object is
  recovered when some hypothesis carries its majority-vote label; duplicates cost
  precision) as the headline, **detection level** (majority-vote label per hypothesis)
  beside it, and a per-object table (recovered / split / merged / lost).
  Pair counting sits beside them again — not as a headline (it weights an error by the
  square of the blocks it mixes) but for `falseMergePairs`, the count to minimise
  first — together with VI split/merge in bits and the impure-cluster / split-object
  counts. Each headline score hides one failure: the detection score barely charges
  fragmentation, and a majority vote hides an impure cluster. The two metrics fail in
  opposite directions on purpose — a split leaves the detection score perfect and
  costs the object score. `groupAnnotations.ts`
  (client + `useGroupAnnotations`, key = `keyframe:theta:phi` rounded to **6**
  decimals, matching the backend) and `GroupPicker.tsx` drive `/group-annotations`;
  `basket.ts` holds the session-scoped set of detections an inspection runs over
  (item = a keyframe direction with a `rays` list, so a SAM2 mask fits later). Both
  are filled from the Object Search cluster list, which is why they live outside it.
- `export-roi/` — the selection session and the two dialogs. The ROI toolbar in the
  Explorer gained a **target**: *count annotations* (what it always did) or *select
  keyframes*. Drawing is unchanged — one polygon, `LivemapAnnotation`'s existing
  tool — and a closed ring in keyframe mode joins the session with the current level
  **frozen at close time**, which is the whole point: draw on one floor, switch,
  draw again. Saved regions ride the existing `polygons` prop, so Mapbox's per-floor
  filter displays them for free. The session lives in `localStorage` per map
  (`session.ts`), like `matching/basket.ts` and for the same reason — `App.tsx`
  mounts one panel at a time. `selection.ts` is the live client-side count;
  `mapTree.ts` nests sub-maps under their parent on the home page, tolerating a
  missing parent and a `parent_map` cycle. `ExportDialog.tsx` shows **two counts**:
  its own and python's, which resolves the floor from the manifest's altitude bands
  rather than from the Wemap SDK. A gap between them is displayed, not hidden — it
  would mean the two floor definitions disagree.
- `annotations/` — point/polygon annotations, **in `object-search-annotations.db`**
  (`api.ts`, `ExplorerAnnotationWorkspace.tsx`, `LivemapAnnotation.tsx`,
  `livemapHost.ts`; `geojson.ts` is now only the "Save As…" download). They used to
  go to `annotations/annotations.geojson`, a file **nothing ever read back**: the
  office opened empty every time, its work had to be re-imported by hand, and it
  never reached the benchmark. Points are now stored as `ground_truth_point` — they
  *are* the positional ground truth the benchmark scores against, so the office is
  its editor: opening it shows the map's existing points (258 on vinci, 674 on
  bbhotel) and deleting one removes it from the benchmark, which is why the UI says
  so. Polygons get `annotation_polygon`, out of the ground truth: they have no
  single position to score a localization against. The palette lives in
  `annotation_class`, so a class created and not yet drawn on survives a reload;
  classes met only in pre-existing ground truth are backfilled with a deterministic,
  collision-free colour. Loading is automatic and every create/delete/rename writes
  through — the "Save" button is gone, the file import stays as the way in for an
  old file. `canonicalLevel` and `pointInPolygon` moved out to
  `annotations/geometry.ts` when the export needed the same floor rule and the
  same ray cast; two copies that must agree with the backend's is exactly the
  drift this file exists to prevent.
- `object-search-review/` — detection-review API client, review controls, and
  per-query TP/FP state with undo/redo whose history survives a re-search. Its
  per-query annotation list is independent of displayed results, and counters
  cover the whole query. Mounted through `ObjectSearchPanel` in the dedicated
  `/ui/maps/:mapId/annotation` tab; `annotation-store.ts` in the backend owns the
  compatible per-map SQLite file directly.
- The object-search panel's "Online overrides" are exactly the fields
  `LocalizeParams` reads; `min_keyframes_per_cluster` defaults to **2** there, in the
  Benchmark tab and in the service, so the three paths build the same clusters. The
  four standalone-era knobs (`use_stored_positions`, `robust_centroid`,
  `embedding_similarity_threshold`, `include_debug`) were removed — nothing consumed
  them.
- `benchmark/` — run + view benchmark results (`BenchmarkPanel.tsx`). The run form
  exposes the review-feedback gains and `feedback_normalization`, and every stored
  run shows its own parameters through `config-summary.ts` — without that, a boosted
  run and a baseline are indistinguishable in the run list.
- The Annotation tab's review toolbar explicitly scores its current prompt through
  `benchmark/score-prompt` and compares it with the same prompt in the newest full
  run. That comparison is *not* guaranteed to be a baseline: it is simply the newest
  run, so both sides are labelled with their parameters. The score sends the panel's
  own `min_similarity` and the Sensitivity slider as `acceptance_threshold`, so it
  measures the clusters actually listed rather than the script's defaults.
  Its ✓/× detection reviews affect feedback only; they do not create the
  positional manual annotations used as benchmark ground truth.
- `App.tsx`, `main.tsx`, top-level `api.ts` — shell + shared client.

## Run

```bash
cd toolbox && npm install
npm run dev:backend -- --config /path/config.json5
VITE_API_PROXY_TARGET=http://127.0.0.1:45700 npm run dev:frontend   # UI at :5173/ui/
```

Search and localization additionally need the mirrored online service, which you
start yourself, plus the database:

```bash
docker compose -f infra/postgres/compose.yml up -d
scripts/run-online-service.sh
```

## Livemap cache (`annotations/livemapHost.ts`)

`App.tsx` renders one panel at a time, so switching tabs unmounts
`LivemapAnnotation`. The Wemap SDK is a page-level singleton
(`window.wemap.v1.getPrivateInterface`), so destroying/recreating the livemap
left a stale facade → blank map until a page reload. The livemap is therefore
created **once per `emmid`** in a detached `<div>` and cached:

- `acquireLivemapHost(emmid)` → cached `LivemapHost` (SDK loader, livemap
  instance, Mapbox map, `LivemapState`, shared `callbacks` slot); a different
  `emmid` destroys the previous one.
- Mount re-parents `host.el` into `.livemap-canvas`; unmount calls
  `resetConsumerState()` + clears the overlay sources, then
  `releaseLivemapHost()` parks `host.el` off-screen — **never `destroy()`**.
- SDK/Mapbox handlers are registered once per host and dispatch through
  `host.callbacks`, which the mounted component overwrites each render — they
  must never close over a component's props or its own state ref.
- Camera and current floor persist across tabs; overlays (markers, segments,
  cone, ROI) are re-synced from the incoming panel's props (`adoptProps`).

## Gotchas

- Backend is plain `node:http` (no framework); routes are matched manually in
  `workbench-api.ts` / `main.ts`.
- **A map's `geo_ref_id` comes from its manifest, and it partitions a shared table.**
  Deleting a map means `DELETE ... WHERE geo_ref_id = %s` on rows every map lives in,
  so both the export (allocate a free id) and the deletion (refuse a shared one) are
  built around that. See gotcha 12 in [`bricks.md`](bricks.md).
- **Only a map with `parent_map` is deletable from the UI.** A map received from
  production was not produced here and its pixels come from `../retrieve-map-data`;
  the repo could not rebuild it. The rule is enforced in `delete_map.py`, not in the
  React component.
- **Map availability means the map directory exists and its v2 manifest parses.** A
  map that passes but was never prepared or ingested shows up as an empty result list,
  not an error; the explorer's own status route is where that shows (`available`,
  `postprocessed`, `summary.coverage`).
- **"Has a pose", "has proposals" and "is in pgvector" are three different sets.**
  The manifest has 3674 keyframes; the parquet lists the ones prepare ran on; ingest
  prunes keyframes closer than 1.5 m. The panel reports all three separately and
  colours markers by indexed / pruned / not-prepared / unknown state — conflating them is the
  mistake this panel exists to prevent.
- **Area thresholds are square degrees, not ERP pixels.** A v2 proposal has an angular
  extent and no fixed raster. `bboxPostProcess.ts` bumped its `localStorage` key for
  that reason; the backend applies the same rule in `filterRows`.
- **Boxes are reconstructed from float16 angles** — good to a few ERP pixels, not
  exact. Say so in anything user-facing.
- **The stored thumbnail is the default preview**, not the re-render: it is the only
  image certain to be what MetaCLIP2 embedded. The exception is an index converted from
  v1, which has no thumbnails at all: there `thumbnail_key` is the virtual
  `object-search/rows/{row_index}.png` and every preview *is* a re-render, so it shows
  the stored angles re-projected rather than the pixels that were embedded (v1 cut its
  crops from cubemap faces). The re-render exists for maps prepared
  without crops and for a widened context view, and it differs from the thumbnail
  (which went through `build_padding_mask`, cropping wide boxes).
- **A proposal can span ≥180°** (a real GroundingDINO `handrail` at 357°). No
  rectilinear view of it exists and its stored thumbnail is degenerate too; the render
  route answers 422 rather than clamping to a different region.
- `src/erp-cutout-fidelity.test.ts` is the load-bearing geometry test: it re-renders
  real rows and cross-correlates against the thumbnails on disk. Opt-in via
  `OBJECT_SEARCH_TEST_MAP=/path/to/map`. Run it after touching `erp-geometry.ts`.
- **`point_world_wds` is WDS, and deliberately so.** The manifest is EUS, production
  is EUS, the bricks are EUS — but this backend's depth-pin / view-cone /
  project-world-point payloads speak WDS world-to-camera, and the frontend
  round-trips that field. `map-manifest.ts` adapts at the boundary instead, so the
  wire format never changed. Do not "modernise" it without changing both sides.
- **Keyframe ids are `geo_keyframes` indices.** An old bookmarked URL holding a v1
  `GeoRefKeyframe.id` will resolve to whatever sits at that index rather than 404,
  which is why `/keyframes/{id}` reports `image_filename`.
- The standalone's unprefixed `/object-search/encode-text` and `/encode-image` are
  gone; the mirrored pipeline has no such endpoints.
- Dev-only, with no production counterpart. The contracts to validate against are the
  mirrored trees, which are copies of `wemap-vision-backend`.
