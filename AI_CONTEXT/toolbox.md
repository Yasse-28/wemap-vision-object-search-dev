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
| `http-utils.ts` | Options/args, static serving, proxy | `parseArgs`, `WorkbenchOptions`; `pythonApiBaseUrl` = **bricks** service (`OBJECT_SEARCH_PYTHON_API`, :45678), `annApiBaseUrl` = **mirrored online** service (`OBJECT_SEARCH_ANN_URL`, :8000), `repoRoot`, `OBJECT_SEARCH_WORKBENCH_PORT=45700` |
| `config.ts` | Load map config | `loadMapEntries`, `loadGlobalObjectSearch`, `MapEntry` (`id, path, emmid, geo_ref_id, object_search`) |
| `annotation-store.ts` | Integrated per-map SQLite annotation CRUD, legacy migration, one-shot benchmark GeoJSON import, and ground-truth assembly | `annotationDatabasePath`, `listAnnotations`, `upsertDetectionReview`, `deleteDetectionReview`, `buildGroundTruth` |
| `object-search-metadata.ts` | **Reads `{map}/object-search/metadata.parquet` natively** (`hyparquet`, pure JS). Replaces the retired `object-search.db` reader: no cutout→objects hierarchy, one row *is* one proposal and one detection. Caches by mtime and **drops the entry when the read fails**. | `resolveMetadataPath`, `loadMetadata`, `requireMetadata`, `rowsForKeyframe`, `rowByIndex`, `MetadataRow`, `LoadedMetadata`, `MetadataError` |
| `erp-geometry.ts` | The only place the four stored angle columns become something drawable. Replaces the cubemap algebra. | `erpBboxRatios`, `erpRectsWrapped`, `cutoutRatioToErpUv`, `assertEquirect2to1`, `gnomonicFfmpegFilter`, `paddingMask`, `isRenderableGnomonic`, `angularAreaDeg2` |
| `workbench-index.ts` | Manifest-backed routes (keyframe markers, depth pin, view cone, world-point projection, ERP + depth previews, `previewFromPathPng`) **plus** the parquet-backed row routes. The manifest half needs no metadata and must never be gated on it. | `objectSearchMetadataStatusPayload`, `metadataRowsPayload`, `metadataRowPayload`, `metadataRowRenderPng`, `indexKeyframeEquirectPreviewPng`, `indexDepthPinPayload`, `indexViewConePayload`, `indexProjectWorldPointPayload`, `keyframeMetadataPayload`, `previewFromPathPng`, `rowFilterParamsFromQuery`, `keyframeHeadingDegreesFromPose`; shells to a Python interp for uint16-TIFF depth decode (`OBJECT_SEARCH_WORKBENCH_PYTHON`) and to `ffmpeg`/`convert` for renders |
| `workbench-api.ts` | `/ui/api/...` route handling | `isWorkbenchUiMapRoute`, `handleWorkbenchUiMapRoute` |
| `keyframe-graph.ts` | 360-viewer graph payload, read straight from `360-viewer/graph.geojson` — no pose source involved | `parseKeyframeGraph`, `keyframeGraphPayload` |
| `map-manifest.ts` | **The v2 manifest reader**, mirroring `toolbox/bricks/map_manifest.py`, plus the EUS→WDS pose adapter every route below depends on | `loadMapManifest`, `findManifest`, `parseManifest`, `MANIFEST_PATTERN`, `assetBasename`, `formatLevel`, `quaternionToMatrix3`, `manifestKeyframeToWorldToCameraWds` |
| `benchmark-runner.ts` | Run the HTTP benchmark | **Spawns the bricks service** (`python -m toolbox.bricks.service --config <the toolbox config>`) when unreachable, and keeps it alive across runs. **Only health-checks the mirrored online service** — never spawns it (it loads MetaCLIP on the GPU); `assertAnnServiceReachable` fails early with instructions. Before each run exports the integrated annotation SQLite store to `{map}/benchmark/annotations.geojson` atomically (best-effort: falls back to the file on disk). Spawns `toolbox/benchmark/object_search_http_benchmark.py` with `cwd=repoRoot` and `PYTHONPATH` set by `pythonEnv()`. Results in `{map_path}/benchmark/{runId}/`. `startBenchmarkRun`, `benchmark{Run,Status}Payload` |

## Frontend (`toolbox/frontend/src/`, React 19 + Vite + three.js)

### Route surface (`/ui/api/maps/{map_id}/…`)

| Route | Notes |
|---|---|
| `GET /object-search-metadata` | Status + `summary.keyframes[]` + markers for **every** manifest keyframe. `summary` is null with no parquet; `coverage` is null when the bricks service does not answer. Never fails on either. |
| `GET /object-search-metadata/rows` | Flat row table: `keyframe_ids`, `offset`, `limit`, `detector_source`, `label`, `with_depth`. One request per explorer page. |
| `GET /object-search-metadata/rows/{row_index}` | One row + `preview_path` (its `thumbnail_key`) + `preview_debug`. |
| `GET /object-search-metadata/rows/{row_index}/render.png` | Re-render from the ERP (`size`, `fov_scale`). **Not** the default preview — 422 for a proposal spanning ≥180°. |
| `GET /object-search-metadata/keyframes/{id}/equirect-preview.png` | Bare ERP (`draw_boxes=false`, JPEG) or with reconstructed boxes (PNG). |
| `GET /object-search-metadata/keyframes/{id}/depth-preview.png`, `POST …/depth-pin`, `…/view-cone`, `…/project-world-point` | Manifest-only, except `depth-pin` with `projection: "cutout"`, which needs `row_index`. |
| `GET /object-search-metadata/keyframe-graph` | From `360-viewer/graph.geojson`. |
| `GET /preview.png?preview_path=` | Serves a file from the map directory. **The default preview** for a proposal (`thumbnail_key`) and for a search result. |
| `GET /review-annotations`, `POST\|DELETE /review-annotations/detection-review` | Integrated review annotations in `{map}/object-search-annotations.db`; no external service. |

The prefix was `/object-search-index`, which promised a database that no longer
exists; and `/cutouts/{id}/{preview.png,detections}`, `…/objects/{id}/crop.png`,
`…/cluster-ocr` and `…/prompt-latent` are gone.

Panels (each in its own dir with `api.ts` + `types.ts`):
- `object-search/` — **text search and online localization both work.** Text search
  builds its rows from the bricks `text` endpoint's already-enriched `candidates`
  (`enrichedFromCandidates`), so it no longer touches the retired SQLite index.
  `localize-offline` is gone: the mode toggle no longer offers it, and the bricks
  service answers 501 for anything still calling the path. `ObjectSearchPanel.tsx`.
- `object-search-explorer/` — proposals, keyframes, previews; livemap + photosphere
  side-by-side; bbox post-process controls; depth-based annotation. Reads the parquet
  rows through `/object-search-metadata/*`. Pagination is **keyframe-major** (one ERP
  and its proposals), and the panel renders as soon as the map is known — the
  metadata warning is scoped to the row table, not the panel
  (`ObjectSearchExplorerPanel.tsx`, `EquirectPhotoSphereViewer.tsx`, `bboxPostProcess.ts`).
- `index-explorer/` — `api.ts` + `types.ts` only, shared by both panels. The
  directory name is a leftover of `IndexExplorerPanel.tsx`/`LatentScatter.tsx`, which
  were never mounted and were deleted with this migration.
- `annotations/` — point/polygon annotations → `annotations/annotations.geojson`
  (`ExplorerAnnotationWorkspace.tsx`, `geojson.ts`, `LivemapAnnotation.tsx`,
  `livemapHost.ts`).
- `object-search-review/` — detection-review API client, review controls, and
  per-query TP/FP state with undo/redo whose history survives a re-search. Its
  per-query annotation list is independent of displayed results, and counters
  cover the whole query. Mounted through `ObjectSearchPanel` in the dedicated
  `/ui/maps/:mapId/annotation` tab; `annotation-store.ts` in the backend owns the
  compatible per-map SQLite file directly.
- `benchmark/` — run + view benchmark results (`BenchmarkPanel.tsx`).
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
- **Map availability means the map directory exists and its v2 manifest parses.** A
  map that passes but was never prepared or ingested shows up as an empty result list,
  not an error; the explorer's own status route is where that shows (`available`,
  `postprocessed`, `summary.coverage`).
- **"Has a pose", "has proposals" and "is in pgvector" are three different sets.**
  The manifest has 3674 keyframes; the parquet lists the ones prepare ran on; ingest
  prunes keyframes closer than 1.5 m. The panel reports all three separately and
  colours markers tri-state (indexed / pruned / unknown) — conflating them is the
  mistake this panel exists to prevent.
- **Area thresholds are square degrees, not ERP pixels.** A v2 proposal has an angular
  extent and no fixed raster. `bboxPostProcess.ts` bumped its `localStorage` key for
  that reason; the backend applies the same rule in `filterRows`.
- **Boxes are reconstructed from float16 angles** — good to a few ERP pixels, not
  exact. Say so in anything user-facing.
- **The stored thumbnail is the default preview**, not the re-render: it is the only
  image certain to be what MetaCLIP2 embedded. The re-render exists for maps prepared
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
