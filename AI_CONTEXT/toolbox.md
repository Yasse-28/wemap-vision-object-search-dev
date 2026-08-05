# toolbox — dev/test/annotate/benchmark tool

**Purpose:** local TypeScript app (React frontend + Node backend) to inspect
indexes, run/annotate/benchmark search. It **proxies** the Python services and
implements no ML, ranking, embedding, or localization itself.

The Python half of the toolbox (`toolbox/bricks`, `toolbox/georef`,
`toolbox/benchmark`) is documented separately in [`bricks.md`](bricks.md).

**Read this when:** changing the workbench UI/API, the index explorer,
annotation, or the benchmark runner.

## Backend (`toolbox/backend/src/`, plain Node http, run with `tsx`)

| Path | Responsibility | Key symbols |
|---|---|---|
| `main.ts` | Entry: HTTP server; routes `/ui/api/...`, serves UI, proxies `/{map_id}/object-search/…` to the bricks service | `server.listen(options.port…)`, `isObjectSearchRoute`, `mapSummaries`, `proxyRequest` |
| `http-utils.ts` | Options/args, static serving, proxy | `parseArgs`, `WorkbenchOptions`; `pythonApiBaseUrl` = **bricks** service (`OBJECT_SEARCH_PYTHON_API`, :45678), `annApiBaseUrl` = **mirrored online** service (`OBJECT_SEARCH_ANN_URL`, :8000), `annotationBaseUrl` (`OBJECT_SEARCH_ANNOTATION_URL`, :8001), `repoRoot`, `OBJECT_SEARCH_WORKBENCH_PORT=45700` |
| `config.ts` | Load map config | `loadMapEntries`, `loadGlobalObjectSearch`, `MapEntry` (`id, path, emmid, geo_ref_id, object_search_index_path` ⟵ legacy, `object_search`) |
| `workbench-index.ts` | Two halves. **`georef.db`-backed** (keyframe graph, depth pin, view cone, world-point projection, `previewFromPathPng`) — fine. **`object-search.db`-backed** (index explorer) — the SQLite index is retired, so `requireIndex` throws **501** with an explanation. | `indexStatusPayload`, `indexObjectsPayload`, `indexCutoutPayload`, `cutoutDetectionsPayload`, `keyframeMetadataPayload`, `previewFromPathPng`, `resolveIndexPath`, `requireIndex`, `keyframeHeadingDegreesFromPose`; can shell to a Python interp for TIFF/Zarr depth decode (`OBJECT_SEARCH_WORKBENCH_PYTHON`) |
| `workbench-api.ts` | `/ui/api/...` route handling | `isWorkbenchUiMapRoute`, `handleWorkbenchUiMapRoute` |
| `keyframe-graph.ts` | 360-viewer graph payload | `parseKeyframeGraph`, `keyframeGraphPayload` |
| `benchmark-runner.ts` | Run the HTTP benchmark | **Spawns the bricks service** (`python -m toolbox.bricks.service --config <the toolbox config>`) when unreachable, and keeps it alive across runs. **Only health-checks the mirrored online service** — never spawns it (it loads MetaCLIP on the GPU); `assertAnnServiceReachable` fails early with instructions. Before each run **GETs `{annotationBaseUrl}/{mapId}/object-search/ground-truth`** and writes it to `{map}/benchmark/annotations.geojson` atomically, validating it is a FeatureCollection first (best-effort: falls back to the file on disk). Spawns `toolbox/benchmark/object_search_http_benchmark.py` with `cwd=repoRoot` and `PYTHONPATH` set by `pythonEnv()`. Results in `{map_path}/benchmark/{runId}/`. `startBenchmarkRun`, `benchmark{Run,Status}Payload` |

## Frontend (`toolbox/frontend/src/`, React 19 + Vite + three.js)

Panels (each in its own dir with `api.ts` + `types.ts`):
- `object-search/` — **text search and online localization both work.** Text search
  builds its rows from the bricks `text` endpoint's already-enriched `candidates`
  (`enrichedFromCandidates`), so it no longer touches the retired SQLite index.
  `localize-offline` is gone: the mode toggle no longer offers it, and the bricks
  service answers 501 for anything still calling the path. `ObjectSearchPanel.tsx`.
- `object-search-explorer/` — index metadata, keyframes, cutouts, detections,
  previews; livemap + photosphere side-by-side; bbox post-process controls;
  depth-based annotation. **Its data came from `object-search.db`, so most of it is
  legacy-maps-only.** (`ObjectSearchExplorerPanel.tsx`, `EquirectPhotoSphereViewer.tsx`, `bboxPostProcess.ts`).
- `index-explorer/` — `api.ts` + `types.ts` are imported widely and live;
  `IndexExplorerPanel.tsx` and `LatentScatter.tsx` are **never mounted** by
  `App.tsx` (dead before this migration, still dead).
- `annotations/` — point/polygon annotations → `annotations/annotations.geojson`
  (`ExplorerAnnotationWorkspace.tsx`, `geojson.ts`, `LivemapAnnotation.tsx`,
  `livemapHost.ts`).
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
- **Map availability no longer means "has an object-search.db".** It now means the
  map directory and a **pose source** exist — a v2 manifest or a legacy `georef.db`.
  A map that passes but was never ingested shows up as an empty result list, not an
  error; that is the one honest gap in the signal, which is why
  `legacy_index_available` and `georef_db_available` are reported separately.
- **The TS `georef.db` readers have no v2 equivalent yet.** `keyframe-graph.ts` and the
  `georef.db` half of `workbench-index.ts` (depth pin, view cone, world-point
  projection) parse that SQLite file directly, so those routes are v1-only. The
  `georef_db_available` flag exists so the UI can say so rather than just fail.
- The standalone's unprefixed `/object-search/encode-text` and `/encode-image` are
  gone; the mirrored pipeline has no such endpoints.
- Dev-only, with no production counterpart. The contracts to validate against are the
  mirrored trees, which are copies of `wemap-vision-backend`.
