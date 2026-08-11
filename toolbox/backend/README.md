# Object Search Toolbox Backend

Local TypeScript backend for the object-search toolbox UI.

The Python services own object-search search/localization only (the bricks
service on 45678, backed by the mirrored online service on 45677 — see
`../README.md`). This backend serves the UI and owns the local workbench /
explorer metadata and preview routes.

Object Search Explorer routes, including image-based annotation metadata and
inverse projection, are implemented in TypeScript here and call no Python
service.

## Route Ownership

Owned by this backend:

- `GET /health`
- `GET /ui/api/maps`
- `GET /workbench/api/maps`
- `/ui/api/maps/:mapId/...` workbench routes, including depth projection,
  keyframe graph, view cones and local annotation map metadata. The
  `object-search-index/*` subset reads the retired standalone `object-search.db`
  and returns `501` when absent; everything else is backed by the map's v2
  manifest, read by `src/map-manifest.ts`.
- `POST /ui/api/maps/:mapId/annotations`, which writes the supplied
  FeatureCollection to `{map.path}/annotations/annotations.geojson`
- `GET /ui/api/maps/:mapId/review-annotations` and
  `POST|DELETE …/detection-review`, which own the integrated
  `{map.path}/object-search-annotations.db` store
- `GET /ui/*` static frontend files from `object-search-toolbox/frontend/dist`

Proxied unchanged to the bricks service:

- `/:mapId/object-search/...`

The standalone service's unprefixed `/object-search/encode-text` and
`/encode-image` are gone — the mirrored pipeline has no such endpoints.

## Run From Toolbox Root

Run the backend:

```bash
cd object-search-toolbox
npm install
npm run dev:backend -- --config /path/to/config.json5 --python-api http://127.0.0.1:45678
```

The Object Search Explorer, including annotation mode, only needs this backend.
Search and localization additionally need the bricks service (spawned on demand
on `45678`) and the mirrored online service (start it yourself:
`../../scripts/run-online-service.sh`).

Depth-pin projection in Object Search Explorer is owned by this backend, but it
decodes uint16 TIFF depth maps (and pre-migration Blosc/Zarr chunks) through a
local Python environment. Install `tifffile`, plus `numcodecs` if you still have
Zarr depth. Override the executable if needed:

```bash
OBJECT_SEARCH_WORKBENCH_PYTHON=/path/to/python npm run dev:backend -- --config /path/to/config.json5
```

Then open:

```text
http://127.0.0.1:45700/ui/
```

The frontend Vite dev server can also target this backend:

```bash
cd object-search-toolbox
VITE_API_PROXY_TARGET=http://127.0.0.1:45700 npm run dev:frontend
```
