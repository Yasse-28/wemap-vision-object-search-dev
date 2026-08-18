# Object Search Toolbox

Local developer toolbox for inspecting object-search indexes, testing search
and localization, and annotating map data. It groups the React frontend and
the TypeScript workbench backend in one npm workspace.

## Layout

- `frontend/`: React and Vite UI.
- `backend/`: TypeScript workbench API, static file server, and Python API proxy.

Two separate Python services are required for the `/:mapId/object-search/...`
routes (see [ADR 0002](../docs/adr/0002-align-on-backend-pipeline.md)):

| Service | Port | What it does | Started by |
|---|---|---|---|
| `toolbox.bricks.service` | 45678 | `/{map_id}/object-search/localize` — enrichment, clustering, ranking. The dev-only stand-in for Django's object-search API. | this backend, on demand |
| `third_party/object_search/services/object_search_online` | 45677 | `/object-search/by-text\|by-image` — MetaCLIP embedding + HNSW. This is production's GPU service, mirrored verbatim. | you, via `scripts/run-online-service.sh` |

The toolbox never starts the second one: it loads MetaCLIP on the GPU, which is
not something to trigger behind a button press. It also needs the local database
(`docker compose -f ../infra/postgres/compose.yml up -d`).

## Features

After selecting a configured map, the UI provides:

- **Object Search**: run text searches, offline localization, and online
  localization; inspect scores, observations, keyframes, headings, and map
  positions.
- **Object Search Explorer**: inspect index metadata, keyframes, cutouts,
  detections, OCR, and previews. The livemap and photosphere are displayed
  side-by-side with a draggable splitter. Moving the photosphere updates its
  viewing arc on the map. Bounding boxes can be enabled with `Show boxes`.
  Its annotation mode creates reviewable point annotations by inverse-projecting
  photosphere or cutout clicks through the keyframe depth map.
- **Annotation**: run localized searches and review whole clusters or individual
  detections as correct/incorrect. Reviews are restored per raw query, saved to
  the map's `object-search-annotations.db` by the Toolbox backend, and support
  toolbar or keyboard undo/redo. No separate annotation service is required.
- **OS Data Explorer** *(legacy maps only)*: inspect indexed cutouts and latent
  data. It reads the standalone `object-search.db`, which is no longer produced —
  the index lives in pgvector now. Routes return `501` with an explanation for
  maps built after the migration. Prompt-latent computation was never ported.

Text search and localization both work against the live pgvector index. Offline
localization is gone — it meant exact cosine over an index held in RAM, which pgvector
HNSW replaced — so the mode toggle now offers Text search and Localize only.

Annotations live in `{map}/object-search-annotations.db`, beside the reviews and the
reference partition. They load when the map opens and every create, delete or class
rename is written straight through — there is no Save button, and nothing is held in
memory waiting to be lost.

**Points are stored as `ground_truth_point`: they are the benchmark's ground truth.**
The office is therefore its editor — it opens on the points the map already has, and
deleting one there removes it from what the benchmark measures against. Polygons go
to `annotation_polygon` and stay out of the ground truth, having no single position
to score a localization against.

Each point can carry an optional search prompt beside its class, and keeps its
resolved location, image-click provenance, normalized ERP coordinates and depth.
**Load GeoJSON** still imports a file — including the `annotations/annotations.geojson`
this store replaced, which is also imported once automatically the first time the
database is opened. **Save As...** still downloads a GeoJSON. Polygons are displayed
and preserved but new polygons cannot be drawn.

Keyframe images are resolved from `GeoRefKeyframe.image_filename` (or the
legacy `filename` column) when available, with `{keyframe_id}.jpg` used only as
a fallback. This supports maps whose image files are named with UUIDs.

## Install

```bash
cd object-search-toolbox
npm install
```

## Development

Run the workbench backend:

```bash
npm run dev:backend -- --config /path/to/config.json5
```

Both service URLs have working defaults (`--python-api http://127.0.0.1:45678`,
`--ann-api http://127.0.0.1:45677`); pass them only to override.

The config must contain a `maps` array. Relative map paths are resolved from
the config file directory:

```json5
{
  maps: [
    {
      id: "example-map",
      path: "./maps/example-map",
      emmid: 123,
      // LEGACY, ignored: the georef id now comes from the map's v2 manifest,
      // which is also where ingest took it. The python service warns if set.
      geo_ref_id: 1,
      // LEGACY, optional: only the OS Data Explorer reads this.
      object_search_index_path: "/absolute/path/to/object-search.db",
    },
  ],
}
```

Run the Vite frontend against the backend:

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:45700 npm run dev:frontend
```

Open the Vite UI at `http://127.0.0.1:5173/ui/`.

To use the backend-served UI instead, build first and open:

```text
http://127.0.0.1:45700/ui/
```

The Object Search Explorer decodes depth maps through a local Python
environment. Install `tifffile` for the uint16 TIFF depth the pipeline uses
(`numcodecs` too, if you still have Zarr depth from before the migration), then
select the interpreter when needed:

```bash
OBJECT_SEARCH_WORKBENCH_PYTHON=/path/to/python \
  npm run dev:backend -- --config /path/to/config.json5
```

## Build

```bash
npm run build
npm run build:frontend
npm run build:backend
```

Start the compiled backend after building:

```bash
npm run start -- --config /path/to/config.json5 --python-api http://127.0.0.1:45678
```

## Architecture Boundary

The TypeScript backend owns `/ui/api/...` workbench routes, serves the frontend,
and owns each map's `object-search-annotations.db`. It proxies object-search model
routes to the Python service without implementing ranking, embedding, or
localization itself. Before a benchmark, it exports ground truth from its SQLite
store to `benchmark/annotations.geojson` atomically.

This toolbox is dev-only and has no production counterpart. The contracts it
must match are the mirrored trees in `../third_party/object_search/` and
`../third_party/object_search/services/object_search_online/`, which are byte-for-byte copies of
`wemap-vision-backend`.
