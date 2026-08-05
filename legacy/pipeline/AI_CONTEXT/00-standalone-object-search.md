# Standalone object search (sandbox)

Scope: `object-search/pipeline/`  
Read this when: working on the minimal file-backed object-search service  
Last updated: 2026-05-21

## Purpose and boundaries

This package is an **experimental** replacement for coupling object search to the GeoPose service layout (`MapManager`, map-scoped URLs, pgvector). It is **not** production canon; compare behavior with `wemap-vision-python` when in doubt.

- **Offline**: `python -m pipeline.offline.build_index` builds a single **`object-search.db`** (SQLite) index from equirectangular images under the **map directory** (`images_360`); it also ensures `object-search-history.db` under that map.
- **Online**: `python -m pipeline.online.app` reads a **GeoPose-style JSON5 config** (`--config_file_path`), loads one index + history per map, and serves the standalone object-search endpoints.

## Layout

| Path | Role |
|------|------|
| `core/` | `types` (incl. `ObjectSearchIndexMetadata`, `LoadedIndex`), `io` (load/save `object-search.db`), `database` (DDL + `index_metadata_json` migration), `search_numpy`, `models/` (MetaCLIP, GroundingDINO, OCR) |
| `offline/ingest/` | Image list loading (`image_io`) and equirectangular cutout extraction (`equirect_extract`) |
| `offline/detect/` | `object_cutouts` (uses `core.models` + `core.detectors` for GroundingDINO and post-process) |
| `offline/localize/` | Depth, georef, 3D localization (`localize_3d`, `depth_io`, `georef`) |
| `offline/refine/` | Visual and OCR cluster refinement |
| `offline/config/` | YAML loader for `offline`, `keyframe_filters`, `gdino_params`, and eval-style `experiments` (`prompts_yaml`) |
| `offline/schema/` | `CutoutRecord` and cutout id encoding (`cutout_schema`) |
| `offline/shared/` | Shared geometry / projection helpers (`geometry`) |
| `offline/build_index.py` | CLI entry and orchestration (map-centric paths) |
| `online/config_loader.py` | Parse config JSON, resolve `map_path` per map |
| `online/map_registry.py` | Per-map `FileBackedObjectSearchService` + `ObjectSearchHistoryService` |
| `online/app.py` | FastAPI + argparse + `uvicorn.run` |
| `example_object_search_config.json` | Sample online map config (copy and edit) |
| `tests/` | Unit tests without GPU |

## Directory layout (GeoPose-style)

Place a config file anywhere (e.g. `.../deploy/config.json`). Its parent is the **config root**.

- Default map path for entry `{ "id": "my-map" }`: **`{config.parent}/maps/my-map/`**
- With `{ "id": "x", "path": "maps/custom" }`: **`{config.parent}/maps/custom/`** (path relative to config parent, same idea as GeoPose `config_folder / map_["path"]`).

Per **map directory** on disk:

| Path | Role |
|------|------|
| `images_360/` | Equirectangular keyframe images (**integer stem = `keyframe_id`**). |
| `object_search_prompts.yaml` | Default offline build config filename. Override path with `--config`. |
| **`object-search.db`** | SQLite file-backed object-search index (offline build output). |
| **`object-search-history.db`** | SQLite request history (created at offline index time and appended by the API). |

## Offline Build YAML

`build_index.py` reads `{map_path}/object_search_prompts.yaml` by default, or the
explicit YAML passed with `--config`.

Supported top-level sections:

- `offline`: device, object skipping, inference, geometry, localization, `visual_refinement`, and `ocr_refinement`
- `keyframe_filters`: optional level and polygon filters
- `gdino_params`: GroundingDINO detection and post-process parameters
- `experiments`: optional eval-style prompt entries selected with `--experiment`

## GroundingDINO Config (`gdino_params`)

Minimal shape (all detection/post-process knobs under `gdino_params`):

```yaml
gdino_params:
  prompts:
    - "object . sign ."
    - "bench . trash bin ."
  score: 0.10              # optional → GroundingDINO processor threshold (default model 0.15 if omitted)
  iou_threshold: 0.50      # optional; greedy NMS IoU cutoff (use 1.0 to effectively disable)
  min_area_ratio: 0.01     # optional; min bbox area / image area
  max_area_ratio: 0.50     # optional; max bbox area / image area
  min_confidence: 0.10     # optional; second filter after model scores
```

**Eval-style configs:** if the file has top-level `experiments` (list of dicts with `name`, `prompts`, `confidence_threshold`), pass **`--experiment <name>`** to select one entry (`confidence_threshold` maps to **`score`**; other fields use pipeline defaults).

**Skipping objects:** set `offline.skip_objects: true` in YAML to skip GroundingDINO
and write an empty object index.

## Offline pipeline

### Equirectangular source

1. Read images from **`{map_path}/images_360`**.
2. **Geometry** (default **`cubemap`**): six faces per keyframe, or **`horizon`** with horizon settings.
3. **Cutout IDs**: `cutout_id = keyframe_id * id_stride + local_index`.
4. **Objects** (unless `offline.skip_objects: true`): for each batch, run GroundingDINO **once per prompt**, merge raw boxes per cutout, then one **`_postprocess_detections`** pass (confidence, area, NMS), then MetaCLIP crops. Keys remain `(cutout_id, obj_idx)`.
5. Optional 3D localization runs when enabled and `depths/` plus `georef.db` are available.
6. Optional visual and OCR refinement run from `offline.visual_refinement` and `offline.ocr_refinement`.
7. Write **`object-search.db`** and ensure **`object-search-history.db`** under **`map_path`**.

### CLI summary

| Flag | Role |
|------|------|
| `--map_path` | **Map directory**; required |
| `--config` | Offline build config YAML (default `{map_path}/object_search_prompts.yaml`) |
| `--experiment` | Select named entry from eval-style `experiments` list |

Device, geometry, object skipping, localization, visual refinement, and OCR
refinement are YAML settings, not CLI flags.

```bash
cd object-search
PYTHONPATH=. python -m pipeline.offline.build_index \
  --map_path /path/to/maps/my-map-id \
  --config pipeline/config/config_station.yaml
```

### Index format (`object-search.db`)

The index is a SQLite database: `params` holds JSON **object-search index metadata** under **`index_metadata_json`** (legacy `manifest_json` is migrated on first load) plus optional packed arrays, plus relational `cutout`, `object`, and `cluster` tables.

| Data | Meaning |
|------|---------|
| **Metadata** (JSON in `params`) | `schema_version` 3, geometry, `source_images_dir`, `gdino_params_json`, … |
| Cutout rows | cutout ids, keyframe ids, centers, rotations, embeddings |
| Object rows | per-object bbox, embedding, optional localization and OCR fields |
| Cluster rows | centroids, observation counts, optional OCR summaries when enabled |

**`gdino_params_json`** stores the resolved GroundingDINO + post-process snapshot for reproducibility. **`build_params_json`** stores resolved non-model build parameters. **`object_detector_prompt`** is a short human-readable summary of the prompt list.

## Online pipeline

1. **CLI** (like GeoPose `main`):  
   `python -m pipeline.online.app --config_file_path /path/to/config.json [--host 0.0.0.0] [--port 45678] [--cors]`
2. **Startup**: For each map in config, resolve `map_path`, require `object-search.db` (or a path that resolves to it), open history at `map_path/object-search-history.db`.
3. **Text search**: `POST /{map_id}/object-search/text` with body `text`, `num_results`, `search_type`. Unknown `map_id` -> **404**.
4. **Current text route behavior**: FastAPI calls `FileBackedObjectSearchService.search`, which computes object and cutout similarities, merges results to keyframe IDs, and returns a list response. `search_with_router` exists, but the current text endpoint does not call it.
5. **Health**: `GET /health` returns `{"status": "ok", "maps": ["...", ...]}`.

```bash
cd object-search
PYTHONPATH=. python -m pipeline.online.app \
  --config_file_path /path/to/config.json \
  --host 0.0.0.0 --port 45678 --cors
```

Example config: see `example_object_search_config.json` in this package.

### Endpoint summary

| Endpoint | Role |
|----------|------|
| `GET /health` | Service status and configured map IDs |
| `POST /{map_id}/object-search/text` | Standalone text search |
| `POST /{map_id}/object-search/localize-offline` | Search object embeddings from a JSON text query or multipart image query and return stored 3D clusters |
| `POST /{map_id}/object-search/localize` | Search object embeddings and localize top matches at request time |
| `POST /object-search/encode-text` | Encode text prompts with MetaCLIP |
| `POST /object-search/encode-image` | Encode uploaded JPEG/PNG images with MetaCLIP |

## Standalone runtime

`pipeline/` now carries local copies or minimal equivalents of the runtime pieces
it previously imported from `object-search/wemap_vision`:

- `pipeline.core.models.metaclip` (model + device/`get_shared_metaclip` runtime), `pipeline.core.models.detection.grounding_dino`, and `pipeline.core.models.base_model.load_model`
- cubemap/equirectangular projection helpers in `pipeline.offline.shared.geometry`
- detection types/post-process in `pipeline.core.detectors`
- logging in `pipeline.core.logging`
- depth/georef helpers used by localization in `pipeline.offline.localize.depth_io` and `pipeline.offline.localize.georef`
- a dependency-light router in `pipeline.online.router`; `search_with_router` can use it, but the FastAPI text route currently calls `search`

## What changed vs GeoPose-backed search

- No full `Map` / reloc stack; only the SQLite object-search index and history per map.
- Config lists maps; paths default under `maps/{id}`.

## Future work

- Optional config hot-reload (GeoPose-style timer).
- Optional ANN when brute force is too slow.
- Richer post-processing (geo/depth).
