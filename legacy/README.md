# `legacy/` — the standalone v1.5 lineage, archived

Everything here is the **pre-0.3.0 standalone pipeline**: the lineage this repo ran
before it was realigned on the production pipeline in `wemap-vision-backend`
(see [ADR 0002](../docs/adr/0002-align-on-backend-pipeline.md)).

It is kept for reference, not for use. It is **excluded from packaging, tests,
`ruff`, `black` and `mypy`** — nothing here is maintained, and it will not be kept
compiling. Do not import from it.

## Why each piece was retired

| Area | Files | Why |
|---|---|---|
| **OCR** | `core/models/ocr/`, `offline/refine/ocr_refine.py`, `online/ocr_scoring.py` | Never ported to production. Text ranking in prod is pure MetaCLIP2 cosine similarity; the OCR-weighted `match_score` variant exists only here. |
| **Visual refine** | `offline/refine/visual_refine.py` | Never ported to production. |
| **SQLite index** | `core/io.py`, `core/io_streaming.py`, `core/database.py`, `core/search_numpy.py`, `core/types.py` | Replaced by Postgres/pgvector (`object_search_candidate` + partial HNSW per georef). |
| **Offline build** | `offline/build_index.py`, `offline/ingest_pgvector.py`, `offline/{config,detect,schema,shared}/` | Replaced by `third_party/object_search/prepare/` (multi-pass YOLO-World + GroundingDINO, gnomonic cutouts). |
| **Cubemap extraction** | `offline/ingest/equirect_extract.py` | Prod dropped cubemap faces entirely — every indexed row is now a detector proposal. |
| **Venue YAML configs** | `config/` | Replaced by `prepare/prompts.py` (`VENUE_YOLO_VOCABS`, `VENUE_PROMPTS`), keyed on `Map.venue_type`. |
| **Standalone online service** | `online/` (17 files) | Replaced by `services/object_search_online/` (embed → HNSW, flat hit list) plus `toolbox/bricks/` for the geometry/clustering that Django owns in prod. |
| **Standalone MetaCLIP wrapper** | `core/models/` | Superseded by `inference/embedder.py`. |
| **Standalone localization** | `offline/localize/{localize_3d,clustering_online,cluster_cutout_membership,depth_io}.py` | Superseded by the ports in `toolbox/bricks/`. |

## Salvaged rather than archived

Four things were dev-only but still useful, so they moved to `toolbox/` instead:
`benchmark/` (HTTP benchmark), `offline/localize/georef.py` (the `georef.db`
reader, now the substitute for the Django ORM), `offline/ingest/keyframe_id.py`
+ `image_io.py`, and `core/logging.py`.

## Known-dead on arrival

`online/index_explorer_helpers.py` (480 lines) had **zero importers** when it was
archived — the index explorer was served by TypeScript (`workbench-index.ts`
shelling out to the `sqlite3` CLI), not by this module. There is no replacement to
look for.
