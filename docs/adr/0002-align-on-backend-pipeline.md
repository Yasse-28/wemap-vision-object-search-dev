# ADR 0002 — Align the platform on the production pipeline

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0001](0001-object-search-platform-structure.md) (three decisions superseded, see below)
- **Amended by:** [ADR 0004](0004-v2-map-data-only.md) — decision 3's second pose
  format (`georef.db`) is removed; the v2 manifest is now the only one.

## Context

This repository is the **standalone v1.5 lineage**: the research codebase where
object search was built. It was the *source* of the port into
`wemap-vision-backend`, where the pipeline was then productionised — see that
repo's `backend/object_search/V1_5_STANDALONE_PORT.md`, which is addressed to the
maintainer of this one. The two repositories share **no git history**.

Since that port, the two lineages diverged, and production became the truth:

| | standalone (`dev/0.2.0`) | backend (production) |
|---|---|---|
| Index | SQLite `.db` + exact numpy cosine | Postgres/pgvector `object_search_candidate` + partial HNSW per georef |
| Cutouts | cubemap, 6 faces per keyframe | gnomonic θ/φ projection, one per detector proposal |
| Detection | `hybrid_detector.py` | multi-pass YOLO-World + GroundingDINO, pooled by NMS |
| OCR | PaddleOCR, ~1 600 lines, OCR-weighted ranking | **absent** — never ported |
| Visual refine | `visual_refine.py` | absent |
| Depth | zarr + TIFF | frozen sqrt-quantised uint16 TIFF |

Keeping both lineages alive means every experiment here has to be re-derived
before it can ship, and every production fix has to be hand-translated back. The
repo had stopped being a faithful place to iterate on the real service.

## Decision

**The mirrored trees are copies of production, and production owns them.**

1. `third_party/object_search/{prepare,inference,indexing,benchmarks,annotation_service}`
   and `services/object_search_online/` are **byte-for-byte copies** of the
   backend, at the backend's own paths. Do not edit them here; fix the backend and
   re-sync. `scripts/check-mirror.sh` enforces this, and
   `third_party/PROVENANCE.md` records the sync point.

2. **Django is not mirrored** — but it is not merely plumbing. It owns five things
   the pipeline cannot work without, so those are **ported to pure Python under
   `toolbox/bricks/`**, from the backend, keeping the algorithms behaviourally
   identical:

   | Brick | Ported from |
   |---|---|
   | 3D lifting + binary COPY + partial HNSW | `db/ingest.py`, `object_search_ingest.py` |
   | candidate enrichment | `candidates.py` |
   | leader-canopy clustering + ranking | `v1_5_logic.py` |
   | keyframe-id resolution before prepare | `object_search_prepare.py` (`image_entries`) |
   | `thumbnail_key` + `depth` bridge | `object_search_prepare.py::_sample_depths` |

   They live under `toolbox/` rather than beside the mirror because they are
   **dev-only**: production gets them from Django.

3. **Poses come from the map directory**, not the ORM, so the pipeline runs against a
   plain checkout of map data with no Django install. Two formats:
   - **v2 (current)** — `{map_id}_{version}_{date}_{time}.json`, a dump of the
     production objects: EUS positions, `[w, x, y, z]` orientations, level bands,
     `venue_type` and the real `geo_ref_id`. **No frame conversion applies.**
   - **v1 (legacy)** — `georef.db`, read by `toolbox/georef/` (salvaged from the
     standalone), whose poses need three composing frame flips.
     **Removed by [ADR 0004](0004-v2-map-data-only.md).**

   `load_pose_source` prefers the manifest. Its keyframe ids are `geo_keyframes`
   indices, which is safe because prepare and ingest read the same file — but it means
   a re-export renumbers them, so both must be re-run together.

4. **Nothing is deleted; the standalone lineage moves to `legacy/`**, excluded from
   packaging, tests, lint and mypy. See `legacy/README.md` for the per-area
   rationale.

5. **`consumers/` is removed.** Both submodules are gone, along with `.gitmodules`
   and `scripts/sync-consumers.sh`. `annotation_service` — which the backend
   already runs — becomes the owner of annotation ground truth, replacing
   wemap-vision-tools on this side.

### What went to `legacy/`, and why

| Retired | Why |
|---|---|
| OCR (models, refine, scoring) | Never reached production. Text ranking there is pure MetaCLIP2 cosine. |
| Visual refine | Never reached production. |
| SQLite index, in-RAM cosine | Replaced by pgvector + HNSW. |
| `build_index.py`, cubemap extraction | Replaced by `prepare/`; production dropped cubemap faces entirely — every indexed row is a detector proposal. |
| Venue YAML configs | Replaced by `prepare/prompts.py`, keyed on `Map.venue_type`. |
| Standalone online service (17 files) | Replaced by the mirrored service plus the bricks. |
| 22 of 24 test files | They test the above. |

The research value here is real, which is why this is an archive and not a
deletion — OCR in particular is a plausible future direction. But it is archived
honestly: nothing in `legacy/` is maintained or kept compiling.

## Amendments to ADR 0001

Three of its decisions no longer hold:

1. **"Pipeline package import path — keep `pipeline.*`."** Superseded. The
   importable names are now `prepare`, `inference` and `indexing`, because that is
   what production puts on `PYTHONPATH`. Renaming them would break the mirror,
   which is the whole point. `pipeline/` no longer exists.
2. **"`consumers/` as submodules."** Superseded — removed (decision 5 above). ADR
   0001 argued the inversion was acceptable because it cost only a checkout
   convenience; with ground truth moving to `annotation_service`, it stopped
   buying even that.
3. **"The toolbox's `python -m pipeline.online.app` spawn contract."** Superseded.
   The toolbox now spawns `toolbox.bricks.service`, and deliberately does *not*
   spawn the mirrored online service.

ADR 0001's core decision — that this repo plays the **platform** role, owning the
service and the toolbox — stands unchanged.

## Consequences

**Positive**
- Porting between the two repos becomes a `diff`, not archaeology.
- The pipeline that runs here is the pipeline that runs in production, so a result
  measured here means something.
- The chain runs locally end to end: prepare → postprocess → ingest → HNSW → serve
  → localize → benchmark. Its second half (ingest → HNSW → enrichment → localize) is
  covered by integration tests against a real Postgres; the `prepare` half needs a
  GPU, MetaCLIP weights and map data, so it has **not** been exercised on the
  migration branch — that is the first thing to run on a machine that has them.

**Negative / costs accepted**
- **Test coverage dropped sharply**: 2 of 24 files survived (~5% of 4 606 lines).
  Partly offset by 77 new tests over the bricks — the algorithms production has none
  for — of which 10 run against a real Postgres and are skipped otherwise. The
  offline build path (`prepare`) is nonetheless untested here; it is covered in the
  backend.
- **Two services to run, not one.** The bricks service is spawned for you; the
  mirrored online service you start yourself, because it loads MetaCLIP on the GPU.
- **PostGIS is now required locally.** `pgvector/pgvector:pg17` does not ship it,
  so `infra/postgres/` builds its own image.
- **Toolbox degradations** from the retired SQLite index: the index-explorer panel
  answers `501`, and `localize-offline` is gone — "offline" meant exact cosine over
  an in-RAM index, a distinction pgvector HNSW removes. Text search was degraded too
  and has since been re-wired: it reads the rows the bricks `text` endpoint already
  enriched, so it no longer touches the index (`enrichedFromCandidates` in
  `toolbox/frontend/src/object-search/api.ts`). Cutout thumbnails now come from the
  path-based preview route, since the index-backed one existed only to draw detection
  boxes on a crop that is already the cutout.
- The vendored helpers under `toolbox/bricks/vendored/` are copies that must be
  re-synced by hand; `PROVENANCE.md` there lists each one and its intended delta.
- **One mirrored function is overridden at runtime.**
  `toolbox/bricks/vendored/proposal_cutouts.py` replaces the mirror's
  `create_proposal_cutouts` via `install()`, which rebinds the name in
  `prepare.pipeline`. The mirror file is untouched and `check-mirror.sh` still passes,
  but this is the one place where what runs locally is not what the mirror says. It
  copies only the orchestration; all projection maths is imported from the mirror.

## Owed to wemap-vision-backend

**`prepare/proposal_cutouts.py` keeps two replicated ERPs alive at once.**
`img_batch = img.repeat(grid_chunk.shape[0], 1, 1, 1)` evaluates the new tensor
*before* rebinding the name, so the previous iteration's copy is still referenced. At
`BATCH = 10` on a 5760×2880 ERP that is 2 × 1.85 GiB instead of 1.85, and it OOMs an
8 GB card outright. Wasted peak memory on any GPU.

The fix upstream is two lines: `del img_batch` at the end of the loop body, and
`BATCH` promoted to a `PrepareConfig` field (keeping 10 as the default). Deliberately
not filed yet — the maintainer's call — so the workaround lives here meanwhile. When
it lands upstream, delete the vendored file and the `install()` call in
`prepare_runner.run`.

Note the `del` is necessary but not sufficient across a run: freeing and
re-allocating 1.85 GiB blocks fragments the caching allocator, so a later image can
fail with 2.14 GiB reserved-but-unallocated. On 8 GB, pair the default with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or pass `--cutout-batch 4`.
Verified over 8 real keyframes: both produce identical `metadata.parquet` and the
same `embeddings.npy` SHA-256 as each other and as `--cutout-batch 2`.

**Neutral**
- Version bumped to 0.3.0. `main` and `dev/0.1.0`/`dev/0.2.0` are untouched, so
  the standalone lineage remains reachable in full.

## Traps worth knowing about

Four failure modes here are **silent** — they produce plausible wrong answers
rather than errors. Each is pinned by a test.

0. **Positional keyframe ids.** `python -m prepare` numbers keyframes with
   `enumerate`, so the `video_keyframe_id` it writes is a position in sorted-path
   order, not `GeoRefKeyframe.id`. Chain that into ingest and candidates attach to
   whichever keyframes happen to share those ids — every object in the wrong place,
   no error. Production avoids it by building real `(keyframe.id, path)` pairs in the
   Django command; `toolbox.bricks.prepare_runner` is the local counterpart, and
   `scripts/build-index.sh` calls it rather than the CLI.
1. **Missing depth.** `prepare` emits no `depth` column. Skip
   `toolbox.bricks.prepare_postprocess` and `bulk_copy` writes NULL, every
   `object_position` is NULL, enrichment filters every row, and `localize` returns
   an empty list that looks exactly like "the model found nothing".
2. **Frame conventions — v1 only.** `georef.db` poses are transposed,
   world-to-camera, and in WDS/OpenCV; the bricks need camera-to-world in EUS/OpenGL.
   Three flips compose; drop one and objects land mirrored or 180° off, with no error.
   v2 manifests sidestep this entirely. *(Superseded by ADR 0004: the v1 path is
   gone. The composition survives only as the TS backend's EUS→WDS adapter, pinned by
   `toolbox/backend/src/map-manifest.test.ts`; the EUS axis convention itself is
   pinned by `toolbox/tests/test_manifest_frames.py`.)*
3. **Level datum.** Level bands are heights *above the georef origin* in both formats.
   Feed `levels_for_altitudes` a WGS84 altitude instead of the EUS up coordinate
   and every level comes back `None`, which silently disables the
   level-compatibility guard and merges objects across floors.

Plus two hard stops, which at least fail loudly: `object_position` must be
declared `geometry(PointZ, 0)` — **SRID 0, not 4326**, the EWKB prefix hardcodes
it — and the angular columns must be `DOUBLE PRECISION`, because binary COPY
encodes at fixed widths with no server-side coercion. Both are asserted against a
live server in `toolbox/tests/test_integration_db.py`.
