# ADR 0006 — Integrate annotations into the Toolbox backend

- **Status:** Accepted
- **Date:** 2026-08-07
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0002](0002-align-on-backend-pipeline.md) — the production
  annotation service remains mirrored, but is no longer a runtime dependency of
  the local Toolbox.

## Context

The annotation review UI originally lived in `livemap-tools` and wrote to a
dedicated annotation service. Its production replacement is the mirrored Python
FastAPI service under `third_party/object_search/annotation_service/`, with one
SQLite database per map.

The first Toolbox port preserved that process boundary: the React UI called the
TypeScript backend, which proxied requests to the Python service on port 8001.
This made a lightweight local feature depend on a third independently started
process. A correctly started Toolbox therefore displayed `fetch failed` as soon
as a user opened the Annotation tab without also knowing to start that service.

The boundary bought little locally. The Toolbox backend already resolves the map
configuration, owns `/ui/api/maps/:mapId/...`, and orchestrates benchmark ground
truth. The annotation service performs no ML or remote work; for the review path
it is SQLite CRUD over a file inside the configured map directory.

Moving that CRUD introduces a real cost: production Python and the dev Toolbox
TypeScript become two implementations of the same SQLite contract. Schema,
migrations and wire behavior must remain compatible so existing databases open
unchanged and review feedback can still be consumed by `toolbox.bricks`.

## Decision

1. **The Toolbox TypeScript backend owns local annotation persistence.**
   `annotation-store.ts` opens `{map.path}/object-search-annotations.db` through
   `better-sqlite3` and implements the review CRUD used by the Annotation tab.

2. **The SQLite contract stays compatible with production.** The table names,
   review status values, unique key, timestamps, geometry encoding, legacy
   class-name migration, one-shot benchmark GeoJSON import, and ground-truth
   output follow the mirrored Python service. Compatibility tests pin review
   replacement, historical migration and GeoJSON import.

3. **One connection is cached per resolved database path.** Schema setup,
   migrations and the one-shot import run when the connection first opens, not
   on every HTTP request. Connections close on process exit. SQLite operations
   remain synchronous: writes are tiny, serialized local actions, and this keeps
   transactions and error propagation straightforward.

4. **The browser only calls the Toolbox.** Review endpoints live under
   `/ui/api/maps/:mapId/review-annotations`; `OBJECT_SEARCH_ANNOTATION_URL`,
   `--annotation-url`, CORS configuration and port 8001 leave the local runtime.

5. **The benchmark uses the same store directly.** Before a run, the backend
   builds the FeatureCollection from SQLite and atomically writes
   `benchmark/annotations.geojson`. A failed refresh keeps the existing file.

6. **`toolbox.bricks` reads by configured map path.** Feedback lookup receives
   `MapEntry.path`, so ranking reads the exact database the TypeScript backend
   writes. `ANNOTATION_DATA_DIR/<slug>` remains only as a backwards-compatible
   fallback for direct callers and tests.

7. **The mirrored Python service is not removed or modified.** It remains the
   production implementation and the reference contract. Changes to that
   contract require an explicit compatibility update in `annotation-store.ts`
   and its tests.

Rejected alternatives:

- **Keep the proxy.** Rejected because a local SQLite feature should not fail
  merely because an undocumented third process is absent.
- **Auto-start the Python service.** Better than a manual process, but still adds
  interpreter discovery, lifecycle management, port allocation and a second
  configuration path for data the backend can access directly.
- **Use a Toolbox-specific schema or GeoJSON-only storage.** Rejected because it
  would strand existing annotation databases and disconnect review feedback from
  the production-compatible data model.

## Consequences

The Toolbox is self-contained for annotation review: starting its normal backend
is sufficient, and the browser no longer knows about an annotation service URL.
Benchmark export and review feedback share the same per-map database without an
HTTP round trip.

`better-sqlite3` becomes a native Node dependency. Installation therefore needs
a compatible prebuilt binary or local build toolchain, unlike the otherwise
pure-JavaScript parquet reader.

Contract duplication is accepted, not hidden. The Python service remains
authoritative for production behavior; the TypeScript compatibility tests are
the guard against silent drift. Any future port of missed-detection or class CRUD
must extend the integrated store using the same tables rather than introduce a
third representation.

Cached connections assume a map's annotation database is not replaced underneath
a running backend. Copying or restoring a database requires restarting the
Toolbox, which is acceptable for a local developer tool and avoids stale prepared
statements or inode replacement surprises.
