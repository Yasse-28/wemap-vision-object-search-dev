# ADR 0003 — Split the pipeline out as a submodule

- **Status:** Accepted
- **Date:** 2026-08-05
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0001](0001-object-search-platform-structure.md) (its core
  decision falls), [ADR 0002](0002-align-on-backend-pipeline.md) (the mirror
  becomes a submodule, and moves)

## Context

ADR 0002 made this repo a **copy** of the production pipeline, with
`scripts/check-mirror.sh` proving byte-identity. That works, but the copy is
maintained by hand: every backend change has to be re-copied, and the only thing
stopping drift is a gate someone remembers to run.

The target is the inverse: **`wemap-vision-object-search` becomes a git submodule
inside `wemap-vision-backend`**, supplying the pipeline the backend currently
carries as real files. Porting stops being an activity.

That target dictates the split. A backend checkout must not acquire a TypeScript
workbench, a retired research lineage, or a local Postgres compose file — so
everything that is not production pipeline code has to live somewhere else.

## Decision

**Two repos, one submodule relationship.**

| Repo | Contains | Role |
|---|---|---|
| `wemap-vision-object-search` | the pipeline, byte-identical to the backend | mirrored; **verifies** |
| `wemap-vision-object-search-dev` (this one) | `toolbox/`, `legacy/`, `infra/`, the dev scripts, the ADRs | owns its code; **develops** |

1. **The submodule's root *is* the backend's `third_party/object_search/`.**
   `prepare/`, `inference/`, `indexing/`, `benchmarks/` and `annotation_service/`
   sit at its root, not one level down. A submodule cannot populate a path whose
   contents it nests deeper, so this is what makes the eventual backend mount
   possible at all.

2. **`services/object_search_online/` keeps its relative path inside the
   submodule.** It has a *different parent* in the backend (`services/`, shared
   with the unrelated `vps_online`), and one submodule cannot populate two paths.
   Nesting it costs the backend two `COPY` lines in
   `docker/object_search_online/Dockerfile` and no `PYTHONPATH` change — cheaper
   than a second submodule for four files.

3. **The submodule is mounted here at `third_party/object_search/`** — the same
   path the backend uses. This is the decision that pays: `scripts/lib.sh`,
   `scripts/check-types.sh` and the TypeScript `PYTHONPATH` join in
   `benchmark-runner.ts` are all **unchanged**, because
   `$REPO_ROOT/third_party/object_search` still means what it meant. The whole
   split cost two path edits.

4. **The submodule verifies the mirror; the parent develops against it.**
   `check-mirror.sh` lives in the submodule (it guards the submodule's only
   invariant, and must work when the submodule is cloned alone); this repo keeps a
   two-line wrapper so the gate list still runs from the root. Every other script
   is a dev launcher and lives here.

5. **The submodule carries no lint or type configuration.** Every Python file in it
   is a byte-for-byte copy, so a linter could only ever demand an edit that
   `check-mirror.sh` would then reject. "The mirror is not in a linted tree"
   replaces ADR 0002's "ignore every rule inside the mirror" — strictly less
   machinery. Linting lives here, with the code we actually write.

6. **`dev/0.3.0` is kept.** It is the last mirror-path state, where `check-mirror.sh`
   is a trivial `diff -r`. `dev/0.3.1` carries the re-rooting and is what this repo
   pins.

## Amendment to ADR 0001

Its core decision — *this repo is the platform, owning both the service and the
toolbox* — no longer holds. The service and the toolbox are now in different repos,
and the platform role belongs to this one. ADR 0001 survives only as the record of
why the toolbox exists at all.

## Consequences

**Positive**
- The re-sync obligation disappears the day the backend mounts the submodule; until
  then it is one `git submodule update` instead of a hand copy.
- The mirror is now provably minimal: anything added to the submodule that the
  backend does not have fails `check-mirror.sh`, in both directions.
- A backend checkout will not acquire 141 MB of `node_modules`.

**Negative / costs accepted**
- **Two `pytest` invocations** (136 here, 20 in the submodule). Re-coupling would
  put the `annotation_service` and `object_search_online` roots back on this repo's
  `sys.path`, where their flat `app.py` modules collide by name — the collision
  ADR 0002 already had to work around. `scripts/check-all.sh` runs both.
- **Two `pip install -e`**, since the submodule declares the pipeline's dependencies
  and this repo declares only what the dev code adds. Not expressible as one without
  hardcoding an absolute path.
- **The mirror's own docs go stale.** `third_party/object_search/AI_CONTEXT.md` and
  `benchmarks/README.md` document `PYTHONPATH=third_party/object_search`, correct for
  the backend and wrong for the re-rooted submodule. They are mirrored files, so they
  stay untouched by definition.
- **Cross-repo references.** `toolbox/bricks/vendored/PROVENANCE.md` and the
  submodule's `AI_CONTEXT/mirror-*.md` now point at each other by repo name rather
  than by relative path.

**Neutral**
- Version 0.3.1 on both sides. Git history for `toolbox/`, `legacy/` and `infra/`
  was not carried across — it remains reachable in `wemap-vision-object-search`
  at `dev/0.3.0` and earlier.

## Still owed to wemap-vision-backend

Mounting the submodule there is **not** part of this change. It needs:

- two `COPY` lines in `docker/object_search_online/Dockerfile` (decision 2);
- `--recurse-submodules` in the `annotation_service` EC2 `user_data` clone —
  today it is a plain `git clone --depth 1`, which would silently leave
  `third_party/object_search/annotation_service` **empty**;
- a `submodules:` flag in the backend's CI checkout (`.github/workflows/tests.yml`
  has none today; harmless now, not harmless then);
- reconciling `goals/object-search-consolidation.md:40-41`, which records the
  *opposite* direction (backend object-search as a submodule of this lineage).

Also unchanged and still owed: the `proposal_cutouts` double-allocation fix
(ADR 0002, "Owed to wemap-vision-backend").
