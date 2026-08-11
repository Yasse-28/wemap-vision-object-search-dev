# ADR 0001 — Object Search Platform Repository Structure

- **Status:** Accepted
- **Date:** 2026-06-25
- **Deciders:** Yacine (maintainer)
- **Supersedes:** —

## Context

Several projects revolve around the object-search capability, currently spread across the workspace:

| Project | Language | Repo today | Relationship to object search |
|---|---|---|---|
| **object-search pipeline** | Python | `wemap/wemap-vision-object-search` | The service itself: offline `build_index` + online FastAPI `pipeline.online.app`. |
| **livemap** | TypeScript | `wemap/livemap` (own remote) | **HTTP consumer** — AR components call the service. |
| **wemap-vision-tools** | TypeScript | `wemap/wemap-vision-tools` (own remote) | **HTTP consumer** — `object-search-*` router/controller. |
| **object-search-toolbox** | TypeScript | *not a standalone repo* (`local-sandbox/`) | First-party dev/test/annotate/benchmark tool; proxies the Python API and spawns `build_index` / `pipeline.online.app`. |

Today these are scattered, the toolbox has no home, and there is no single place a new developer can go to understand or run the whole object-search world. We want to **centralize the ecosystem** for future developers while following sound structure, and crucially **iterate quickly on the pipeline while keeping the consumers compatible**.

Key facts that shape the decision:

- The consumers depend on the pipeline **over HTTP only** — no Python is imported. Co-locating them therefore introduces **no code/build dependency**, only a checkout/coordination convenience.
- The consumers are large, independent product repos with their own teams, CI, and release cadence.
- The toolbox is specific to *this* pipeline and has no independent repo or history worth preserving as a separate unit.

## Decision

The existing `wemap-vision-object-search` repository **takes on a "platform" identity** (no new Git repository is created). It hosts:

1. **`pipeline/`** — the deployable object-search service (offline build + online serving). It keeps **self-contained packaging, versioning, and CI** (its own `pyproject.toml`, its own SemVer line, currently `0.2.0`). This is the only component that iterates fast.
2. **`toolbox/`** — **owned outright** by the platform repo, ported from `local-sandbox/object-search-toolbox`. No submodule.
3. **`consumers/`** — the two HTTP consumers as **Git submodules**, each pinned to a **dedicated integration branch** that the platform maintainer **solely owns**. Consumer-repo owners are not involved until a feature matures and the maintainer notifies them to merge the integration branch upstream.
4. A **README** that states the platform role explicitly.

### Target layout

```
wemap-vision-object-search/          ← repo plays the "object-search platform" role
├── pipeline/                        ← THE service (build + serve); own pyproject, version, CI
│   ├── core/  offline/  online/  tests/  config/
│   └── pyproject.toml               (scoped to the service)
├── toolbox/                         ← owned dev/test/annotate/benchmark tool (ported from local-sandbox)
│   ├── backend/  frontend/  shared/
├── consumers/                       ← integration/E2E co-checkout only — NOT build dependencies
│   ├── livemap/                     (submodule → wemap/livemap,            branch: feature/object-search-v2-yacine)
│   └── wemap-vision-tools/          (submodule → wemap/wemap-vision-tools, branch: feature/object-search-v2-yacine)
├── docs/
│   └── adr/0001-object-search-platform-structure.md
├── scripts/
│   └── sync-consumers.sh            (push submodule branches, then bump pointers — see workflow)
├── .gitmodules                      (branch = <integration-branch> per consumer)
└── README.md                        (states the platform identity + how the pieces relate)
```

### Why this shape (alternatives considered)

- **New umbrella meta-repo** (everything as submodules under a fresh repo) — *rejected*: maintainer prefers not to create a new Git repo. "Umbrella" is a *role*, and the existing repo can take it on.
- **True monorepo** (single history via Nx/Turborepo/Bazel) — *rejected*: too heavy; the consumers are already-separate product repos with independent lifecycles; merging them in is disproportionate.
- **Don't embed consumers; rely only on a published API contract** — *deferred*: a good complement (see Open Items) but does not by itself give the co-checkout / staged-compatibility workflow the maintainer wants.

### The "consumers as submodules" smell, and why it is acceptable here

Embedding *consumers* (things that depend on us) as submodules inverts the usual "submodules = my dependencies" convention. That convention is about **code/build** dependencies; since the pipeline never imports the consumers, this is only a *semantic/organizational* inversion, not a real one. It is acceptable here because three guardrails neutralize the downsides:

1. **README framing** — the repo declares itself the *platform*; `pipeline/` is the deployable service; `consumers/` are checked out for integration/E2E only.
2. **Walled-off service identity** — the pipeline's packaging, version, and CI live in `pipeline/` and do not blend with the platform-level aggregation.
3. **CI does not recurse submodules** for the pipeline's build/test/scan jobs; only an explicit E2E job checks out `consumers/`.

## Submodule + integration-branch workflow

- Each consumer submodule pins a **dedicated integration branch** (declared via `branch = …` in `.gitmodules`). The maintainer is the **sole owner** of these branches, and **all consumer updates for object-search are made from within this platform checkout**.
- A submodule always records a **specific commit SHA** in the superproject — `branch = …` is only used by `git submodule update --remote` to fast-forward the pin to the branch tip. "Everything matches" is enforced at each **deliberate pointer bump**, not silently.

**To adapt a consumer to a pipeline change:**
1. `cd consumers/livemap`, edit, commit **to the integration branch**.
2. **Push that branch to the consumer's remote** (`wemap/livemap`).
3. Back at the platform root, commit the **bumped submodule pointer**.

**Golden rule:** always **push the submodule branch (step 2) before pushing the platform pointer (step 3)**. Pushing a superproject commit that references an unpushed submodule SHA leaves a dangling reference for any fresh `--recurse` clone. `scripts/sync-consumers.sh` should encode this order.

- Pin bumps are **human-driven** (no automation for now). Pushing integration branches will trigger the consumers' own CI — expected and usually harmless.
- When a feature matures, the maintainer notifies the consumer owners to merge the integration branch upstream.

## Consequences

**Positive**
- One `git clone --recurse-submodules` gives a developer the entire object-search world: service, tooling, and the consumers it must stay compatible with.
- The pipeline can iterate freely; compatibility is staged on integration branches the maintainer controls.
- The toolbox finally has a home and sits next to the service it orchestrates.

**Negative / costs accepted**
- The repo's identity broadens from "service" to "platform" — must be communicated via README to avoid confusing newcomers.
- Submodule ergonomics (the push-order trap, remembering `--recurse-submodules`).
- Submodule-walking tools (scanners, SBOM) must be scoped so they don't drag in the consumers.

**Neutral**
- Pipeline keeps its own SemVer; the platform repo as a whole is not separately versioned for now.

## Migration checklist (executed later — not part of this ADR)

- [ ] Reframe `README.md` to state the platform identity and how `pipeline/`, `toolbox/`, `consumers/` relate.
- [ ] Ensure `pipeline/` has a self-contained `pyproject.toml` and scoped CI.
- [ ] Port `local-sandbox/object-search-toolbox` → `toolbox/` (decide: plain copy vs `git subtree` to preserve history).
- [ ] Create the `feature/object-search-v2-yacine` branch in `wemap/livemap` and `wemap/wemap-vision-tools`.
- [ ] Add submodules under `consumers/` with `branch = feature/object-search-v2-yacine` in `.gitmodules`.
- [ ] Add `scripts/sync-consumers.sh` encoding the push-then-bump order.
- [ ] Scope CI: pipeline jobs do **not** recurse submodules; a separate E2E job does.
- [ ] Update `.gitignore` and developer setup docs (`clone --recurse-submodules`, bootstrap script).

## Resolved decisions

- **Integration-branch name** — `feature/object-search-v2-yacine` on **all** consumer repos. Sole owner: the maintainer.
- **Toolbox port method** — **plain copy** (do not preserve `local-sandbox` history); sandbox history is not worth the `git subtree`/`filter-repo` complexity.
- **Pipeline package import path** — **keep `pipeline.*`**. Inside the platform repo `pipeline` reads as "the object-search pipeline", and it avoids breaking every import, the test suite, and the toolbox's `python -m pipeline.online.app` spawn contract. The `src/`-layout benefits (scoped `pyproject.toml`, tests) can be adopted without renaming the package.

## Deferred (revisit later)

- **API contract** — optionally publish an OpenAPI spec / generated TS client from `pipeline/` as a single source of truth the consumers import. Complements (does not replace) the submodule approach. Deferred by decision; revisit once the integration branches stabilize.

### Known integration gaps (address/port/path) — to fix later

Survey of how each component reaches the object-search service, as of this ADR:

| Component | Listens on | Calls the service at | Path shape |
|---|---|---|---|
| `pipeline` (Python service) | `0.0.0.0:45678` (default) | — (is the service) | `/{map_id}/object-search/…` |
| `toolbox` backend | `:45700` (`OBJECT_SEARCH_WORKBENCH_PORT`) | `http://127.0.0.1:45678` (`OBJECT_SEARCH_PYTHON_API`, env-driven) | `/{map_id}/object-search/…` |
| `consumers/wemap-vision-tools` backend | `:8000` (default) | `http://127.0.0.1:45678` (`LOCALIZE_SERVICE_BASE_URL`, **hardcoded**) | `/{map_id}/object-search/{localize,localize-offline}` |
| `consumers/livemap` (frontend) | — | configured `providers.vps.endpoint` (**remote**, not localhost) | `…/geopose/object-search/text` |

**Compatible today:** `pipeline` ⇄ `toolbox` ⇄ `wemap-vision-tools` all agree on port **45678** and the `/{map_id}/object-search/…` path; their own listen ports don't collide (45678 / 45700 / 45677).

**Two gaps to fix later:**
1. **livemap path prefix mismatch** — livemap calls `…/geopose/object-search/text`, but the standalone `pipeline` serves `…/object-search/text` (no `geopose/` segment). livemap is written against the full geopose VPS service (`wemap-vision-python`); pointed at this pipeline it would 404. Fix options: a `geopose/`-stripping gateway, livemap's `vps.endpoint` pointing at a gateway that mounts the pipeline under `geopose/`, or the pipeline also registering the `geopose/`-prefixed routes. (The deferred API-contract item would surface this automatically.)
2. **wemap-vision-tools hardcoded base URL** — `LOCALIZE_SERVICE_BASE_URL = 'http://127.0.0.1:45678'` should become env-driven (mirroring the toolbox's `OBJECT_SEARCH_PYTHON_API`) so all consumers share one configuration knob.
