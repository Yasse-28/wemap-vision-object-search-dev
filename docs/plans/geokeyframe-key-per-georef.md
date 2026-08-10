# Spec — key the local `geokeyframe` table per georef

Status: **not started**. Written 2026-08-10, branch `feat/explorer-reads-parquet`, on top
of the uncommitted `level_strategy` + `UNRESOLVED_LEVEL` work.

**Dev-only. Production is not affected** and must not be touched — see "Why this is not
a production bug" below.

## The bug, measured

`geokeyframe.id` is the primary key and holds the **manifest array index**, which restarts
at 0 for every map (`db_schema.py:42`, `ingest_cli._upsert_geokeyframes:231`). The upsert
deletes only its own georef's rows and then does `ON CONFLICT (id) DO UPDATE`
(`ingest_cli.py:244`), so ingesting a second map **overwrites the first map's rows**, and
`object_search_candidate` joins on `k.id = c.geokeyframe_id` with no georef predicate
(`candidates.py:51`).

State of the local database when this was found, after ingesting `bbhotel-choisy`
(georef 149) and then `VINCI_Saint_Domingue` (georef 30):

| | |
|---|---|
| rows in `geokeyframe` | 74 988, ids 0…74 987, **all `geo_ref_id = 30`** |
| candidates for georef 149 | 27 226 |
| whose join lands on a row owned by **another** georef | **27 226 / 27 226** |

Keyframe 423 of `bbhotel-choisy`: the manifest says `y = 15.00` → level 5, and the
toolbox UI (which reads the manifest per map) agrees. `geokeyframe` says `y = 0.32` →
level 0, and that is what every localization reported. The symptom is a cluster claiming
one floor while its keyframe images are visibly from another — with no error anywhere.

**A second, worse path exists**: `object_search_candidate.geokeyframe_id` is
`REFERENCES geokeyframe(id) ON DELETE CASCADE` (`db_schema.py:57`). Any ingest that
deletes an id another map's candidates point at silently deletes **those candidates**.

## Why this is not a production bug — do not "fix" the backend

`backend/api/models.py:382`'s `GeoKeyframe` has no explicit `id`, so its primary key is a
Django `AutoField`: a serial that is globally unique across georefs. `geo_ref` and
`video_keyframe` are two separate FKs, with `UniqueConstraint(["geo_ref", "video_keyframe"])`
on top. Production's enrichment traverses the FK through the ORM
(`geokeyframe__position`, `backend/object_search/candidates.py:83`), so a candidate can
only ever reach the one row its FK names.

The collision is created by the port, which collapses `geokeyframe_id` and
`video_keyframe_id` into the per-map manifest index (`AI_CONTEXT/bricks.md`, "Frames":
"one id plays the role of both … Production distinguishes them, we do not") and then
promotes that index to primary key. This spec brings the local schema *closer* to
production semantics, it does not diverge from them.

## The change

### 1. `toolbox/bricks/db_schema.py`

- `geokeyframe`: primary key becomes **`(geo_ref_id, id)`**. Keep the `id` column (it is
  the manifest index and the value candidates carry) and keep
  `UNIQUE (geo_ref_id, video_keyframe_id)`.
- `object_search_candidate`: replace the single-column FK with a **composite** one:
  `FOREIGN KEY (geo_ref_id, geokeyframe_id) REFERENCES geokeyframe (geo_ref_id, id) ON DELETE CASCADE`.
  This is what makes the cascade scoped to the right map.
- Keep the existing indexes; add nothing that the composite key already provides (the
  PK's own index covers `(geo_ref_id, id)`).

### 2. Refuse to run against the legacy shape — loudly

`ensure_schema` uses `CREATE TABLE IF NOT EXISTS`, so an existing database keeps the old
single-column key and every symptom above. Add a check: if `geokeyframe` exists and its
primary key is `(id)` alone, raise `RuntimeError` naming the exact recovery, e.g.

```
geokeyframe still uses the legacy single-column primary key, which lets one map
overwrite another's poses. Drop both tables and re-ingest each map:
  psql … -c 'DROP TABLE object_search_candidate, geokeyframe'
  python -m toolbox.bricks.ingest_cli <each map dir>
```

Detect it with `pg_index`/`pg_constraint` on `geokeyframe_pkey`. **Do not drop or migrate
anything automatically** — this deletes embeddings the user paid GPU time for, and the
decision is theirs. A clear refusal is the deliverable, not a silent repair.

### 3. `toolbox/bricks/candidates.py` — the join

`_ENRICH_SQL` (`:51`) becomes:

```sql
JOIN geokeyframe AS k
  ON k.geo_ref_id = c.geo_ref_id
 AND k.id = c.geokeyframe_id
```

Note this in the module docstring as a **deliberate, dev-only divergence** from the
ported production query: production joins an ORM FK to a globally unique pk and needs no
georef predicate. Same bookkeeping style as `UNRESOLVED_LEVEL` in `localize.py`.

Check the feedback-prototype subquery too (`_PROTOTYPE_SIM_SQL`): it already carries
`p.geo_ref_id = %s` for exactly this class of bug, and its comment says so. Leave it.

### 4. `toolbox/bricks/ingest_cli.py`

`_upsert_geokeyframes`: `ON CONFLICT (id)` must become `ON CONFLICT (geo_ref_id, id)`.
Leave the `DELETE FROM geokeyframe WHERE geo_ref_id = %s` — it is now correct rather than
misleading.

### 5. Audit, and report what you find

Grep every reader of `geokeyframe` and every place a `geokeyframe_id` is resolved without
a georef, in `toolbox/bricks/` and `toolbox/backend/src/`. `service.index_coverage` is the
one to look at first — it reports per-keyframe `ingested`/`no_position` counts and would
have been wrong the same way. Fix what is wrong, and list in the summary what you checked
and found correct. Do not touch `third_party/` or `legacy/`.

## Tests

`toolbox/tests/test_integration_db.py` already builds a schema and inserts `geokeyframe`
rows by hand (`:90`). Add there:

1. **Two maps coexist.** Insert poses for two georefs using **overlapping** ids (0,1,2 in
   both) with clearly different positions, then run `load_enriched_candidates` for each
   and assert each gets **its own** poses. This is the test that would have caught the
   bug; without it the fix is unverified.
2. **Ingesting map B does not disturb map A.** Call `_upsert_geokeyframes` for georef A,
   then for georef B with overlapping ids, then re-read A's rows and assert they are
   unchanged.
3. **The cascade is scoped.** Delete one georef's `geokeyframe` rows and assert only that
   georef's candidates went away.
4. **The legacy-shape guard fires.** Create the table with the old `PRIMARY KEY (id)` and
   assert `ensure_schema` raises, with the message naming the recovery.

These need PostgreSQL. If it is unreachable from your sandbox, **say so explicitly in the
summary and do not substitute a hand-rolled simulation for the measurement** — the
reviewer has database access and will run them.

## Documentation (required by the maintenance contract)

- `AI_CONTEXT/bricks.md`: the "Local schema" section states `geokeyframe` is keyed by a
  single id; correct it. Add a gotcha, in the same measured style as gotcha 5, saying the
  local table is per-georef *because* the ids are manifest indices, that this is a
  divergence from the ported query, and that a legacy database is refused rather than
  migrated.
- The "Frames" paragraph says "one id plays the role of both `geokeyframe_id` and
  `video_keyframe_id` here … Production distinguishes them, we do not." That sentence is
  what made this bug possible; extend it with the consequence and the key that now
  contains it.

## Verification

- `pytest toolbox/tests/` — 24 in `test_localize.py`, and the DB tests if you can reach
  Postgres. `black --check`, `ruff check`, `mypy toolbox/bricks/`.
- `cd toolbox && npm run type-check`.
- Do **not** drop the user's tables and do **not** re-ingest anything: the running
  database holds one map's poses and the user decides when to rebuild. Verify against
  temporary georef ids in the tests instead.
