# ADR 0008 — The Annotation office writes to the SQLite store, and its points are ground truth

- **Status:** Accepted
- **Date:** 2026-08-18
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0006](0006-integrate-annotations-into-toolbox.md) — the integrated
  SQLite store now also holds the office's annotations, not only reviews.

## Context

The Annotation office kept its annotations in React state and, on demand, wrote
`{map}/annotations/annotations.geojson`.

**Nothing ever read that file.** There was no `GET /annotations` route; the
benchmark, the sweeps and every script read `{map}/benchmark/annotations.geojson`,
which `buildGroundTruth` regenerates from the SQLite store before each run. The
consequences were not cosmetic:

- the office opened empty on every page load, and the only way back in was dragging
  the file onto it by hand;
- a whole-file rewrite was the unit of saving, so unsaved work was lost by switching
  tabs — `App.tsx` mounts one panel at a time;
- the annotations never reached the benchmark at all. Someone could annotate a
  venue for an afternoon and change no measurement.

Meanwhile `{map}/object-search-annotations.db` already held the detection reviews,
the reference partition and `ground_truth_point`, with real CRUD routes.

## Decision

The office reads and writes that database.

### Points go to `ground_truth_point`

Not to a table of its own. The office produces exactly what the benchmark scores
against — a class, a prompt and a map position — and the point of the migration is
to connect the two. On the maps in use this means the office now opens on the ground
truth that already exists: 258 points on vinci, 674 on bbhotel.

**This makes the office an editor of measurement data.** Deleting an annotation
there removes a benchmark ground-truth point, and there is no undo. The delete and
clear-all confirmations name that consequence rather than talking about
"annotations". We accepted the risk deliberately: the alternative — a parallel table
plus an explicit promotion step — keeps the annotations disconnected, which is the
state this change exists to end.

The migration itself changes no score: the backfill only adds palette rows, and
`buildGroundTruth` returns the same 258 and 674 features before and after. Scores
move when someone annotates, which is the intent.

### Polygons go to `annotation_polygon`

A polygon has no single position to score a localization against, and
`ground_truth_point` has no geometry column. A nullable geometry on that table would
force everything reading ground truth to learn to skip rows; a separate table does
not.

### The palette is a table

`annotation_class` (name, type, colour, prompt) rather than deriving classes from
the annotations that use them, so a class created and not yet drawn on survives a
reload — which is exactly when a palette is being set up. Classes met only in
pre-existing ground truth are backfilled with a deterministic colour, collisions
resolved by taking the next free one: twelve classes on bbhotel with eight colours
otherwise produced three greens.

The class is referenced **by name**, not by a foreign key, because
`ground_truth_point.class` is the column `buildGroundTruth` emits. A rename cascades
in a transaction instead.

### Loading is automatic, writing is per annotation

There is no Save button. `GET /annotations` on open; a create, a delete or a class
change is one request. The file import stays — a store that cannot read what the
previous one wrote strands whatever is already on disk — and
`annotations/annotations.geojson` is imported once automatically the first time the
database is opened, with a `.pre-import.bak` beside it, mirroring the benchmark
import ADR 0006 introduced. Neither map in use has such a file; this is for the ones
that might.

"Save As…" still downloads a GeoJSON, which is now an export and nothing else.

## Consequences

- `saveAnnotations` and the `annotations/annotations.geojson` writer are gone;
  `POST /annotations` now means "merge this FeatureCollection into the store".
- `geojson.ts` shrank from 338 lines to 122: parsing moved server-side, where the
  tables can reject a malformed ring before it is stored.
- Benchmark ground truth can now change from the UI. Any comparison across runs must
  check that the ground truth did not move in between — see the project memory on
  what a measurement used.
