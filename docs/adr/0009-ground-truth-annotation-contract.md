# ADR 0009 — What a ground-truth annotation must carry

- **Status:** Accepted
- **Date:** 2026-08-18
- **Deciders:** Yacine (maintainer)
- **Amends:** [ADR 0008](0008-annotations-move-into-the-sqlite-store.md) — the store
  and its table stay as they are; this adds required content to
  `ground_truth_point.extra_properties` and states what each field buys.

## Context

The benchmark's ground truth is one point per object, carrying a class, an optional
prompt and an `accuracy` radius. That was enough to ask "was this object found". It is
not enough for the two measurements now wanted, and the reason is measurable rather
than aesthetic.

**Error decomposition** (TIDE, six error types with a contribution each) needs to tell a
prediction that landed on the wrong *class* from one that landed on nothing, and a
redundant prediction from a genuinely new one. Measured 2026-08-18 over both maps:

| distance to the nearest annotation of a **different** class | vinci | bbhotel |
|---|---|---|
| median | 2.01 m | 0.70 m |
| share within 5 m — the radius every annotation actually uses | 60.5 % | **98.8 %** |

`accuracy` is 5.0 m on 931 of the 932 annotations of both maps. On bbhotel a prediction
matched inside its radius is therefore almost always inside another class's radius too,
so *classification* and *localisation* errors cannot be attributed there at all.
Separating a chair from its table would need a radius under ~0.35 m, which is below the
depth error the position is built from — no annotation effort fixes that.

**Open-vocabulary scoring** (label-set ground truth, top-N frequency and set ranking)
needs each object to carry a *set* of acceptable labels in ranked categories rather than
one string, because today a cluster labelled `seat` against an annotation labelled
`chair` is scored as a false positive. The two maps also disagree on language: bbhotel is
annotated in French (`chaise`, `detecteur de fumée`), vinci in English (`check in
counter`, `FIDS`).

Two defects found while measuring this bound what can be asked of the current data:

- **duplicated annotations.** 394 of bbhotel's 674 rows are in strictly identical groups
  — 197 pairs sharing class, source keyframe, ERP `u`/`v`, depth and `created_at`. One
  click inserted twice. vinci has none. Any denominator counting annotations is
  overstated by ~29 % on bbhotel.
- **unknown exhaustivity.** 74.8 % of vinci's annotations have their source panorama
  absent from the index. Nothing records where the ground truth is complete, so an
  unannotated real object and a genuine background error are indistinguishable.

## Decision

### 1. Classification and localisation errors are gated by measurement, not by map name

The code **measures the separability and withholds those two columns when it fails**,
printing the number it refused on. A future map is then handled with no code change.

The gate's threshold is a third: beyond that, one attribution in three would be a coin
toss. The four remaining types — *correct*, *duplicate*, *background*, *missed* — stay
available on every map whatever the gate says.

**Amended the same day, after measuring.** The 98.8 % figure above comes from the flat
5 m `accuracy`, not from any real footprint. Re-measured against radii derived from a
plausible `extent_m`, the share of annotations with another class inside their own radius
is:

| `extent_m` | radius | vinci | bbhotel |
|---|---|---|---|
| 0.5 m | 0.25 m | 0.8 % | 7.3 % |
| 1.0 m | 0.50 m | 6.2 % | 34.2 % |
| 2.0 m | 1.00 m | 25.2 % | **49.6 %** |
| 4.0 m | 2.00 m | 49.6 % | 84.9 % |

So the original framing — "only vinci, bbhotel is hopeless" — was too pessimistic, and
the reason is worth keeping visible: it was an artefact of the 5 m radius, not a property
of the venue. bbhotel's chairs and smoke detectors are small, and at an honest 0.5 m
extent they separate at 7.3 %. Both maps are therefore worth annotating with real
extents, and which columns each one earns is decided after the fact by the gate. What
stays accepted is that **some maps will be refused**, and that this is reported rather
than worked around.

### 2. Required fields, written into `extra_properties`

No schema migration. `ground_truth_point.extra_properties` is already a free JSON blob
carrying the ERP source of the click, and it already travels through the GeoJSON export
into `feature.properties`, which `load_annotations` reads. New keys go there.

| key | type | required | what it buys |
|---|---|---|---|
| `object_id` | string | yes | Object identity. Two annotations sharing it are one object seen twice, not two objects — without it the *duplicate* error type cannot be defined, and the 197 pairs above cannot be told from a real pair of adjacent objects. |
| `extent_m` | number | yes | Approximate largest horizontal footprint, metres. The match radius is **derived** from it, never asked for directly: an annotator can judge "this chair is 0.5 m wide", not "0.5 m is a good matching threshold". |
| `exhaustive_zone` | string | yes | Identifier of a region within which the annotation is complete. Precondition of the *background* and *missed* types: outside a declared zone, a prediction on an unannotated object is not an error and must not be counted as one. |
| `labels.synonyms` | string[] | yes | Labels naming this object exactly. A prediction carrying any of them is correct. Replaces the single `class` for scoring; `class` stays as the display name. |
| `labels.depictions` | string[] | no | Labels that describe an image *of* the object rather than the object. Non-anecdotal in an airport and a hotel. |
| `labels.visually_similar` | string[] | no | Labels for objects that look like this one. Ranked below synonyms, above clutter. |
| `labels.clutter` | string[] | no | Labels induced by an imprecise crop — what is also in the box. |
| `is_depiction` | boolean | no | This annotation *is* a printed image of its class, not the class. Defaults to false. |

Every field is optional to the *reader*: a map annotated under the old convention keeps
working, and each metric that needs a field says which annotations it had to skip. A
field is required of the *annotator* going forward.

### 3. The vocabulary is shared and English

One vocabulary across maps, in English, with the local-language term as a synonym. Two
maps with disjoint vocabularies cannot be compared, and comparison between maps is the
only use the measured data supports (relative, never absolute — see the project memory
on which benchmark data to trust). bbhotel's existing French classes become synonyms of
their English entry rather than being rewritten.

### 4. Co-located objects of one class are annotated individually, with a shared zone

A row of chairs is a row of objects, each with its own `object_id`. The *duplicate* type
then means "two predictions for one `object_id`", which is well defined regardless of how
close two real chairs are. Grouping them into one "row of chairs" annotation was
considered and rejected: it would move the granularity confound from the metric into the
ground truth, where it can no longer be measured.

### 5. A double-annotated subset is part of the deliverable

At least 50 objects annotated independently by two people, on each map. Without an
inter-annotator agreement figure, the six error types inherit the spec's ambiguity and it
cannot be told from a pipeline defect. This is cheap and it is the only check that the
spec was followed.

### 6. Uniqueness is enforced at the store

`ground_truth_point` gains the unique index its sibling `detection_review` already has
(`uq_detection_review_key`), over the click identity: class plus source keyframe plus ERP
`u`/`v`. The 197 existing pairs are removed, keeping the lowest id.

## Consequences

- The benchmark's match radius stops being a single `--default-accuracy` and becomes
  per-object, derived from `extent_m`. Every recall figure moves; historical numbers are
  not comparable across the change, which has to be stated wherever they are quoted.
- bbhotel's annotation count drops from 674 to about 477. Its recall figures rise
  accordingly, and the improvement is a correction, not progress.
- Two of TIDE's six columns may be unavailable on a given map, decided by the gate and
  with a measured reason printed in their place. With today's annotations — no
  `extent_m`, so a 5 m radius everywhere — they are withheld on **both** maps: vinci
  measures 60.5 % overlap. They become available as extents land, which makes `extent_m`
  the field to annotate first.
- The annotation tool must write six new keys. The tool and the annotation work are
  owned separately from the measurement code; this table is the whole contract between
  them, so a change to it is a change to both sides.
- `class` keeps its current meaning for display and grouping. Scoring reads
  `labels.synonyms`, falling back to `class` when absent, so nothing breaks before the
  labels exist.
