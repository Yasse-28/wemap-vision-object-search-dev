/**
 * What every number in the Analysis panel means, and what counts as a bad one.
 *
 * One place, deliberately. The panel renders the same explanation in the hover overlay
 * and derives the badge colour from the same bands, so a threshold cannot drift
 * between the text that explains it and the colour that judges it.
 *
 * **The bands are calibration marks, not truth.** They were set from the constants the
 * pipeline itself uses (15 m trusted depth, 5° useful parallax, `min_keyframes_per_cluster`
 * = 2) and from one measured map, `bbhotel-choisy`. Two maps of this repo disagree on
 * tuned values while agreeing on the *sign* of a conclusion, so read a badge as "worth
 * looking at" and compare maps against each other — never quote one as a precision.
 */

export type Verdict = "good" | "warn" | "bad" | "neutral";

export type MetricExplanation = {
  /** Short name shown on the tile or row. */
  label: string;
  /** What the number is, in one sentence. */
  measures: string;
  /** How to read it: what moves it, and what it must not be mistaken for. */
  reading: string;
  /** The bands, as text — kept next to `verdict` so the two cannot disagree. */
  bands?: string;
  /** Why the bands sit where they do. */
  why: string;
  /** What to look at when the value is bad. */
  remedy?: string;
  /** Higher is better unless this says otherwise — drives the arrow on the tile. */
  direction?: "higher" | "lower" | "none";
  verdict?: (value: number) => Verdict;
};

/** Shown at the foot of every overlay: the bands come from one map, not from theory. */
export const CALIBRATION_NOTE =
  "Bands are calibration marks from bbhotel-choisy and from the pipeline's own"
  + " constants — a prompt to look, not a precision to quote. Compare maps against"
  + " each other.";

/** Bands where a higher number is better. */
function above(good: number, warn: number): (value: number) => Verdict {
  return (value) => {
    if (!Number.isFinite(value)) {
      return "neutral";
    }
    if (value >= good) {
      return "good";
    }
    return value >= warn ? "warn" : "bad";
  };
}

/** Bands where a lower number is better. */
function below(good: number, warn: number): (value: number) => Verdict {
  return (value) => {
    if (!Number.isFinite(value)) {
      return "neutral";
    }
    if (value <= good) {
      return "good";
    }
    return value <= warn ? "warn" : "bad";
  };
}

export const METRICS: Record<string, MetricExplanation> = {
  // -------------------------------------------------------------- s0, inventory
  placed_share: {
    label: "Rows placed in 3D",
    measures:
      "Share of parquet rows that got a position: a pose for their keyframe and a"
      + " depth value at their box centre.",
    reading:
      "This is arithmetic, not quality. Rows that are not placed are invisible to"
      + " every 3D measurement below, so a low value silently shrinks the sample of"
      + " every other section.",
    bands: "≥ 99 % expected · 90–99 % worth explaining · < 90 % a prepare defect",
    why:
      "Ingestion places a row from depth and the two ERP angles; failing to do so"
      + " means a missing pose or a hole in the depth map, both of which are upstream"
      + " defects rather than pipeline behaviour.",
    remedy: "Check the v2 manifest poses and the depth maps for the affected keyframes.",
    verdict: above(0.99, 0.9),
  },
  missing_pieces: {
    label: "Missing from the index",
    measures: "What this prepare run did NOT write: labels, scores, embeddings, reviews.",
    reading:
      "Read this before any table below. A section fed by something missing reports"
      + " emptiness, and a table of zeros looks exactly like a measured zero.",
    why:
      "This is the failure mode the analysis tool exists to prevent: concluding that a"
      + " cue carries no signal when in fact nothing wrote the column it reads.",
    direction: "none",
  },

  // ------------------------------------------------------------- s1, detections
  range_p90_m: {
    label: "Range p90",
    measures: "The 90th percentile distance from a keyframe to its detection's depth point.",
    reading:
      "A spread, not an error. The tail is what matters: rows beyond the depth map's"
      + " trusted range carry a position nothing downstream can rely on.",
    bands: "< 15 m good · 15–25 m thin · > 25 m past what depth can support",
    why:
      "The tool treats 15 m as the trusted range (MAX_TRUSTED_RANGE_M), which is also"
      + " the radius the observability pass searches keyframes in. Past it a depth"
      + " value is extrapolation.",
    remedy: "Cap the depth, or drop far rows before clustering — do not tune around them.",
    direction: "lower",
    verdict: below(15, 25),
  },
  range_max_m: {
    label: "Range max",
    measures: "The single furthest depth point in the map.",
    reading:
      "A diagnostic for depth blow-ups, not a distribution. One row at 100 m says the"
      + " depth map saturated somewhere, usually on sky or glass.",
    why:
      "Values far past the trusted range are how a handful of rows drag a cluster"
      + " centroid across a room.",
    direction: "lower",
    verdict: below(30, 60),
  },
  detector_score_median: {
    label: "Detector score median",
    measures: "Median confidence of this detector's boxes.",
    reading:
      "Compare detectors against each other, never against an absolute bar: an"
      + " open-vocabulary detector and a closed-set one do not share a score scale.",
    why:
      "The two detectors in this repo sit an order of magnitude apart (≈0.26 vs ≈0.08"
      + " on bbhotel), so one threshold applied to both silently deletes one of them.",
    direction: "none",
  },

  // ----------------------------------------------------------------- s2, labels
  base_rate_2m: {
    label: "Base rate at 2 m",
    measures:
      "Share of ALL detections that sit within 2 m of some annotation, regardless of"
      + " label.",
    reading:
      "The floor every label and every cue must beat. It is a property of how densely"
      + " the map is annotated, not a result.",
    why:
      "On a densely annotated map half of all detections are within 2 m of something,"
      + " so an uninformative label already scores 50 %. Reading an attachment rate"
      + " without it is the most common way to conclude a label works when it does not.",
    direction: "none",
  },
  attachment_lift: {
    label: "Lift over base rate",
    measures:
      "Attachment at 2 m minus the base rate at 2 m, in percentage points.",
    reading:
      "The only form in which an attachment rate means anything. Near zero, labels and"
      + " geometry agree no better than chance on this map.",
    bands: "≥ 5 pts informative · 1–5 pts marginal · < 1 pt indistinguishable from chance",
    why:
      "Attachment is measured at a fixed radius over the whole index, so the more"
      + " annotations a map carries the higher it climbs on its own. Only the gap"
      + " survives that.",
    direction: "higher",
    verdict: above(0.05, 0.01),
  },
  mutual_information: {
    label: "Label ↔ class MI",
    measures:
      "Normalised mutual information between a detector's free label and the annotated"
      + " class it attaches to.",
    reading:
      "How much a label tells you about what the thing is. It is symmetric and says"
      + " nothing about which label maps to which class — the two conditional tables"
      + " below do that.",
    bands: "≥ 0.5 informative · 0.2–0.5 weak · < 0.2 near-independent",
    why:
      "An open vocabulary spreads one object over many labels ('sofa', 'lounge chair',"
      + " 'booth seat'), which caps MI even when every label is individually sensible.",
    direction: "higher",
    verdict: above(0.5, 0.2),
  },
  purity_same_class: {
    label: "Neighbour purity, class",
    measures:
      "Among a row's 10 nearest embedding neighbours, the share attached to the same"
      + " annotated class.",
    reading:
      "How well the embedding space groups kinds of thing. This is what a text query"
      + " actually rides on.",
    bands: "≥ 0.5 usable · 0.3–0.5 weak · < 0.3 the space is not separating classes",
    why:
      "Retrieval is the pipeline's largest measured loss on this repo's maps, and this"
      + " is the property it depends on before any ranking runs.",
    direction: "higher",
    verdict: above(0.5, 0.3),
  },
  purity_same_object: {
    label: "Neighbour purity, object",
    measures:
      "Among a row's 10 nearest neighbours, the share attached to the same individual"
      + " annotation.",
    reading:
      "Expected to be low and not a defect on its own — one object is a handful of rows"
      + " among many similar ones. Read it against the class purity above: a large gap"
      + " means the space knows the kind but not the instance.",
    why:
      "Instance identity is what association has to recover, and this says how much of"
      + " it the embedding alone provides. On this repo's maps: very little.",
    direction: "none",
  },

  // ----------------------------------------------------------- s3, ground truth
  covered_in_own_keyframe: {
    label: "Depth-free coverage",
    measures:
      "Share of measurable annotations whose clicked pixel falls inside some detection"
      + " box of the very panorama it was clicked in.",
    reading:
      "The one measurement in this tool that owes nothing to depth — two angles in one"
      + " frame. It is the detector's recall, and it is a ceiling on everything"
      + " downstream: no association recovers an object no box ever covered.",
    bands: "≥ 80 % good · 50–80 % the detector is the limit · < 50 % detection is the problem",
    why:
      "Every other ground-truth number shares the annotation's own construction"
      + " (ray(u, v) × depth(u, v)), so it measures agreement, never accuracy. This"
      + " one escapes that entirely, which is why it is reported first.",
    remedy:
      "Look at the per-class table: a class at 0 % is a prompt or vocabulary problem,"
      + " a uniform sag is a detector-threshold problem.",
    direction: "higher",
    verdict: above(0.8, 0.5),
  },
  source_keyframe_indexed: {
    label: "Source panorama indexed",
    measures:
      "Share of annotations whose clicked panorama is in the index at all.",
    reading:
      "A hard cap on the line above: an annotation whose panorama was never indexed"
      + " cannot be measured depth-free, and cannot be found by any query either.",
    bands: "≥ 90 % good · 70–90 % thinning is biting · < 70 % an index-coverage defect",
    why:
      "Ingest thins keyframes by distance. Thinning over the manifest instead of over"
      + " the keyframes the index covers is exactly the defect that once made a whole"
      + " map look signal-free — it was coverage, not data.",
    remedy: "Check the thinning step against the keyframes the index actually holds.",
    direction: "higher",
    verdict: above(0.9, 0.7),
  },
  keyframe_threshold_cost: {
    label: "min_keyframes cost",
    measures:
      "Annotations lost, in percentage points, by raising min_keyframes_per_cluster"
      + " from 1 to its online default of 2.",
    reading:
      "The threshold buys precision by discarding evidence; this is the bill. Read it"
      + " with the curve: if the covered share does not rise as the threshold does,"
      + " the filter is not selecting for quality, it is only losing recall.",
    bands: "< 5 pts cheap · 5–15 pts a real trade · > 15 pts the threshold sets your ceiling",
    why:
      "It is applied before any ranking, so whatever it drops is unreachable no matter"
      + " how good the retrieval is.",
    direction: "lower",
    verdict: below(0.05, 0.15),
  },
  useful_pair_share: {
    label: "Useful parallax",
    measures:
      "Median share of an annotation's observation pairs separated by at least 5° as"
      + " seen from the object.",
    reading:
      "Whether repeated observations add geometry or only photometric redundancy. Below"
      + " 5° two views intersect too shallowly to settle their disagreement about depth.",
    bands: "≥ 50 % good · 20–50 % thin · < 20 % the capture cannot triangulate",
    why:
      "A capture that walked past one side of a room in a straight line produces many"
      + " observations and no baseline. That is a capture ceiling, not an algorithm one.",
    remedy: "Read it next to the trajectory anisotropy: 0 means a straight line.",
    direction: "higher",
    verdict: above(0.5, 0.2),
  },
  unseen_never_looked: {
    label: "Never looked at",
    measures:
      "Annotations with no keyframe within the trusted range at all.",
    reading:
      "A capture failure, not a detector failure. Separating these from 'looked and"
      + " missed' matters because the two have unrelated remedies.",
    why:
      "Anything above zero is ground truth the pipeline cannot reach by construction,"
      + " and it should be excluded before a recall number is quoted.",
    direction: "lower",
    verdict: below(0, 5),
  },

  // ------------------------------------------------------------------ s4, cues
  best_conditional_auc: {
    label: "Best cue (conditional)",
    measures:
      "The strongest pairwise cue's AUC once conditioned on geometry — its ability to"
      + " tell 'same object' from 'different object' among pairs geometry already"
      + " brought close.",
    reading:
      "Conditional, always. The raw AUC of a distance cue is near 1 by construction and"
      + " has already mispredicted an association idea twice in this repo.",
    bands: "≥ 0.70 usable · 0.55–0.70 weak · ≤ 0.55 no signal beyond geometry",
    why:
      "0.5 is a coin toss. A cue that only re-states what distance already said adds"
      + " nothing to a pipeline that has distance, however impressive its raw number.",
    remedy: "Also read the share of pairs the cue applies to: coverage caps its value.",
    direction: "higher",
    verdict: above(0.7, 0.55),
  },
  cue_raw_auc: {
    label: "Raw AUC",
    measures: "The cue's AUC over all sampled pairs, unconditioned.",
    reading:
      "Kept for contrast, not for decisions. Any cue correlated with distance inherits"
      + " distance's separability here.",
    why:
      "The gap between raw and conditional is the interesting quantity: a large gap"
      + " means the cue was riding geometry all along.",
    direction: "none",
  },

  // -------------------------------------------------------------- s5, counting
  cross_source_share: {
    label: "Cross-detector overlap",
    measures:
      "Share of overlapping in-view box pairs that come from two different detectors.",
    reading:
      "How much of the duplication is the two detectors seeing the same thing, rather"
      + " than one detector firing twice.",
    why:
      "The two cases need different remedies: cross-detector duplication is a merge"
      + " problem, same-detector duplication is an NMS problem.",
    direction: "none",
  },

  // --------------------------------------------------------------- s6, hubness
  hubness_skewness: {
    label: "Hubness skewness",
    measures:
      "Skewness of how often each row appears in others' 10 nearest neighbours.",
    reading:
      "Needs no ground truth at all. High means a few rows are everyone's neighbour and"
      + " crowd out the rest of the index.",
    bands: "< 1 healthy · 1–2 noticeable · > 2 a few rows dominate retrieval",
    why:
      "Hub rows answer queries they have nothing to do with, which costs precision"
      + " everywhere at once and looks like a ranking bug.",
    remedy:
      "Compare with the recentred figure next to it: if recentring fixes most of it,"
      + " the cause is a mean offset, not the data.",
    direction: "lower",
    verdict: below(1, 2),
  },
  hub_mass: {
    label: "Hub mass",
    measures: "Share of all neighbour slots taken by the hub rows.",
    reading: "How much of the retrieval budget the hubs actually consume.",
    why:
      "Skewness says the shape is uneven; this says whether that unevenness is big"
      + " enough to matter.",
    direction: "lower",
    verdict: below(0.05, 0.15),
  },

  // ------------------------------------------------------------ s7, label sets
  label_set_coverage: {
    label: "Label-set coverage",
    measures:
      "Share of annotations carrying a set of acceptable labels (ADR 0009) rather than"
      + " a single class name.",
    reading:
      "At zero this section is inert — not a measured zero. Nothing has label sets yet,"
      + " so the scores below it cannot be computed at all.",
    why:
      "A single class name cannot judge an open vocabulary: 'sofa' and 'booth seat' are"
      + " both right, and only a set can say so.",
    direction: "higher",
    verdict: above(0.5, 0.01),
  },
};

/ The explanation for a metric id, or null when nothing was written for it. */
export function explain(id: string): MetricExplanation | null {
  return METRICS[id] ?? null;
}

/ The band verdict for a value, neutral when the metric carries no bands. */
export function verdictFor(id: string, value: number): Verdict {
  const metric = METRICS[id];
  if (!metric?.verdict) {
    return "neutral";
  }
  return metric.verdict(value);
}
