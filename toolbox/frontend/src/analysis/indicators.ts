/**
 * The score-card indicators, derived from the raw `map_analysis` payload.
 *
 * Kept out of the panel because these are the only *computed* numbers in the feature —
 * everything else is a value the python tool already reported. A lift that subtracts
 * the wrong base rate, or a threshold cost read at the wrong point of the curve, is a
 * wrong verdict with a confident colour, so they are unit-tested rather than eyeballed.
 */
import type { AnalysisPayload } from "./types";

/** The value the online path uses, and the point the retention curve is read at. */
export const ONLINE_MIN_KEYFRAMES = 2;

export type Indicator = {
  /** Key into `METRICS`, so the tile and its overlay cannot disagree. */
  id: string;
  value: number;
  display: string;
  /** Shown under the value: what the number is made of, so the tile is auditable. */
  detail: string;
};

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export function percent(value: number | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(digits)}%`;
}

export function points(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }
  // A delta that rounds to nothing gets no sign: "+0.0 pts" reads as a gain that is
  // not there, and "-0.0 pts" as a loss, when the honest statement is "no difference".
  const rounded = Number((value * 100).toFixed(1));
  if (rounded === 0) {
    return "0.0 pts";
  }
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)} pts`;
}

export function number(value: number | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function integer(value: number | undefined): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : Math.round(value).toLocaleString("en-US");
}

export function metres(value: number | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)} m`;
}

// ---------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------

/**
 * A radius key off one of the tool's radius-keyed maps.
 *
 * The python side builds these with `str(radius)` over floats, so 2 m is `"2.0"` —
 * but a hand-written or older payload may carry `"2"`. Both are accepted rather than
 * silently missing, which would drop the lift tile with no explanation.
 */
function atRadius(
  table: Record<string, number> | undefined,
  radius: number,
): number | undefined {
  if (!table) {
    return undefined;
  }
  return table[radius.toFixed(1)] ?? table[String(radius)];
}

/** The best cue AND its name: the reader has to see WHICH cue won. */
export function bestConditionalCue(
  analysis: AnalysisPayload,
): { name: string; conditional: number; raw: number } | null {
  const auc = analysis.s4?.auc;
  if (!auc) {
    return null;
  }
  let best: { name: string; conditional: number; raw: number } | null = null;
  for (const [name, value] of Object.entries(auc)) {
    if (!best || value.conditional > best.conditional) {
      best = { name, conditional: value.conditional, raw: value.raw };
    }
  }
  return best;
}

/**
 * Annotations lost by raising `min_keyframes_per_cluster` from 1 to the online default,
 * as a share. `NaN` when the curve is absent — an older run — rather than a false zero.
 */
export function keyframeThresholdCost(analysis: AnalysisPayload): number {
  const curve = analysis.s3?.keyframe_threshold;
  if (!curve || curve.length === 0) {
    return Number.NaN;
  }
  const first = curve.find((point) => point.min_keyframes === 1);
  const atDefault = curve.find(
    (point) => point.min_keyframes === ONLINE_MIN_KEYFRAMES,
  );
  if (!first || !atDefault) {
    return Number.NaN;
  }
  return first.retained_share - atDefault.retained_share;
}

/** Attachment at 2 m minus the base rate at 2 m — the only honest form of either. */
export function attachmentLift(analysis: AnalysisPayload): number {
  const detections = analysis.s0?.detections ?? 0;
  const attached = atRadius(analysis.s3?.attachment_by_radius, 2);
  const baseRate = atRadius(analysis.s2?.base_rate, 2);
  if (!detections || attached == null || baseRate == null) {
    return Number.NaN;
  }
  return attached / detections - baseRate;
}

export function buildIndicators(analysis: AnalysisPayload): Indicator[] {
  const out: Indicator[] = [];
  const { s0, s1, s2, s3, s6 } = analysis;

  if (s0 && s0.detections > 0) {
    out.push({
      id: "placed_share",
      value: s0.placed / s0.detections,
      display: percent(s0.placed / s0.detections),
      detail: `${integer(s0.placed)} of ${integer(s0.detections)} rows`,
    });
  }
  if (s3) {
    out.push({
      id: "covered_in_own_keyframe",
      value: s3.covered_in_own_keyframe,
      display: percent(s3.covered_in_own_keyframe),
      detail: "depth-free, in the clicked panorama",
    });
    out.push({
      id: "source_keyframe_indexed",
      value: s3.source_keyframe_indexed,
      display: percent(s3.source_keyframe_indexed),
      detail: "annotations with an indexed panorama",
    });
  }
  const lift = attachmentLift(analysis);
  if (Number.isFinite(lift)) {
    const attached = atRadius(s3?.attachment_by_radius, 2) ?? 0;
    out.push({
      id: "attachment_lift",
      value: lift,
      display: points(lift),
      detail: `${percent(attached / (s0?.detections ?? 1))} attached vs ${percent(
        atRadius(s2?.base_rate, 2),
      )} base rate`,
    });
  }
  const cue = bestConditionalCue(analysis);
  if (cue) {
    out.push({
      id: "best_conditional_auc",
      value: cue.conditional,
      display: number(cue.conditional, 3),
      detail: `${cue.name} — raw ${number(cue.raw, 3)}`,
    });
  }
  const cost = keyframeThresholdCost(analysis);
  if (Number.isFinite(cost)) {
    out.push({
      id: "keyframe_threshold_cost",
      value: cost,
      // The metric is a cost, so it is shown as the loss it is: a negative delta.
      display: points(-cost),
      detail: `raising min_keyframes 1 → ${ONLINE_MIN_KEYFRAMES}`,
    });
  }
  if (s3?.observability.useful_pair_share) {
    const value = s3.observability.useful_pair_share.median;
    out.push({
      id: "useful_pair_share",
      value,
      display: percent(value),
      detail: "median over annotations, ≥ 5° apart",
    });
  }
  if (s1) {
    out.push({
      id: "range_p90_m",
      value: s1.range_m.p90,
      display: metres(s1.range_m.p90),
      detail: `max ${metres(s1.range_m.max)}, trusted to 15 m`,
    });
  }
  if (s2) {
    out.push({
      id: "purity_same_class",
      value: s2.embedding_purity.same_class,
      display: percent(s2.embedding_purity.same_class),
      detail: `object purity ${percent(s2.embedding_purity.same_object)}`,
    });
  }
  if (s6) {
    out.push({
      id: "hubness_skewness",
      value: s6.raw.skewness,
      display: number(s6.raw.skewness),
      detail: `${number(s6.centred.skewness)} once recentred`,
    });
  }
  if (s3) {
    out.push({
      id: "unseen_never_looked",
      value: s3.unseen.never_looked,
      display: integer(s3.unseen.never_looked),
      detail: `${integer(s3.unseen.looked_and_missed)} looked at and missed`,
    });
  }
  return out;
}
