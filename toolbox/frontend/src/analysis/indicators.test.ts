/**
 * The derived indicators and the bands that colour them.
 *
 * Two things are worth pinning here. The lift subtracts the base rate, because an
 * attachment rate on its own climbs with annotation density and reads as a result when
 * it is not one. And the retention curve is read at the threshold the online path
 * actually uses, not at the first point of the array — an older payload without the
 * curve has to yield NaN rather than a confident zero.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  attachmentLift,
  bestConditionalCue,
  buildIndicators,
  keyframeThresholdCost,
  percent,
  points,
} from "./indicators";
import { verdictFor } from "./metricCatalogue";
import type { AnalysisPayload } from "./types";

const PERCENTILES = { n: 10, p10: 0, median: 0.5, p90: 1, max: 2 };

/** A payload shaped like bbhotel-choisy's, trimmed to what the indicators read. */
function payload(): AnalysisPayload {
  return {
    s0: {
      map: "test",
      keyframes_manifest: 100,
      keyframes_with_detections: 100,
      detections: 1000,
      placed: 1000,
      ground_truth: 50,
      reviews: 0,
      group_labels: 0,
      embeddings: [1000, 512],
      missing: [],
    },
    s1: {
      range_m: { n: 1000, p10: 1, median: 2.5, p90: 9, max: 100 },
      angular_diagonal_deg: PERCENTILES,
      phi_deg: PERCENTILES,
      per_keyframe: PERCENTILES,
      by_source: {},
    },
    s2: {
      attachment_radius_m: 2,
      embedding_purity: {
        neighbours: 10,
        queries: 100,
        same_object: 0.13,
        same_class: 0.44,
      },
      base_rate: { "0.5": 0.07, "1.0": 0.18, "2.0": 0.5 },
      mutual_information_label_gt_class: 0.146,
      p_label_given_gt_class: {},
      p_gt_class_given_label: {},
    },
    s3: {
      source_keyframe_indexed: 0.62,
      covered_in_own_keyframe: 0.63,
      covered_by_class: {},
      // 600 of 1000 rows attached, against a base rate of 0.5 → a lift of 10 points.
      attachment_by_radius: { "0.5": 100, "1.0": 300, "2.0": 600 },
      observability: { useful_pair_share: PERCENTILES },
      keyframe_threshold: [
        {
          min_keyframes: 1,
          retained: 80,
          retained_share: 0.8,
          covered_share: 0.66,
          measurable: 50,
        },
        {
          min_keyframes: 2,
          retained: 75,
          retained_share: 0.75,
          covered_share: 0.66,
          measurable: 48,
        },
      ],
      degenerate_baseline: 0,
      unseen: { total: 10, never_looked: 0, looked_and_missed: 10 },
    },
    s4: {
      sampled_pairs: 100,
      same_object: 40,
      different_object: 40,
      contaminated: 20,
      auc: {
        distance_m: { raw: 0.99, conditional: 0.78 },
        cosine: { raw: 0.64, conditional: 0.56 },
      },
    },
    s6: {
      sampled: 100,
      k: 10,
      raw: { skewness: 2.78, hub_share: 0.01, hub_mass: 0.06, antihub_share: 0.05 },
      centred: { skewness: 1.5, hub_share: 0.001, hub_mass: 0.04, antihub_share: 0.02 },
      hub_labels: {},
    },
  };
}

test("the lift is attachment minus the base rate, not attachment", () => {
  assert.ok(Math.abs(attachmentLift(payload()) - 0.1) < 1e-9);
});

test("a base rate keyed without a decimal is still found", () => {
  const data = payload();
  data.s2!.base_rate = { "2": 0.5 };
  data.s3!.attachment_by_radius = { "2": 600 };
  assert.ok(Math.abs(attachmentLift(data) - 0.1) < 1e-9);
});

test("no lift without a base rate to subtract", () => {
  const data = payload();
  delete data.s2;
  assert.ok(Number.isNaN(attachmentLift(data)));
});

test("the threshold cost is read at the online default", () => {
  assert.ok(Math.abs(keyframeThresholdCost(payload()) - 0.05) < 1e-9);
});

test("a payload from before the curve yields NaN, not a free threshold", () => {
  const data = payload();
  delete data.s3!.keyframe_threshold;
  assert.ok(Number.isNaN(keyframeThresholdCost(data)));
});

test("the best cue is named, and it is the conditional winner", () => {
  const best = bestConditionalCue(payload());
  assert.equal(best?.name, "distance_m");
  assert.equal(best?.conditional, 0.78);
});

test("every indicator carries a metric the catalogue can explain", () => {
  for (const indicator of buildIndicators(payload())) {
    assert.notEqual(
      verdictFor(indicator.id, indicator.value),
      undefined,
      `${indicator.id} has no catalogue entry`,
    );
  }
});

test("sections absent from the payload produce no tile", () => {
  const indicators = buildIndicators({});
  assert.equal(indicators.length, 0);
});

test("the bands judge the measured map the way the report reads it", () => {
  // Depth-free coverage at 63 % is the detector being the limit, not a pass.
  assert.equal(verdictFor("covered_in_own_keyframe", 0.63), "warn");
  // A cue that only re-states geometry.
  assert.equal(verdictFor("best_conditional_auc", 0.56), "warn");
  assert.equal(verdictFor("best_conditional_auc", 0.78), "good");
  // Range past the trusted depth range.
  assert.equal(verdictFor("range_p90_m", 9), "good");
  assert.equal(verdictFor("range_p90_m", 30), "bad");
  // Hubness: raw is suspect, recentred is not.
  assert.equal(verdictFor("hubness_skewness", 2.78), "bad");
  assert.equal(verdictFor("hubness_skewness", 0.9), "good");
});

test("a metric without bands never gets a colour", () => {
  assert.equal(verdictFor("cue_raw_auc", 0.99), "neutral");
  assert.equal(verdictFor("purity_same_object", 0.13), "neutral");
});

test("a non-finite value is not judged", () => {
  assert.equal(verdictFor("covered_in_own_keyframe", Number.NaN), "neutral");
});

test("formatters mark absent numbers rather than printing 0", () => {
  assert.equal(percent(undefined), "—");
  assert.equal(percent(Number.NaN), "—");
  assert.equal(percent(0.634), "63.4%");
  assert.equal(points(-0.05), "-5.0 pts");
  assert.equal(points(0.05), "+5.0 pts");
  // A delta that rounds to nothing must not read as a gain or a loss.
  assert.equal(points(0), "0.0 pts");
  assert.equal(points(-0.00001), "0.0 pts");
});
