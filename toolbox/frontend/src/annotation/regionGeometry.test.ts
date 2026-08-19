import assert from "node:assert/strict";
import test from "node:test";

import {
  estimatedExtentM,
  extentFromAngularWidth,
  ratioDistance,
  ratioToPhiRad,
  regionAngularWidthRad,
  regionCentroid,
  type RegionVertex,
} from "./regionGeometry";

const QUAD: RegionVertex[] = [
  [0.40, 0.45],
  [0.46, 0.45],
  [0.46, 0.60],
  [0.40, 0.60],
];

/** The same quad, moved so that it straddles u = 0. */
const SEAM_QUAD: RegionVertex[] = [
  [0.97, 0.45],
  [0.03, 0.45],
  [0.03, 0.60],
  [0.97, 0.60],
];

test("the centroid of an ordinary region is its middle", () => {
  const centroid = regionCentroid(QUAD);
  assert.ok(centroid);
  assert.ok(Math.abs(centroid[0] - 0.43) < 1e-9);
  assert.ok(Math.abs(centroid[1] - 0.525) < 1e-9);
});

test("a region across the seam centres on the seam, not on the far side", () => {
  const centroid = regionCentroid(SEAM_QUAD);
  assert.ok(centroid);
  // Near u = 0, from either side. A plain mean would answer 0.5 — the far wall.
  assert.ok(centroid[0] < 0.01 || centroid[0] > 0.99, `got ${centroid[0]}`);
});

test("width is measured the same on either side of the seam", () => {
  const plain = regionAngularWidthRad(QUAD);
  const across = regionAngularWidthRad(SEAM_QUAD);
  assert.ok(Math.abs(plain - 0.06 * 2 * Math.PI) < 1e-9);
  assert.ok(Math.abs(across - plain) < 1e-9);
});

test("a single vertex has no width and no extent", () => {
  assert.equal(regionAngularWidthRad([[0.5, 0.5]]), 0);
  assert.equal(estimatedExtentM([[0.5, 0.5]], 4), null);
  assert.equal(regionCentroid([]), null);
});

test("the extent estimate follows depth and stays a rounded guess", () => {
  const near = estimatedExtentM(QUAD, 2);
  const far = estimatedExtentM(QUAD, 8);
  assert.ok(near && far && far > near);
  assert.equal(near, Math.round(near * 100) / 100);
});

test("no depth, or a region wider than a half turn, yields no estimate", () => {
  assert.equal(estimatedExtentM(QUAD, null), null);
  assert.equal(estimatedExtentM(QUAD, 0), null);
  assert.equal(
    estimatedExtentM([[0.0, 0.5], [0.35, 0.5], [0.7, 0.5]], 4),
    null,
  );
});

test("closing distance takes the short way round the seam", () => {
  // A pixel apart on either side of u = 0, not almost a whole turn apart.
  assert.ok(ratioDistance([0.999, 0.5], [0.001, 0.5]) < 0.01);
  assert.ok(Math.abs(ratioDistance([0.2, 0.5], [0.3, 0.5]) - 0.1) < 1e-9);
  assert.ok(Math.abs(ratioDistance([0.2, 0.4], [0.2, 0.5]) - 0.1) < 1e-9);
});

test("the same yaw span means a smaller object away from eye level", () => {
  const atEyeLevel = extentFromAngularWidth(0.2, 0, 5);
  const nearTheCeiling = extentFromAngularWidth(0.2, ratioToPhiRad(0.2), 5);
  assert.ok(atEyeLevel && nearTheCeiling);
  // Meridians converge: the same yaw span covers less of the object up there.
  assert.ok(nearTheCeiling < atEyeLevel, `${nearTheCeiling} !< ${atEyeLevel}`);
});

test("a suggestion needs a positive width and a usable depth", () => {
  assert.equal(extentFromAngularWidth(0.2, 0, null), null);
  assert.equal(extentFromAngularWidth(0.2, 0, 0), null);
  assert.equal(extentFromAngularWidth(0, 0, 5), null);
  assert.equal(extentFromAngularWidth(Number.NaN, 0, 5), null);
  // Wider than a half turn once corrected: no answer rather than a wrong one.
  assert.equal(extentFromAngularWidth(Math.PI + 0.2, 0, 5), null);
});

test("the suggestion scales with distance", () => {
  const near = extentFromAngularWidth(0.2, 0, 2);
  const far = extentFromAngularWidth(0.2, 0, 8);
  assert.ok(near && far);
  // Four times the distance, four times the size — within the 1 cm rounding.
  assert.ok(Math.abs(far - 4 * near) <= 0.02, `${far} vs ${4 * near}`);
});
