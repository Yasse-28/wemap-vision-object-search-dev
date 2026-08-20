import assert from "node:assert/strict";
import test from "node:test";

import { rulerTicks, thetaToRatio } from "./ribbonGeometry";

test("theta maps to the same u the equirectangular image was built with", () => {
  assert.equal(thetaToRatio(-Math.PI), 0);
  assert.equal(thetaToRatio(0), 0.5);
  assert.ok(Math.abs(thetaToRatio(Math.PI / 2) - 0.75) < 1e-12);
});

test("an angle past the seam lands back on the ruler", () => {
  assert.ok(Math.abs(thetaToRatio(Math.PI + 0.1) - thetaToRatio(-Math.PI + 0.1)) < 1e-12);
  assert.ok(thetaToRatio(-Math.PI - 0.1) > 0.9);
});

test("the ruler spans the whole turn", () => {
  const ticks = rulerTicks();
  assert.equal(ticks.length, 13);
  assert.deepEqual(ticks[0], { degrees: -180, ratio: 0 });
  assert.deepEqual(ticks[12], { degrees: 180, ratio: 1 });
});
