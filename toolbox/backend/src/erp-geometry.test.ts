import assert from "node:assert/strict";
import test from "node:test";

import {
  angularAreaDeg2,
  assertEquirect2to1,
  cutoutRatioToErpUv,
  erpBboxRatios,
  erpRectsWrapped,
  gnomonicFfmpegFilter,
  isRenderableGnomonic,
  paddingMask,
  type ErpViewAngles,
} from "./erp-geometry.js";

/**
 * Round to the nearest float16, as `prepare/writer.py` does on the way out.
 *
 * Node 20 has no `Float16Array`, so this goes through the IEEE half encoding by hand:
 * the point of the test is that the *stored* precision is enough, and reproducing the
 * quantisation is the only way to check that.
 */
function toFloat16(value: number): number {
  const float32 = new Float32Array([value]);
  const bits = new Uint32Array(float32.buffer)[0];
  const sign = (bits >>> 31) & 0x1;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  const unbiased = exponent - 127 + 15;
  if (unbiased <= 0 || unbiased >= 0x1f) {
    // Out of half range: no test case goes there (angles live in [-pi, pi]).
    return value;
  }
  // Round-to-nearest-even on the 13 dropped mantissa bits.
  let half = (sign << 15) | (unbiased << 10) | (mantissa >>> 13);
  const dropped = mantissa & 0x1fff;
  if (dropped > 0x1000 || (dropped === 0x1000 && (half & 1) === 1)) {
    half += 1;
  }
  const halfSign = (half >>> 15) & 0x1 ? -1 : 1;
  const halfExponent = (half >>> 10) & 0x1f;
  const halfMantissa = half & 0x3ff;
  return halfSign * (1 + halfMantissa / 1024) * 2 ** (halfExponent - 15);
}

/** `prepare/convention.py::erp_pixel_centers_to_spherical` + the angular extents. */
function anglesFromErpBox(
  box: { x0: number; y0: number; x1: number; y1: number },
  width: number,
  height: number,
): ErpViewAngles {
  const cx = (box.x0 + box.x1) / 2;
  const cy = (box.y0 + box.y1) / 2;
  return {
    thetaCenter: (cx / width) * 2 * Math.PI - Math.PI,
    phiCenter: Math.PI / 2 - (cy / height) * Math.PI,
    angularWidth: ((box.x1 - box.x0) / width) * 2 * Math.PI,
    angularHeight: ((box.y1 - box.y0) / height) * Math.PI,
  };
}

test("erpBboxRatios round-trips a known ERP box through the stored angles", () => {
  const width = 5760;
  const height = 2880;
  const box = { x0: 1200, y0: 640, x1: 1560, y1: 900 };
  const rect = erpBboxRatios(anglesFromErpBox(box, width, height));

  // Sub-pixel: this leg of the chain is exact, float16 quantisation happens on the
  // way into the parquet, not here.
  assert.ok(Math.abs(rect.u0 * width - box.x0) < 1e-6);
  assert.ok(Math.abs(rect.u1 * width - box.x1) < 1e-6);
  assert.ok(Math.abs(rect.v0 * height - box.y0) < 1e-6);
  assert.ok(Math.abs(rect.v1 * height - box.y1) < 1e-6);
});

test("erpBboxRatios survives float16 quantisation to a few pixels", () => {
  const width = 5760;
  const height = 2880;
  const box = { x0: 5000, y0: 200, x1: 5120, y1: 320 };
  const exact = anglesFromErpBox(box, width, height);
  const rect = erpBboxRatios({
    thetaCenter: toFloat16(exact.thetaCenter),
    phiCenter: toFloat16(exact.phiCenter),
    angularWidth: toFloat16(exact.angularWidth),
    angularHeight: toFloat16(exact.angularHeight),
  });
  assert.ok(Math.abs(rect.u0 * width - box.x0) < 4, `u0 off by ${rect.u0 * width - box.x0}px`);
  assert.ok(Math.abs(rect.v1 * height - box.y1) < 4, `v1 off by ${rect.v1 * height - box.y1}px`);
});

test("erpRectsWrapped splits a seam-crossing box in two and keeps the total width", () => {
  const angles: ErpViewAngles = {
    // Centred on the seam: theta = +pi is the same column as -pi.
    thetaCenter: Math.PI - 0.01,
    phiCenter: 0,
    angularWidth: 0.1,
    angularHeight: 0.1,
  };
  const rects = erpRectsWrapped(angles);
  assert.equal(rects.length, 2);
  const total = rects.reduce((sum, rect) => sum + (rect.u1 - rect.u0), 0);
  assert.ok(Math.abs(total - 0.1 / (2 * Math.PI)) < 1e-9);
  for (const rect of rects) {
    assert.ok(rect.u0 >= 0 && rect.u1 <= 1);
  }
});

test("cutoutRatioToErpUv puts the cutout centre back on the view centre", () => {
  const angles: ErpViewAngles = {
    thetaCenter: 0.7,
    phiCenter: -0.4,
    angularWidth: 0.3,
    angularHeight: 0.2,
  };
  const [u, v] = cutoutRatioToErpUv(angles, 0.5, 0.5);
  assert.ok(Math.abs(u - (angles.thetaCenter + Math.PI) / (2 * Math.PI)) < 1e-12);
  assert.ok(Math.abs(v - (Math.PI / 2 - angles.phiCenter) / Math.PI) < 1e-12);
});

test("cutoutRatioToErpUv sends the top of the cutout to +phi", () => {
  const angles: ErpViewAngles = {
    thetaCenter: 0,
    phiCenter: 0,
    angularWidth: 0.4,
    angularHeight: 0.4,
  };
  const [, vTop] = cutoutRatioToErpUv(angles, 0.5, 0);
  const [, vBottom] = cutoutRatioToErpUv(angles, 0.5, 1);
  // v grows downwards, and the top of the grid is +phi — the sign convention that
  // silently mirrors every preview if it is reversed.
  assert.ok(vTop < 0.5, `top ratio ${vTop} should be above the equator`);
  assert.ok(vBottom > 0.5, `bottom ratio ${vBottom} should be below the equator`);
});

test("assertEquirect2to1 rejects a non-2:1 raster", () => {
  assert.doesNotThrow(() => assertEquirect2to1(5760, 2880));
  assert.throws(() => assertEquirect2to1(5760, 2000), /not 2:1 equirectangular/);
});

test("gnomonicFfmpegFilter converts to degrees without negating an axis", () => {
  const filter = gnomonicFfmpegFilter(
    { thetaCenter: 0.5, phiCenter: -0.25, angularWidth: 0.2, angularHeight: 0.1 },
    { size: 336 },
  );
  assert.match(filter, /^v360=input=equirect:output=flat:/);
  assert.match(filter, new RegExp(`yaw=${0.5 * 180 / Math.PI}`));
  assert.match(filter, new RegExp(`pitch=${-0.25 * 180 / Math.PI}`));
  assert.match(filter, new RegExp(`h_fov=${0.2 * 180 / Math.PI}`));
  assert.match(filter, new RegExp(`v_fov=${0.1 * 180 / Math.PI}`));
  // Aspect follows the FOVs: 2:1 here, so 336x168 rather than a square.
  assert.match(filter, /w=336:h=168$/);
  // ffmpeg's default rotation order matches the pipeline; pinning it would break it.
  assert.ok(!filter.includes("rorder"));
});

test("isRenderableGnomonic rejects a proposal spanning 180 degrees or more", () => {
  assert.ok(
    isRenderableGnomonic({
      thetaCenter: 0,
      phiCenter: 0,
      angularWidth: 0.2,
      angularHeight: 0.1,
    }),
  );
  // Observed in real data: a GroundingDINO "handrail" at angular_width = 357°. There
  // is no rectilinear view of it, and the stored thumbnail is degenerate too —
  // `create_proposal_cutouts` evaluated tan(3.12), which is negative.
  assert.equal(
    isRenderableGnomonic({
      thetaCenter: 0,
      phiCenter: -1.13,
      angularWidth: 6.238,
      angularHeight: 0.842,
    }),
    false,
  );
});

test("paddingMask crops a wide proposal vertically and a tall one horizontally", () => {
  const wide = paddingMask(
    { thetaCenter: 0, phiCenter: 0, angularWidth: 0.4, angularHeight: 0.1 },
    9,
    9,
  );
  // ratio 4 > 1: keep |base_y| <= 1/4, i.e. the middle rows only.
  assert.equal(wide[4 * 9 + 4], 1, "centre kept");
  assert.equal(wide[0 * 9 + 4], 0, "top row dropped");
  assert.equal(wide[4 * 9 + 0], 1, "left edge kept at mid-height");

  const tall = paddingMask(
    { thetaCenter: 0, phiCenter: 0, angularWidth: 0.1, angularHeight: 0.4 },
    9,
    9,
  );
  assert.equal(tall[4 * 9 + 4], 1);
  assert.equal(tall[4 * 9 + 0], 0, "left column dropped");
  assert.equal(tall[0 * 9 + 4], 1, "top edge kept at mid-width");
});

test("angularAreaDeg2 is the product of the two extents in degrees", () => {
  const area = angularAreaDeg2({
    thetaCenter: 0,
    phiCenter: 0,
    angularWidth: Math.PI / 180,
    angularHeight: (2 * Math.PI) / 180,
  });
  assert.ok(Math.abs(area - 2) < 1e-9);
});
