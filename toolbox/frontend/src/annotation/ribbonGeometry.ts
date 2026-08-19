/**
 * Placing a proposal on the 360° ruler.
 *
 * The ruler is the keyframe's whole turn drawn at full width, so a proposal's stored
 * angle maps straight onto it: `u = (θ + π) / 2π`, the same relation
 * `backend/src/erp-geometry.ts` used to build the equirectangular image. The angles
 * are **unwrapped** — θ may sit just past ±π near the seam — so wrapping is the
 * whole job here, and getting it wrong puts a notch off the ruler with no symptom.
 */

function wrapUnit(value: number): number {
  // Short-circuit the common case: the modulo pair drifts by an ulp on values that
  // were already in range, which is enough to make a notch and its tick disagree.
  if (value >= 0 && value < 1) {
    return value;
  }
  return ((value % 1) + 1) % 1;
}

/** `u = (θ + π) / 2π`, wrapped into `[0, 1)`. Radians in. */
export function thetaToRatio(theta: number): number {
  return wrapUnit((theta + Math.PI) / (2 * Math.PI));
}

/** Ruler ticks every `stepDegrees`, from −180° to +180° inclusive. */
export function rulerTicks(stepDegrees = 30): { degrees: number; ratio: number }[] {
  const ticks: { degrees: number; ratio: number }[] = [];
  for (let degrees = -180; degrees <= 180; degrees += stepDegrees) {
    ticks.push({ degrees, ratio: (degrees + 180) / 360 });
  }
  return ticks;
}
