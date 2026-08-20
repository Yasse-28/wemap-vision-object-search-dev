/**
 * Measuring a region drawn on the panorama.
 *
 * Every function here has to cope with the seam: a region around θ = ±180° has
 * vertices at both ends of the `u` range, and treating `u` as a plain number there
 * puts the centroid on the opposite side of the room and the width at nearly a full
 * turn. Both failures are silent — a saved point simply lands somewhere wrong.
 */

export type RegionVertex = [number, number];

const TWO_PI = 2 * Math.PI;

function wrapUnit(value: number): number {
  if (value >= 0 && value < 1) {
    return value;
  }
  return ((value % 1) + 1) % 1;
}

/**
 * The region's centre in texture ratios. `u` is averaged as an angle, so a region
 * straddling the seam averages to the seam rather than to the far side.
 */
export function regionCentroid(vertices: RegionVertex[]): RegionVertex | null {
  if (!vertices.length) {
    return null;
  }
  let sumSin = 0;
  let sumCos = 0;
  let sumV = 0;
  for (const [u, v] of vertices) {
    const angle = wrapUnit(u) * TWO_PI;
    sumSin += Math.sin(angle);
    sumCos += Math.cos(angle);
    sumV += v;
  }
  const v = sumV / vertices.length;
  // A degenerate ring (vertices evenly spread around the turn) has no meaningful
  // mean direction; fall back to the first vertex rather than to an arbitrary angle.
  if (Math.hypot(sumSin, sumCos) < 1e-9) {
    return [wrapUnit(vertices[0][0]), v];
  }
  return [wrapUnit(Math.atan2(sumSin, sumCos) / TWO_PI), v];
}

/**
 * How wide the region is, in radians of yaw: the turn minus its largest empty gap.
 * That is the only definition that stays right across the seam.
 */
export function regionAngularWidthRad(vertices: RegionVertex[]): number {
  if (vertices.length < 2) {
    return 0;
  }
  const angles = vertices.map(([u]) => wrapUnit(u) * TWO_PI).sort((a, b) => a - b);
  let largestGap = angles[0] + TWO_PI - angles[angles.length - 1];
  for (let index = 1; index < angles.length; index += 1) {
    largestGap = Math.max(largestGap, angles[index] - angles[index - 1]);
  }
  return Math.max(0, TWO_PI - largestGap);
}

/**
 * Distance between two texture ratios, with `u` measured the short way round.
 *
 * Used to decide whether a click landed on the first vertex and therefore closes the
 * outline. Without the wrap, a region traced across the seam could never be closed:
 * two points a pixel apart read as almost a full turn.
 */
export function ratioDistance(a: RegionVertex, b: RegionVertex): number {
  const rawU = Math.abs(wrapUnit(a[0]) - wrapUnit(b[0]));
  const deltaU = Math.min(rawU, 1 - rawU);
  return Math.hypot(deltaU, a[1] - b[1]);
}

/**
 * How wide a thing is, from how wide it looks and how far away it is.
 *
 * `yawWidthRad` is a span in **yaw**, which is not the object's apparent angular
 * size: meridians converge towards the poles, so the same object seen high up spans
 * more yaw than one at eye level. The `cos(φ)` factor is that correction — without
 * it, a sign near the ceiling is reported far wider than it is.
 *
 * This is a suggestion for a field the annotator owns, never a measurement: it
 * assumes the object faces the camera and that the depth belongs to it rather than
 * to the wall behind, and neither can be checked from here.
 */
export function extentFromAngularWidth(
  yawWidthRad: number,
  phiRad: number,
  depthM: number | null,
): number | null {
  if (depthM === null || !Number.isFinite(depthM) || depthM <= 0) {
    return null;
  }
  if (!Number.isFinite(yawWidthRad) || yawWidthRad <= 0) {
    return null;
  }
  const apparent = yawWidthRad * Math.cos(phiRad);
  // Past a half turn the tangent is meaningless, and so is a fronto-parallel object
  // that wide: answer nothing rather than a number nobody can read.
  if (apparent <= 0 || apparent >= Math.PI) {
    return null;
  }
  return Math.round(2 * depthM * Math.tan(apparent / 2) * 100) / 100;
}

/** `φ` for a texture-ratio `v`: the top of the image is +π/2. */
export function ratioToPhiRad(vRatio: number): number {
  return (0.5 - vRatio) * Math.PI;
}

/** The same suggestion for a drawn outline, measured across its own vertices. */
export function estimatedExtentM(
  vertices: RegionVertex[],
  depthM: number | null,
): number | null {
  const centroid = regionCentroid(vertices);
  if (!centroid) {
    return null;
  }
  return extentFromAngularWidth(
    regionAngularWidthRad(vertices),
    ratioToPhiRad(centroid[1]),
    depthM,
  );
}
