/**
 * One similarity → colour scale, shared by the cosine matrix and the map rays.
 *
 * Blue is a weak match, red a strong one. Two detections of the same object must look
 * the same in the table and on the map, or the two views cannot be read together.
 */

/** Endpoints of the ramp, blue → violet → red. */
const WEAK: [number, number, number] = [37, 99, 235];
const MID: [number, number, number] = [147, 51, 234];
const STRONG: [number, number, number] = [220, 38, 38];

function mix(
  from: [number, number, number],
  to: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ];
}

/**
 * `[min, max]` of the finite values, for the stretched scale.
 *
 * Absolute [0, 1] is the honest default, but MetaCLIP image↔image similarity lives in
 * a narrow band around 0.69 — on an absolute ramp a whole basket is one shade of red,
 * whatever it contains. Stretching to the observed range is what makes the *relative*
 * structure visible; it must stay a deliberate choice, since it also makes a
 * uniformly-high matrix look like it has contrast.
 */
export function similarityRange(
  values: Array<number | null>,
): [number, number] | null {
  const finite = values.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  );
  if (finite.length < 2) {
    return null;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return max - min < 1e-6 ? null : [min, max];
}

export function matchColor(
  value: number | null,
  range: [number, number] | null = null,
): string {
  if (value === null || !Number.isFinite(value)) {
    return "#cbd5e1";
  }
  const [low, high] = range ?? [0, 1];
  const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
  const [r, g, b] = t < 0.5 ? mix(WEAK, MID, t * 2) : mix(MID, STRONG, (t - 0.5) * 2);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Distinct hues for the object partition — a different question from similarity, so
 * a different palette rather than another ramp of the same one.
 */
const OBJECT_COLORS = [
  "#0ea5e9",
  "#f97316",
  "#8b5cf6",
  "#16a34a",
  "#e11d48",
  "#0d9488",
  "#a16207",
  "#4f46e5",
];

/** `null` = claimed by no hypothesis. */
export function objectColor(hypothesisIndex: number | null): string {
  return hypothesisIndex === null
    ? "#94a3b8"
    : OBJECT_COLORS[hypothesisIndex % OBJECT_COLORS.length];
}
