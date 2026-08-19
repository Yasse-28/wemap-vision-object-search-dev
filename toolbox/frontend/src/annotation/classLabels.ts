import type { AnnotationFeature } from "../annotations/types";

/**
 * The label sets that belong to a class rather than to a point.
 *
 * Synonyms and visually-similar terms describe what a *chair* is and what gets
 * mistaken for one — they are properties of the class, and retyping them per point is
 * both tedious and a source of drift: two spellings of the same set turn one class
 * into two as far as any consumer of the ground truth is concerned.
 *
 * They are not stored on the class (`annotation_class` holds name, colour, prompt and
 * type), so they are read back from the points already saved under it. The most
 * frequent set wins, ties going to the most recent, which lets a correction spread
 * without one stray point rewriting the class.
 */
export type ClassLabelDefaults = {
  synonyms: string[];
  visuallySimilar: string[];
};

function mostRepresentative(sets: string[][]): string[] {
  const counts = new Map<string, { value: string[]; count: number; lastIndex: number }>();
  sets.forEach((value, index) => {
    if (!value.length) {
      return;
    }
    const key = value.join(" ");
    const entry = counts.get(key);
    if (entry) {
      entry.count += 1;
      entry.lastIndex = index;
    } else {
      counts.set(key, { value, count: 1, lastIndex: index });
    }
  });
  let best: { value: string[]; count: number; lastIndex: number } | null = null;
  for (const entry of counts.values()) {
    if (
      !best ||
      entry.count > best.count ||
      (entry.count === best.count && entry.lastIndex > best.lastIndex)
    ) {
      best = entry;
    }
  }
  return best?.value ?? [];
}

export function classLabelDefaults(
  annotations: AnnotationFeature[],
  className: string | null,
): ClassLabelDefaults | null {
  if (!className) {
    return null;
  }
  const ofClass = annotations.filter(
    (item) => item.annotationType === "point" && item.className === className,
  );
  if (!ofClass.length) {
    return null;
  }
  const defaults: ClassLabelDefaults = {
    synonyms: mostRepresentative(ofClass.map((item) => item.groundTruth.labels.synonyms)),
    visuallySimilar: mostRepresentative(
      ofClass.map((item) => item.groundTruth.labels.visuallySimilar),
    ),
  };
  if (!defaults.synonyms.length && !defaults.visuallySimilar.length) {
    return null;
  }
  return defaults;
}
