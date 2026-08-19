/**
 * Client for the Annotation office's store.
 *
 * The office used to keep everything in React state and write one
 * `annotations/annotations.geojson` on demand — a file nothing ever read back, so
 * the workspace opened empty every time and the work never reached the benchmark.
 * It now reads and writes `{map}/object-search-annotations.db`, where the reviews
 * and the reference partition already live.
 *
 * Points are stored as `ground_truth_point`: they *are* the benchmark's ground
 * truth. Deleting one here removes it from the benchmark, which is why the UI
 * confirms it.
 */

import { fetchJson } from "../http";
import type { AnnotationClass, AnnotationFeature, AnnotationType } from "./types";

type WireClass = {
  name: string;
  annotationType: AnnotationType;
  color: string;
  prompt: string | null;
};

type WireAnnotation = {
  id: string;
  className: string;
  annotationType: AnnotationType;
  prompt: string | null;
  level: string | null;
  altitude: number | null;
  accuracyM: number | null;
  coordinates: number[] | number[][];
  source: AnnotationFeature["source"];
  groundTruth?: AnnotationFeature["groundTruth"];
};

export const DEFAULT_ACCURACY_FALLBACK_M = 2;

function emptyGroundTruth(): AnnotationFeature["groundTruth"] {
  return {
    objectId: null,
    extentM: null,
    exhaustiveZone: null,
    isDepiction: false,
    labels: {
      synonyms: [],
      depictions: [],
      visuallySimilar: [],
      clutter: [],
    },
  };
}

function base(mapId: string): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}/annotations`;
}

function send(path: string, method: string, body: unknown): Promise<unknown> {
  return fetchJson(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/**
 * Rebuild the shape the workspace renders.
 *
 * `classColor` is denormalised onto each annotation because every overlay and list
 * row reads it directly; the store keeps it once, on the class.
 */
export async function fetchWorkspaceAnnotations(mapId: string): Promise<{
  classes: AnnotationClass[];
  annotations: AnnotationFeature[];
}> {
  const data = (await fetchJson(base(mapId))) as {
    classes: WireClass[];
    annotations: WireAnnotation[];
  };
  const classes: AnnotationClass[] = data.classes.map((item) => ({
    name: item.name,
    color: item.color,
    prompt: item.prompt,
    annotationType: item.annotationType,
  }));
  const colorFor = new Map(
    classes.map((item) => [`${item.annotationType}:${item.name}`, item.color]),
  );
  const annotations: AnnotationFeature[] = data.annotations.map((item) => ({
    id: item.id,
    className: item.className,
    prompt: item.prompt,
    classColor:
      colorFor.get(`${item.annotationType}:${item.className}`) ?? "#2563eb",
    annotationType: item.annotationType,
    geometryType: item.annotationType === "polygon" ? "Polygon" : "Point",
    coordinates:
      item.annotationType === "polygon"
        ? [item.coordinates as number[][]]
        : (item.coordinates as number[]),
    altitude: item.altitude,
    level: item.level,
    accuracyM: item.accuracyM ?? DEFAULT_ACCURACY_FALLBACK_M,
    source: item.source ?? null,
    groundTruth: item.groundTruth ?? emptyGroundTruth(),
  }));
  return { classes, annotations };
}

/** Store one annotation; the server's id replaces the client's placeholder. */
export async function createAnnotation(
  mapId: string,
  annotation: AnnotationFeature,
): Promise<string> {
  const data = (await send(
    `${base(mapId)}/annotation`,
    "POST",
    annotationPayload(annotation),
  )) as { id: string };
  return data.id;
}

function annotationPayload(annotation: AnnotationFeature): Record<string, unknown> {
  const coordinates =
    annotation.annotationType === "polygon"
      ? (annotation.coordinates as number[][][])[0]
      : (annotation.coordinates as number[]);
  return {
    id: annotation.id,
    className: annotation.className,
    annotationType: annotation.annotationType,
    prompt: annotation.prompt,
    level: annotation.level,
    altitude: annotation.altitude,
    accuracyM: annotation.accuracyM,
    coordinates,
    source: annotation.source,
    groundTruth: annotation.groundTruth,
  };
}

export async function updateAnnotation(
  mapId: string,
  annotation: AnnotationFeature,
): Promise<void> {
  await send(`${base(mapId)}/annotation`, "PUT", annotationPayload(annotation));
}

export async function deleteAnnotation(mapId: string, id: string): Promise<void> {
  await send(`${base(mapId)}/annotation`, "DELETE", { id });
}

export async function clearAnnotations(mapId: string): Promise<number> {
  const data = (await send(base(mapId), "DELETE", {})) as { deleted: number };
  return data.deleted;
}

/** Create, rename or recolour a class. A rename cascades to its annotations. */
export async function saveClass(
  mapId: string,
  next: AnnotationClass,
  original: { name: string; annotationType: AnnotationType } | null,
): Promise<void> {
  await send(`${base(mapId)}/class`, "POST", { class: next, original });
}

export async function deleteClass(
  mapId: string,
  name: string,
  annotationType: AnnotationType,
): Promise<void> {
  await send(`${base(mapId)}/class`, "DELETE", { name, annotationType });
}

/** Merge a GeoJSON FeatureCollection — the file-import path, kept for old files. */
export async function importAnnotations(
  mapId: string,
  collection: unknown,
): Promise<{ imported: number; skipped: number }> {
  return (await send(base(mapId), "POST", collection)) as {
    imported: number;
    skipped: number;
  };
}
