import {
  AnnotationClass,
  AnnotationFeature,
  AnnotationSource,
  AnnotationType,
  isHexColor,
} from "./types";

const DEFAULT_COLOR = "#ff6b6b";
const DEFAULT_ACCURACY_M = 5.0;

type Properties = Record<string, unknown>;
type Geometry = { type?: unknown; coordinates?: unknown };
type FileSystemWritable = {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
};
type SaveFileHandle = {
  createWritable: () => Promise<FileSystemWritable>;
};
type SaveFilePickerWindow = Window & {
  showSaveFilePicker?: (options: {
    suggestedName: string;
    types: Array<{
      description: string;
      accept: Record<string, string[]>;
    }>;
  }) => Promise<SaveFileHandle>;
};

export type SaveGeoJSONResult = "saved" | "downloaded" | "cancelled";
export type SaveMapGeoJSONResult = {
  saved: boolean;
  annotation_count: number;
  path: string;
  relative_path: string;
};

function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `ann-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function buildAnnotationGeoJSON(
  mapId: string,
  annotations: AnnotationFeature[],
): {
  type: "FeatureCollection";
  name: string;
  features: unknown[];
} {
  const features = annotations.map((annotation) => ({
    type: "Feature" as const,
    id: annotation.id,
    geometry: {
      type: annotation.geometryType,
      coordinates: annotation.coordinates,
    },
    properties: {
      class: annotation.className,
      ...(annotation.prompt ? { prompt: annotation.prompt } : {}),
      "marker-color": annotation.classColor,
      altitude: annotation.altitude,
      level: annotation.level,
      accuracy: annotation.accuracyM,
      annotation_type: annotation.annotationType,
      ...(annotation.source
        ? {
            source_keyframe_id: annotation.source.keyframeId,
            source_projection: annotation.source.projection,
            ...(annotation.source.cutoutId
              ? { source_cutout_id: annotation.source.cutoutId }
              : {}),
            source_x_ratio: annotation.source.xRatio,
            source_y_ratio: annotation.source.yRatio,
            source_erp_u: annotation.source.erpU,
            source_erp_v: annotation.source.erpV,
            depth_m: annotation.source.depthM,
          }
        : {}),
    },
  }));

  return {
    type: "FeatureCollection",
    name: `${mapId}_annotations`,
    features,
  };
}

export type ParseResult = {
  classes: AnnotationClass[];
  annotations: AnnotationFeature[];
  createdClasses: number;
  importedAnnotations: number;
};

export function parseAnnotationGeoJSON(
  text: string,
  existingClasses: AnnotationClass[],
  existingAnnotations: AnnotationFeature[],
): ParseResult {
  const data = JSON.parse(text) as { type?: unknown; features?: unknown };
  const features = data.features;
  if (data.type !== "FeatureCollection" || !Array.isArray(features)) {
    throw new Error("GeoJSON must be a FeatureCollection.");
  }

  const classes = [...existingClasses];
  const annotations = [...existingAnnotations];
  const classIndex = new Map<string, AnnotationClass>();
  for (const item of classes) {
    classIndex.set(classKey(item.name, item.annotationType), item);
  }

  let createdClasses = 0;
  let importedAnnotations = 0;

  for (const raw of features) {
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const feature = raw as {
      id?: unknown;
      geometry?: Geometry;
      properties?: Properties;
    };
    const geometry = (feature.geometry ?? {}) as Geometry;
    const properties = (feature.properties ?? {}) as Properties;
    const geometryType = geometry.type;
    const coordinates = geometry.coordinates;
    if (
      (geometryType !== "Point" && geometryType !== "Polygon") ||
      coordinates === undefined ||
      coordinates === null
    ) {
      continue;
    }

    const annotationType: AnnotationType =
      geometryType === "Polygon" ? "polygon" : "point";

    let className = String(
      properties.class ??
        properties.name ??
        `imported_${annotationType}`,
    );
    const markerColor = String(properties["marker-color"] ?? DEFAULT_COLOR);
    const classColor = isHexColor(markerColor) ? markerColor : DEFAULT_COLOR;

    let key = classKey(className, annotationType);
    let existingClass = classIndex.get(key);
    const conflicting = classes.find(
      (item) =>
        item.name === className && item.annotationType !== annotationType,
    );
    if (conflicting) {
      className = `${className}_${annotationType}`;
      key = classKey(className, annotationType);
      existingClass = classIndex.get(key);
    }

    if (!existingClass) {
      existingClass = {
        name: className,
        color: classColor,
        annotationType,
      };
      classes.push(existingClass);
      classIndex.set(key, existingClass);
      createdClasses += 1;
    }

    const level =
      properties.level !== undefined && properties.level !== null
        ? String(properties.level)
        : null;
    const altitude =
      typeof properties.altitude === "number"
        ? properties.altitude
        : properties.altitude === null || properties.altitude === undefined
          ? null
          : Number(properties.altitude);
    const accuracyRaw = Number(properties.accuracy ?? DEFAULT_ACCURACY_M);
    const accuracyM = Number.isFinite(accuracyRaw)
      ? accuracyRaw
      : DEFAULT_ACCURACY_M;
    const prompt =
      typeof properties.prompt === "string" && properties.prompt.trim()
        ? properties.prompt.trim()
        : null;
    const sourceProjection: AnnotationSource["projection"] | null =
      properties.source_projection === "erp" || properties.source_projection === "cutout"
        ? properties.source_projection
        : null;
    const sourceKeyframeId =
      typeof properties.source_keyframe_id === "string"
        ? properties.source_keyframe_id
        : null;
    const sourceValues = [
      properties.source_x_ratio,
      properties.source_y_ratio,
      properties.source_erp_u,
      properties.source_erp_v,
      properties.depth_m,
    ].map(Number);
    const source =
      sourceProjection &&
      sourceKeyframeId &&
      sourceValues.every(Number.isFinite)
        ? {
            keyframeId: sourceKeyframeId,
            projection: sourceProjection,
            cutoutId:
              typeof properties.source_cutout_id === "string"
                ? properties.source_cutout_id
                : null,
            xRatio: sourceValues[0],
            yRatio: sourceValues[1],
            erpU: sourceValues[2],
            erpV: sourceValues[3],
            depthM: sourceValues[4],
          }
        : null;

    annotations.push({
      id: String(feature.id ?? newId()),
      className: existingClass.name,
      prompt,
      classColor: existingClass.color,
      annotationType,
      geometryType,
      coordinates: coordinates as number[] | number[][][],
      altitude: altitude === null ? null : Number.isFinite(altitude) ? altitude : null,
      level,
      accuracyM,
      source,
    });
    importedAnnotations += 1;
  }

  return { classes, annotations, createdClasses, importedAnnotations };
}

function classKey(name: string, type: AnnotationType): string {
  return `${name}::${type}`;
}

export async function saveGeoJSONToMap(
  mapId: string,
  annotations: AnnotationFeature[],
): Promise<SaveMapGeoJSONResult> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/annotations`,
    {
      method: "POST",
      headers: { "Content-Type": "application/geo+json" },
      body: JSON.stringify(buildAnnotationGeoJSON(mapId, annotations)),
    },
  );
  const responseText = await response.text();
  let payload: SaveMapGeoJSONResult & { detail?: string };
  try {
    payload = JSON.parse(responseText) as SaveMapGeoJSONResult & {
      detail?: string;
    };
  } catch {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("text/html") || responseText.trimStart().startsWith("<")) {
      throw new Error(
        "The backend returned the UI page instead of the annotation save API. Restart the object-search toolbox backend.",
      );
    }
    throw new Error(
      `The annotation save API returned an invalid response${
        responseText.trim() ? `: ${responseText.slice(0, 160)}` : "."
      }`,
    );
  }
  if (!response.ok) {
    throw new Error(payload.detail ?? `${response.status} ${response.statusText}`);
  }
  return payload;
}

export async function saveGeoJSONAs(
  mapId: string,
  annotations: AnnotationFeature[],
): Promise<SaveGeoJSONResult> {
  const geojson = buildAnnotationGeoJSON(mapId, annotations);
  const blob = new Blob([JSON.stringify(geojson, null, 2)], {
    type: "application/geo+json",
  });
  const filename = `${mapId}_annotations.geojson`;
  const saveFilePicker = (window as SaveFilePickerWindow).showSaveFilePicker;
  if (saveFilePicker) {
    try {
      const handle = await saveFilePicker({
        suggestedName: filename,
        types: [
          {
            description: "GeoJSON file",
            accept: {
              "application/geo+json": [".geojson"],
              "application/json": [".json"],
            },
          },
        ],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return "saved";
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return "cancelled";
      }
      throw error;
    }
  }

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  return "downloaded";
}

export { newId as newAnnotationId };
