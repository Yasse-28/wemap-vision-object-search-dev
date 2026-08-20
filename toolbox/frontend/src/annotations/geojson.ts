import { AnnotationFeature } from "./types";

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
      ...(annotation.groundTruth.objectId
        ? { object_id: annotation.groundTruth.objectId }
        : {}),
      ...(annotation.groundTruth.extentM !== null
        ? { extent_m: annotation.groundTruth.extentM }
        : {}),
      ...(annotation.groundTruth.exhaustiveZone
        ? { exhaustive_zone: annotation.groundTruth.exhaustiveZone }
        : {}),
      is_depiction: annotation.groundTruth.isDepiction,
      labels: {
        synonyms: annotation.groundTruth.labels.synonyms,
        depictions: annotation.groundTruth.labels.depictions,
        visually_similar: annotation.groundTruth.labels.visuallySimilar,
        clutter: annotation.groundTruth.labels.clutter,
      },
      ...(annotation.source
        ? {
            source: annotation.source,
            source_keyframe_id: annotation.source.keyframeId,
            source_projection: annotation.source.projection,
            ...(annotation.source.rowIndex !== null
              ? { source_row_index: annotation.source.rowIndex }
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
