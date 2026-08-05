import { readFile } from "node:fs/promises";
import path from "node:path";

import type { MapEntry } from "./config.js";

export type KeyframeGraphEdge = {
  id: string;
  keyframe_id_1: string | null;
  keyframe_id_2: string | null;
  from: [number, number];
  to: [number, number];
  levels: string[];
};

export type KeyframeGraphPayload = {
  available: boolean;
  path: string;
  edges: KeyframeGraphEdge[];
  skipped_feature_count: number;
  error?: string;
};

function finiteCoordinate(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) {
    return null;
  }
  const longitude = Number(value[0]);
  const latitude = Number(value[1]);
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) {
    return null;
  }
  return [longitude, latitude];
}

function graphLevels(value: unknown): string[] {
  const values = Array.isArray(value) ? value : [value];
  return [
    ...new Set(
      values.flatMap((item) =>
        item === null || item === undefined || item === "" ? [] : [String(item)],
      ),
    ),
  ];
}

function optionalId(value: unknown): string | null {
  return value === null || value === undefined || value === "" ? null : String(value);
}

export function parseKeyframeGraph(data: unknown): {
  edges: KeyframeGraphEdge[];
  skippedFeatureCount: number;
} {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("Graph GeoJSON must be an object.");
  }
  const collection = data as Record<string, unknown>;
  if (collection.type !== "FeatureCollection" || !Array.isArray(collection.features)) {
    throw new Error("Graph GeoJSON must be a FeatureCollection.");
  }

  const edges: KeyframeGraphEdge[] = [];
  let skippedFeatureCount = 0;
  for (const [featureIndex, value] of collection.features.entries()) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      skippedFeatureCount += 1;
      continue;
    }
    const feature = value as Record<string, unknown>;
    const geometry =
      feature.geometry && typeof feature.geometry === "object" && !Array.isArray(feature.geometry)
        ? (feature.geometry as Record<string, unknown>)
        : null;
    if (geometry?.type !== "LineString") {
      continue;
    }
    const coordinates = Array.isArray(geometry.coordinates) ? geometry.coordinates : [];
    const from = finiteCoordinate(coordinates[0]);
    const to = finiteCoordinate(coordinates[coordinates.length - 1]);
    if (!from || !to) {
      skippedFeatureCount += 1;
      continue;
    }
    const properties =
      feature.properties &&
      typeof feature.properties === "object" &&
      !Array.isArray(feature.properties)
        ? (feature.properties as Record<string, unknown>)
        : {};
    const keyframeId1 = optionalId(properties.keyframeId1);
    const keyframeId2 = optionalId(properties.keyframeId2);
    edges.push({
      id: `graph-edge-${keyframeId1 ?? "unknown"}-${keyframeId2 ?? "unknown"}-${featureIndex}`,
      keyframe_id_1: keyframeId1,
      keyframe_id_2: keyframeId2,
      from,
      to,
      levels: graphLevels(properties.level),
    });
  }
  return { edges, skippedFeatureCount };
}

export async function keyframeGraphPayload(map: MapEntry): Promise<KeyframeGraphPayload> {
  const graphPath = path.join(map.path, "360-viewer", "graph.geojson");
  let raw: string;
  try {
    raw = await readFile(graphPath, "utf8");
  } catch (error) {
    const code =
      error && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : "";
    if (code === "ENOENT") {
      return {
        available: false,
        path: graphPath,
        edges: [],
        skipped_feature_count: 0,
      };
    }
    throw error;
  }

  try {
    const parsed = parseKeyframeGraph(JSON.parse(raw) as unknown);
    return {
      available: true,
      path: graphPath,
      edges: parsed.edges,
      skipped_feature_count: parsed.skippedFeatureCount,
    };
  } catch (error) {
    return {
      available: false,
      path: graphPath,
      edges: [],
      skipped_feature_count: 0,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
