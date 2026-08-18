import {
  DEFAULT_ONLINE_OVERRIDES,
  type ApiDebugInfo,
  type EnrichedResult,
  type KeyframeDebugMetadata,
  type ObjectLocalization,
  type ObjectObservation,
  type ObjectSearchRunResult,
  type OnlineLocalizeOverrides,
  type RunObjectSearchParams,
} from "./types";
import { fetchJson } from "../http";

const ASSOCIATION_OVERRIDE_KEYS: ReadonlySet<keyof OnlineLocalizeOverrides> = new Set([
  "association",
  "combination",
  "association_sim_threshold",
  "max_depth_m",
  "max_depth_stage",
  "max_cluster_spread_m",
  "min_observations_per_cluster",
]);

/*
 * The service owns these defaults. Omitting unchanged association controls keeps
 * requests from an untouched panel byte-for-byte equivalent to the previous UI.
 */
function isDefaultAssociationOverride(
  key: keyof OnlineLocalizeOverrides,
  value: OnlineLocalizeOverrides[keyof OnlineLocalizeOverrides],
): boolean {
  return ASSOCIATION_OVERRIDE_KEYS.has(key) && value === DEFAULT_ONLINE_OVERRIDES[key];
}

function parseTextPairs(raw: unknown): {
  pairs: Array<[string, number]>;
  routerObjectType: string | null;
} {
  let routerObjectType: string | null = null;
  let pairs: unknown = raw;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const obj = raw as { results?: unknown; router_object_type?: unknown };
    pairs = obj.results;
    routerObjectType =
      typeof obj.router_object_type === "string" ? obj.router_object_type : null;
  }
  if (!Array.isArray(pairs)) {
    return { pairs: [], routerObjectType };
  }
  const normalized: Array<[string, number]> = [];
  for (const item of pairs) {
    if (!Array.isArray(item) || item.length < 2) {
      continue;
    }
    const score = Number(item[1]);
    if (!Number.isFinite(score)) {
      continue;
    }
    normalized.push([String(item[0]), score]);
  }
  return { pairs: normalized, routerObjectType };
}

function parseBbox(item: Record<string, unknown>): [number, number, number, number] | null {
  const raw = Array.isArray(item.bbox)
    ? item.bbox
    : Array.isArray(item.bbox_coordinates)
      ? item.bbox_coordinates
      : [];
  if (raw.length < 4) {
    return null;
  }
  const bbox: [number, number, number, number] = [
    Number(raw[0]),
    Number(raw[1]),
    Number(raw[2]),
    Number(raw[3]),
  ];
  return bbox.every(Number.isFinite) ? bbox : null;
}

function finiteOrNull(value: unknown): number | null {
  const parsed = Number(value);
  return value != null && Number.isFinite(parsed) ? parsed : null;
}

function parseObservation(item: Record<string, unknown>): ObjectObservation | null {
  const coordinates = Array.isArray(item.coordinates) ? item.coordinates : null;
  const bbox = parseBbox(item);
  if (!bbox) {
    return null;
  }
  const quaternionValues = Array.isArray(item.quaternion)
    ? item.quaternion.map((value) => Number(value))
    : null;
  const quaternion =
    quaternionValues &&
    quaternionValues.length === 4 &&
    quaternionValues.every(Number.isFinite)
      ? (quaternionValues as [number, number, number, number])
      : null;
  const heading = item.heading == null ? null : Number(item.heading);
  return {
    objectIdx: Number(item.object_idx ?? 0),
    cutoutId: String(item.cutout_id ?? ""),
    keyframeId: String(item.keyframe_id ?? ""),
    thumbnail:
      typeof item.thumbnail === "string" && item.thumbnail.length ? item.thumbnail : null,
    coordinates:
      coordinates && coordinates.length >= 3
        ? [Number(coordinates[0]), Number(coordinates[1]), Number(coordinates[2])]
        : null,
    bbox,
    thetaCenter: finiteOrNull(item.theta_center),
    phiCenter: finiteOrNull(item.phi_center),
    similarityScore: Number(item.similarity_score ?? 0),
    quaternion,
    heading: heading == null || Number.isFinite(heading) ? heading : null,
  };
}

function parseLocalizations(raw: unknown): ObjectLocalization[] {
  if (!raw || typeof raw !== "object") {
    return [];
  }
  const items = (raw as { localizations?: unknown }).localizations;
  if (!Array.isArray(items)) {
    return [];
  }
  const similarities = items
    .filter((item) => item && typeof item === "object")
    .map((item) => Number((item as Record<string, unknown>).similarity_score ?? 0));
  const bestSimilarity = similarities.length ? Math.max(...similarities) : 0;

  return items.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const loc = item as Record<string, unknown>;
    const coordinates = Array.isArray(loc.coordinates) ? loc.coordinates : [];
    const lat = Number(coordinates[0] ?? loc.lat);
    const lng = Number(coordinates[1] ?? loc.lng);
    const alt = Number(coordinates[2] ?? loc.alt);
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || !Number.isFinite(alt)) {
      return [];
    }
    const similarityScore = Number(loc.similarity_score ?? 0);
    const confidence = Number(loc.confidence ?? 0);
    const keyframeIds = Array.isArray(loc.keyframe_ids)
      ? loc.keyframe_ids.map((value) => String(value))
      : [];
    const matchScore = Number.isFinite(Number(loc.match_score))
      ? Number(loc.match_score)
      : 0.5 * (similarityScore / Math.max(bestSimilarity, 1e-6)) +
        0.3 * confidence +
        0.2 * Math.min(1, keyframeIds.length / 3);

    const observations: ObjectObservation[] = [];
    if (Array.isArray(loc.observations)) {
      for (const obs of loc.observations) {
        if (!obs || typeof obs !== "object") {
          continue;
        }
        const parsed = parseObservation(obs as Record<string, unknown>);
        if (parsed) {
          observations.push(parsed);
        }
      }
    }

    return [
      {
        lat,
        lng,
        alt,
        level: loc.level == null ? null : String(loc.level),
        confidence,
        observationCount: Number(loc.observation_count ?? observations.length),
        spreadM: Number(loc.spread_m ?? 0),
        similarityScore,
        matchScore,
        keyframeIds,
        observations,
      },
    ];
  });
}

function parseDebugKeyframes(raw: unknown): Record<string, KeyframeDebugMetadata> {
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const debug = (raw as { debug?: unknown }).debug;
  if (!debug || typeof debug !== "object") {
    return {};
  }
  const keyframes = (debug as { keyframes?: unknown }).keyframes;
  if (!keyframes || typeof keyframes !== "object" || Array.isArray(keyframes)) {
    return {};
  }

  const parsed: Record<string, KeyframeDebugMetadata> = {};
  for (const [id, value] of Object.entries(keyframes as Record<string, unknown>)) {
    if (!value || typeof value !== "object") {
      continue;
    }
    const meta = value as Record<string, unknown>;
    const lat = Number(meta.lat);
    const lon = Number(meta.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      continue;
    }
    const alt = meta.alt == null ? null : Number(meta.alt);
    parsed[id] = {
      lat,
      lon,
      alt: alt == null || Number.isFinite(alt) ? alt : null,
      level: meta.level == null ? null : String(meta.level),
    };
  }
  return parsed;
}

function onlineOverrideEntries(
  overrides: OnlineLocalizeOverrides,
): Array<[string, boolean | number | string]> {
  return Object.entries(overrides).flatMap(([rawKey, value]) => {
    const key = rawKey as keyof OnlineLocalizeOverrides;
    if (value === null || isDefaultAssociationOverride(key, value)) {
      return [];
    }
    return [[key === "merge_radius" ? "clustering_eps_m" : key, value]];
  });
}

/**
 * Candidate rows as returned by the bricks service's `/object-search/text`.
 *
 * These come out of pgvector already enriched (WGS84, level, thumbnail), so text
 * search needs no second round-trip. Before ADR 0002 the frontend re-enriched
 * through the workbench's `enrich-text-results` route, which read the standalone
 * SQLite index — that index is retired, so this replaced it.
 */
type ServiceCandidate = {
  id: number;
  similarity: number;
  lat: number | null;
  lng: number | null;
  alt: number | null;
  level: number | null;
  thumbnail_key: string;
  video_keyframe_id: number | null;
};

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Map the service's candidate rows onto the panel's `EnrichedResult` shape. */
export function enrichedFromCandidates(raw: unknown): EnrichedResult[] {
  const candidates =
    raw && typeof raw === "object" && Array.isArray((raw as { candidates?: unknown }).candidates)
      ? ((raw as { candidates: unknown[] }).candidates as ServiceCandidate[])
      : [];

  return candidates.map((candidate, index) => {
    const lat = numberOrNull(candidate.lat);
    const lon = numberOrNull(candidate.lng);
    const previewPath = candidate.thumbnail_key || null;
    return {
      rank: index + 1,
      id: String(candidate.id),
      score: numberOrNull(candidate.similarity) ?? 0,
      // Every indexed row is a detector-proposal cutout: production dropped
      // cubemap faces, and the query router that would have said otherwise was a
      // stub that always answered "cutout".
      kind: "cutout",
      cutout_id: String(candidate.id),
      keyframe_id:
        candidate.video_keyframe_id == null ? null : String(candidate.video_keyframe_id),
      lat,
      lon,
      alt: numberOrNull(candidate.alt),
      level: candidate.level == null ? null : String(candidate.level),
      has_preview: previewPath != null,
      preview_path: previewPath,
      // A row only reaches us with a position at all — enrichment filters out
      // candidates whose object_position is NULL — so anything here is resolved.
      resolution_status: lat != null && lon != null ? "resolved" : "unresolved",
      resolution_error: null,
    };
  });
}

export async function runObjectSearch(
  params: RunObjectSearchParams,
): Promise<ObjectSearchRunResult> {
  const encodedMapId = encodeURIComponent(params.mapId);
  const started = performance.now();

  if (params.searchMode === "text") {
    const raw = await fetchJson(`/${encodedMapId}/object-search/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: params.text,
        num_results: params.numResults,
      }),
    });
    const { routerObjectType } = parseTextPairs(raw);
    const enriched = enrichedFromCandidates(raw);
    const debug: ApiDebugInfo = {
      url: `/${encodedMapId}/object-search/text`,
      statusCode: 200,
      elapsedMs: performance.now() - started,
      raw,
    };
    return { mode: "text", routerObjectType, enriched, debug };
  }

  // Only localization remains here — the text branch returned above, and
  // `localize-offline` was retired with the in-RAM index (ADR 0002). `remoteUrl`
  // still wins when the panel is pointed at a deployed service.
  const url = params.remoteUrl || `/${encodedMapId}/object-search/localize`;

  let raw: unknown;
  if (params.queryInputMode === "image") {
    if (!params.imageFile) {
      throw new Error("Select an image to run localization.");
    }
    const form = new FormData();
    form.append("image", params.imageFile, params.imageFile.name);
    form.append("num_results", String(params.numResults));
    form.append("max_observations_per_cluster", String(params.maxObservationsPerCluster));
    for (const [key, value] of onlineOverrideEntries(params.onlineOverrides)) {
      form.append(key, typeof value === "boolean" ? (value ? "true" : "false") : String(value));
    }
    raw = await fetchJson(url, { method: "POST", body: form });
  } else {
    const payload: Record<string, unknown> = {
      text: params.text,
      num_results: params.numResults,
      max_observations_per_cluster: params.maxObservationsPerCluster,
      ...Object.fromEntries(onlineOverrideEntries(params.onlineOverrides)),
    };
    raw = await fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  const localizations = parseLocalizations(raw);
  const keyframes = parseDebugKeyframes(raw);
  const debug: ApiDebugInfo = {
    url,
    statusCode: 200,
    elapsedMs: performance.now() - started,
    raw,
  };
  return {
    mode: params.searchMode,
    localizations,
    keyframes,
    debug,
  };
}
