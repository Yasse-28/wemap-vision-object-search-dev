import type { SearchType } from "../api";

/**
 * `"localize-offline"` is gone (ADR 0002): "offline" meant exact cosine over an
 * index held in RAM, a distinction pgvector HNSW removes. `"localize-online"` keeps
 * its name so the request paths and the benchmark's `--api-style standalone --online`
 * contract stay as they are.
 */
export type ObjectSearchMode = "text" | "localize-online";
export type QueryInputMode = "text" | "image";

export type ApiDebugInfo = {
  url: string;
  statusCode: number;
  elapsedMs: number;
  raw: unknown;
};

export type EnrichedResult = {
  rank: number;
  id: string;
  score: number;
  kind: "cutout" | "object" | null;
  cutout_id: string | null;
  keyframe_id: string | null;
  lat: number | null;
  lon: number | null;
  alt: number | null;
  level: string | null;
  has_preview: boolean;
  preview_path: string | null;
  resolution_status: "resolved" | "partial" | "unresolved";
  resolution_error: string | null;
};

export type ObjectObservation = {
  objectIdx: number;
  cutoutId: string;
  keyframeId: string;
  /** `thumbnail_key` of the candidate — the only preview an observation has. */
  thumbnail: string | null;
  coordinates: [number, number, number] | null;
  bbox: [number, number, number, number];
  similarityScore: number;
  quaternion: [number, number, number, number] | null;
  heading: number | null;
};

export type ObjectLocalization = {
  lat: number;
  lng: number;
  alt: number;
  level: string | null;
  confidence: number;
  observationCount: number;
  similarityScore: number;
  matchScore: number;
  keyframeIds: string[];
  observations: ObjectObservation[];
};

export type KeyframeDebugMetadata = {
  lat: number;
  lon: number;
  alt: number | null;
  level: string | null;
};

export type OnlineLocalizeOverrides = {
  use_stored_positions: boolean;
  robust_centroid: boolean;
  merge_radius: number;
  embedding_similarity_threshold: number;
  min_keyframes_per_cluster: number;
  candidate_count: number;
  include_debug: boolean;
};

export const DEFAULT_ONLINE_OVERRIDES: OnlineLocalizeOverrides = {
  use_stored_positions: true,
  robust_centroid: false,
  merge_radius: 2.0,
  embedding_similarity_threshold: 0.85,
  min_keyframes_per_cluster: 3,
  candidate_count: 1000,
  // On by default so the keyframe overlay keeps rendering; turn off to drop the
  // per-keyframe georef lookup (adds seconds on large maps).
  include_debug: true,
};

export type TextSearchState = {
  mode: "text";
  routerObjectType: string | null;
  enriched: EnrichedResult[];
  debug: ApiDebugInfo;
};

export type LocalizeSearchState = {
  mode: "localize-online";
  localizations: ObjectLocalization[];
  keyframes: Record<string, KeyframeDebugMetadata>;
  debug: ApiDebugInfo;
};

export type ObjectSearchRunResult = TextSearchState | LocalizeSearchState;

export type RunObjectSearchParams = {
  mapId: string;
  mapPath: string;
  searchMode: ObjectSearchMode;
  queryInputMode: QueryInputMode;
  text: string;
  imageFile: File | null;
  numResults: number;
  searchType: SearchType;
  maxObservationsPerCluster: number;
  onlineOverrides: OnlineLocalizeOverrides;
  remoteUrl: string | null;
};

export type KeyframeGroup = {
  keyframeId: string;
  lat: number | null;
  lon: number | null;
  alt: number | null;
  level: string | null;
  bestScore: number;
  results: EnrichedResult[];
};
