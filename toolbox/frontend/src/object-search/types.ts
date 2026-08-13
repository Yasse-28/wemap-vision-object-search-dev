import type { FeedbackNormalization } from "../benchmark/types";

export type { FeedbackNormalization };

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
  /**
   * Mean per-axis standard deviation of the cluster's members, in metres. Reported
   * on its own rather than folded into `matchScore`: geometric support is a filter
   * (`max_cluster_spread_m`) now, not a score term.
   */
  spreadM: number;
  similarityScore: number;
  /** `similarityScore` as a fraction of the best cluster of the *same* query. */
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

/**
 * Every field here is a parameter the bricks service actually reads. Four more used
 * to sit alongside them — `use_stored_positions`, `robust_centroid`,
 * `embedding_similarity_threshold` and `include_debug` — inherited from the retired
 * standalone service (`legacy/pipeline/online/`). Neither bricks nor the mirrored
 * production service accepts them: pydantic dropped them silently, so the controls
 * did nothing. Do not reintroduce one without a field to receive it.
 */
export type OnlineLocalizeOverrides = {
  merge_radius: number;
  feedback_alpha: number;
  feedback_beta: number;
  feedback_normalization: FeedbackNormalization;
  /**
   * Sent explicitly rather than left to the service default, so the same number
   * reaches both `/localize` and the benchmark behind "Score this prompt". It is the
   * absolute eligibility floor; `match_score` is the *relative* gate (a ratio to the
   * query's best cluster) and no longer depends on this value, so the two can now be
   * swept independently. Two different floors still admit two different candidate
   * sets, hence two different top clusters — so it still has to match.
   */
  min_similarity: number;
  min_keyframes_per_cluster: number;
  candidate_count: number;
  /**
   * How a cluster picks the floor it claims. `"seed"` is production's behaviour and
   * the default; `"median"` lets the majority of a cluster's detections outvote its
   * highest-similarity one. Dev-only — see `toolbox/bricks/localize.py`.
   */
  level_strategy: "seed" | "median";
};

export const DEFAULT_ONLINE_OVERRIDES: OnlineLocalizeOverrides = {
  merge_radius: 2.0,
  feedback_alpha: 0.0,
  feedback_beta: 0.0,
  feedback_normalization: "none",
  min_similarity: 0.2,
  // The service's own default, and the benchmark's. It used to be 3 here alone,
  // which meant the panel and every benchmark number described differently-built
  // clusters for the same prompt.
  min_keyframes_per_cluster: 2,
  candidate_count: 1000,
  level_strategy: "seed",
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
