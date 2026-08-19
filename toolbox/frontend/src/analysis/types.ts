/**
 * The `map_analysis.py` payload, section by section.
 *
 * These mirror `report.data` in `toolbox/benchmark/map_analysis.py` — that tool is the
 * schema. Everything is optional at the section level because a map missing labels,
 * annotations or embeddings makes whole sections empty, and an absent section must
 * read as "not measurable here", never as a measured zero.
 */

/** `_percentiles()`: the same five numbers everywhere the tool reports a spread. */
export type Percentiles = {
  n: number;
  p10: number;
  median: number;
  p90: number;
  max: number;
};

export type SourceSummary = {
  n: number;
  range_m: Percentiles;
  score: Percentiles;
};

export type InventorySection = {
  map: string;
  keyframes_manifest: number;
  keyframes_with_detections: number;
  detections: number;
  placed: number;
  ground_truth: number;
  reviews: number;
  group_labels: number;
  embeddings: [number, number] | null;
  /** Human-readable list of what this index does NOT hold. Empty is the good case. */
  missing: string[];
};

export type DetectionsSection = {
  range_m: Percentiles;
  angular_diagonal_deg: Percentiles;
  phi_deg: Percentiles;
  per_keyframe: Percentiles;
  by_source: Record<string, SourceSummary>;
};

export type LabelsSection = {
  attachment_radius_m: number;
  embedding_purity: {
    neighbours: number;
    queries: number;
    same_object: number;
    same_class: number;
  };
  /** Radius (as a string key) → share of detections within it of ANY annotation. */
  base_rate: Record<string, number>;
  mutual_information_label_gt_class: number;
  p_label_given_gt_class: Record<string, unknown[]>;
  p_gt_class_given_label: Record<string, unknown[]>;
};

/** One point of the `min_keyframes_per_cluster` retention curve. */
export type KeyframeThresholdPoint = {
  min_keyframes: number;
  retained: number;
  retained_share: number;
  covered_share: number;
  measurable: number;
};

export type GroundTruthSection = {
  source_keyframe_indexed: number;
  covered_in_own_keyframe: number;
  covered_by_class: Record<string, number>;
  attachment_by_radius: Record<string, number>;
  observability: Record<string, Percentiles>;
  keyframe_threshold?: KeyframeThresholdPoint[];
  degenerate_baseline: number;
  unseen: {
    total: number;
    never_looked: number;
    looked_and_missed: number;
  };
};

export type PairsSection = {
  sampled_pairs: number;
  same_object: number;
  different_object: number;
  contaminated: number;
  /** Cue name → raw AUC and the AUC conditioned on geometry. */
  auc: Record<string, { raw: number; conditional: number }>;
};

export type CountingSection = {
  sampled_keyframes: number;
  overlapping_pairs: number;
  cross_source_share: number;
  overlap_cosine: Percentiles;
};

export type HubnessSection = {
  sampled: number;
  k: number;
  raw: HubnessStats;
  centred: HubnessStats;
  hub_labels: Record<string, unknown>;
};

export type HubnessStats = {
  skewness: number;
  hub_share: number;
  hub_mass: number;
  antihub_share: number;
};

export type LabelSetsSection = {
  coverage: number;
  annotations: number;
  [key: string]: unknown;
};

export type AnalysisPayload = {
  s0?: InventorySection;
  s1?: DetectionsSection;
  s2?: LabelsSection;
  s3?: GroundTruthSection;
  s4?: PairsSection;
  s5?: CountingSection;
  s6?: HubnessSection;
  s7?: LabelSetsSection;
};

export type AnalysisJob = {
  run_id: string;
  map_id: string;
  state: "running" | "done" | "error";
  /** `started` counts sections ENTERED: the last one is still running. */
  progress: { started: number; total: number; sections: string[] };
  error: string | null;
  started_at: string;
  finished_at: string | null;
};

export type AnalysisStatus = {
  map_id: string;
  runs: string[];
  latest_run_id: string | null;
  analysis: AnalysisPayload | null;
  /** Layer names the latest run wrote, from `map_layers.LAYERS`. */
  layers: string[];
  job: AnalysisJob | null;
};
