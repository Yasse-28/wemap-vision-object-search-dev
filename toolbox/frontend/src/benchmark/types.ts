export type BenchmarkServiceStatus = {
  base_url: string;
  reachable: boolean;
  status: string | null;
  map_state: string | null;
  managed: boolean;
  pid: number | null;
};

export type BenchmarkAnnotationsStatus = {
  exists: boolean;
  path: string;
  annotation_count: number;
  prompt_count: number;
};

export type PromptProgress = {
  index: number;
  total: number;
  class_name: string;
  prompt: string;
  precision: number;
  recall: number;
  f1: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  elapsed_ms: number;
  error: string | null;
};

export type BenchmarkSummary = {
  prompt_count: number;
  error_count: number;
  /**
   * Threshold-free. Averaged over prompts rather than pooled over predictions:
   * `match_score` is normalised by the best cluster of its own query, so scores from
   * two prompts are not on one scale. These are what two runs should be compared on.
   */
  scored_prompt_count?: number;
  mean_average_precision?: number;
  mean_best_f1?: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
};

export type BenchmarkJobState = "ensuring-service" | "running" | "done" | "error";

export type BenchmarkJob = {
  run_id: string;
  map_id: string;
  state: BenchmarkJobState;
  service_state: string | null;
  progress: { completed: number; total: number | null };
  prompts: PromptProgress[];
  summary: BenchmarkSummary | null;
  error: string | null;
  started_at: string;
};

export type BenchmarkRunListItem = {
  run_id: string;
  summary: BenchmarkSummary | null;
  has_raw: boolean;
};

export type BenchmarkStatus = {
  python_service: BenchmarkServiceStatus;
  annotations: BenchmarkAnnotationsStatus;
  job: BenchmarkJob | null;
  runs: BenchmarkRunListItem[];
};

export type ByClassRow = {
  class_name: string;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  prompt_count: number;
  errors: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ByPromptRow = {
  class_name: string;
  prompt: string;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  accepted_predictions: number;
  rejected_predictions: number;
  ground_truth: number;
  error: string | null;
  elapsed_ms: number;
  /** Optional: runs stored before the sweep existed do not carry them. */
  average_precision?: number;
  best_f1?: number;
  /** Accept predictions scoring **>= this** to obtain `best_f1`. */
  best_f1_threshold?: number;
  best_f1_precision?: number;
  best_f1_recall?: number;
};

export type BenchmarkMetrics = {
  config: Record<string, unknown>;
  summary: BenchmarkSummary;
  by_class: ByClassRow[];
  by_prompt: ByPromptRow[];
  grouped?: {
    group_annotation_radius_m: number;
    summary: BenchmarkSummary;
    by_class: ByClassRow[];
    by_prompt: ByPromptRow[];
  };
};

export type BenchmarkRunParams = {
  target_server: "local" | "remote";
  remote_localize_url: string;
  acceptance_threshold: number;
  num_results: number;
  min_similarity: number;
  clustering_eps_m: number;
  candidate_count: number;
  group_annotation_radius_m: number;
  default_accuracy: number;
  feedback_alpha?: number;
  feedback_beta?: number;
  feedback_normalization?: FeedbackNormalization;
  min_keyframes_per_cluster?: number;
  max_observations_per_cluster?: number;
};

/**
 * How the review-feedback prototype similarities are rescaled before the gains apply.
 * `"none"` is the raw term the baseline was measured with; see
 * `toolbox/bricks/candidates.py::normalize_prototype_similarities`.
 */
export type FeedbackNormalization = "none" | "center" | "standardize";

export const FEEDBACK_NORMALIZATIONS: readonly FeedbackNormalization[] = [
  "none",
  "center",
  "standardize",
];

/**
 * Match radius for annotations that carry no `accuracy` of their own — which is all of
 * them created by the review tool.
 *
 * 2 m, not the script's own 5 m: on a real venue the annotated instances of one class
 * sit closer together than 5 m (measured on `vinci-st-domingue`: median nearest
 * neighbour 1.0 m for FIDS, 3.1 m for check-in counters), so a 5 m radius encloses
 * *other* targets and the 1-1 assignment picks between them on differences far below
 * the pipeline's own localisation error.
 */
export const DEFAULT_MATCH_ACCURACY_M = 2.0;

export type PromptScore = {
  prompt: string;
  row: ByPromptRow;
  config: Record<string, unknown>;
  scored_at: string;
};

export type BenchmarkRawAnnotation = {
  id: string;
  class_name: string;
  lat: number;
  lng: number;
  accuracy_m: number;
  prompt: string;
  level: string | null;
};

export type BenchmarkRawPrompt = {
  prompt: string;
  class_name: string;
  annotations: BenchmarkRawAnnotation[];
  predictions: Record<string, unknown>[];
  matches: {
    prediction_id: string;
    annotation_id: string;
    distance_m: number;
    score: number;
  }[];
  error: string | null;
};

export type BenchmarkRawResults = {
  config: Record<string, unknown>;
  prompts: BenchmarkRawPrompt[];
};
