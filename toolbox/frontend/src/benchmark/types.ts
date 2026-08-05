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
