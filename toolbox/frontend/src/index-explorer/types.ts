export type ExplorerView = "browse" | "prompt-latent";

export type IndexObjectRecord = {
  id: string;
  keyframe_id: string;
  cutout_id: string;
  bbox: [number, number, number, number];
  bbox_coordinates?: [number, number, number, number];
  label: string;
  detection_source: string;
  ocr_text: string;
  ocr_tokens: string;
  ocr_key: string;
  ocr_source: string;
  textness: number | null;
  ocr_candidate: boolean;
  ocr_assigned: boolean;
  cluster_id: number | null;
};

export type IndexSummary = {
  index_path: string;
  manifest: Record<string, unknown>;
  keyframe_ids: string[];
  cutout_ids_by_keyframe: Record<string, string[]>;
  object_count_by_cutout: Record<string, number>;
  ocr_text_count: number;
  ocr_key_count: number;
  ocr_candidate_count: number;
  ocr_assigned_count: number;
  ocr_source_counts: Record<string, number>;
  cluster_ocr_count: number;
  default_prompts: string[];
};

export type IndexStatusResponse = {
  available: boolean;
  index_path: string | null;
  map_path?: string;
  object_search_index_path?: string | null;
  checked_paths?: string[];
  summary?: IndexSummary;
  markers?: Array<{
    id: string;
    latitude: number;
    longitude: number;
    level: string | null;
    heading_deg: number | null;
    color: string;
    radius: number;
  }>;
  resolved_marker_count?: number;
};

export type KeyframeGraphResponse = {
  available: boolean;
  path: string;
  edges: Array<{
    id: string;
    keyframe_id_1: string | null;
    keyframe_id_2: string | null;
    from: [number, number];
    to: [number, number];
    levels: string[];
  }>;
  skipped_feature_count: number;
  error?: string;
};

export type CutoutIndexPayload = {
  detections: IndexObjectRecord[];
  preview_ref: string;
  preview_debug: Record<string, unknown>;
};

export type DepthPinResponse = {
  available: boolean;
  keyframe_id: string;
  cutout_id: string | null;
  projection: "erp" | "cutout";
  face: string | null;
  erp_u: number;
  erp_v: number;
  erp_col: number;
  erp_row: number;
  erp_width: number;
  erp_height: number;
  depth_m: number;
  sample_count: number;
  point_keyframe: [number, number, number];
  point_local: [number, number, number];
  point_world: [number, number, number];
  point_world_wds: [number, number, number];
  latitude: number;
  longitude: number;
  altitude: number;
  level: string | null;
};

export type ProjectWorldPointResponse = {
  available: boolean;
  keyframe_id: string;
  erp_u: number;
  erp_v: number;
  distance_m: number;
  point_keyframe: [number, number, number];
  point_world_wds: [number, number, number];
};

export type ViewConeResponse = {
  available: boolean;
  keyframe_id: string;
  yaw_rad: number;
  pitch_rad: number;
  center: {
    latitude: number;
    longitude: number;
    altitude: number;
    level: string | null;
  };
  polygon: Array<[number, number]>;
  level: string | null;
};

export type ClusterOcrRecord = {
  cluster_id: number;
  ocr_text: string;
  ocr_tokens: string;
  ocr_key: string;
  ocr_observation_count: number;
  ocr_source: string;
};

export type LatentPoint = {
  object_id: string;
  keyframe_id: string;
  cutout_id: string;
  x: number;
  y: number;
  size: number;
  similarity: number;
  active: boolean;
  color: string;
};

export type PromptLatentResponse = {
  available: boolean;
  default_prompts?: string[];
  message?: string;
  query_labels?: string[];
  selected_query?: string;
  embedding_backend?: string;
  warnings?: string[];
  color_min?: number;
  color_max?: number;
  threshold?: number;
  projection?: string;
  projection_meta?: Record<string, number>;
  points?: LatentPoint[];
  topk?: Array<Record<string, unknown>>;
};
