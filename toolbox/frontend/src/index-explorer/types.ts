/**
 * Wire types for the object-search **metadata** routes.
 *
 * These used to describe the retired standalone SQLite index, and with it a
 * cutout → objects hierarchy, per-object OCR and cluster records. None of that
 * exists in v2: a `metadata.parquet` row *is* both the cutout and the detection,
 * OCR was never ported, and clustering happens at query time. What is left is a
 * flat row table plus a per-keyframe range index.
 *
 * The directory name is a leftover of the panel that lived here
 * (`IndexExplorerPanel`, deleted with this migration); the module is now shared by
 * the object-search and object-search-explorer panels.
 */

/** One `metadata.parquet` row, as `/object-search-metadata/rows` serves it. */
export type MetadataRowRecord = {
  row_index: number;
  keyframe_id: string;
  label: string | null;
  detector_source: string;
  detection_score: number | null;
  /** Metres at the view centre; null means invisible to `localize`. */
  depth: number | null;
  /** Path relative to the map directory, or null when crops were not stored. */
  thumbnail_key: string | null;
  theta_center: number;
  phi_center: number;
  angular_width: number;
  angular_height: number;
  /** Angular footprint, the v2 replacement for the cubemap era's pixel bbox area. */
  angular_area_deg2: number;
  /**
   * Reconstructed ERP box `[u0, v0, u1, v1]` in ratio units, **unwrapped** — `u`
   * may fall slightly outside `[0, 1]` near the seam, which is what the photosphere
   * overlay wants (it reads `u` as a yaw). Reconstructed from float16 angles, so it
   * is accurate to a few ERP pixels, not exact.
   */
  erp_bbox_ratios: [number, number, number, number];
};

/** Materialized in-memory view of one keyframe's parquet slice and pgvector state. */
export type KeyframeSummary = {
  id: string;
  row_count: number;
  row_start: number;
  row_end: number;
  with_depth: number;
  /** null = unknown (no pgvector answer), 0 = pruned at ingest, >0 = indexed. */
  ingested: number | null;
  /** Rows ingested but with a NULL `object_position` — invisible to `localize`. */
  no_position: number | null;
};

export type ColumnarKeyframeSummaries = {
  ids: string[];
  row_start: number[];
  row_count: number[];
  with_depth: number[];
  ingested: Array<number | null>;
  no_position: Array<number | null>;
};

export type KeyframeMarker = {
  id: string;
  latitude: number;
  longitude: number;
  level: string | null;
  heading_deg: number | null;
};

export type ColumnarKeyframeMarkers = {
  ids_are_dense: boolean;
  ids?: string[];
  latitude: number[];
  longitude: number[];
  levels: string[];
  level_codes: number[];
  heading_deg: Array<number | null>;
};

export type MetadataSummary = {
  metadata_path: string;
  row_count: number;
  postprocessed: boolean;
  detector_source_counts: Record<string, number>;
  with_depth_count: number;
  coverage: {
    ingested_total: number;
    no_position_total: number;
    keyframe_count: number;
  } | null;
};

export type MetadataStatusResponse = {
  /** Whether a `metadata.parquet` was found and parsed. */
  available: boolean;
  metadata_path: string | null;
  map_path?: string;
  checked_paths?: string[];
  capture_count: number;
  postprocessed: boolean;
  manifest_keyframe_count: number;
  error: string | null;
  summary: MetadataSummary | null;
  marker_count: number;
  first_keyframe_id: string | null;
};

export type MetadataMarkersWireResponse = ColumnarKeyframeMarkers;

export type MetadataKeyframesResponse = {
  total: number;
  keyframes: KeyframeSummary[];
};

export type MetadataKeyframesWireResponse = ColumnarKeyframeSummaries & {
  total: number;
};

export type MetadataRowsResponse = {
  total: number;
  offset: number;
  limit: number;
  postprocessed: boolean;
  rows: MetadataRowRecord[];
};

export type MetadataRowDetail = {
  row: MetadataRowRecord;
  /** The stored thumbnail, served through `/preview.png?preview_path=`. */
  preview_path: string | null;
  preview_debug: Record<string, unknown>;
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

export type DepthPinResponse = {
  available: boolean;
  keyframe_id: string;
  row_index: number | null;
  projection: "erp" | "cutout";
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

export type ProposalNeighborProjection = {
  keyframe_id: string;
  distance_from_source_m: number;
  distance_to_proposal_m: number;
  erp_u: number;
  erp_v: number;
  theta_center: number;
  phi_center: number;
  angular_width: number;
  angular_height: number;
  viewpoint_angle_deg: number;
  apparent_area_ratio: number;
  distance_score: number;
  view_angle_score: number;
  apparent_size_score: number;
  geometric_confidence: number;
  observed_depth_m: number | null;
  occlusion_margin_m: number | null;
  occlusion_confidence: number | null;
  occlusion_status: "clear" | "occluded" | "depth_mismatch" | "unknown";
  feature_match_score: number | null;
  feature_matches: number;
  feature_inliers: number;
  feature_inlier_ratio: number | null;
  feature_match_status: "matched" | "weak" | "no_features" | "unavailable";
};

export type ProposalNeighborProjectionsResponse = {
  row_index: number;
  source_keyframe_id: string;
  requested_count: number;
  projections: ProposalNeighborProjection[];
  note: string;
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
