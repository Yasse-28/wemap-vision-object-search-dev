import type { RoiPolygon } from "../annotations/types";

/**
 * One region drawn during a selection session.
 *
 * `level` is captured **when the polygon closes**, not read at export time: the
 * point of the session is that you draw on one floor, switch, and draw on another,
 * so the floor the map happens to show at the end says nothing about a region drawn
 * five minutes earlier.
 */
export type RoiEntry = {
  id: string;
  ring: RoiPolygon;
  level: string | null;
  createdAt: number;
};

export type ExportSessionState = {
  active: boolean;
  rois: RoiEntry[];
};

/** Keyframes a region set selects, resolved client-side from the marker table. */
export type RoiSelection = {
  keyframeIds: string[];
  total: number;
  perLevel: Array<{ level: string | null; count: number }>;
  /** Per region id, so the list can show each one's own contribution. */
  countByRoi: Record<string, number>;
};

export type DirectoryEntry = { name: string; path: string };

export type DirectoryListing = {
  path: string;
  parent: string | null;
  roots: string[];
  entries: DirectoryEntry[];
};

/** The authoritative count, resolved by the python selection rather than the SDK. */
export type ExportPreview = {
  total: number;
  per_level: Record<string, number>;
  keyframe_ids: number[];
  geo_ref_id: number;
  geo_ref_id_allocated_against_database: boolean;
};

export type ExportRequest = {
  map_id: string;
  destination_path: string;
  rois: Array<{ ring: Array<[number, number]>; level: number | null }>;
  include_artifacts: boolean;
  register_in_config: boolean;
};

export type ExportJob = {
  jobId: string;
  mapId: string;
  state: "idle" | "running" | "succeeded" | "failed";
  step: string;
  detail: string;
  startedAt: string;
  finishedAt: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
  configSnippet: string | null;
  configWritten: boolean;
};

export type DeletionPreview = {
  map_id: string;
  path: string;
  parent_map: string | null;
  geo_ref_id: number | null;
  keyframe_count: number | null;
  candidate_rows: number | null;
  geokeyframe_rows: number | null;
  index_name: string | null;
  removes_directory: boolean;
  warnings: string[];
};
