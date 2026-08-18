/**
 * Client for the `/object-search-metadata/*` routes.
 *
 * The prefix was `/object-search-index`, which promised a database that no longer
 * exists. Renaming it means an old bookmark 404s instead of half-working — the right
 * trade for a local dev tool.
 */

import type { BboxPostProcessParams } from "../object-search-explorer/bboxPostProcess";
import { bboxPostProcessQueryParams } from "../object-search-explorer/bboxPostProcess";
import type {
  DepthPinResponse,
  KeyframeGraphResponse,
  KeyframeMarker,
  MetadataKeyframesResponse,
  MetadataKeyframesWireResponse,
  MetadataMarkersWireResponse,
  MetadataRowDetail,
  MetadataRowRecord,
  MetadataRowsResponse,
  MetadataStatusResponse,
  ProjectWorldPointResponse,
  ProposalNeighborProjectionsResponse,
  ViewConeResponse,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function metadataUrl(mapId: string, suffix: string, query?: URLSearchParams): string {
  const search = query?.toString();
  return (
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-metadata${suffix}`
    + (search ? `?${search}` : "")
  );
}

function normalizeRow(item: unknown): MetadataRowRecord | null {
  if (!item || typeof item !== "object") {
    return null;
  }
  const record = item as Record<string, unknown>;
  const rowIndex = Number(record.row_index);
  const ratios = Array.isArray(record.erp_bbox_ratios)
    ? record.erp_bbox_ratios.map(Number)
    : [];
  if (!Number.isInteger(rowIndex) || ratios.length < 4 || !ratios.every(Number.isFinite)) {
    return null;
  }
  const nullableNumber = (value: unknown): number | null =>
    value == null || !Number.isFinite(Number(value)) ? null : Number(value);
  return {
    row_index: rowIndex,
    keyframe_id: String(record.keyframe_id ?? ""),
    label: record.label == null ? null : String(record.label),
    detector_source: String(record.detector_source ?? ""),
    detection_score: nullableNumber(record.detection_score),
    depth: nullableNumber(record.depth),
    thumbnail_key:
      typeof record.thumbnail_key === "string" && record.thumbnail_key.length
        ? record.thumbnail_key
        : null,
    theta_center: Number(record.theta_center),
    phi_center: Number(record.phi_center),
    angular_width: Number(record.angular_width),
    angular_height: Number(record.angular_height),
    angular_area_deg2: Number(record.angular_area_deg2),
    erp_bbox_ratios: [ratios[0], ratios[1], ratios[2], ratios[3]],
  };
}

function assertAlignedColumns(
  label: string,
  columns: ReadonlyArray<readonly unknown[]>,
): void {
  const expected = columns[0]?.length ?? 0;
  if (!columns.every((column) => column.length === expected)) {
    throw new Error(`${label} column lengths are not aligned.`);
  }
}

export async function fetchMetadataStatus(mapId: string): Promise<MetadataStatusResponse> {
  return readJson(await fetch(metadataUrl(mapId, "")));
}

/** Materialize the optional marker table, including its dense ids and level dictionary. */
export async function fetchMetadataMarkers(mapId: string): Promise<KeyframeMarker[]> {
  const columns = await readJson<MetadataMarkersWireResponse>(
    await fetch(metadataUrl(mapId, "/markers")),
  );
  const ids = columns.ids_are_dense
    ? columns.latitude.map((_value, index) => String(index))
    : columns.ids;
  if (!ids) {
    throw new Error("markers.ids is required when ids_are_dense is false.");
  }
  assertAlignedColumns("markers", [
    ids,
    columns.latitude,
    columns.longitude,
    columns.level_codes,
    columns.heading_deg,
  ]);
  return ids.map((id, index) => {
    const levelCode = columns.level_codes[index];
    const level = levelCode === -1 ? null : columns.levels[levelCode];
    if (levelCode < -1 || (levelCode >= 0 && level === undefined)) {
      throw new Error(`markers.level_codes[${index}] is outside the level dictionary.`);
    }
    return {
      id,
      latitude: columns.latitude[index],
      longitude: columns.longitude[index],
      level,
      heading_deg: columns.heading_deg[index],
    };
  });
}

export type MetadataKeyframesOptions = {
  offset?: number;
  limit?: number;
  sort?: "parquet" | "objects-desc" | "objects-asc";
  includeEmpty?: boolean;
  keyframeId?: string | null;
};

export async function fetchMetadataKeyframes(
  mapId: string,
  options: MetadataKeyframesOptions = {},
): Promise<MetadataKeyframesResponse> {
  const params = new URLSearchParams();
  if (options.offset != null) params.set("offset", String(options.offset));
  if (options.limit != null) params.set("limit", String(options.limit));
  if (options.sort) params.set("sort", options.sort);
  if (options.includeEmpty != null) {
    params.set("include_empty", String(options.includeEmpty));
  }
  if (options.keyframeId) params.set("keyframe_id", options.keyframeId);
  const payload = await readJson<MetadataKeyframesWireResponse>(
    await fetch(metadataUrl(mapId, "/keyframes", params)),
  );
  const { total, ...columns } = payload;
  assertAlignedColumns("keyframes", Object.values(columns));
  return {
    total,
    keyframes: columns.ids.map((id, index) => ({
      id,
      row_start: columns.row_start[index],
      row_count: columns.row_count[index],
      row_end: columns.row_start[index] + columns.row_count[index],
      with_depth: columns.with_depth[index],
      ingested: columns.ingested[index],
      no_position: columns.no_position[index],
    })),
  };
}

/**
 * Rows for a set of keyframes, in one request.
 *
 * The explorer used to issue one request per keyframe per page; the parquet is read
 * whole and cached server-side, so a single call is both simpler and cheaper.
 */
export async function fetchMetadataRows(
  mapId: string,
  options: { keyframeIds?: string[]; offset?: number; limit?: number } = {},
): Promise<MetadataRowsResponse> {
  const params = new URLSearchParams();
  if (options.keyframeIds?.length) {
    params.set("keyframe_ids", options.keyframeIds.join(","));
  }
  if (options.offset != null) {
    params.set("offset", String(options.offset));
  }
  if (options.limit != null) {
    params.set("limit", String(options.limit));
  }
  const payload = await readJson<MetadataRowsResponse>(
    await fetch(metadataUrl(mapId, "/rows", params)),
  );
  return {
    ...payload,
    rows: Array.isArray(payload.rows)
      ? payload.rows.flatMap((item) => {
          const parsed = normalizeRow(item);
          return parsed ? [parsed] : [];
        })
      : [],
  };
}

export async function fetchMetadataRow(
  mapId: string,
  rowIndex: number,
): Promise<MetadataRowDetail> {
  const payload = await readJson<MetadataRowDetail>(
    await fetch(metadataUrl(mapId, `/rows/${rowIndex}`)),
  );
  const row = normalizeRow(payload.row);
  if (!row) {
    throw new Error(`Row ${rowIndex} came back malformed.`);
  }
  return { ...payload, row };
}

export async function fetchKeyframeGraph(mapId: string): Promise<KeyframeGraphResponse> {
  return readJson(await fetch(metadataUrl(mapId, "/keyframe-graph")));
}

/** The stored 336px thumbnail — the default, and the only image MetaCLIP2 saw. */
export function rowThumbnailUrl(mapId: string, thumbnailKey: string): string {
  const params = new URLSearchParams({ preview_path: thumbnailKey });
  return `/ui/api/maps/${encodeURIComponent(mapId)}/preview.png?${params}`;
}

/**
 * A re-rendered gnomonic view of one row, for context or when no thumbnail exists.
 *
 * `fovScale > 1` widens the view around the same centre. At `fovScale = 1` this is
 * the proposal's true angular extent, which is **not** identical to the stored
 * thumbnail: that one went through `build_padding_mask`, which crops wide boxes.
 */
export function rowRenderUrl(
  mapId: string,
  rowIndex: number,
  options: { size?: number; fovScale?: number } = {},
): string {
  const params = new URLSearchParams();
  if (options.size != null) {
    params.set("size", String(options.size));
  }
  if (options.fovScale != null) {
    params.set("fov_scale", String(options.fovScale));
  }
  return metadataUrl(mapId, `/rows/${rowIndex}/render.png`, params);
}

function appendBboxPostProcessParams(
  params: URLSearchParams,
  bboxPostProcess?: BboxPostProcessParams,
): void {
  if (!bboxPostProcess) {
    return;
  }
  bboxPostProcessQueryParams(bboxPostProcess).forEach((value, key) => {
    params.set(key, value);
  });
}

export function indexKeyframeEquirectPreviewUrl(
  mapId: string,
  keyframeId: string,
  selectedRowIndex: number | null,
  bboxPostProcess?: BboxPostProcessParams,
  options?: { drawBoxes?: boolean },
): string {
  const params = new URLSearchParams();
  if (selectedRowIndex !== null) {
    params.set("selected_row_index", String(selectedRowIndex));
  }
  if (options?.drawBoxes === false) {
    params.set("draw_boxes", "false");
  }
  appendBboxPostProcessParams(params, bboxPostProcess);
  return metadataUrl(
    mapId,
    `/keyframes/${encodeURIComponent(keyframeId)}/equirect-preview.png`,
    params,
  );
}

export function indexKeyframeDepthPreviewUrl(mapId: string, keyframeId: string): string {
  return metadataUrl(mapId, `/keyframes/${encodeURIComponent(keyframeId)}/depth-preview.png`);
}

export async function resolveDepthPin(
  mapId: string,
  keyframeId: string,
  body: {
    projection: "erp" | "cutout";
    x_ratio: number;
    y_ratio: number;
    row_index?: number;
    sample_radius?: number;
    min_depth_m?: number;
  },
): Promise<DepthPinResponse> {
  const response = await fetch(
    metadataUrl(mapId, `/keyframes/${encodeURIComponent(keyframeId)}/depth-pin`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  return readJson(response);
}

export async function projectWorldPoint(
  mapId: string,
  keyframeId: string,
  pointWorldWds: [number, number, number],
): Promise<ProjectWorldPointResponse> {
  const response = await fetch(
    metadataUrl(mapId, `/keyframes/${encodeURIComponent(keyframeId)}/project-world-point`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_world_wds: pointWorldWds }),
    },
  );
  return readJson(response);
}

export async function fetchProposalNeighborProjections(
  mapId: string,
  rowIndex: number,
  count: number,
  diverseViews: boolean = true,
): Promise<ProposalNeighborProjectionsResponse> {
  const params = new URLSearchParams({
    count: String(count),
    diverse: String(diverseViews),
  });
  return readJson(
    await fetch(metadataUrl(mapId, `/rows/${rowIndex}/neighbor-projections`, params)),
  );
}

export function proposalNeighborProjectionRenderUrl(
  mapId: string,
  rowIndex: number,
  targetKeyframeId: string,
  options: { size?: number; fovScale?: number } = {},
): string {
  const params = new URLSearchParams();
  if (options.size != null) {
    params.set("size", String(options.size));
  }
  if (options.fovScale != null) {
    params.set("fov_scale", String(options.fovScale));
  }
  return metadataUrl(
    mapId,
    `/rows/${rowIndex}/neighbor-projections/${encodeURIComponent(targetKeyframeId)}.png`,
    params,
  );
}

export async function resolveViewCone(
  mapId: string,
  keyframeId: string,
  body: {
    yaw_rad: number;
    pitch_rad?: number;
    distance_m?: number;
    half_angle_deg?: number;
  },
): Promise<ViewConeResponse> {
  const response = await fetch(
    metadataUrl(mapId, `/keyframes/${encodeURIComponent(keyframeId)}/view-cone`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  return readJson(response);
}
