import type { BboxPostProcessParams } from "../object-search-explorer/bboxPostProcess";
import { bboxPostProcessQueryParams } from "../object-search-explorer/bboxPostProcess";
import type {
  ClusterOcrRecord,
  CutoutIndexPayload,
  DepthPinResponse,
  IndexObjectRecord,
  IndexStatusResponse,
  KeyframeGraphResponse,
  ProjectWorldPointResponse,
  PromptLatentResponse,
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

function normalizeIndexObjectRecord(item: unknown): IndexObjectRecord | null {
  if (!item || typeof item !== "object") {
    return null;
  }
  const record = item as Record<string, unknown>;
  const bbox = parseBbox(record);
  if (!bbox) {
    return null;
  }
  return {
    id: String(record.id ?? ""),
    keyframe_id: String(record.keyframe_id ?? ""),
    cutout_id: String(record.cutout_id ?? ""),
    bbox,
    bbox_coordinates: bbox,
    label: String(record.label ?? ""),
    detection_source: String(record.detection_source ?? ""),
    ocr_text: String(record.ocr_text ?? ""),
    ocr_tokens: String(record.ocr_tokens ?? ""),
    ocr_key: String(record.ocr_key ?? ""),
    ocr_source: String(record.ocr_source ?? ""),
    textness:
      record.textness == null || !Number.isFinite(Number(record.textness))
        ? null
        : Number(record.textness),
    ocr_candidate: Boolean(record.ocr_candidate),
    ocr_assigned: Boolean(record.ocr_assigned),
    cluster_id:
      record.cluster_id == null || !Number.isFinite(Number(record.cluster_id))
        ? null
        : Number(record.cluster_id),
  };
}

export async function fetchIndexStatus(
  mapId: string,
  selectedKeyframeId: string | null,
): Promise<IndexStatusResponse> {
  const params = new URLSearchParams();
  if (selectedKeyframeId) {
    params.set("selected_keyframe_id", selectedKeyframeId);
  }
  const query = params.toString();
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index${query ? `?${query}` : ""}`,
  );
  return readJson(response);
}

export async function fetchIndexObjects(
  mapId: string,
  keyframeId?: string | null,
): Promise<IndexObjectRecord[]> {
  const params = new URLSearchParams();
  if (keyframeId) {
    params.set("keyframe_id", keyframeId);
  }
  const query = params.toString();
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/objects${query ? `?${query}` : ""}`,
  );
  const payload = await readJson<{ objects: unknown[] }>(response);
  return Array.isArray(payload.objects)
    ? payload.objects.flatMap((item) => {
        const parsed = normalizeIndexObjectRecord(item);
        return parsed ? [parsed] : [];
      })
    : [];
}

export async function fetchKeyframeGraph(mapId: string): Promise<KeyframeGraphResponse> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframe-graph`,
  );
  return readJson(response);
}

export async function fetchClusterOcr(mapId: string): Promise<ClusterOcrRecord[]> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/cluster-ocr`,
  );
  const payload = await readJson<{ records: ClusterOcrRecord[] }>(response);
  return payload.records;
}

export async function fetchIndexCutout(
  mapId: string,
  cutoutId: string,
): Promise<CutoutIndexPayload> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/cutouts/${encodeURIComponent(cutoutId)}`,
  );
  const payload = await readJson<CutoutIndexPayload>(response);
  return {
    ...payload,
    detections: Array.isArray(payload.detections)
      ? payload.detections.flatMap((item) => {
          const parsed = normalizeIndexObjectRecord(item);
          return parsed ? [parsed] : [];
        })
      : [],
  };
}

function appendBboxPostProcessParams(
  params: URLSearchParams,
  bboxPostProcess?: BboxPostProcessParams,
): void {
  if (!bboxPostProcess) {
    return;
  }
  const extra = bboxPostProcessQueryParams(bboxPostProcess);
  extra.forEach((value, key) => {
    params.set(key, value);
  });
}

export function indexKeyframeEquirectPreviewUrl(
  mapId: string,
  keyframeId: string,
  selectedObjectId: string | null,
  bboxPostProcess?: BboxPostProcessParams,
  options?: { drawBoxes?: boolean },
): string {
  const params = new URLSearchParams();
  if (selectedObjectId) {
    params.set("selected_object_id", selectedObjectId);
  }
  if (options?.drawBoxes === false) {
    params.set("draw_boxes", "false");
  }
  appendBboxPostProcessParams(params, bboxPostProcess);
  const query = params.toString();
  return `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframes/${encodeURIComponent(keyframeId)}/equirect-preview.png${query ? `?${query}` : ""}`;
}

export function indexKeyframeDepthPreviewUrl(
  mapId: string,
  keyframeId: string,
): string {
  return `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframes/${encodeURIComponent(keyframeId)}/depth-preview.png`;
}

export function indexCutoutPreviewUrl(
  mapId: string,
  cutoutId: string,
  selectedObjectId: string | null,
  options?: { drawBoxes?: boolean; bboxPostProcess?: BboxPostProcessParams },
): string {
  const params = new URLSearchParams();
  if (selectedObjectId) {
    params.set("selected_object_id", selectedObjectId);
  }
  if (options?.drawBoxes === false) {
    params.set("draw_boxes", "false");
  }
  appendBboxPostProcessParams(params, options?.bboxPostProcess);
  const query = params.toString();
  return `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/cutouts/${encodeURIComponent(cutoutId)}/preview.png${query ? `?${query}` : ""}`;
}

export function indexObjectCropUrl(
  mapId: string,
  cutoutId: string,
  objectId: string,
  bbox: [number, number, number, number],
): string {
  const params = new URLSearchParams({
    x0: String(bbox[0]),
    y0: String(bbox[1]),
    x1: String(bbox[2]),
    y1: String(bbox[3]),
  });
  return `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/cutouts/${encodeURIComponent(cutoutId)}/objects/${encodeURIComponent(objectId)}/crop.png?${params}`;
}

export async function resolveDepthPin(
  mapId: string,
  keyframeId: string,
  body: {
    projection: "erp" | "cutout";
    x_ratio: number;
    y_ratio: number;
    cutout_id?: string;
    sample_radius?: number;
    min_depth_m?: number;
  },
): Promise<DepthPinResponse> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframes/${encodeURIComponent(keyframeId)}/depth-pin`,
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
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframes/${encodeURIComponent(keyframeId)}/project-world-point`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_world_wds: pointWorldWds }),
    },
  );
  return readJson(response);
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
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/keyframes/${encodeURIComponent(keyframeId)}/view-cone`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  return readJson(response);
}

export async function computePromptLatent(
  mapId: string,
  body: Record<string, unknown>,
): Promise<PromptLatentResponse> {
  const response = await fetch(
    `/ui/api/maps/${encodeURIComponent(mapId)}/object-search-index/prompt-latent`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  return readJson(response);
}
