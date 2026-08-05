import type { IndexObjectRecord } from "../index-explorer/types";

export type BboxPostProcessParams = {
  enabled: boolean;
  nmsIou: number;
  minBboxArea: number;
  maxBboxArea: number;
  showYolo: boolean;
  showGdino: boolean;
};

export const DEFAULT_BBOX_POST_PROCESS: BboxPostProcessParams = {
  enabled: true,
  nmsIou: 0.5,
  minBboxArea: 0,
  maxBboxArea: 0,
  showYolo: true,
  showGdino: true,
};

const STORAGE_KEY = "object-search-gui.explorerBboxPostProcess";

export function readStoredBboxPostProcess(): BboxPostProcessParams {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_BBOX_POST_PROCESS;
    }
    const parsed = JSON.parse(raw) as Partial<BboxPostProcessParams>;
    return {
      enabled:
        typeof parsed.enabled === "boolean"
          ? parsed.enabled
          : DEFAULT_BBOX_POST_PROCESS.enabled,
      nmsIou: clampNumber(parsed.nmsIou, 0.05, 1, DEFAULT_BBOX_POST_PROCESS.nmsIou),
      minBboxArea: clampNumber(
        parsed.minBboxArea,
        0,
        1_000_000,
        DEFAULT_BBOX_POST_PROCESS.minBboxArea,
      ),
      maxBboxArea: clampNumber(
        parsed.maxBboxArea,
        0,
        1_000_000,
        DEFAULT_BBOX_POST_PROCESS.maxBboxArea,
      ),
      showYolo:
        typeof parsed.showYolo === "boolean"
          ? parsed.showYolo
          : DEFAULT_BBOX_POST_PROCESS.showYolo,
      showGdino:
        typeof parsed.showGdino === "boolean"
          ? parsed.showGdino
          : DEFAULT_BBOX_POST_PROCESS.showGdino,
    };
  } catch {
    return DEFAULT_BBOX_POST_PROCESS;
  }
}

export function storeBboxPostProcess(params: BboxPostProcessParams): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  } catch {
    /* ignore */
  }
}

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, num));
}

export function bboxArea(bbox: [number, number, number, number]): number {
  const [x0, y0, x1, y1] = bbox;
  return Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
}

function bboxIou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter <= 0) {
    return 0;
  }
  const areaA = bboxArea(a);
  const areaB = bboxArea(b);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function detectionScore(item: IndexObjectRecord): number {
  if (item.textness !== null && item.textness !== undefined) {
    return item.textness;
  }
  return bboxArea(item.bbox);
}

function detectionSourceFamily(item: IndexObjectRecord): "yolo" | "gdino" | "other" {
  const source = item.detection_source.toLowerCase();
  if (source.includes("yolo")) {
    return "yolo";
  }
  if (source.includes("gdino") || source.includes("g-dino") || source.includes("grounding")) {
    return "gdino";
  }
  return "other";
}

export function postProcessDetections(
  detections: IndexObjectRecord[],
  params: BboxPostProcessParams,
): IndexObjectRecord[] {
  if (!params.enabled) {
    return detections;
  }
  const maxLimit = params.maxBboxArea <= 0 ? Number.POSITIVE_INFINITY : params.maxBboxArea;
  const minLimit = Math.max(0, params.minBboxArea);
  const filteredBySourceAndArea = detections.filter((item) => {
    const family = detectionSourceFamily(item);
    if (family === "yolo" && !params.showYolo) {
      return false;
    }
    if (family === "gdino" && !params.showGdino) {
      return false;
    }
    if (family === "other" && (!params.showYolo || !params.showGdino)) {
      return false;
    }
    const area = bboxArea(item.bbox);
    return area >= minLimit && area <= maxLimit;
  });
  if (params.nmsIou >= 1 || filteredBySourceAndArea.length <= 1) {
    return filteredBySourceAndArea;
  }
  const ordered = [...filteredBySourceAndArea].sort((a, b) => detectionScore(b) - detectionScore(a));
  const kept: IndexObjectRecord[] = [];
  for (const item of ordered) {
    if (kept.every((other) => bboxIou(item.bbox, other.bbox) < params.nmsIou)) {
      kept.push(item);
    }
  }
  return kept;
}

export function bboxPostProcessCacheKey(params: BboxPostProcessParams): string {
  return `${params.enabled}-${params.nmsIou}-${params.minBboxArea}-${params.maxBboxArea}-${params.showYolo}-${params.showGdino}`;
}

export function bboxPostProcessQueryParams(
  params: BboxPostProcessParams,
): URLSearchParams {
  const query = new URLSearchParams();
  if (!params.enabled) {
    query.set("nms_iou", "1");
    query.set("min_bbox_area", "0");
    query.set("max_bbox_area", "0");
    query.set("show_yolo", "true");
    query.set("show_gdino", "true");
    return query;
  }
  query.set("nms_iou", String(params.nmsIou));
  query.set("min_bbox_area", String(params.minBboxArea));
  if (params.maxBboxArea > 0) {
    query.set("max_bbox_area", String(params.maxBboxArea));
  }
  query.set("show_yolo", params.showYolo ? "true" : "false");
  query.set("show_gdino", params.showGdino ? "true" : "false");
  return query;
}
