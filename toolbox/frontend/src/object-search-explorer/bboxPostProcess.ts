/**
 * Client-side box post-processing for the explorer.
 *
 * Two things changed with the v2 metadata. Areas are **square degrees**
 * (`angular_area_deg2`), because a proposal has an angular extent and no fixed
 * cutout raster to measure pixels in. And NMS runs across the whole keyframe rather
 * than per cutout — there are no cutouts to group by, and collapsing the two
 * detectors' overlapping proposals is the point. The backend applies the identical
 * rules for the drawn ERP preview (`workbench-index.ts::filterRows`); the query
 * parameter names are unchanged so the two stay in step.
 */

import type { MetadataRowRecord } from "../index-explorer/types";

export type BboxPostProcessParams = {
  enabled: boolean;
  nmsIou: number;
  /** Square degrees, not ERP pixels. */
  minBboxArea: number;
  maxBboxArea: number;
  showYolo: boolean;
  showGdino: boolean;
};

/** A full sphere is 360x180 deg; nothing legitimate approaches it. */
export const MAX_AREA_DEG2 = 360 * 180;

export const DEFAULT_BBOX_POST_PROCESS: BboxPostProcessParams = {
  enabled: true,
  nmsIou: 0.5,
  minBboxArea: 0,
  maxBboxArea: 0,
  showYolo: true,
  showGdino: true,
};

/**
 * The four corners of a row's reconstructed ERP box, in texture ratios.
 *
 * The box arrives ready-made in `erp_bbox_ratios`, so there is no geometry here any
 * more — the 45 lines of cubemap algebra this replaces lived in two copies (here and
 * in the backend) and had to agree. `u` is intentionally left unwrapped: the viewer
 * reads it as a yaw, where 1.02 and 0.02 are the same direction.
 */
export function bboxPolygonRatios(row: MetadataRowRecord): Array<[number, number]> {
  const [u0, v0, u1, v1] = row.erp_bbox_ratios;
  return [
    [u0, v0],
    [u1, v0],
    [u1, v1],
    [u0, v1],
  ];
}

// v2: the area thresholds changed unit (ERP pixels -> square degrees), so a stored
// v1 value like 5000 would silently filter every row out. New key, fresh defaults.
const STORAGE_KEY = "object-search-gui.explorerBboxPostProcess.v2";

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
        MAX_AREA_DEG2,
        DEFAULT_BBOX_POST_PROCESS.minBboxArea,
      ),
      maxBboxArea: clampNumber(
        parsed.maxBboxArea,
        0,
        MAX_AREA_DEG2,
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

/** Angular footprint of a row, in square degrees. */
export function bboxArea(row: MetadataRowRecord): number {
  return Number.isFinite(row.angular_area_deg2) ? Math.max(0, row.angular_area_deg2) : 0;
}

/** IoU of two reconstructed ERP boxes, in ratio units. */
function bboxIou(a: MetadataRowRecord, b: MetadataRowRecord): number {
  const [ax0, ay0, ax1, ay1] = a.erp_bbox_ratios;
  const [bx0, by0, bx1, by1] = b.erp_bbox_ratios;
  const interW = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0));
  const interH = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  const inter = interW * interH;
  if (inter <= 0) {
    return 0;
  }
  const areaA = Math.max(0, ax1 - ax0) * Math.max(0, ay1 - ay0);
  const areaB = Math.max(0, bx1 - bx0) * Math.max(0, by1 - by0);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function detectionScore(item: MetadataRowRecord): number {
  // Detector confidence, where the cubemap code used an OCR textness score that
  // v2 never had. Falls back to angular area when the column is NULL.
  return item.detection_score ?? bboxArea(item);
}

function detectionSourceFamily(item: MetadataRowRecord): "yolo" | "gdino" | "other" {
  const source = item.detector_source.toLowerCase();
  if (source.includes("yolo")) {
    return "yolo";
  }
  if (source.includes("gdino") || source.includes("g-dino") || source.includes("grounding")) {
    return "gdino";
  }
  return "other";
}

export function postProcessDetections(
  detections: MetadataRowRecord[],
  params: BboxPostProcessParams,
): MetadataRowRecord[] {
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
    const area = bboxArea(item);
    return area >= minLimit && area <= maxLimit;
  });
  if (params.nmsIou >= 1 || filteredBySourceAndArea.length <= 1) {
    return filteredBySourceAndArea;
  }
  const ordered = [...filteredBySourceAndArea].sort((a, b) => detectionScore(b) - detectionScore(a));
  const kept: MetadataRowRecord[] = [];
  for (const item of ordered) {
    if (kept.every((other) => bboxIou(item, other) < params.nmsIou)) {
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
