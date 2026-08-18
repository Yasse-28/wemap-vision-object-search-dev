import { execFile, spawn } from "node:child_process";
import { access, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import type { MapEntry } from "./config.js";
import {
  angularAreaDeg2,
  assertEquirect2to1,
  erpBboxRatios,
  erpRectsWrapped,
  cutoutRatioToErpUv,
  gnomonicFfmpegFilter,
  isRenderableGnomonic,
  type ErpViewAngles,
} from "./erp-geometry.js";
import {
  formatLevel,
  loadMapManifest,
  manifestKeyframeToWorldToCameraWds,
  type ManifestKeyframe,
  type MapManifest,
} from "./map-manifest.js";
import {
  MetadataError,
  metadataRowMatches,
  requireMetadata,
  resolveMetadataPath,
  rowByIndex,
  rowSlice,
  rowsForKeyframe,
  type LoadedMetadata,
  type MetadataRow,
} from "./object-search-metadata.js";
import { pythonBinaryCandidates } from "./python-process.js";

const execFileAsync = promisify(execFile);

const BOX_COLOR = "#11b5ae";
const BOX_COLOR_SELECTED = "#ff6b6b";
const MAX_PREVIEW_WIDTH = 8192;
const MAX_DEPTH_PREVIEW_WIDTH = 2048;
const VIEW_CONE_ARC_STEPS = 48;
/** Default long side of a re-rendered proposal view, in pixels. */
const DEFAULT_ROW_RENDER_SIZE = 512;
const MAX_ROW_RENDER_SIZE = 2048;
const DEFAULT_NEIGHBOR_PROJECTION_COUNT = 5;
const MAX_NEIGHBOR_PROJECTION_COUNT = 12;
const MIN_NEIGHBOR_PROJECTION_BASELINE_M = 0.5;
const PROJECTION_DISTANCE_CONFIDENCE_SCALE_M = 5;
const PROJECTION_VIEW_ANGLE_CONFIDENCE_SCALE_RAD = Math.PI / 4;
/** How long the status route waits on the bricks service for pgvector coverage. */
const COVERAGE_TIMEOUT_MS = 2500;

export class WorkbenchRouteError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Metadata failures already carry a status; re-label them as route errors. */
function asRouteError(error: unknown): never {
  if (error instanceof MetadataError) {
    throw new WorkbenchRouteError(error.status, error.message);
  }
  throw error;
}

/** Post-processing knobs the explorer sends with every preview request. */
type RowFilterParams = {
  nmsIou: number;
  /** Angular area in **square degrees** — not ERP pixels; v2 has no fixed raster. */
  minAreaDeg2: number;
  maxAreaDeg2: number;
  showYolo: boolean;
  showGdino: boolean;
};

// Rendered images, keyed by source path + mtime. Both drop the entry when the
// promise rejects, so one failed render is not cached as "no preview" forever.
const cutoutPreviewCache = new Map<string, Promise<Buffer>>();
const equirectPreviewCache = new Map<string, Promise<Buffer>>();

function ffmpegBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_FFMPEG ?? "ffmpeg";
}

function convertBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_CONVERT ?? "convert";
}

function identifyBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_IDENTIFY ?? "identify";
}

function asNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ row filtering

function rowSourceFamily(row: MetadataRow): "yolo" | "gdino" | "other" {
  const source = row.detectorSource.toLowerCase();
  if (source.includes("yolo")) {
    return "yolo";
  }
  if (source.includes("gdino") || source.includes("g-dino") || source.includes("grounding")) {
    return "gdino";
  }
  return "other";
}

/** IoU of two reconstructed ERP boxes, in ratio units (unwrapped — see erp-geometry). */
function rowIou(a: MetadataRow, b: MetadataRow): number {
  const ra = erpBboxRatios(rowAngles(a));
  const rb = erpBboxRatios(rowAngles(b));
  const interW = Math.max(0, Math.min(ra.u1, rb.u1) - Math.max(ra.u0, rb.u0));
  const interH = Math.max(0, Math.min(ra.v1, rb.v1) - Math.max(ra.v0, rb.v0));
  const inter = interW * interH;
  if (inter <= 0) {
    return 0;
  }
  const areaA = Math.max(0, ra.u1 - ra.u0) * Math.max(0, ra.v1 - ra.v0);
  const areaB = Math.max(0, rb.u1 - rb.u0) * Math.max(0, rb.v1 - rb.v0);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * The explorer's box post-processing, on rows instead of cutout-local detections.
 *
 * Two things changed with v2 and both are visible here. NMS now runs across the
 * whole keyframe rather than per cutout — there are no cutouts to group by, and
 * overlapping proposals from the two detectors are exactly what a reviewer wants
 * collapsed. And the area thresholds are **square degrees**, because there is no
 * fixed cutout raster to measure pixels against any more.
 */
function filterRows(rows: MetadataRow[], params: RowFilterParams): MetadataRow[] {
  const maxArea = params.maxAreaDeg2 <= 0 ? Number.POSITIVE_INFINITY : params.maxAreaDeg2;
  const minArea = Math.max(0, params.minAreaDeg2);
  const kept = rows.filter((row) => {
    const family = rowSourceFamily(row);
    if (family === "yolo" && !params.showYolo) {
      return false;
    }
    if (family === "gdino" && !params.showGdino) {
      return false;
    }
    if (family === "other" && (!params.showYolo || !params.showGdino)) {
      return false;
    }
    const area = angularAreaDeg2(rowAngles(row));
    return area >= minArea && area <= maxArea;
  });
  if (params.nmsIou >= 1 || kept.length <= 1) {
    return kept;
  }
  // Highest detector confidence wins; the cubemap code sorted on textness, which
  // never existed in v2 (OCR was not ported), and fell back to area.
  const ordered = [...kept].sort(
    (a, b) =>
      (b.detectionScore ?? angularAreaDeg2(rowAngles(b)))
      - (a.detectionScore ?? angularAreaDeg2(rowAngles(a))),
  );
  const survivors: MetadataRow[] = [];
  for (const row of ordered) {
    if (survivors.every((other) => rowIou(row, other) < params.nmsIou)) {
      survivors.push(row);
    }
  }
  return survivors;
}

/** Query-string post-processing knobs, with the same defaults the old routes had. */
export function rowFilterParamsFromQuery(read: (name: string) => string | null): RowFilterParams {
  const number = (name: string, fallback: number): number => {
    const value = Number(read(name));
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    nmsIou: number("nms_iou", 1),
    minAreaDeg2: number("min_bbox_area", 0),
    maxAreaDeg2: number("max_bbox_area", 0),
    showYolo: read("show_yolo") !== "false",
    showGdino: read("show_gdino") !== "false",
  };
}

// -------------------------------------------------------------------- wire shapes

/**
 * One row on the wire.
 *
 * `erp_bbox_ratios` is the reconstructed ERP box, **unwrapped** — `u` may fall
 * slightly outside `[0, 1]` for a box near the seam. That is what the photosphere
 * overlay wants (it turns `u` into a yaw, where 1.02 and 0.02 are the same
 * direction); anything drawing into the raster must wrap it itself. Shipping the box
 * from here is also why the frontend needs no copy of the geometry.
 */
function metadataRowRecord(row: MetadataRow): Record<string, unknown> {
  const angles = rowAngles(row);
  const rect = erpBboxRatios(angles);
  return {
    row_index: row.rowIndex,
    keyframe_id: String(row.videoKeyframeId),
    label: row.label,
    detector_source: row.detectorSource,
    detection_score: row.detectionScore,
    depth: row.depth,
    thumbnail_key: row.thumbnailKey,
    theta_center: row.thetaCenter,
    phi_center: row.phiCenter,
    angular_width: row.angularWidth,
    angular_height: row.angularHeight,
    angular_area_deg2: angularAreaDeg2(angles),
    erp_bbox_ratios: [rect.u0, rect.v0, rect.u1, rect.v1],
  };
}

// ---------------------------------------------------------------------- coverage

type KeyframeCoverage = { ingested: number; noPosition: number };

type IndexCoverage = {
  byKeyframe: Map<number, KeyframeCoverage>;
  ingestedTotal: number;
  noPositionTotal: number;
  keyframeCount: number;
};

/**
 * Per-keyframe pgvector coverage, from the bricks service — or `null`.
 *
 * `null` on any failure, deliberately and without propagating: Postgres being down
 * must leave keyframes in an "unknown" state, not break the whole explorer. Bringing
 * back a hard dependency on a datastore is precisely the failure mode this migration
 * is getting out of.
 *
 * It goes through the python service rather than a `pg` client here so the DSN stays
 * built in one place (`bricks/db.py`), which is that module's whole reason to exist.
 */
async function fetchIndexCoverage(
  map: MapEntry,
  pythonApiBaseUrl: string | null,
): Promise<IndexCoverage | null> {
  if (!pythonApiBaseUrl) {
    return null;
  }
  try {
    const response = await fetch(
      `${pythonApiBaseUrl}/${encodeURIComponent(map.id)}/object-search/index-coverage`,
      { signal: AbortSignal.timeout(COVERAGE_TIMEOUT_MS) },
    );
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as { keyframes?: unknown };
    if (!Array.isArray(payload.keyframes)) {
      return null;
    }
    const byKeyframe = new Map<number, KeyframeCoverage>();
    let ingestedTotal = 0;
    let noPositionTotal = 0;
    for (const entry of payload.keyframes) {
      const item = (entry ?? {}) as Record<string, unknown>;
      const keyframeId = Number(item.video_keyframe_id);
      if (!Number.isInteger(keyframeId)) {
        continue;
      }
      const ingested = asNumber(item.ingested);
      const noPosition = asNumber(item.no_position);
      byKeyframe.set(keyframeId, { ingested, noPosition });
      ingestedTotal += ingested;
      noPositionTotal += noPosition;
    }
    return {
      byKeyframe,
      ingestedTotal,
      noPositionTotal,
      keyframeCount: byKeyframe.size,
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------- payloads

/** Every keyframe the manifest knows about, as string ids. */
async function manifestKeyframeIds(map: MapEntry): Promise<string[]> {
  try {
    const manifest = await loadMapManifest(map.path);
    return manifest.keyframes.map((keyframe) => String(keyframe.keyframeId));
  } catch {
    return [];
  }
}

/** What the explorer needs before its optional marker table and keyframe page. */
export async function objectSearchMetadataStatusPayload(
  map: MapEntry,
  pythonApiBaseUrl: string | null = null,
): Promise<Record<string, unknown>> {
  const { metadataPath, checkedPaths, captureCount } = await resolveMetadataPath(map);
  const keyframeIds = await manifestKeyframeIds(map);
  const base = {
    metadata_path: metadataPath,
    map_path: map.path,
    checked_paths: checkedPaths,
    capture_count: captureCount,
    manifest_keyframe_count: keyframeIds.length,
    marker_count: keyframeIds.length,
    first_keyframe_id:
      keyframeIds.reduce<number | null>((smallest, id) => {
        const numericId = Number(id);
        return smallest === null || numericId < smallest ? numericId : smallest;
      }, null)?.toString() ?? null,
  };

  let metadata: LoadedMetadata;
  try {
    metadata = await requireMetadata(map);
  } catch (error) {
    return {
      ...base,
      available: false,
      postprocessed: false,
      summary: null,
      error:
        error instanceof MetadataError || error instanceof Error
          ? error.message
          : String(error),
    };
  }

  const coverage = await fetchIndexCoverage(map, pythonApiBaseUrl);
  return {
    ...base,
    available: true,
    postprocessed: metadata.postprocessed,
    error: null,
    summary: {
      metadata_path: metadata.metadataPath,
      row_count: metadata.rowCount,
      postprocessed: metadata.postprocessed,
      detector_source_counts: metadata.detectorSourceCounts,
      with_depth_count: metadata.withDepthCount,
      coverage: coverage
        ? {
            ingested_total: coverage.ingestedTotal,
            no_position_total: coverage.noPositionTotal,
            keyframe_count: coverage.keyframeCount,
          }
        : null,
    },
  };
}

/** Every manifest keyframe pose, fetched only when a panel actually needs it. */
export async function objectSearchMetadataMarkersPayload(
  map: MapEntry,
): Promise<Record<string, unknown>> {
  const records = await keyframeMarkerRecords(map, await manifestKeyframeIds(map));
  return columnarKeyframeMarkers(records);
}

export type MetadataKeyframesSort = "parquet" | "objects-desc" | "objects-asc";

export type MetadataKeyframesQuery = {
  offset: number;
  limit: number;
  sort: MetadataKeyframesSort;
  includeEmpty: boolean;
  keyframeId: string | null;
};

/** Per-keyframe parquet ranges, filtered and paged in the legacy panel order. */
export async function metadataKeyframesPayload(
  map: MapEntry,
  query: MetadataKeyframesQuery,
  pythonApiBaseUrl: string | null = null,
): Promise<Record<string, unknown>> {
  const metadata = await requireMetadata(map).catch(asRouteError);
  const coverage = await fetchIndexCoverage(map, pythonApiBaseUrl);
  let records: KeyframeSummaryRecord[] = metadata.keyframeIds.map((keyframeId) => {
    const range = metadata.rangeByKeyframe.get(keyframeId)!;
    const covered = coverage?.byKeyframe.get(keyframeId);
    return {
      id: String(keyframeId),
      row_count: range.rowEnd - range.rowStart,
      row_start: range.rowStart,
      row_end: range.rowEnd,
      with_depth: range.withDepth,
      ingested: coverage ? (covered?.ingested ?? 0) : null,
      no_position: coverage ? (covered?.noPosition ?? 0) : null,
    };
  });
  if (query.keyframeId !== null) {
    records = records.filter((record) => record.id === query.keyframeId);
  }
  if (!query.includeEmpty) {
    records = records.filter((record) => record.row_count > 0);
  }
  if (query.sort === "objects-desc") {
    records.sort(
      (a, b) => b.row_count - a.row_count || a.id.localeCompare(b.id),
    );
  } else if (query.sort === "objects-asc") {
    records.sort(
      (a, b) => a.row_count - b.row_count || a.id.localeCompare(b.id),
    );
  }
  const total = records.length;
  const offset = Math.max(0, Math.round(query.offset));
  const limit = Math.max(1, Math.min(20_000, Math.round(query.limit)));
  return {
    total,
    ...columnarKeyframeSummaries(records.slice(offset, offset + limit)),
  };
}

export type MetadataRowsQuery = {
  keyframeIds: string[] | null;
  offset: number;
  limit: number;
  detectorSource: string | null;
  labelQuery: string | null;
  withDepthOnly: boolean;
};

/** The flat row table, filtered and paged. */
export async function metadataRowsPayload(
  map: MapEntry,
  query: MetadataRowsQuery,
): Promise<Record<string, unknown>> {
  const metadata = await requireMetadata(map).catch(asRouteError);
  const wanted = query.keyframeIds?.length
    ? query.keyframeIds.map(Number).filter(Number.isInteger)
    : null;
  const labelQuery = query.labelQuery?.toLowerCase() ?? null;
  const filter = {
    detectorSource: query.detectorSource,
    labelQuery,
    withDepthOnly: query.withDepthOnly,
  };
  const offset = Math.max(0, Math.round(query.offset));
  const limit = Math.max(1, Math.min(20_000, Math.round(query.limit)));
  const rows: MetadataRow[] = [];
  let total = 0;
  const ranges = wanted
    ? wanted.flatMap((keyframeId) => {
        const range = metadata.rangeByKeyframe.get(keyframeId);
        return range ? [[range.rowStart, range.rowEnd] as const] : [];
      })
    : [[0, metadata.rowCount] as const];
  for (const [start, end] of ranges) {
    for (let rowPosition = start; rowPosition < end; rowPosition += 1) {
      if (!metadataRowMatches(metadata, rowPosition, filter)) {
        continue;
      }
      if (total >= offset && rows.length < limit) {
        rows.push(...rowSlice(metadata, rowPosition, rowPosition + 1));
      }
      total += 1;
    }
  }
  return {
    total,
    offset,
    limit,
    postprocessed: metadata.postprocessed,
    rows: rows.map(metadataRowRecord),
  };
}

/** One row, plus what is needed to preview it. */
export async function metadataRowPayload(
  map: MapEntry,
  rowIndex: number,
): Promise<Record<string, unknown>> {
  const metadata = await requireMetadata(map).catch(asRouteError);
  const row = rowByIndex(metadata, rowIndex);
  if (!row) {
    throw new WorkbenchRouteError(404, `Row ${rowIndex} not found in ${metadata.metadataPath}.`);
  }
  const sourceImagePath = await resolveSourceImagePath(map, String(row.videoKeyframeId));
  return {
    row: metadataRowRecord(row),
    // The thumbnail is the default preview: it is the only image certain to match
    // what MetaCLIP2 embedded. `/preview.png?preview_path=` serves it directly.
    preview_path: row.thumbnailKey,
    preview_debug: {
      metadata_path: metadata.metadataPath,
      postprocessed: metadata.postprocessed,
      keyframe_id: String(row.videoKeyframeId),
      source_image_path: sourceImagePath,
      thumbnail_key: row.thumbnailKey,
      // Reconstructed from float16 angles: good to a few ERP pixels, not exact.
      geometry: "erp-gnomonic",
    },
  };
}

export async function keyframeMetadataPayload(
  map: MapEntry,
  keyframeId: string,
): Promise<Record<string, unknown>> {
  const meta = await lookupFlatKeyframes(map, [keyframeId]);
  const found = meta.get(keyframeId);
  // `image_filename` makes a stale id visible. Ids are `geo_keyframes` indices, so a
  // bookmarked URL from before the v2 migration — or from an earlier export — lands
  // on whatever keyframe now sits at that index rather than 404ing. Reporting the
  // filename is the only thing that turns a silent mismatch into a checkable one.
  const imageFilename = await lookupKeyframeImageFilename(map, keyframeId);
  if (!found) {
    return {
      id: keyframeId,
      lat: null,
      lon: null,
      alt: null,
      level: null,
      image_filename: imageFilename,
    };
  }
  return {
    id: keyframeId,
    lat: found.lat,
    lon: found.lon,
    alt: found.alt,
    level: found.level,
    image_filename: imageFilename,
  };
}

type FlatKeyframe = { lat: number; lon: number; alt: number | null; level: string | null };

/**
 * Look keyframes up in the map's v2 manifest.
 *
 * Ids are `geo_keyframes` indices, and an id that is not in the manifest is simply
 * absent from the returned map — `keyframeMetadataPayload` turns that into the
 * all-null record, as it did when a row was missing from `georef.db`.
 */
async function lookupFlatKeyframes(
  map: MapEntry,
  keyframeIds: string[],
): Promise<Map<string, FlatKeyframe>> {
  const uniqueIds = [...new Set(keyframeIds.filter(Boolean))];
  const out = new Map<string, FlatKeyframe>();
  if (!uniqueIds.length) {
    return out;
  }
  let manifest: MapManifest;
  try {
    manifest = await loadMapManifest(map.path);
  } catch {
    return out;
  }
  const context = geoRefContextFromManifest(manifest);
  for (const id of uniqueIds) {
    const keyframe = manifest.keyframeById.get(Number(id));
    if (!keyframe) {
      continue;
    }
    const pose = manifestKeyframeToWorldToCameraWds(keyframe);
    const cameraToWorld = invertRigid4(pose);
    const geo = localWorldPointToGeo(context, [
      cameraToWorld[0][3],
      cameraToWorld[1][3],
      cameraToWorld[2][3],
    ]);
    out.set(id, {
      lat: geo.latitude,
      lon: geo.longitude,
      alt: geo.altitude,
      level: geo.level,
    });
  }
  return out;
}

type GeoRefContext = {
  originLatDeg: number;
  originLonDeg: number;
  originAlt: number;
  originTr: number[][];
  levels: Array<{ value: number; min: number; max: number }>;
};

/** The on-disk image basename for a keyframe, from the manifest's `image_url`. */
async function lookupKeyframeImageFilename(
  map: MapEntry,
  keyframeId: string,
): Promise<string | null> {
  try {
    const manifest = await loadMapManifest(map.path);
    return manifest.keyframeById.get(Number(keyframeId))?.imageFilename ?? null;
  } catch {
    return null;
  }
}

/** The on-disk depth basename for a keyframe, from the manifest's `depth_url`. */
async function lookupKeyframeDepthFilename(
  map: MapEntry,
  keyframeId: string,
): Promise<string | null> {
  try {
    const manifest = await loadMapManifest(map.path);
    return manifest.keyframeById.get(Number(keyframeId))?.depthFilename || null;
  } catch {
    return null;
  }
}

async function lookupKeyframeHeadings(
  map: MapEntry,
  keyframeIds: string[],
): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  const uniqueIds = [...new Set(keyframeIds.filter(Boolean))];
  if (!uniqueIds.length) {
    return out;
  }
  let manifest: MapManifest;
  try {
    manifest = await loadMapManifest(map.path);
  } catch {
    return out;
  }
  for (const id of uniqueIds) {
    const keyframe = manifest.keyframeById.get(Number(id));
    if (keyframe) {
      out.set(id, keyframeHeadingDegreesFromPose(manifestKeyframeToWorldToCameraWds(keyframe)));
    }
  }
  return out;
}

/**
 * The map's origin frame and level bands, from its manifest.
 *
 * `originTr` is still built from the same three numbers the `GeoRef` row used to
 * supply, so every consumer downstream is unchanged. Two things differ from the v1
 * reader and both matter:
 *
 * - `local_origin` is `[lng, lat, alt]` (Django `PointField` order), the opposite
 *   of the old `SELECT latitude, longitude, altitude`.
 * - Levels are keyed by `GeoLevel.value` — the actual floor — rather than by the
 *   `Level` row id, which only coincidentally matched it. The livemap consumes this
 *   string as a floor number, so this is a fix rather than a rename.
 */
function geoRefContextFromManifest(manifest: MapManifest): GeoRefContext {
  const originEcef = geodeticToEcef(manifest.origin.lat, manifest.origin.lng, manifest.origin.alt);
  const rot = wdsToEcefRotation(manifest.origin.lat, manifest.origin.lng);
  return {
    originLatDeg: manifest.origin.lat,
    originLonDeg: manifest.origin.lng,
    originAlt: manifest.origin.alt,
    originTr: [
      [rot[0][0], rot[0][1], rot[0][2], originEcef[0]],
      [rot[1][0], rot[1][1], rot[1][2], originEcef[1]],
      [rot[2][0], rot[2][1], rot[2][2], originEcef[2]],
      [0, 0, 0, 1],
    ],
    // Already sorted by value in the parser; first matching band wins below.
    levels: manifest.levels.map((level) => ({
      value: level.value,
      min: level.minAltitude,
      max: level.maxAltitude,
    })),
  };
}

function invertRigid4(m: number[][]): number[][] {
  const rT = [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
  const t = [m[0][3], m[1][3], m[2][3]];
  const invT = [
    -(rT[0][0] * t[0] + rT[0][1] * t[1] + rT[0][2] * t[2]),
    -(rT[1][0] * t[0] + rT[1][1] * t[1] + rT[1][2] * t[2]),
    -(rT[2][0] * t[0] + rT[2][1] * t[1] + rT[2][2] * t[2]),
  ];
  return [
    [rT[0][0], rT[0][1], rT[0][2], invT[0]],
    [rT[1][0], rT[1][1], rT[1][2], invT[1]],
    [rT[2][0], rT[2][1], rT[2][2], invT[2]],
    [0, 0, 0, 1],
  ];
}

function matMul3(a: number[][], b: number[][]): number[][] {
  return [0, 1, 2].map((row) =>
    [0, 1, 2].map((col) =>
      a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col],
    ),
  );
}

function rx(theta: number): number[][] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rz(theta: number): number[][] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

function wdsToEcefRotation(latDeg: number, lonDeg: number): number[][] {
  const lat = latDeg * Math.PI / 180;
  const lon = lonDeg * Math.PI / 180;
  return matMul3(rz(-Math.PI / 2 + lon), rx(Math.PI + lat));
}

function geodeticToEcef(latDeg: number, lonDeg: number, alt: number): [number, number, number] {
  const a = 6378137.0;
  const e2 = 6.69437999014e-3;
  const lat = latDeg * Math.PI / 180;
  const lon = lonDeg * Math.PI / 180;
  const sinLat = Math.sin(lat);
  const cosLat = Math.cos(lat);
  const n = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  return [
    (n + alt) * cosLat * Math.cos(lon),
    (n + alt) * cosLat * Math.sin(lon),
    (n * (1 - e2) + alt) * sinLat,
  ];
}

function ecefToGeodetic(x: number, y: number, z: number): { lat: number; lon: number; alt: number } {
  const a = 6378137.0;
  const e2 = 6.69437999014e-3;
  const b = a * Math.sqrt(1 - e2);
  const ep2 = (a * a - b * b) / (b * b);
  const p = Math.hypot(x, y);
  const theta = Math.atan2(z * a, p * b);
  const sinTheta = Math.sin(theta);
  const cosTheta = Math.cos(theta);
  const lat = Math.atan2(
    z + ep2 * b * sinTheta * sinTheta * sinTheta,
    p - e2 * a * cosTheta * cosTheta * cosTheta,
  );
  const lon = Math.atan2(y, x);
  const sinLat = Math.sin(lat);
  const n = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  const alt = p / Math.cos(lat) - n;
  return {
    lat: lat * 180 / Math.PI,
    lon: lon * 180 / Math.PI,
    alt,
  };
}

function transformPoint4(m: number[][], p: [number, number, number]): [number, number, number] {
  return [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
  ];
}

function wdsToEnu(pointWds: [number, number, number]): [number, number, number] {
  return [-pointWds[0], -pointWds[2], -pointWds[1]];
}

export function keyframeHeadingDegreesFromPose(worldToCameraPose: number[][]): number {
  const cameraToWorld = invertRigid4(worldToCameraPose);
  const originWorld = transformPoint4(cameraToWorld, [0, 0, 0]);
  const forwardWorld = transformPoint4(cameraToWorld, [0, 0, 1]);
  const forwardEnu = wdsToEnu([
    forwardWorld[0] - originWorld[0],
    forwardWorld[1] - originWorld[1],
    forwardWorld[2] - originWorld[2],
  ]);
  const heading = Math.atan2(forwardEnu[0], forwardEnu[1]) * 180 / Math.PI;
  return ((heading % 360) + 360) % 360;
}

function localWorldPointToGeo(
  context: GeoRefContext,
  pointWorld: [number, number, number],
): { latitude: number; longitude: number; altitude: number; level: string | null } {
  const ecef = transformPoint4(context.originTr, pointWorld);
  const coords = ecefToGeodetic(ecef[0], ecef[1], ecef[2]);
  const heightFromGround = coords.alt - context.originAlt;
  // Altitude band only. The manifest also carries a per-level footprint, which the
  // bricks use to disambiguate stacked buildings sharing a band; this route has
  // never done that, and still does not — do not assume parity with the Python.
  const level = context.levels.find((item) => heightFromGround >= item.min && heightFromGround <= item.max);
  return {
    latitude: coords.lat,
    longitude: coords.lon,
    altitude: coords.alt,
    level: level ? formatLevel(level.value) : null,
  };
}

function equirectUvToRay(u: number, v: number): [number, number, number] {
  const lon = (u - 0.5) * 2 * Math.PI;
  const lat = (v - 0.5) * Math.PI;
  const cosLat = Math.cos(lat);
  return [
    Math.sin(lon) * cosLat,
    Math.sin(lat),
    Math.cos(lon) * cosLat,
  ];
}

function cameraPointToEquirectUv(point: [number, number, number]): [number, number] | null {
  const distance = Math.hypot(point[0], point[1], point[2]);
  if (!Number.isFinite(distance) || distance < 1e-9) {
    return null;
  }
  const lon = Math.atan2(point[0], point[2]);
  const lat = Math.asin(Math.max(-1, Math.min(1, point[1] / distance)));
  return [
    (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1,
    Math.max(0, Math.min(1, lat / Math.PI + 0.5)),
  ];
}

function yawPitchToEquirectUv(yawRad: number, pitchRad: number): [number, number] {
  const twoPi = 2 * Math.PI;
  const u = (((yawRad / twoPi + 0.5) % 1) + 1) % 1;
  const v = Math.max(0, Math.min(1, pitchRad / Math.PI + 0.5));
  return [u, v];
}

/**
 * One keyframe's world-to-camera pose plus the map's origin frame.
 *
 * Feeds depth-pin, view-cone and project-world-point. The pose is in WDS, which is
 * what those payloads speak — `point_world_wds` crosses the network in both
 * directions — so the manifest's EUS pose is adapted rather than propagated.
 */
async function lookupKeyframePose(
  map: MapEntry,
  keyframeId: string,
): Promise<{ context: GeoRefContext; pose: number[][] } | null> {
  let manifest: MapManifest;
  try {
    manifest = await loadMapManifest(map.path);
  } catch {
    return null;
  }
  const keyframe = manifest.keyframeById.get(Number(keyframeId));
  if (!keyframe) {
    return null;
  }
  return {
    context: geoRefContextFromManifest(manifest),
    pose: manifestKeyframeToWorldToCameraWds(keyframe),
  };
}

type KeyframeMarkerRecord = {
  id: string;
  latitude: number;
  longitude: number;
  level: string | null;
  heading_deg: number | null;
};

type KeyframeSummaryRecord = {
  id: string;
  row_count: number;
  row_start: number;
  row_end: number;
  with_depth: number;
  ingested: number | null;
  no_position: number | null;
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

type ColumnarKeyframeSummaries = {
  ids: string[];
  row_start: number[];
  row_count: number[];
  with_depth: number[];
  ingested: Array<number | null>;
  no_position: Array<number | null>;
};

function assertParallelLengths(label: string, arrays: readonly unknown[][]): void {
  const expected = arrays[0]?.length ?? 0;
  if (!arrays.every((values) => values.length === expected)) {
    throw new Error(`${label} column lengths are not aligned.`);
  }
}

function roundWire(value: number, decimals: number): number {
  return Number(value.toFixed(decimals));
}

export function columnarKeyframeMarkers(
  records: KeyframeMarkerRecord[],
): ColumnarKeyframeMarkers {
  const idsAreDense = records.every((record, index) => record.id === String(index));
  const levels: string[] = [];
  const levelCodeByValue = new Map<string, number>();
  const levelCodes = records.map((record) => {
    if (record.level === null) {
      return -1;
    }
    const existing = levelCodeByValue.get(record.level);
    if (existing !== undefined) {
      return existing;
    }
    const code = levels.length;
    levels.push(record.level);
    levelCodeByValue.set(record.level, code);
    return code;
  });
  const columns: ColumnarKeyframeMarkers = {
    ids_are_dense: idsAreDense,
    ...(idsAreDense ? {} : { ids: records.map((record) => record.id) }),
    // Wire precision only: the manifest-derived values remain full precision.
    latitude: records.map((record) => roundWire(record.latitude, 7)),
    longitude: records.map((record) => roundWire(record.longitude, 7)),
    levels,
    level_codes: levelCodes,
    heading_deg: records.map((record) =>
      record.heading_deg === null ? null : roundWire(record.heading_deg, 2),
    ),
  };
  assertParallelLengths("markers", [
    columns.latitude,
    columns.longitude,
    columns.level_codes,
    columns.heading_deg,
    ...(columns.ids ? [columns.ids] : []),
  ]);
  return columns;
}

function columnarKeyframeSummaries(
  records: KeyframeSummaryRecord[],
): ColumnarKeyframeSummaries {
  const columns: ColumnarKeyframeSummaries = {
    ids: records.map((record) => record.id),
    row_start: records.map((record) => record.row_start),
    row_count: records.map((record) => record.row_count),
    with_depth: records.map((record) => record.with_depth),
    ingested: records.map((record) => record.ingested),
    no_position: records.map((record) => record.no_position),
  };
  assertParallelLengths("summary.keyframes", Object.values(columns));
  return columns;
}

async function keyframeMarkerRecords(
  map: MapEntry,
  keyframeIds: string[],
): Promise<KeyframeMarkerRecord[]> {
  const meta = await lookupFlatKeyframes(map, keyframeIds);
  const headings = await lookupKeyframeHeadings(map, [...meta.keys()]);
  return [...meta.entries()].map(([id, item]) => ({
    id,
    latitude: item.lat,
    longitude: item.lon,
    level: item.level,
    heading_deg: headings.get(id) ?? null,
  }));
}

/**
 * The ERP image on disk for a keyframe.
 *
 * Only the manifest's `image_url` basename is tried. There is deliberately no
 * `{keyframeId}.jpg` fallback: ids are `geo_keyframes` indices, so a stray `0.jpg`
 * would serve a completely unrelated image for a valid-looking request, silently.
 * A wrong *directory*, by contrast, can only fail loudly, so both conventions stay
 * in the search list.
 */
async function resolveSourceImagePath(
  map: MapEntry,
  keyframeId: string,
): Promise<string | null> {
  const filename = await lookupKeyframeImageFilename(map, keyframeId);
  if (!filename) {
    return null;
  }
  for (const dirname of ["images", "images_360"]) {
    const candidate = path.resolve(map.path, dirname, filename);
    if (await pathExists(candidate)) {
      return candidate;
    }
  }
  return null;
}

type ImageSize = { width: number; height: number };

async function imageSize(imagePath: string): Promise<ImageSize> {
  const { stdout } = await execFileAsync(
    identifyBinary(),
    ["-format", "%w %h", imagePath],
    { maxBuffer: 1024 * 1024 },
  );
  const [width, height] = stdout.trim().split(/\s+/).map(Number);
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new WorkbenchRouteError(500, `Could not identify image size for ${imagePath}`);
  }
  return { width, height };
}

/** The four canonical angle columns of a row, as the geometry module expects. */
function rowAngles(row: MetadataRow): ErpViewAngles {
  return {
    thetaCenter: row.thetaCenter,
    phiCenter: row.phiCenter,
    angularWidth: row.angularWidth,
    angularHeight: row.angularHeight,
  };
}

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
  appearance_ssim_score: number | null;
  appearance_ssim_status: "matched" | "weak" | "unavailable";
};

type ProposalProjectionGeometry = {
  row: MetadataRow;
  manifest: MapManifest;
  pointWorldWds: [number, number, number];
  sourceOriginWorldWds: [number, number, number];
  physicalWidthM: number;
  physicalHeightM: number;
};

type ProposalProjectionCandidate = {
  projection: ProposalNeighborProjection;
  originWorldWds: [number, number, number];
};

function keyframeOriginWorldWds(keyframe: ManifestKeyframe): [number, number, number] {
  return transformPoint4(invertRigid4(manifestKeyframeToWorldToCameraWds(keyframe)), [0, 0, 0]);
}

function proposalProjectionGeometry(
  row: MetadataRow,
  manifest: MapManifest,
): ProposalProjectionGeometry {
  if (row.depth === null || !Number.isFinite(row.depth) || row.depth <= 0) {
    throw new WorkbenchRouteError(
      422,
      `Row ${row.rowIndex} has no positive depth, so it cannot be projected into neighbours.`,
    );
  }
  const angles = rowAngles(row);
  if (!isRenderableGnomonic(angles)) {
    throw new WorkbenchRouteError(
      422,
      `Row ${row.rowIndex} has a degenerate angular extent and cannot be projected.`,
    );
  }
  const sourceKeyframe = manifest.keyframeById.get(row.videoKeyframeId);
  if (!sourceKeyframe) {
    throw new WorkbenchRouteError(
      404,
      `GeoRef pose not found for source keyframe ${row.videoKeyframeId}.`,
    );
  }
  const sourcePose = manifestKeyframeToWorldToCameraWds(sourceKeyframe);
  const [u, v] = yawPitchToEquirectUv(row.thetaCenter, row.phiCenter);
  const ray = equirectUvToRay(u, v);
  const pointWorldWds = transformPoint4(invertRigid4(sourcePose), [
    ray[0] * row.depth,
    ray[1] * row.depth,
    ray[2] * row.depth,
  ]);
  return {
    row,
    manifest,
    pointWorldWds,
    sourceOriginWorldWds: keyframeOriginWorldWds(sourceKeyframe),
    physicalWidthM: 2 * row.depth * Math.tan(row.angularWidth / 2),
    physicalHeightM: 2 * row.depth * Math.tan(row.angularHeight / 2),
  };
}

function projectProposalIntoKeyframe(
  geometry: ProposalProjectionGeometry,
  target: ManifestKeyframe,
): ProposalNeighborProjection | null {
  const targetPose = manifestKeyframeToWorldToCameraWds(target);
  const pointKeyframe = transformPoint4(targetPose, geometry.pointWorldWds);
  const uv = cameraPointToEquirectUv(pointKeyframe);
  if (!uv) {
    return null;
  }
  const distanceToProposal = Math.hypot(
    pointKeyframe[0],
    pointKeyframe[1],
    pointKeyframe[2],
  );
  const targetOrigin = keyframeOriginWorldWds(target);
  const distanceFromSource = Math.hypot(
    targetOrigin[0] - geometry.sourceOriginWorldWds[0],
    targetOrigin[1] - geometry.sourceOriginWorldWds[1],
    targetOrigin[2] - geometry.sourceOriginWorldWds[2],
  );
  const angularWidth = 2 * Math.atan(geometry.physicalWidthM / (2 * distanceToProposal));
  const angularHeight = 2 * Math.atan(geometry.physicalHeightM / (2 * distanceToProposal));
  const viewpointAngle = angleBetweenVectors(
    [
      geometry.sourceOriginWorldWds[0] - geometry.pointWorldWds[0],
      geometry.sourceOriginWorldWds[1] - geometry.pointWorldWds[1],
      geometry.sourceOriginWorldWds[2] - geometry.pointWorldWds[2],
    ],
    [
      targetOrigin[0] - geometry.pointWorldWds[0],
      targetOrigin[1] - geometry.pointWorldWds[1],
      targetOrigin[2] - geometry.pointWorldWds[2],
    ],
  );
  const apparentAreaRatio = Math.max(
    0,
    (angularWidth * angularHeight)
      / (geometry.row.angularWidth * geometry.row.angularHeight),
  );
  const distanceScore = Math.exp(
    -distanceFromSource / PROJECTION_DISTANCE_CONFIDENCE_SCALE_M,
  );
  const viewAngleScore = Math.exp(
    -viewpointAngle / PROJECTION_VIEW_ANGLE_CONFIDENCE_SCALE_RAD,
  );
  const apparentSizeScore = Math.min(1, Math.sqrt(apparentAreaRatio));
  // Weighted geometric mean: one weak dimension must lower the result without a
  // single noisy input collapsing it. Distance and angle use 5 m / 45° e-folding
  // scales; apparent size compares angular area with the source proposal.
  const geometricConfidence =
    distanceScore ** 0.35
    * viewAngleScore ** 0.4
    * apparentSizeScore ** 0.25;
  return {
    keyframe_id: String(target.keyframeId),
    distance_from_source_m: distanceFromSource,
    distance_to_proposal_m: distanceToProposal,
    erp_u: uv[0],
    erp_v: uv[1],
    theta_center: (uv[0] - 0.5) * 2 * Math.PI,
    phi_center: (uv[1] - 0.5) * Math.PI,
    angular_width: angularWidth,
    angular_height: angularHeight,
    viewpoint_angle_deg: viewpointAngle * 180 / Math.PI,
    apparent_area_ratio: apparentAreaRatio,
    distance_score: confidencePercent(distanceScore),
    view_angle_score: confidencePercent(viewAngleScore),
    apparent_size_score: confidencePercent(apparentSizeScore),
    geometric_confidence: confidencePercent(geometricConfidence),
    observed_depth_m: null,
    occlusion_margin_m: null,
    occlusion_confidence: null,
    occlusion_status: "unknown",
    feature_match_score: null,
    feature_matches: 0,
    feature_inliers: 0,
    feature_inlier_ratio: null,
    feature_match_status: "unavailable",
    appearance_ssim_score: null,
    appearance_ssim_status: "unavailable",
  };
}

function cameraBaselineM(
  left: ProposalProjectionCandidate,
  right: ProposalProjectionCandidate,
): number {
  return Math.hypot(
    left.originWorldWds[0] - right.originWorldWds[0],
    left.originWorldWds[1] - right.originWorldWds[1],
    left.originWorldWds[2] - right.originWorldWds[2],
  );
}

function angleBetweenVectors(left: number[], right: number[]): number {
  const denominator = Math.hypot(...left) * Math.hypot(...right);
  if (denominator <= Number.EPSILON) {
    return 0;
  }
  const cosine = left.reduce((sum, value, index) => sum + value * right[index], 0)
    / denominator;
  return Math.acos(Math.max(-1, Math.min(1, cosine)));
}

function confidencePercent(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

export function featureMatchAssessment(
  rawMatches: number,
  rawInliers: number,
  hasFeatures: boolean = true,
): FeatureMatchAssessment {
  const matches = Math.max(0, Math.round(rawMatches));
  const inliers = Math.max(0, Math.min(matches, Math.round(rawInliers)));
  if (!hasFeatures) {
    return {
      feature_match_score: 0,
      feature_matches: 0,
      feature_inliers: 0,
      feature_inlier_ratio: 0,
      feature_match_status: "no_features",
    };
  }
  const inlierRatio = matches > 0 ? inliers / matches : 0;
  // A high ratio from only two points is not evidence. Saturate support at twenty
  // verified correspondences, then combine support and purity symmetrically.
  const support = Math.min(1, inliers / 20);
  return {
    feature_match_score: confidencePercent(Math.sqrt(inlierRatio * support)),
    feature_matches: matches,
    feature_inliers: inliers,
    feature_inlier_ratio: inlierRatio,
    feature_match_status:
      inliers >= 8 && inlierRatio >= 0.25 ? "matched" : "weak",
  };
}

export function appearanceSsimAssessment(
  rawScore: number | null,
): AppearanceSsimAssessment {
  if (rawScore === null || !Number.isFinite(rawScore)) {
    return {
      appearance_ssim_score: null,
      appearance_ssim_status: "unavailable",
    };
  }
  const score = confidencePercent(rawScore);
  return {
    appearance_ssim_score: score,
    appearance_ssim_status: score >= 60 ? "matched" : "weak",
  };
}

/** Project one depth-backed proposal into nearby, spatially distinct camera poses. */
export function projectProposalIntoNearestKeyframes(
  row: MetadataRow,
  manifest: MapManifest,
  requestedCount: number = DEFAULT_NEIGHBOR_PROJECTION_COUNT,
): ProposalNeighborProjection[] {
  const geometry = proposalProjectionGeometry(row, manifest);
  const count = Math.max(
    1,
    Math.min(MAX_NEIGHBOR_PROJECTION_COUNT, Math.round(requestedCount)),
  );
  const candidates = manifest.keyframes
    .filter((keyframe) => keyframe.keyframeId !== row.videoKeyframeId)
    .flatMap((keyframe): ProposalProjectionCandidate[] => {
      const projection = projectProposalIntoKeyframe(geometry, keyframe);
      return projection
        ? [{ projection, originWorldWds: keyframeOriginWorldWds(keyframe) }]
        : [];
    })
    .sort(
      (a, b) =>
        a.projection.distance_from_source_m - b.projection.distance_from_source_m
        || Number(a.projection.keyframe_id) - Number(b.projection.keyframe_id),
    );

  const selected: ProposalProjectionCandidate[] = [];
  const deferred: ProposalProjectionCandidate[] = [];
  for (const candidate of candidates) {
    if (
      candidate.projection.distance_from_source_m >= MIN_NEIGHBOR_PROJECTION_BASELINE_M
      && selected.every(
        (other) => cameraBaselineM(candidate, other) >= MIN_NEIGHBOR_PROJECTION_BASELINE_M,
      )
    ) {
      selected.push(candidate);
    } else {
      deferred.push(candidate);
    }
    if (selected.length === count) {
      break;
    }
  }
  // Dense or very small captures may not offer N distinct origins. Preserve the
  // requested result count by falling back to the nearest deferred poses.
  if (selected.length < count) {
    selected.push(...deferred.slice(0, count - selected.length));
  }
  return selected.map((candidate) => candidate.projection);
}

async function proposalProjectionSource(
  map: MapEntry,
  rowIndex: number,
): Promise<{ row: MetadataRow; manifest: MapManifest }> {
  const metadata = await requireMetadata(map).catch(asRouteError);
  const row = rowByIndex(metadata, rowIndex);
  if (!row) {
    throw new WorkbenchRouteError(404, `Row ${rowIndex} not found in ${metadata.metadataPath}.`);
  }
  let manifest: MapManifest;
  try {
    manifest = await loadMapManifest(map.path);
  } catch (error) {
    throw new WorkbenchRouteError(404, error instanceof Error ? error.message : String(error));
  }
  return { row, manifest };
}

export async function proposalNeighborProjectionsPayload(
  map: MapEntry,
  rowIndex: number,
  requestedCount: number,
): Promise<Record<string, unknown>> {
  const { row, manifest } = await proposalProjectionSource(map, rowIndex);
  const baseProjections = projectProposalIntoNearestKeyframes(
    row,
    manifest,
    requestedCount,
  );
  const [occlusionScored, featureAssessments] = await Promise.all([
    scoreProjectionOcclusion(map, baseProjections),
    scoreProjectionFeatures(map, row, baseProjections),
  ]);
  const projections = occlusionScored.map((projection, index) => ({
    ...projection,
    ...(featureAssessments?.[index] ?? {}),
  }));
  return {
    row_index: row.rowIndex,
    source_keyframe_id: String(row.videoKeyframeId),
    requested_count: Math.max(
      1,
      Math.min(MAX_NEIGHBOR_PROJECTION_COUNT, Math.round(requestedCount)),
    ),
    projections,
    note:
      "Nearby views are spaced by at least 0.5 m when possible. Geometric "
      + "confidence combines distance, viewpoint angle and apparent size. Visibility "
      + "compares the projected distance with target depth. Feature match uses "
      + "COLMAP-style SIFT ratio matching with geometric verification. Appearance "
      + "uses translation-aligned SSIM on the projected proposal region.",
  };
}

/**
 * Re-render one proposal's rectilinear view from the keyframe's ERP.
 *
 * This is **not** the default preview. The 336px JPEG `prepare` already wrote is,
 * because it is the only image guaranteed to be what MetaCLIP2 actually embedded —
 * the panel serves it through `/preview.png?preview_path=<thumbnail_key>`, which
 * needs no code at all. This path exists for the two cases the thumbnail cannot
 * cover: a map prepared without `--store-crops` (empty `thumbnail_key`), and a
 * context view at widened FOV and full resolution.
 *
 * `fovScale = 1` reproduces the patch *before* `build_padding_mask`, which crops
 * wide boxes and pads tall ones. The rendered region is therefore the proposal's
 * true angular extent, not the masked square on disk — the UI must say which it is
 * showing.
 */
async function renderGnomonicPng(
  map: MapEntry,
  keyframeId: string,
  angles: ErpViewAngles,
  options: { size?: number; fovScale?: number },
): Promise<Buffer> {
  if (!isRenderableGnomonic(angles)) {
    throw new WorkbenchRouteError(
      422,
      "The requested angular extent cannot be rendered as a rectilinear view.",
    );
  }
  const sourceImagePath = await resolveSourceImagePath(map, keyframeId);
  if (!sourceImagePath) {
    throw new WorkbenchRouteError(
      404,
      `Equirectangular source image not found for keyframe ${keyframeId}.`,
    );
  }
  const source = await imageSize(sourceImagePath);
  try {
    assertEquirect2to1(source.width, source.height);
  } catch (error) {
    throw new WorkbenchRouteError(422, error instanceof Error ? error.message : String(error));
  }
  const size = Math.max(
    16,
    Math.min(MAX_ROW_RENDER_SIZE, Math.round(options.size ?? DEFAULT_ROW_RENDER_SIZE)),
  );
  const fovScale = Math.max(1, Math.min(8, options.fovScale ?? 1));
  const filter = gnomonicFfmpegFilter(angles, { size, fovScale });

  const sourceMtime = (await stat(sourceImagePath)).mtimeMs;
  const cacheKey = `${sourceImagePath}:${sourceMtime}:${filter}`;
  const cached = cutoutPreviewCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const promise = execFileAsync(
    ffmpegBinary(),
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      sourceImagePath,
      "-vf",
      filter,
      "-frames:v",
      "1",
      "-f",
      "image2pipe",
      "-vcodec",
      "png",
      "-",
    ],
    { encoding: "buffer", maxBuffer: 64 * 1024 * 1024 },
  )
    .then(({ stdout }) => Buffer.from(stdout))
    .catch((error: unknown) => {
      cutoutPreviewCache.delete(cacheKey);
      throw error;
    });
  cutoutPreviewCache.set(cacheKey, promise);
  if (cutoutPreviewCache.size > 256) {
    cutoutPreviewCache.delete(cutoutPreviewCache.keys().next().value as string);
  }
  return promise;
}

async function rowRenderPng(
  map: MapEntry,
  row: MetadataRow,
  options: { size?: number; fovScale?: number },
): Promise<Buffer> {
  if (!isRenderableGnomonic(rowAngles(row))) {
    throw new WorkbenchRouteError(
      422,
      `Row ${row.rowIndex} spans ${(row.angularWidth * 180 / Math.PI).toFixed(1)}x`
      + `${(row.angularHeight * 180 / Math.PI).toFixed(1)} degrees, so no rectilinear view `
      + "of it exists (a gnomonic projection diverges at 180). The stored thumbnail is "
      + "degenerate for the same reason — this is an upstream property of the proposal, "
      + "not a rendering limit.",
    );
  }
  return renderGnomonicPng(
    map,
    String(row.videoKeyframeId),
    rowAngles(row),
    options,
  );
}

/**
 * ImageMagick arguments outlining one row's reconstructed ERP box.
 *
 * Wrapped rectangles are drawn as one or two axis-aligned boxes rather than the
 * cubemap era's projected polyline: an ERP box *is* axis-aligned in v2, so there is
 * no quadrilateral to trace.
 */
function drawArgsForRow(
  row: MetadataRow,
  target: { width: number; height: number; scale: number; selected: boolean },
): string[] {
  const color = target.selected ? BOX_COLOR_SELECTED : BOX_COLOR;
  const strokeWidth = target.selected ? "4" : "2";
  const args: string[] = [];
  for (const rect of erpRectsWrapped(rowAngles(row))) {
    const x0 = rect.u0 * (target.width - 1) * target.scale;
    const x1 = rect.u1 * (target.width - 1) * target.scale;
    const y0 = rect.v0 * (target.height - 1) * target.scale;
    const y1 = rect.v1 * (target.height - 1) * target.scale;
    args.push(
      "-fill", "none",
      "-stroke", color,
      "-strokewidth", strokeWidth,
      "-draw", `rectangle ${x0},${y0} ${x1},${y1}`,
    );
    const label = row.label || row.detectorSource;
    if (label) {
      const safeLabel = label.replaceAll("'", "\\'");
      args.push(
        "-fill", color,
        "-stroke", "none",
        "-pointsize", "12",
        "-draw", `text ${x0 + 2},${Math.max(12, y0 - 3)} '${safeLabel}'`,
      );
    }
  }
  return args;
}

async function convertPng(input: Buffer | string, args: string[]): Promise<Buffer> {
  const inputArg = Buffer.isBuffer(input) ? "png:-" : input;
  const child = spawn(convertBinary(), [inputArg, ...args, "png:-"], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  if (Buffer.isBuffer(input)) {
    child.stdin.end(input);
  } else {
    child.stdin.end();
  }
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new WorkbenchRouteError(
      500,
      `Image conversion failed: ${Buffer.concat(stderr).toString("utf8").trim()}`,
    );
  }
  return Buffer.concat(stdout);
}

async function convertPpmToPng(input: Buffer, args: string[] = []): Promise<Buffer> {
  const child = spawn(convertBinary(), ["ppm:-", ...args, "png:-"], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(input);
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new WorkbenchRouteError(
      500,
      `Depth preview conversion failed: ${Buffer.concat(stderr).toString("utf8").trim()}`,
    );
  }
  return Buffer.concat(stdout);
}

async function convertJpeg(input: Buffer | string, args: string[]): Promise<Buffer> {
  const inputArg = Buffer.isBuffer(input) ? "png:-" : input;
  const child = spawn(convertBinary(), [inputArg, ...args, "-quality", "82", "jpg:-"], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  if (Buffer.isBuffer(input)) {
    child.stdin.end(input);
  } else {
    child.stdin.end();
  }
  const code = await new Promise<number | null>((resolve) => child.on("close", resolve));
  if (code !== 0) {
    throw new WorkbenchRouteError(500, `convert failed: ${Buffer.concat(stderr).toString("utf8").trim()}`);
  }
  return Buffer.concat(stdout);
}

export async function metadataRowRenderPng(
  map: MapEntry,
  rowIndex: number,
  options: { size?: number; fovScale?: number },
): Promise<Buffer> {
  const metadata = await requireMetadata(map).catch(asRouteError);
  const row = rowByIndex(metadata, rowIndex);
  if (!row) {
    throw new WorkbenchRouteError(404, `Row ${rowIndex} not found in ${metadata.metadataPath}.`);
  }
  return rowRenderPng(map, row, options);
}

export async function proposalNeighborProjectionRenderPng(
  map: MapEntry,
  rowIndex: number,
  targetKeyframeId: string,
  options: { size?: number; fovScale?: number },
): Promise<Buffer> {
  const { row, manifest } = await proposalProjectionSource(map, rowIndex);
  const geometry = proposalProjectionGeometry(row, manifest);
  const target = manifest.keyframeById.get(Number(targetKeyframeId));
  if (!target) {
    throw new WorkbenchRouteError(
      404,
      `GeoRef pose not found for target keyframe ${targetKeyframeId}.`,
    );
  }
  if (target.keyframeId === row.videoKeyframeId) {
    throw new WorkbenchRouteError(400, "The source keyframe is not a neighbour.");
  }
  const projection = projectProposalIntoKeyframe(geometry, target);
  if (!projection) {
    throw new WorkbenchRouteError(
      422,
      `The proposal coincides with target keyframe ${targetKeyframeId}.`,
    );
  }
  return renderGnomonicPng(
    map,
    targetKeyframeId,
    {
      thetaCenter: projection.theta_center,
      phiCenter: projection.phi_center,
      angularWidth: projection.angular_width,
      angularHeight: projection.angular_height,
    },
    options,
  );
}

/**
 * Virtual `thumbnail_key` for an index with no thumbnail JPEGs on disk.
 *
 * An index converted from the v1 SQLite lineage has no crops: v1 rendered its previews
 * from the ERP on request and stored none. Writing a million of them is possible but
 * costly (12.6 GB of JPEGs, ~139 GB once exFAT's 128 KB clusters are counted), and the
 * renderer already exists for the explorer — so `v1_index_convert` writes this path
 * instead of a file, and the preview route renders it on demand.
 *
 * It is deliberately the *same* field the UI already reads, so nothing downstream
 * changes: pgvector carries the string, the response carries it, and the panel asks for
 * it through `/preview.png?preview_path=…` exactly as it does for a real thumbnail.
 */
const VIRTUAL_ROW_PREVIEW = /^object-search\/rows\/(\d+)\.(?:png|jpg)$/;

export async function previewFromPathPng(map: MapEntry, previewPath: string | null): Promise<Buffer> {
  if (!previewPath) {
    throw new WorkbenchRouteError(400, "Missing preview_path");
  }
  const virtual = VIRTUAL_ROW_PREVIEW.exec(previewPath);
  if (virtual) {
    return metadataRowRenderPng(map, Number(virtual[1]), {});
  }
  const resolved = path.isAbsolute(previewPath) ? previewPath : path.resolve(map.path, previewPath);
  if (!(await pathExists(resolved))) {
    throw new WorkbenchRouteError(404, "Preview file not found");
  }
  return convertPng(resolved, []);
}

type DepthSample = {
  depth_m: number;
  row: number;
  col: number;
  width: number;
  height: number;
  sample_count: number;
};

type DepthBatchSample = {
  sample: DepthSample | null;
  error: string | null;
};

type FeatureMatchAssessment = Pick<
  ProposalNeighborProjection,
  | "feature_match_score"
  | "feature_matches"
  | "feature_inliers"
  | "feature_inlier_ratio"
  | "feature_match_status"
>;

type AppearanceSsimAssessment = Pick<
  ProposalNeighborProjection,
  "appearance_ssim_score" | "appearance_ssim_status"
>;

type ProjectionVisualAssessment = FeatureMatchAssessment & AppearanceSsimAssessment;

type RawFeatureMatch = {
  matches: number;
  inliers: number;
  has_features: boolean;
  ssim: number | null;
};

const FEATURE_MATCH_SCRIPT = String.raw`
import base64
import json
import sys

import cv2
import numpy as np


def decode_gray(encoded):
    payload = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("could not decode projection PNG")
    return image


def extract_sift(image):
    # COLMAP defaults to SIFT. A slightly lower contrast threshold keeps useful
    # points in these small, already-localised proposal views.
    sift = cv2.SIFT_create(nfeatures=2048, contrastThreshold=0.02)
    return sift.detectAndCompute(image, None)


def proposal_region(image):
    # The proposal occupies half of a 2x-FOV render. Keep a small tolerance for
    # viewpoint changes while excluding most scene context from the score.
    height, width = image.shape[:2]
    half_height = max(1, int(height * 0.30))
    half_width = max(1, int(width * 0.30))
    center_y = height // 2
    center_x = width // 2
    return image[
        center_y - half_height:center_y + half_height,
        center_x - half_width:center_x + half_width,
    ]


def appearance_region(image):
    # The exact proposal spans half of a 2x-FOV render. SSIM uses that region,
    # while SIFT keeps the slightly wider crop above for geometric tolerance.
    height, width = image.shape[:2]
    quarter_height = max(1, height // 4)
    quarter_width = max(1, width // 4)
    center_y = height // 2
    center_x = width // 2
    return image[
        center_y - quarter_height:center_y + quarter_height,
        center_x - quarter_width:center_x + quarter_width,
    ]


def ssim(left, right):
    left_float = left.astype(np.float32)
    right_float = right.astype(np.float32)
    mu_left = cv2.GaussianBlur(left_float, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right_float, (11, 11), 1.5)
    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_cross = mu_left * mu_right
    sigma_left = cv2.GaussianBlur(left_float * left_float, (11, 11), 1.5) - mu_left_sq
    sigma_right = (
        cv2.GaussianBlur(right_float * right_float, (11, 11), 1.5)
        - mu_right_sq
    )
    sigma_cross = (
        cv2.GaussianBlur(left_float * right_float, (11, 11), 1.5)
        - mu_cross
    )
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (
        sigma_left + sigma_right + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-6)))


def aligned_ssim(source, target):
    # Projection already normalises scale and centre. A small translation search
    # absorbs residual pose/depth error without letting unrelated context win.
    size = 160
    source_small = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA)
    max_shift = 8
    best = -1.0
    for dy in (-max_shift, -max_shift // 2, 0, max_shift // 2, max_shift):
        for dx in (-max_shift, -max_shift // 2, 0, max_shift // 2, max_shift):
            source_y = slice(max(0, dy), min(size, size + dy))
            target_y = slice(max(0, -dy), min(size, size - dy))
            source_x = slice(max(0, dx), min(size, size + dx))
            target_x = slice(max(0, -dx), min(size, size - dx))
            best = max(
                best,
                ssim(
                    source_small[source_y, source_x],
                    target_small[target_y, target_x],
                ),
            )
    return max(0.0, min(1.0, best))


def match_target(
    source_keypoints,
    source_descriptors,
    source_appearance,
    target_encoded,
):
    target_full = decode_gray(target_encoded)
    target = proposal_region(target_full)
    appearance = aligned_ssim(source_appearance, appearance_region(target_full))
    target_keypoints, target_descriptors = extract_sift(target)
    if (
        source_descriptors is None
        or target_descriptors is None
        or len(source_descriptors) < 2
        or len(target_descriptors) < 2
    ):
        return {
            "matches": 0,
            "inliers": 0,
            "has_features": False,
            "ssim": appearance,
        }

    # COLMAP's SIFT_BRUTEFORCE matcher uses L2 descriptors and a 0.8 ratio test.
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        source_descriptors,
        target_descriptors,
        k=2,
    )
    matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance
    ]
    inliers = 0
    if len(matches) >= 4:
        source_points = np.float32(
            [source_keypoints[item.queryIdx].pt for item in matches]
        ).reshape(-1, 1, 2)
        target_points = np.float32(
            [target_keypoints[item.trainIdx].pt for item in matches]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(
            source_points,
            target_points,
            cv2.RANSAC,
            4.0,
            maxIters=2000,
            confidence=0.995,
        )
        if mask is not None:
            inliers = int(mask.ravel().sum())

    return {
        "matches": len(matches),
        "inliers": inliers,
        "has_features": True,
        "ssim": appearance,
    }


def main():
    request = json.loads(sys.stdin.read())
    source_full = decode_gray(request["source_png"])
    source = proposal_region(source_full)
    source_appearance = appearance_region(source_full)
    source_keypoints, source_descriptors = extract_sift(source)
    results = [
        match_target(
            source_keypoints,
            source_descriptors,
            source_appearance,
            encoded,
        )
        for encoded in request["target_pngs"]
    ]
    print(json.dumps(results))


if __name__ == "__main__":
    main()
`;

const DEPTH_SAMPLE_SCRIPT = String.raw`
import json
import math
import sys
from pathlib import Path

import numpy as np


_DMIN_M = 0.1
_DMAX_M = 200.0
_EFFECTIVE_BITS = 10
_LEVELS = (1 << _EFFECTIVE_BITS) - 1
_SHIFT = 16 - _EFFECTIVE_BITS
_SQRT_LO = math.sqrt(_DMIN_M)
_SQRT_HI = math.sqrt(_DMAX_M)


def ratio_to_pixel(u, v, width, height):
    col = round(min(max(float(u), 0.0), 1.0) * (width - 1))
    row = round(min(max(float(v), 0.0), 1.0) * (height - 1))
    return int(row), int(col)


def decode_tiff_uint16_meters(encoded):
    values = np.asarray(encoded, dtype=np.uint16)
    valid = values != 0
    raw = values.astype(np.int32) >> _SHIFT
    normed = (raw - 1).astype(np.float32) / max(_LEVELS - 1, 1)
    depth = (_SQRT_LO + normed * (_SQRT_HI - _SQRT_LO)) ** 2
    return np.where(valid, depth.astype(np.float32), np.float32(np.nan))


def sample_window(row, col, width, height, radius, get_value):
    values = []
    for yy in range(max(0, row - radius), min(height, row + radius + 1)):
        for xx in range(max(0, col - radius), min(width, col + radius + 1)):
            depth_m = get_value(yy, xx)
            if math.isfinite(depth_m):
                values.append(depth_m)
    if len(values) < 3:
        raise RuntimeError("not enough valid depth samples around clicked pixel")
    return values



def sample_tiff(depth_path, u, v, radius):
    try:
        import tifffile
    except ModuleNotFoundError as exc:
        raise RuntimeError("tifffile is required to sample TIFF depth maps") from exc

    depth = np.asarray(tifffile.imread(str(depth_path)))
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"expected a 2D TIFF depth map, got shape {depth.shape}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"expected a uint16 TIFF depth map, got {depth.dtype}")

    height, width = [int(v) for v in depth.shape]
    row, col = ratio_to_pixel(u, v, width, height)

    def get_value(yy, xx):
        return float(decode_tiff_uint16_meters(depth[yy, xx]))

    values = sample_window(row, col, width, height, radius, get_value)
    return {
        "depth_m": float(np.median(np.asarray(values, dtype=np.float32))),
        "row": row,
        "col": col,
        "width": width,
        "height": height,
        "sample_count": len(values),
    }


def main():
    req = json.loads(sys.stdin.read())
    if "samples" in req:
        results = []
        for item in req["samples"]:
            try:
                result = sample_tiff(
                    Path(item["depth_path"]),
                    float(item["u"]),
                    float(item["v"]),
                    max(0, int(item.get("radius", 3))),
                )
                results.append({"sample": result, "error": None})
            except Exception as exc:
                results.append({"sample": None, "error": str(exc)})
        print(json.dumps(results))
        return
    depth_path = Path(req["depth_path"])
    u = float(req["u"])
    v = float(req["v"])
    radius = max(0, int(req.get("radius", 3)))
    if depth_path.suffix.lower() not in {".tif", ".tiff"}:
        raise RuntimeError(f"unsupported depth map format: {depth_path}")
    result = sample_tiff(depth_path, u, v, radius)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
`;

const DEPTH_PREVIEW_SCRIPT = String.raw`
import json
import math
import sys
from pathlib import Path

import numpy as np


_DMIN_M = 0.1
_DMAX_M = 200.0
_EFFECTIVE_BITS = 10
_LEVELS = (1 << _EFFECTIVE_BITS) - 1
_SHIFT = 16 - _EFFECTIVE_BITS
_SQRT_LO = math.sqrt(_DMIN_M)
_SQRT_HI = math.sqrt(_DMAX_M)


def decode_tiff_uint16_meters(encoded):
    values = np.asarray(encoded, dtype=np.uint16)
    valid = values != 0
    raw = values.astype(np.int32) >> _SHIFT
    normed = (raw - 1).astype(np.float32) / max(_LEVELS - 1, 1)
    depth = (_SQRT_LO + normed * (_SQRT_HI - _SQRT_LO)) ** 2
    return np.where(valid, depth.astype(np.float32), np.float32(np.nan))



def load_tiff_preview(depth_path, max_width):
    try:
        import tifffile
    except ModuleNotFoundError as exc:
        raise RuntimeError("tifffile is required to render TIFF depth maps") from exc
    depth = np.asarray(tifffile.imread(str(depth_path)))
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"expected a 2D TIFF depth map, got shape {depth.shape}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"expected a uint16 TIFF depth map, got {depth.dtype}")
    height, width = [int(v) for v in depth.shape]
    out_width = min(width, max(1, int(max_width)))
    out_height = max(1, round(height * out_width / width))
    rows = np.linspace(0, height - 1, out_height).round().astype(np.int64)
    cols = np.linspace(0, width - 1, out_width).round().astype(np.int64)
    return decode_tiff_uint16_meters(depth[np.ix_(rows, cols)])


def colorize_depth(depth):
    valid = np.isfinite(depth) & (depth > 0)
    rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb
    log_depth = np.log(np.clip(depth[valid], _DMIN_M, _DMAX_M))
    lo, hi = np.percentile(log_depth, [2, 98])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(log_depth))
        hi = float(np.max(log_depth))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((np.log(np.clip(depth[valid], _DMIN_M, _DMAX_M)) - lo) / (hi - lo), 0.0, 1.0)
    t = norm[valid]
    rgb[..., 0][valid] = ((1.0 - t) * 255).astype(np.uint8)
    rgb[..., 1][valid] = ((1.0 - np.abs(2.0 * t - 1.0)) * 190).astype(np.uint8)
    rgb[..., 2][valid] = (t * 255).astype(np.uint8)
    return rgb


def write_ppm(rgb):
    height, width = rgb.shape[:2]
    sys.stdout.buffer.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    sys.stdout.buffer.write(np.ascontiguousarray(rgb).tobytes())


def main():
    req = json.loads(sys.stdin.read())
    depth_path = Path(req["depth_path"])
    max_width = int(req.get("max_width", 2048))
    if depth_path.suffix.lower() not in {".tif", ".tiff"}:
        raise RuntimeError(f"unsupported depth map format: {depth_path}")
    depth = load_tiff_preview(depth_path, max_width)
    write_ppm(colorize_depth(depth))


if __name__ == "__main__":
    main()
`;

async function runDepthSamplePython(
  pythonBin: string,
  request: Record<string, unknown>,
): Promise<DepthSample> {
  const child = spawn(pythonBin, ["-c", DEPTH_SAMPLE_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(JSON.stringify(request));
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new Error(Buffer.concat(stderr).toString("utf8").trim() || `python exited with ${code}`);
  }
  return JSON.parse(Buffer.concat(stdout).toString("utf8")) as DepthSample;
}

async function runDepthBatchSamplePython(
  pythonBin: string,
  requests: Record<string, unknown>[],
): Promise<DepthBatchSample[]> {
  const child = spawn(pythonBin, ["-c", DEPTH_SAMPLE_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(JSON.stringify({ samples: requests }));
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new Error(Buffer.concat(stderr).toString("utf8").trim() || `python exited with ${code}`);
  }
  return JSON.parse(Buffer.concat(stdout).toString("utf8")) as DepthBatchSample[];
}

async function runFeatureMatchPython(
  pythonBin: string,
  sourcePng: Buffer,
  targetPngs: Buffer[],
): Promise<ProjectionVisualAssessment[]> {
  const child = spawn(pythonBin, ["-c", FEATURE_MATCH_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(JSON.stringify({
    source_png: sourcePng.toString("base64"),
    target_pngs: targetPngs.map((target) => target.toString("base64")),
  }));
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new Error(Buffer.concat(stderr).toString("utf8").trim() || `python exited with ${code}`);
  }
  const raw = JSON.parse(Buffer.concat(stdout).toString("utf8")) as RawFeatureMatch[];
  return raw.map((item) => ({
    ...featureMatchAssessment(item.matches, item.inliers, item.has_features),
    ...appearanceSsimAssessment(item.ssim),
  }));
}

async function runDepthPreviewPython(
  pythonBin: string,
  request: Record<string, unknown>,
): Promise<Buffer> {
  const child = spawn(pythonBin, ["-c", DEPTH_PREVIEW_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(JSON.stringify(request));
  const code = await new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  if (code !== 0) {
    throw new Error(Buffer.concat(stderr).toString("utf8").trim() || `python exited with ${code}`);
  }
  return Buffer.concat(stdout);
}

async function sampleDepthAtEquirectUv(
  map: MapEntry,
  keyframeId: string,
  u: number,
  v: number,
  radius: number,
): Promise<DepthSample> {
  const depthPath = await resolveDepthPath(map, keyframeId);
  if (!depthPath) {
    throw new WorkbenchRouteError(404, `Depth map not found for keyframe ${keyframeId}.`);
  }
  const errors: string[] = [];
  for (const pythonBin of pythonBinaryCandidates()) {
    try {
      return await runDepthSamplePython(pythonBin, {
        depth_path: depthPath,
        u: Math.max(0, Math.min(1, u)),
        v: Math.max(0, Math.min(1, v)),
        radius,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${pythonBin}: ${message}`);
    }
  }
  throw new WorkbenchRouteError(
    500,
    `Could not sample depth map. Install tifffile, or set OBJECT_SEARCH_WORKBENCH_PYTHON. ${errors.join(" | ")}`,
  );
}

export function projectionOcclusionAssessment(
  expectedDepthM: number,
  observedDepthM: number,
): Pick<
  ProposalNeighborProjection,
  "observed_depth_m" | "occlusion_margin_m" | "occlusion_confidence" | "occlusion_status"
> {
  // Depth is sqrt-quantised and a small proposal can straddle object/background
  // pixels, so require a discrepancy greater than 35 cm or 8% before classifying it.
  const toleranceM = Math.max(0.35, expectedDepthM * 0.08);
  const marginM = observedDepthM - expectedDepthM;
  const mismatchExcessM = Math.max(0, Math.abs(marginM) - toleranceM);
  const confidence = Math.exp(-mismatchExcessM / Math.max(0.5, toleranceM));
  return {
    observed_depth_m: observedDepthM,
    occlusion_margin_m: marginM,
    occlusion_confidence: confidencePercent(confidence),
    occlusion_status:
      mismatchExcessM === 0
        ? "clear"
        : marginM < 0
          ? "occluded"
          : "depth_mismatch",
  };
}

async function scoreProjectionOcclusion(
  map: MapEntry,
  projections: ProposalNeighborProjection[],
): Promise<ProposalNeighborProjection[]> {
  const depthPaths = await Promise.all(
    projections.map((projection) => resolveDepthPath(map, projection.keyframe_id)),
  );
  const available = projections.flatMap((projection, index) => {
    const depthPath = depthPaths[index];
    return depthPath
      ? [
          {
            index,
            request: {
              depth_path: depthPath,
              u: projection.erp_u,
              v: projection.erp_v,
              radius: 3,
            },
          },
        ]
      : [];
  });
  if (!available.length) {
    return projections;
  }

  let samples: DepthBatchSample[] | null = null;
  for (const pythonBin of pythonBinaryCandidates()) {
    try {
      samples = await runDepthBatchSamplePython(
        pythonBin,
        available.map((item) => item.request),
      );
      break;
    } catch {
      // Occlusion is an optional confidence signal. A missing decoder must not hide
      // otherwise valid geometric projections.
    }
  }
  if (!samples) {
    return projections;
  }

  const assessments = new Map<number, ReturnType<typeof projectionOcclusionAssessment>>();
  available.forEach((item, sampleIndex) => {
    const sample = samples?.[sampleIndex]?.sample;
    if (sample) {
      assessments.set(
        item.index,
        projectionOcclusionAssessment(
          projections[item.index].distance_to_proposal_m,
          sample.depth_m,
        ),
      );
    }
  });
  return projections.map((projection, index) => ({
    ...projection,
    ...(assessments.get(index) ?? {}),
  }));
}

async function scoreProjectionFeatures(
  map: MapEntry,
  row: MetadataRow,
  projections: ProposalNeighborProjection[],
): Promise<ProjectionVisualAssessment[] | null> {
  let sourcePng: Buffer;
  let targetPngs: Buffer[];
  try {
    [sourcePng, targetPngs] = await Promise.all([
      rowRenderPng(map, row, { size: 384, fovScale: 2 }),
      Promise.all(
        projections.map((projection) =>
          renderGnomonicPng(
            map,
            projection.keyframe_id,
            {
              thetaCenter: projection.theta_center,
              phiCenter: projection.phi_center,
              angularWidth: projection.angular_width,
              angularHeight: projection.angular_height,
            },
            { size: 384, fovScale: 2 },
          ),
        ),
      ),
    ]);
  } catch {
    return null;
  }

  for (const pythonBin of pythonBinaryCandidates()) {
    try {
      const assessments = await runFeatureMatchPython(
        pythonBin,
        sourcePng,
        targetPngs,
      );
      return assessments.length === projections.length ? assessments : null;
    } catch {
      // Feature scoring is additive. Missing OpenCV/SIFT must not make geometric
      // projection, depth visibility, or rendered cards disappear.
    }
  }
  return null;
}

export async function indexKeyframeDepthPreviewPng(
  map: MapEntry,
  keyframeId: string,
): Promise<Buffer> {
  const depthPath = await resolveDepthPath(map, keyframeId);
  if (!depthPath) {
    throw new WorkbenchRouteError(404, `Depth map not found for keyframe ${keyframeId}.`);
  }
  const sourceMtime = (await stat(depthPath)).mtimeMs;
  const cacheKey = `${depthPath}:${sourceMtime}:${MAX_DEPTH_PREVIEW_WIDTH}:depth-preview`;
  const cached = equirectPreviewCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const errors: string[] = [];
  const promise = (async () => {
    for (const pythonBin of pythonBinaryCandidates()) {
      try {
        const ppm = await runDepthPreviewPython(pythonBin, {
          depth_path: depthPath,
          max_width: MAX_DEPTH_PREVIEW_WIDTH,
        });
        return await convertPpmToPng(ppm);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`${pythonBin}: ${message}`);
      }
    }
    throw new WorkbenchRouteError(
      500,
      `Could not render depth map. Install tifffile, or set OBJECT_SEARCH_WORKBENCH_PYTHON. ${errors.join(" | ")}`,
    );
  })().catch((error) => {
    equirectPreviewCache.delete(cacheKey);
    throw error;
  });
  equirectPreviewCache.set(cacheKey, promise);
  if (equirectPreviewCache.size > 32) {
    equirectPreviewCache.delete(equirectPreviewCache.keys().next().value as string);
  }
  return promise;
}

/**
 * The depth map on disk for a keyframe: the manifest's `depth_url` basename.
 *
 * As with images, there is no id-derived fallback — `{keyframeId}.tif` would match
 * an unrelated keyframe's depth map now that ids are array indices. The format is
 * the frozen sqrt-quantised uint16 TIFF; Zarr depths belonged to the standalone
 * lineage and are gone.
 */
async function resolveDepthPath(map: MapEntry, keyframeId: string): Promise<string | null> {
  const filename = await lookupKeyframeDepthFilename(map, keyframeId);
  if (!filename) {
    return null;
  }
  const candidate = path.join(map.path, "depths", filename);
  return (await pathExists(candidate)) ? candidate : null;
}

/**
 * Depth-sample a click and lift it to a map position.
 *
 * `projection: "erp"` — a click on the photosphere — needs only the manifest and the
 * depth TIFF, so it deliberately does **not** touch the metadata: on a map whose
 * parquet is missing this still answers. `projection: "cutout"` resolves the click
 * through the proposal's gnomonic geometry and therefore does need the row.
 */
export async function indexDepthPinPayload(
  map: MapEntry,
  keyframeId: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const projection = typeof body.projection === "string" ? body.projection : "erp";
  const xRatio = asNumber(body.x_ratio, Number.NaN);
  const yRatio = asNumber(body.y_ratio, Number.NaN);
  if (!Number.isFinite(xRatio) || !Number.isFinite(yRatio)) {
    throw new WorkbenchRouteError(400, "Missing x_ratio/y_ratio click coordinates.");
  }

  let u = Math.max(0, Math.min(1, xRatio));
  let v = Math.max(0, Math.min(1, yRatio));
  let rowIndex: number | null = null;
  if (projection === "cutout") {
    rowIndex = Number.isInteger(Number(body.row_index)) ? Number(body.row_index) : null;
    if (rowIndex === null) {
      throw new WorkbenchRouteError(400, "Missing row_index for cutout depth pin.");
    }
    const metadata = await requireMetadata(map).catch(asRouteError);
    const row = rowByIndex(metadata, rowIndex);
    if (!row) {
      throw new WorkbenchRouteError(404, `Row ${rowIndex} not found in ${metadata.metadataPath}.`);
    }
    if (row.videoKeyframeId !== Number(keyframeId)) {
      throw new WorkbenchRouteError(
        400,
        `Row ${rowIndex} belongs to keyframe ${row.videoKeyframeId}, not ${keyframeId}.`,
      );
    }
    [u, v] = cutoutRatioToErpUv(rowAngles(row), xRatio, yRatio);
  } else if (projection !== "erp") {
    throw new WorkbenchRouteError(400, `Unsupported depth pin projection: ${projection}`);
  }

  const pose = await lookupKeyframePose(map, keyframeId);
  if (!pose) {
    throw new WorkbenchRouteError(404, `GeoRef pose not found for keyframe ${keyframeId}.`);
  }
  const radius = Math.max(0, Math.min(12, Math.round(asNumber(body.sample_radius, 3))));
  const sample = await sampleDepthAtEquirectUv(map, keyframeId, u, v, radius);
  const minDepthM = asNumber(body.min_depth_m, 0.25);
  const maxDepthM = asNumber(body.max_depth_m, 50);
  if (sample.depth_m < minDepthM || sample.depth_m > maxDepthM) {
    throw new WorkbenchRouteError(
      422,
      `Depth ${sample.depth_m.toFixed(2)}m is outside the accepted range ${minDepthM}-${maxDepthM}m.`,
    );
  }

  const ray = equirectUvToRay(u, v);
  const pointKeyframe: [number, number, number] = [
    ray[0] * sample.depth_m,
    ray[1] * sample.depth_m,
    ray[2] * sample.depth_m,
  ];
  const keyframeToWorld = invertRigid4(pose.pose);
  const pointWorldWds = transformPoint4(keyframeToWorld, pointKeyframe);
  const pointLocalEnu = wdsToEnu(pointWorldWds);
  const geo = localWorldPointToGeo(pose.context, pointWorldWds);

  return {
    available: true,
    keyframe_id: keyframeId,
    row_index: rowIndex,
    projection,
    erp_u: u,
    erp_v: v,
    erp_col: sample.col,
    erp_row: sample.row,
    erp_width: sample.width,
    erp_height: sample.height,
    depth_m: sample.depth_m,
    sample_count: sample.sample_count,
    point_keyframe: pointKeyframe,
    point_local: pointLocalEnu,
    point_world: [geo.latitude, geo.longitude, geo.altitude],
    point_world_wds: pointWorldWds,
    latitude: geo.latitude,
    longitude: geo.longitude,
    altitude: geo.altitude,
    level: geo.level,
  };
}

export async function indexViewConePayload(
  map: MapEntry,
  keyframeId: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const pose = await lookupKeyframePose(map, keyframeId);
  if (!pose) {
    throw new WorkbenchRouteError(404, `GeoRef pose not found for keyframe ${keyframeId}.`);
  }

  const yawRad = asNumber(body.yaw_rad, Number.NaN);
  if (!Number.isFinite(yawRad)) {
    throw new WorkbenchRouteError(400, "Missing yaw_rad.");
  }
  const pitchRad = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, asNumber(body.pitch_rad, 0)));
  const distanceM = Math.max(0.5, Math.min(50, asNumber(body.distance_m, 8)));
  const halfAngleRad = Math.max(1, Math.min(90, asNumber(body.half_angle_deg, 25))) * Math.PI / 180;
  const keyframeToWorld = invertRigid4(pose.pose);
  const originWorldWds = transformPoint4(keyframeToWorld, [0, 0, 0]);
  const originGeo = localWorldPointToGeo(pose.context, originWorldWds);

  const arc = Array.from({ length: VIEW_CONE_ARC_STEPS + 1 }, (_, index) => {
    const ratio = index / VIEW_CONE_ARC_STEPS;
    const rayYaw = yawRad - halfAngleRad + ratio * halfAngleRad * 2;
    const [u, v] = yawPitchToEquirectUv(rayYaw, pitchRad);
    const ray = equirectUvToRay(u, v);
    const pointWorldWds = transformPoint4(keyframeToWorld, [
      ray[0] * distanceM,
      ray[1] * distanceM,
      ray[2] * distanceM,
    ]);
    const geo = localWorldPointToGeo(pose.context, pointWorldWds);
    return [geo.longitude, geo.latitude] as [number, number];
  });

  const centerUv = yawPitchToEquirectUv(yawRad, pitchRad);
  const centerRay = equirectUvToRay(centerUv[0], centerUv[1]);
  const centerWorldWds = transformPoint4(keyframeToWorld, [
    centerRay[0] * distanceM,
    centerRay[1] * distanceM,
    centerRay[2] * distanceM,
  ]);
  const centerGeo = localWorldPointToGeo(pose.context, centerWorldWds);

  return {
    available: true,
    keyframe_id: keyframeId,
    yaw_rad: yawRad,
    pitch_rad: pitchRad,
    center: {
      latitude: centerGeo.latitude,
      longitude: centerGeo.longitude,
      altitude: centerGeo.altitude,
      level: centerGeo.level,
    },
    polygon: [
      [originGeo.longitude, originGeo.latitude],
      ...arc,
      [originGeo.longitude, originGeo.latitude],
    ],
    level: originGeo.level,
  };
}

export async function indexProjectWorldPointPayload(
  map: MapEntry,
  keyframeId: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const rawPoint = body.point_world_wds;
  if (
    !Array.isArray(rawPoint) ||
    rawPoint.length !== 3 ||
    rawPoint.some((value) => !Number.isFinite(Number(value)))
  ) {
    throw new WorkbenchRouteError(400, "point_world_wds must contain three finite coordinates.");
  }
  const pointWorldWds = rawPoint.map(Number) as [number, number, number];
  const pose = await lookupKeyframePose(map, keyframeId);
  if (!pose) {
    throw new WorkbenchRouteError(404, `GeoRef pose not found for keyframe ${keyframeId}.`);
  }

  const pointKeyframe = transformPoint4(pose.pose, pointWorldWds);
  const uv = cameraPointToEquirectUv(pointKeyframe);
  if (!uv) {
    throw new WorkbenchRouteError(422, "The selected point coincides with the destination keyframe origin.");
  }

  return {
    available: true,
    keyframe_id: keyframeId,
    erp_u: uv[0],
    erp_v: uv[1],
    distance_m: Math.hypot(pointKeyframe[0], pointKeyframe[1], pointKeyframe[2]),
    point_keyframe: pointKeyframe,
    point_world_wds: pointWorldWds,
  };
}

/**
 * The keyframe's ERP image, optionally with reconstructed proposal boxes drawn on.
 *
 * The bare photosphere — the `drawBoxes === false` fast path below — needs nothing
 * but the manifest and the file on disk. It used to sit behind `requireIndex` and a
 * `geometry === "cubemap"` check, so it answered 501 on every v2 map while both
 * mounted panels swallowed the error and showed no panorama at all. Those two guards
 * were the single highest-leverage line of this migration; do not reintroduce them.
 */
export async function indexKeyframeEquirectPreviewPng(
  map: MapEntry,
  keyframeId: string,
  options: {
    selectedRowIndex: number | null;
    drawBoxes: boolean;
  } & RowFilterParams,
): Promise<Buffer> {
  const sourceImagePath = await resolveSourceImagePath(map, keyframeId);
  if (!sourceImagePath) {
    throw new WorkbenchRouteError(404, `Equirectangular source image not found for keyframe ${keyframeId}.`);
  }
  const size = await imageSize(sourceImagePath);
  const scale = size.width > MAX_PREVIEW_WIDTH ? MAX_PREVIEW_WIDTH / size.width : 1;
  if (!options.drawBoxes) {
    const sourceMtime = (await stat(sourceImagePath)).mtimeMs;
    const cacheKey = `${sourceImagePath}:${sourceMtime}:${MAX_PREVIEW_WIDTH}`;
    const cached = equirectPreviewCache.get(cacheKey);
    if (cached) {
      return cached;
    }
    const extension = path.extname(sourceImagePath).toLowerCase();
    const promise = (
      scale === 1 && (extension === ".jpg" || extension === ".jpeg")
        ? readFile(sourceImagePath)
        : convertJpeg(
            sourceImagePath,
            scale !== 1 ? ["-resize", `${MAX_PREVIEW_WIDTH}x`] : [],
          )
    ).catch((error) => {
      equirectPreviewCache.delete(cacheKey);
      throw error;
    });
    equirectPreviewCache.set(cacheKey, promise);
    if (equirectPreviewCache.size > 32) {
      equirectPreviewCache.delete(equirectPreviewCache.keys().next().value as string);
    }
    return promise;
  }
  const metadata = await requireMetadata(map).catch(asRouteError);
  const rows = filterRows(rowsForKeyframe(metadata, Number(keyframeId)), options);
  const drawArgs: string[] = [];
  if (scale !== 1) {
    drawArgs.push("-resize", `${MAX_PREVIEW_WIDTH}x`);
  }
  for (const row of rows) {
    drawArgs.push(
      ...drawArgsForRow(row, {
        width: size.width,
        height: size.height,
        scale,
        selected: row.rowIndex === options.selectedRowIndex,
      }),
    );
  }
  return convertPng(sourceImagePath, drawArgs);
}
