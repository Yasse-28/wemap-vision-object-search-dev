import { execFile, spawn } from "node:child_process";
import { access, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import type { MapEntry } from "./config.js";

const execFileAsync = promisify(execFile);

const OBJECT_SEARCH_DB = "object-search.db";
const CUBEMAP_FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"] as const;
const OCR_SOURCE_LABELS: Record<number, string> = {
  0: "none",
  1: "lightweight PPOCR",
  2: "PaddleOCR-VL",
};
const BOX_COLOR = "#11b5ae";
const BOX_COLOR_SELECTED = "#ff6b6b";
const MAX_PREVIEW_WIDTH = 8192;
const MAX_DEPTH_PREVIEW_WIDTH = 2048;
const OBJECT_ROW_QUERY_LIMIT = 10000;
const LEGACY_BBOX_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"] as const;
const VIEW_CONE_ARC_STEPS = 48;

type Bbox = [number, number, number, number];

type ObjectRecord = {
  object_idx: number;
  object_id: string;
  keyframe_id: string;
  cutout_id: string;
  bbox: Bbox;
  label: string;
  detection_source: string;
  textness_score: number | null;
  ocr_text: string;
  ocr_tokens: string;
  ocr_key: string;
  ocr_source: number;
  ocr_candidate: boolean;
  ocr_assigned: boolean;
  cluster_id: number | null;
};

type CutoutRecord = {
  cutout_id: string;
  keyframe_id: string;
  center_x: number | null;
  center_y: number | null;
};

type ClusterRecord = {
  cluster_id: number;
  ocr_text: string;
  ocr_tokens: string;
  ocr_key: string;
  ocr_observation_count: number;
  ocr_source: number;
};

type LoadedIndex = {
  indexPath: string;
  mtimeMs: number;
  manifest: Record<string, unknown>;
  cutouts: CutoutRecord[];
  cutoutsById: Map<string, CutoutRecord>;
  cutoutIdsByKeyframe: Map<string, string[]>;
  objects: ObjectRecord[];
  objectsByCutout: Map<string, ObjectRecord[]>;
  objectsByKeyframe: Map<string, ObjectRecord[]>;
  clusters: ClusterRecord[];
};

export class WorkbenchRouteError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function sqliteBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_SQLITE ?? "sqlite3";
}

function ffmpegBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_FFMPEG ?? "ffmpeg";
}

function convertBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_CONVERT ?? "convert";
}

function identifyBinary(): string {
  return process.env.OBJECT_SEARCH_WORKBENCH_IDENTIFY ?? "identify";
}

function pythonBinaryCandidates(): string[] {
  return [
    process.env.OBJECT_SEARCH_WORKBENCH_PYTHON,
    process.env.PYTHON,
    "/home/yacine/anaconda3/envs/wemap-vision/bin/python",
    "python3",
    "python",
  ].filter((item): item is string => Boolean(item));
}

function sqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function sqlIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

function asNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function asNullableNumber(value: unknown): number | null {
  if (value == null) {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function asString(value: unknown): string {
  return value == null ? "" : String(value);
}

function decodeFloat32HexTuple(hex: unknown, expectedLength: number): number[] | null {
  if (typeof hex !== "string" || hex.length < expectedLength * 8) {
    return null;
  }
  const buffer = Buffer.from(hex, "hex");
  if (buffer.byteLength < expectedLength * 4) {
    return null;
  }
  return Array.from({ length: expectedLength }, (_, idx) => buffer.readFloatLE(idx * 4));
}

function bboxFromObjectRow(row: Record<string, unknown>): Bbox {
  const decoded = decodeFloat32HexTuple(row.bbox_coordinates_hex, 4);
  if (decoded) {
    return [decoded[0], decoded[1], decoded[2], decoded[3]];
  }
  return [
    asNumber(row.bbox_x1),
    asNumber(row.bbox_y1),
    asNumber(row.bbox_x2),
    asNumber(row.bbox_y2),
  ];
}

function formatOcrSource(source: number): string {
  return `${source} - ${OCR_SOURCE_LABELS[source] ?? "unknown"}`;
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function sqliteJson<T = Record<string, unknown>>(
  dbPath: string,
  sql: string,
): Promise<T[]> {
  try {
    const { stdout } = await execFileAsync(
      sqliteBinary(),
      ["-readonly", "-json", dbPath, sql],
      { maxBuffer: 128 * 1024 * 1024 },
    );
    const text = stdout.trim();
    return text ? JSON.parse(text) as T[] : [];
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new WorkbenchRouteError(500, `SQLite query failed: ${detail}`);
  }
}

async function maybeSqliteJson<T = Record<string, unknown>>(
  dbPath: string,
  sql: string,
): Promise<T[]> {
  try {
    return await sqliteJson<T>(dbPath, sql);
  } catch {
    return [];
  }
}

async function sqliteObjectRowsPaged(
  dbPath: string,
  selectColumns: string,
  whereSql = "1 = 1",
): Promise<Array<Record<string, unknown>>> {
  const rows: Array<Record<string, unknown>> = [];
  let lastObjectIdx = -1;
  for (;;) {
    const page = await sqliteJson<Record<string, unknown>>(
      dbPath,
      `SELECT ${selectColumns}
       FROM ${sqlIdentifier("object")}
       WHERE (${whereSql}) AND object_idx > ${lastObjectIdx}
       ORDER BY object_idx
       LIMIT ${OBJECT_ROW_QUERY_LIMIT}`,
    );
    if (!page.length) {
      break;
    }
    rows.push(...page);
    lastObjectIdx = asNumber(page[page.length - 1].object_idx, lastObjectIdx);
    if (page.length < OBJECT_ROW_QUERY_LIMIT) {
      break;
    }
  }
  return rows;
}

async function tableExists(dbPath: string, tableName: string): Promise<boolean> {
  const rows = await maybeSqliteJson<{ name: string }>(
    dbPath,
    `SELECT name FROM sqlite_master WHERE type='table' AND name=${sqlString(tableName)} LIMIT 1`,
  );
  return rows.length > 0;
}

async function sqliteTableColumns(dbPath: string, tableName: string): Promise<Set<string> | null> {
  if (!(await pathExists(dbPath)) || !(await tableExists(dbPath, tableName))) {
    return null;
  }
  const rows = await sqliteJson<Record<string, unknown>>(
    dbPath,
    `PRAGMA table_info(${sqlIdentifier(tableName)})`,
  );
  return new Set(rows.map((row) => String(row.name)).filter(Boolean));
}

function buildObjectBboxSelectClause(objectColumns: Set<string>): string {
  if (objectColumns.has("bbox_coordinates")) {
    return "hex(bbox_coordinates) AS bbox_coordinates_hex";
  }
  if (LEGACY_BBOX_COLUMNS.every((name) => objectColumns.has(name))) {
    return LEGACY_BBOX_COLUMNS.join(", ");
  }
  const found = [...objectColumns].sort().join(", ");
  throw new WorkbenchRouteError(
    500,
    `Object-search index object table has no supported bbox columns.${found ? ` Found: ${found}` : ""}`,
  );
}

/**
 * Locate a map's LEGACY SQLite index.
 *
 * `object-search.db` was produced by the standalone `build_index.py`, which was
 * retired with the rest of that lineage (ADR 0002) — the index now lives in
 * pgvector. Nothing writes this file any more, so the `/object-search-index/*`
 * routes that depend on it only work for maps built before the migration.
 *
 * The `georef.db`-backed routes in this module (keyframe graph, depth pin, view
 * cone, world-point projection) are unaffected: `georef.db` is a map-directory
 * artifact, not a pipeline output.
 */
export async function resolveIndexPath(map: MapEntry): Promise<{
  indexPath: string | null;
  checkedPaths: string[];
}> {
  const checkedPaths: string[] = [];
  const candidates = [
    map.object_search_index_path,
    path.join(map.path, "object-search", OBJECT_SEARCH_DB),
    path.join(map.path, OBJECT_SEARCH_DB),
  ].filter((item): item is string => Boolean(item));

  for (const candidate of candidates) {
    const resolved = path.resolve(candidate);
    checkedPaths.push(resolved);
    if (await pathExists(resolved)) {
      return { indexPath: resolved, checkedPaths };
    }
  }
  return { indexPath: null, checkedPaths };
}

const indexCache = new Map<string, Promise<LoadedIndex>>();
const cutoutPreviewCache = new Map<string, Promise<Buffer>>();
const equirectPreviewCache = new Map<string, Promise<Buffer>>();

async function loadIndex(indexPath: string): Promise<LoadedIndex> {
  const resolved = path.resolve(indexPath);
  const mtimeMs = (await stat(resolved)).mtimeMs;
  const cacheKey = `${resolved}:${mtimeMs}`;
  const cached = indexCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const promise = loadIndexUncached(resolved, mtimeMs);
  indexCache.set(cacheKey, promise);
  if (indexCache.size > 8) {
    indexCache.delete(indexCache.keys().next().value as string);
  }
  return promise;
}

async function loadIndexUncached(indexPath: string, mtimeMs: number): Promise<LoadedIndex> {
  const metadataRows = await sqliteJson<{ value: string }>(
    indexPath,
    "SELECT CAST(value AS TEXT) AS value FROM params WHERE key IN ('index_metadata_json', 'manifest_json') ORDER BY key LIMIT 1",
  );
  const manifest = metadataRows[0]?.value ? JSON.parse(metadataRows[0].value) as Record<string, unknown> : {};

  const cutoutRows = await sqliteJson<Record<string, unknown>>(
    indexPath,
    `SELECT cutout_id, keyframe_id, center_x, center_y FROM cutout ORDER BY cutout_id`,
  );
  const cutouts: CutoutRecord[] = cutoutRows.map((row) => ({
    cutout_id: String(row.cutout_id),
    keyframe_id: String(row.keyframe_id),
    center_x: asNullableNumber(row.center_x),
    center_y: asNullableNumber(row.center_y),
  }));

  const clusterRows = await maybeSqliteJson<Record<string, unknown>>(
    indexPath,
    `SELECT cluster_id, ocr_text, ocr_tokens, ocr_key, ocr_observation_count, ocr_source
     FROM cluster
     ORDER BY cluster_id`,
  );
  const clusters: ClusterRecord[] = clusterRows.map((row) => ({
    cluster_id: asNumber(row.cluster_id),
    ocr_text: asString(row.ocr_text),
    ocr_tokens: asString(row.ocr_tokens),
    ocr_key: asString(row.ocr_key),
    ocr_observation_count: asNumber(row.ocr_observation_count),
    ocr_source: asNumber(row.ocr_source),
  }));

  const cutoutsById = new Map<string, CutoutRecord>();
  const cutoutIdsByKeyframe = new Map<string, string[]>();
  for (const cutout of cutouts) {
    cutoutsById.set(cutout.cutout_id, cutout);
    const group = cutoutIdsByKeyframe.get(cutout.keyframe_id) ?? [];
    group.push(cutout.cutout_id);
    cutoutIdsByKeyframe.set(cutout.keyframe_id, group);
  }
  for (const values of cutoutIdsByKeyframe.values()) {
    values.sort((a, b) => Number(a) - Number(b));
  }

  return {
    indexPath,
    mtimeMs,
    manifest,
    cutouts,
    cutoutsById,
    cutoutIdsByKeyframe,
    objects: [],
    objectsByCutout: new Map(),
    objectsByKeyframe: new Map(),
    clusters,
  };
}

async function queryObjects(
  indexPath: string,
  whereSql = "1 = 1",
): Promise<ObjectRecord[]> {
  const objectColumns = await sqliteTableColumns(indexPath, "object");
  if (!objectColumns?.size) {
    throw new WorkbenchRouteError(
      500,
      "Could not read object table schema from object-search index.",
    );
  }
  const bboxSelectColumns = buildObjectBboxSelectClause(objectColumns);
  const objectRows = await sqliteObjectRowsPaged(
    indexPath,
    `object_idx, keyframe_id, cutout_id,
     ${bboxSelectColumns},
     label, detection_source, textness_score,
     ocr_text, ocr_tokens, ocr_key, ocr_candidate, ocr_assigned, ocr_source,
     cluster_id`,
    whereSql,
  );
  return objectRows.map((row) => {
    const objectIdx = asNumber(row.object_idx);
    return {
      object_idx: objectIdx,
      object_id: `object-${objectIdx}`,
      keyframe_id: String(row.keyframe_id),
      cutout_id: String(row.cutout_id),
      bbox: bboxFromObjectRow(row),
      label: asString(row.label),
      detection_source: asString(row.detection_source),
      textness_score: asNullableNumber(row.textness_score),
      ocr_text: asString(row.ocr_text),
      ocr_tokens: asString(row.ocr_tokens),
      ocr_key: asString(row.ocr_key),
      ocr_source: asNumber(row.ocr_source),
      ocr_candidate: Boolean(row.ocr_candidate),
      ocr_assigned: Boolean(row.ocr_assigned),
      cluster_id: asNullableNumber(row.cluster_id),
    };
  });
}

function indexObjectRecord(item: ObjectRecord): Record<string, unknown> {
  return {
    id: item.object_id,
    keyframe_id: item.keyframe_id,
    cutout_id: item.cutout_id,
    bbox: item.bbox.map((v) => Math.round(v * 10) / 10),
    label: item.label,
    detection_source: item.detection_source,
    ocr_text: item.ocr_text,
    ocr_tokens: item.ocr_tokens,
    ocr_key: item.ocr_key,
    ocr_source: formatOcrSource(item.ocr_source),
    textness: item.textness_score == null ? null : Math.round(item.textness_score * 10000) / 10000,
    ocr_candidate: item.ocr_candidate,
    ocr_assigned: item.ocr_assigned,
    cluster_id: item.cluster_id,
  };
}

function splitCatalogPrompt(prompt: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const part of prompt.split(".")) {
    const item = part.trim().replace(/^[,;]+|[,;]+$/g, "");
    if (item && !seen.has(item)) {
      out.push(item);
      seen.add(item);
    }
  }
  return out;
}

function extractDefaultPrompts(manifest: Record<string, unknown>): string[] {
  const rawParams = manifest.gdino_params_json;
  if (typeof rawParams === "string" && rawParams.trim()) {
    try {
      const params = JSON.parse(rawParams) as Record<string, unknown>;
      const promptSources =
        typeof params.prompt === "string" && params.prompt.trim()
          ? [params.prompt.trim()]
          : Array.isArray(params.prompts)
            ? params.prompts.map(String).filter((item) => item.trim())
            : [];
      if (promptSources.length) {
        const parsed: string[] = [];
        const seen = new Set<string>();
        for (const prompt of promptSources) {
          const expanded = splitCatalogPrompt(prompt);
          for (const candidate of expanded.length ? expanded : [prompt]) {
            if (!seen.has(candidate)) {
              parsed.push(candidate);
              seen.add(candidate);
            }
          }
        }
        if (parsed.length) {
          return parsed;
        }
      }
    } catch {
      // fall through to summary prompt
    }
  }
  const summary = asString(manifest.object_detector_prompt).trim();
  if (!summary) {
    return [];
  }
  const expanded = splitCatalogPrompt(summary);
  return expanded.length ? expanded : [summary];
}

function bboxArea(bbox: Bbox): number {
  return Math.max(0, bbox[2] - bbox[0]) * Math.max(0, bbox[3] - bbox[1]);
}

function bboxIou(a: Bbox, b: Bbox): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter <= 0) {
    return 0;
  }
  const union = bboxArea(a) + bboxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

function postprocessDetections(
  detections: ObjectRecord[],
  params: { nmsIou: number; minBBoxArea: number; maxBBoxArea: number; showYolo?: boolean; showGdino?: boolean },
): ObjectRecord[] {
  const maxArea = params.maxBBoxArea <= 0 ? Number.POSITIVE_INFINITY : params.maxBBoxArea;
  const minArea = Math.max(0, params.minBBoxArea);
  const filtered = detections.filter((item) => {
    const source = item.detection_source.toLowerCase();
    const isYolo = source.includes("yolo");
    const isGdino = source.includes("gdino") || source.includes("g-dino") || source.includes("grounding");
    if (isYolo && params.showYolo === false) {
      return false;
    }
    if (isGdino && params.showGdino === false) {
      return false;
    }
    if (!isYolo && !isGdino && (params.showYolo === false || params.showGdino === false)) {
      return false;
    }
    const area = bboxArea(item.bbox);
    return area >= minArea && area <= maxArea;
  });
  if (params.nmsIou >= 1 || filtered.length <= 1) {
    return filtered;
  }
  const ordered = [...filtered].sort((a, b) => {
    const scoreA = a.textness_score ?? bboxArea(a.bbox);
    const scoreB = b.textness_score ?? bboxArea(b.bbox);
    return scoreB - scoreA;
  });
  const kept: ObjectRecord[] = [];
  for (const item of ordered) {
    if (kept.every((existing) => bboxIou(item.bbox, existing.bbox) < params.nmsIou)) {
      kept.push(item);
    }
  }
  return kept;
}

function postprocessByCutout(
  detections: ObjectRecord[],
  params: { nmsIou: number; minBBoxArea: number; maxBBoxArea: number; showYolo?: boolean; showGdino?: boolean },
): ObjectRecord[] {
  const grouped = new Map<string, ObjectRecord[]>();
  for (const item of detections) {
    const group = grouped.get(item.cutout_id) ?? [];
    group.push(item);
    grouped.set(item.cutout_id, group);
  }
  return [...grouped.values()].flatMap((group) => postprocessDetections(group, params));
}

export async function indexStatusPayload(
  map: MapEntry,
  selectedKeyframeId: string | null,
): Promise<Record<string, unknown>> {
  const { indexPath, checkedPaths } = await resolveIndexPath(map);
  if (!indexPath) {
    return {
      available: false,
      index_path: null,
      map_path: map.path,
      object_search_index_path: map.object_search_index_path,
      checked_paths: checkedPaths,
    };
  }

  const index = await loadIndex(indexPath);
  const cutoutIdsByKeyframe = Object.fromEntries(index.cutoutIdsByKeyframe.entries());
  const keyframeIds = [...index.cutoutIdsByKeyframe.keys()].sort((a, b) => Number(a) - Number(b));
  const objectCountRows = await sqliteJson<Record<string, unknown>>(
    indexPath,
    `SELECT cutout_id, COUNT(*) AS object_count
     FROM ${sqlIdentifier("object")}
     GROUP BY cutout_id`,
  );
  const objectCountByCutout = Object.fromEntries(
    objectCountRows.map((row) => [String(row.cutout_id), asNumber(row.object_count)]),
  );
  const markers = await keyframeMarkers(map, keyframeIds, selectedKeyframeId);

  const ocrSummary = (await sqliteJson<Record<string, unknown>>(
    indexPath,
    `SELECT
       SUM(CASE WHEN COALESCE(TRIM(ocr_text), '') != '' THEN 1 ELSE 0 END) AS ocr_text_count,
       SUM(CASE WHEN COALESCE(TRIM(ocr_key), '') != '' THEN 1 ELSE 0 END) AS ocr_key_count,
       SUM(CASE WHEN ocr_candidate THEN 1 ELSE 0 END) AS ocr_candidate_count,
       SUM(CASE WHEN ocr_assigned THEN 1 ELSE 0 END) AS ocr_assigned_count
     FROM ${sqlIdentifier("object")}`,
  ))[0] ?? {};
  const ocrSourceRows = await sqliteJson<Record<string, unknown>>(
    indexPath,
    `SELECT COALESCE(ocr_source, 0) AS ocr_source, COUNT(*) AS object_count
     FROM ${sqlIdentifier("object")}
     GROUP BY COALESCE(ocr_source, 0)
     ORDER BY COALESCE(ocr_source, 0)`,
  );

  return {
    available: true,
    index_path: indexPath,
    summary: {
      index_path: indexPath,
      manifest: index.manifest,
      keyframe_ids: keyframeIds,
      cutout_ids_by_keyframe: cutoutIdsByKeyframe,
      object_count_by_cutout: objectCountByCutout,
      ocr_text_count: asNumber(ocrSummary.ocr_text_count),
      ocr_key_count: asNumber(ocrSummary.ocr_key_count),
      ocr_candidate_count: asNumber(ocrSummary.ocr_candidate_count),
      ocr_assigned_count: asNumber(ocrSummary.ocr_assigned_count),
      ocr_source_counts: Object.fromEntries(
        ocrSourceRows.map(
          (row) => [formatOcrSource(asNumber(row.ocr_source)), asNumber(row.object_count)],
        ),
      ),
      cluster_ocr_count: index.clusters.filter((item) => item.ocr_text.trim() || item.ocr_key.trim()).length,
      default_prompts: extractDefaultPrompts(index.manifest),
    },
    markers,
    resolved_marker_count: markers.length,
  };
}

export async function indexObjectsPayload(
  map: MapEntry,
  keyframeId: string | null,
): Promise<Record<string, unknown>> {
  const index = await requireIndex(map);
  const whereSql = keyframeId === null ? "1 = 1" : `keyframe_id = ${sqlString(keyframeId)}`;
  const objects = await queryObjects(index.indexPath, whereSql);
  return { objects: objects.map(indexObjectRecord) };
}

export async function indexClusterOcrPayload(map: MapEntry): Promise<Record<string, unknown>> {
  const index = await requireIndex(map);
  return {
    records: index.clusters.map((item) => ({
      cluster_id: item.cluster_id,
      ocr_text: item.ocr_text,
      ocr_tokens: item.ocr_tokens,
      ocr_key: item.ocr_key,
      ocr_observation_count: item.ocr_observation_count,
      ocr_source: formatOcrSource(item.ocr_source),
    })),
  };
}

export async function indexCutoutPayload(
  map: MapEntry,
  cutoutId: string,
): Promise<Record<string, unknown>> {
  const index = await requireIndex(map);
  const cutout = index.cutoutsById.get(cutoutId);
  const previewRef = `index:${index.indexPath}::${cutoutId}`;
  const sourceImagePath = cutout ? await resolveSourceImagePath(map, index, cutout.keyframe_id) : null;
  const detections = await queryObjects(index.indexPath, `cutout_id = ${sqlString(cutoutId)}`);
  return {
    detections: detections.map(indexObjectRecord),
    preview_ref: previewRef,
    preview_debug: {
      index_path: index.indexPath,
      cutout_id: cutoutId,
      keyframe_id: cutout?.keyframe_id ?? null,
      source: index.manifest.source ?? null,
      geometry: index.manifest.geometry ?? null,
      source_images_dir: index.manifest.source_images_dir ?? null,
      source_image_path: sourceImagePath,
      params: cutoutParams(index.manifest),
      error: cutout ? null : "Cutout id not found in object-search index",
    },
  };
}

export async function cutoutDetectionsPayload(
  map: MapEntry,
  cutoutId: string,
): Promise<Record<string, unknown>> {
  const index = await requireIndex(map);
  const detections = await queryObjects(index.indexPath, `cutout_id = ${sqlString(cutoutId)}`);
  return {
    detections: detections.map((item) => ({
      id: item.object_id,
      label: item.label || null,
      confidence: null,
      source: item.detection_source || "index",
      bbox: item.bbox,
    })),
  };
}

export async function keyframeMetadataPayload(
  map: MapEntry,
  keyframeId: string,
): Promise<Record<string, unknown>> {
  const meta = await lookupFlatKeyframes(map, [keyframeId]);
  const found = meta.get(keyframeId);
  if (!found) {
    return { id: keyframeId, lat: null, lon: null, alt: null, level: null };
  }
  return { id: keyframeId, lat: found.lat, lon: found.lon, alt: found.alt, level: found.level };
}

async function requireIndex(map: MapEntry): Promise<LoadedIndex> {
  const { indexPath, checkedPaths } = await resolveIndexPath(map);
  if (!indexPath) {
    // 501, not 404: the file is not "missing", the feature is retired. A 404 read
    // as "build the index first", which sent people looking for a build_index.py
    // that no longer exists.
    throw new WorkbenchRouteError(
      501,
      "The index-explorer needs the legacy standalone index (object-search.db), which "
      + "is no longer produced — the object-search index lives in pgvector as of "
      + "ADR 0002. This panel only works for maps built before that migration. "
      + `Looked in: ${checkedPaths.join(", ")}. `
      + "For current maps, use the object-search panel (it queries the live index).",
    );
  }
  return loadIndex(indexPath);
}

type FlatKeyframe = { lat: number; lon: number; alt: number | null; level: string | null };

async function lookupFlatKeyframes(
  map: MapEntry,
  keyframeIds: string[],
): Promise<Map<string, FlatKeyframe>> {
  const uniqueIds = [...new Set(keyframeIds.filter(Boolean))];
  const out = new Map<string, FlatKeyframe>();
  if (!uniqueIds.length) {
    return out;
  }
  for (const dbName of ["georef.db", "reloc.db"]) {
    const dbPath = path.join(map.path, dbName);
    if (!(await pathExists(dbPath)) || !(await tableExists(dbPath, "keyframes"))) {
      continue;
    }
    const rows = await maybeSqliteJson<Record<string, unknown>>(
      dbPath,
      `SELECT id, lat, lon, alt, level FROM keyframes WHERE id IN (${uniqueIds.map(sqlString).join(",")})`,
    );
    for (const row of rows) {
      const lat = asNullableNumber(row.lat);
      const lon = asNullableNumber(row.lon);
      if (lat == null || lon == null) {
        continue;
      }
      out.set(String(row.id), {
        lat,
        lon,
        alt: asNullableNumber(row.alt),
        level: row.level == null ? null : String(row.level),
      });
    }
  }
  const missing = uniqueIds.filter((id) => !out.has(id));
  if (missing.length) {
    const poseRows = await lookupGeoRefPoseKeyframes(map, missing);
    for (const [id, value] of poseRows.entries()) {
      out.set(id, value);
    }
  }
  return out;
}

type GeoRefContext = {
  originLatDeg: number;
  originLonDeg: number;
  originAlt: number;
  originTr: number[][];
  levels: Array<{ id: number; min: number; max: number }>;
};

async function georefTableColumns(dbPath: string, tableName: string): Promise<Set<string> | null> {
  return sqliteTableColumns(dbPath, tableName);
}

function georefKeyframeFilenameColumn(columns: Set<string> | null): string | null {
  if (!columns) {
    return null;
  }
  if (columns.has("image_filename")) {
    return "image_filename";
  }
  if (columns.has("filename")) {
    return "filename";
  }
  return null;
}

async function lookupGeoRefKeyframeImageFilename(
  map: MapEntry,
  keyframeId: string,
): Promise<string | null> {
  const dbPath = path.join(map.path, "georef.db");
  const columns = await georefTableColumns(dbPath, "GeoRefKeyframe");
  const filenameColumn = georefKeyframeFilenameColumn(columns);
  if (!filenameColumn) {
    return null;
  }
  const rows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    `SELECT "${filenameColumn}" AS image_filename FROM GeoRefKeyframe WHERE id = ${sqlString(keyframeId)} LIMIT 1`,
  );
  const value = rows[0]?.image_filename;
  if (value == null) {
    return null;
  }
  const filename = String(value).trim();
  return filename || null;
}

async function lookupGeoRefPoseKeyframes(
  map: MapEntry,
  keyframeIds: string[],
): Promise<Map<string, FlatKeyframe>> {
  const out = new Map<string, FlatKeyframe>();
  const dbPath = path.join(map.path, "georef.db");
  if (!(await pathExists(dbPath)) || !(await tableExists(dbPath, "GeoRefKeyframe"))) {
    return out;
  }
  const context = await loadGeoRefContext(dbPath);
  if (!context) {
    return out;
  }
  const rows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    `SELECT id, hex(pose) AS pose_hex FROM GeoRefKeyframe WHERE id IN (${keyframeIds.map(sqlString).join(",")})`,
  );
  for (const row of rows) {
    const id = String(row.id);
    const poseHex = typeof row.pose_hex === "string" ? row.pose_hex : "";
    if (!poseHex) {
      continue;
    }
    try {
      const pose = transpose4(decodeFloat64Matrix4(poseHex));
      const localPose = invertRigid4(pose);
      const world = matMul4(context.originTr, localPose);
      const coords = ecefToGeodetic(world[0][3], world[1][3], world[2][3]);
      const heightFromGround = coords.alt - context.originAlt;
      const level = context.levels.find((item) => heightFromGround >= item.min && heightFromGround <= item.max);
      out.set(id, {
        lat: coords.lat,
        lon: coords.lon,
        alt: coords.alt,
        level: level ? String(level.id) : null,
      });
    } catch {
      // Ignore malformed pose rows; the UI can still operate without markers.
    }
  }
  return out;
}

async function lookupGeoRefKeyframeHeadings(
  map: MapEntry,
  keyframeIds: string[],
): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  const uniqueIds = [...new Set(keyframeIds.filter(Boolean))];
  if (!uniqueIds.length) {
    return out;
  }
  const dbPath = path.join(map.path, "georef.db");
  if (!(await pathExists(dbPath)) || !(await tableExists(dbPath, "GeoRefKeyframe"))) {
    return out;
  }
  const rows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    `SELECT id, hex(pose) AS pose_hex FROM GeoRefKeyframe WHERE id IN (${uniqueIds.map(sqlString).join(",")})`,
  );
  for (const row of rows) {
    const poseHex = typeof row.pose_hex === "string" ? row.pose_hex : "";
    if (!poseHex) {
      continue;
    }
    try {
      const pose = transpose4(decodeFloat64Matrix4(poseHex));
      out.set(String(row.id), keyframeHeadingDegreesFromPose(pose));
    } catch {
      // Ignore malformed poses; the viewer falls back to captured-forward on that keyframe.
    }
  }
  return out;
}

async function loadGeoRefContext(dbPath: string): Promise<GeoRefContext | null> {
  const originRows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    "SELECT latitude, longitude, altitude FROM GeoRef LIMIT 1",
  );
  const origin = originRows[0];
  if (!origin) {
    return null;
  }
  const originLatDeg = asNumber(origin.latitude);
  const originLonDeg = asNumber(origin.longitude);
  const originAlt = asNumber(origin.altitude);
  const originEcef = geodeticToEcef(originLatDeg, originLonDeg, originAlt);
  const rot = wdsToEcefRotation(originLatDeg, originLonDeg);
  const originTr = [
    [rot[0][0], rot[0][1], rot[0][2], originEcef[0]],
    [rot[1][0], rot[1][1], rot[1][2], originEcef[1]],
    [rot[2][0], rot[2][1], rot[2][2], originEcef[2]],
    [0, 0, 0, 1],
  ];
  const levelRows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    "SELECT id, min_altitude, max_altitude FROM Level",
  );
  return {
    originLatDeg,
    originLonDeg,
    originAlt,
    originTr,
    levels: levelRows.map((row) => ({
      id: asNumber(row.id),
      min: asNumber(row.min_altitude),
      max: asNumber(row.max_altitude),
    })),
  };
}

function decodeFloat64Matrix4(hex: string): number[][] {
  const buffer = Buffer.from(hex, "hex");
  if (buffer.byteLength < 16 * 8) {
    throw new Error("Pose blob is too short");
  }
  const values = Array.from({ length: 16 }, (_, idx) => buffer.readDoubleLE(idx * 8));
  return [
    values.slice(0, 4),
    values.slice(4, 8),
    values.slice(8, 12),
    values.slice(12, 16),
  ];
}

function transpose4(m: number[][]): number[][] {
  return [0, 1, 2, 3].map((row) => [0, 1, 2, 3].map((col) => m[col][row]));
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

function matMul4(a: number[][], b: number[][]): number[][] {
  return [0, 1, 2, 3].map((row) =>
    [0, 1, 2, 3].map((col) =>
      a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col] + a[row][3] * b[3][col],
    ),
  );
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
  const level = context.levels.find((item) => heightFromGround >= item.min && heightFromGround <= item.max);
  return {
    latitude: coords.lat,
    longitude: coords.lon,
    altitude: coords.alt,
    level: level ? String(level.id) : null,
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

async function lookupGeoRefKeyframePose(
  map: MapEntry,
  keyframeId: string,
): Promise<{ context: GeoRefContext; pose: number[][] } | null> {
  const dbPath = path.join(map.path, "georef.db");
  if (!(await pathExists(dbPath)) || !(await tableExists(dbPath, "GeoRefKeyframe"))) {
    return null;
  }
  const context = await loadGeoRefContext(dbPath);
  if (!context) {
    return null;
  }
  const rows = await maybeSqliteJson<Record<string, unknown>>(
    dbPath,
    `SELECT hex(pose) AS pose_hex FROM GeoRefKeyframe WHERE id = ${sqlString(keyframeId)} LIMIT 1`,
  );
  const poseHex = typeof rows[0]?.pose_hex === "string" ? rows[0].pose_hex : "";
  if (!poseHex) {
    return null;
  }
  return {
    context,
    pose: transpose4(decodeFloat64Matrix4(poseHex)),
  };
}

async function keyframeMarkers(
  map: MapEntry,
  keyframeIds: string[],
  selectedKeyframeId: string | null,
): Promise<Array<Record<string, unknown>>> {
  const meta = await lookupFlatKeyframes(map, keyframeIds);
  const headings = await lookupGeoRefKeyframeHeadings(map, [...meta.keys()]);
  return [...meta.entries()].map(([id, item]) => ({
    id,
    latitude: item.lat,
    longitude: item.lon,
    level: item.level,
    heading_deg: headings.get(id) ?? null,
    color: id === selectedKeyframeId ? "#16a34a" : "#9ca3af",
    radius: id === selectedKeyframeId ? 8 : 6,
  }));
}

function cutoutParams(manifest: Record<string, unknown>): Record<string, unknown> | null {
  if (manifest.geometry === "cubemap") {
    return {
      width: asNumber(manifest.cubemap_face_size),
      height: asNumber(manifest.cubemap_face_size),
      fov: asNumber(manifest.cubemap_fov_deg, 90),
      inclination: 0,
      nb_of_cuts: 1,
    };
  }
  if (manifest.geometry === "horizon") {
    return {
      width: asNumber(manifest.horizon_width),
      height: asNumber(manifest.horizon_height),
      fov: asNumber(manifest.horizon_fov),
      inclination: asNumber(manifest.horizon_inclination),
      nb_of_cuts: asNumber(manifest.horizon_nb_of_cuts, 1),
    };
  }
  return null;
}

async function resolveSourceImagePath(
  map: MapEntry,
  index: LoadedIndex,
  keyframeId: string,
): Promise<string | null> {
  const dirs = [
    typeof index.manifest.source_images_dir === "string" ? index.manifest.source_images_dir : null,
    path.join(map.path, "images_360"),
    path.join(path.dirname(index.indexPath), "images_360"),
  ].filter((item): item is string => Boolean(item));
  const georefFilename = await lookupGeoRefKeyframeImageFilename(map, keyframeId);
  const filenames = [
    georefFilename,
    `${keyframeId}.jpg`,
    `${keyframeId}.jpeg`,
    `${keyframeId}.png`,
  ].filter((item): item is string => Boolean(item));
  for (const dir of dirs) {
    for (const filename of filenames) {
      const candidate = path.resolve(dir, filename);
      if (await pathExists(candidate)) {
        return candidate;
      }
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

function cubemapFace(cutoutId: string, manifest: Record<string, unknown>): string {
  const stride = Math.max(1, asNumber(manifest.id_stride, 1024));
  const localIndex = Number(cutoutId) % stride;
  return CUBEMAP_FACE_ORDER[localIndex] ?? "front";
}

function faceYawPitch(faceName: string): { yaw: number; pitch: number } {
  switch (faceName) {
    case "back":
      return { yaw: 180, pitch: 0 };
    case "left":
      return { yaw: -90, pitch: 0 };
    case "right":
      return { yaw: 90, pitch: 0 };
    case "top":
      return { yaw: 0, pitch: 90 };
    case "bottom":
      return { yaw: 0, pitch: -90 };
    default:
      return { yaw: 0, pitch: 0 };
  }
}

async function baseCutoutPng(
  map: MapEntry,
  index: LoadedIndex,
  cutoutId: string,
): Promise<Buffer> {
  const cutout = index.cutoutsById.get(cutoutId);
  if (!cutout) {
    throw new WorkbenchRouteError(404, "Cutout id not found");
  }
  if (index.manifest.geometry !== "cubemap") {
    throw new WorkbenchRouteError(404, `Unsupported cutout preview geometry: ${String(index.manifest.geometry)}`);
  }
  const sourceImagePath = await resolveSourceImagePath(map, index, cutout.keyframe_id);
  if (!sourceImagePath) {
    throw new WorkbenchRouteError(404, "Preview source image not found");
  }
  const faceSize = Math.max(1, asNumber(index.manifest.cubemap_face_size, 512));
  const fov = Math.max(1, asNumber(index.manifest.cubemap_fov_deg, 90));
  const faceName = cubemapFace(cutoutId, index.manifest);
  const { yaw, pitch } = faceYawPitch(faceName);
  const sourceMtime = (await stat(sourceImagePath)).mtimeMs;
  const cacheKey = `${sourceImagePath}:${sourceMtime}:${cutoutId}:${faceSize}:${fov}`;
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
      `v360=input=equirect:output=flat:yaw=${yaw}:pitch=${pitch}:h_fov=${fov}:v_fov=${fov}:w=${faceSize}:h=${faceSize}`,
      "-frames:v",
      "1",
      "-f",
      "image2pipe",
      "-vcodec",
      "png",
      "-",
    ],
    { encoding: "buffer", maxBuffer: 64 * 1024 * 1024 },
  ).then(({ stdout }) => Buffer.from(stdout));
  cutoutPreviewCache.set(cacheKey, promise);
  if (cutoutPreviewCache.size > 256) {
    cutoutPreviewCache.delete(cutoutPreviewCache.keys().next().value as string);
  }
  return promise;
}

function drawArgsForObjects(objects: ObjectRecord[], selectedObjectId: string | null): string[] {
  const args: string[] = [];
  for (const item of objects) {
    const color = item.object_id === selectedObjectId ? BOX_COLOR_SELECTED : BOX_COLOR;
    const width = item.object_id === selectedObjectId ? "4" : "2";
    const [x0, y0, x1, y1] = item.bbox;
    args.push("-fill", "none", "-stroke", color, "-strokewidth", width, "-draw", `rectangle ${x0},${y0} ${x1},${y1}`);
    const label = item.label || item.detection_source;
    if (label) {
      const safeLabel = label.replaceAll("'", "\\'");
      args.push("-fill", color, "-stroke", "none", "-pointsize", "12", "-draw", `text ${x0 + 2},${Math.max(12, y0 - 3)} '${safeLabel}'`);
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

export async function indexCutoutPreviewPng(
  map: MapEntry,
  cutoutId: string,
  options: {
    selectedObjectId: string | null;
    selectedOnly?: boolean;
    selectedBbox?: Bbox | null;
    drawBoxes: boolean;
    nmsIou: number;
    minBBoxArea: number;
    maxBBoxArea: number;
    showYolo: boolean;
    showGdino: boolean;
  },
): Promise<Buffer> {
  const index = await requireIndex(map);
  const base = await baseCutoutPng(map, index, cutoutId);
  if (!options.drawBoxes) {
    return base;
  }
  const cutoutObjects = await queryObjects(index.indexPath, `cutout_id = ${sqlString(cutoutId)}`);
  if (options.selectedOnly) {
    const selectedObject = options.selectedBbox
      ? cutoutObjects.reduce<ObjectRecord | null>((closest, item) => {
          if (!closest) {
            return item;
          }
          const distance = item.bbox.reduce(
            (sum, value, index) => sum + (value - options.selectedBbox![index]) ** 2,
            0,
          );
          const closestDistance = closest.bbox.reduce(
            (sum, value, index) => sum + (value - options.selectedBbox![index]) ** 2,
            0,
          );
          return distance < closestDistance ? item : closest;
        }, null)
      : cutoutObjects.find((item) => item.object_id === options.selectedObjectId);
    return selectedObject
      ? convertPng(base, drawArgsForObjects([selectedObject], options.selectedObjectId))
      : base;
  }
  let objects = postprocessDetections(cutoutObjects, {
    nmsIou: options.nmsIou,
    minBBoxArea: options.minBBoxArea,
    maxBBoxArea: options.maxBBoxArea,
    showYolo: options.showYolo,
    showGdino: options.showGdino,
  });
  return objects.length ? convertPng(base, drawArgsForObjects(objects, options.selectedObjectId)) : base;
}

export async function legacyCutoutPreviewPng(
  map: MapEntry,
  cutoutId: string,
  selectedObjectId: string | null,
  selectedOnly = false,
  selectedBboxRaw: string | null = null,
): Promise<Buffer> {
  const selectedBboxValues = selectedBboxRaw?.split(",").map(Number) ?? [];
  const selectedBbox =
    selectedBboxValues.length === 4 && selectedBboxValues.every(Number.isFinite)
      ? (selectedBboxValues as Bbox)
      : null;
  return indexCutoutPreviewPng(map, cutoutId, {
    selectedObjectId,
    selectedOnly,
    selectedBbox,
    drawBoxes: true,
    nmsIou: 1,
    minBBoxArea: 0,
    maxBBoxArea: 0,
    showYolo: true,
    showGdino: true,
  });
}

export async function previewFromPathPng(map: MapEntry, previewPath: string | null): Promise<Buffer> {
  if (!previewPath) {
    throw new WorkbenchRouteError(400, "Missing preview_path");
  }
  if (previewPath.startsWith("index:") && previewPath.includes("::")) {
    const [indexPath, cutoutId] = previewPath.slice(6).split("::");
    const index = await loadIndex(indexPath);
    return baseCutoutPng(map, index, cutoutId);
  }
  const resolved = path.isAbsolute(previewPath) ? previewPath : path.resolve(map.path, previewPath);
  if (!(await pathExists(resolved))) {
    throw new WorkbenchRouteError(404, "Preview file not found");
  }
  return convertPng(resolved, []);
}

export async function indexObjectCropPng(
  map: MapEntry,
  cutoutId: string,
  bbox: Bbox,
): Promise<Buffer> {
  const index = await requireIndex(map);
  const base = await baseCutoutPng(map, index, cutoutId);
  const x0 = Math.max(0, Math.floor(bbox[0]));
  const y0 = Math.max(0, Math.floor(bbox[1]));
  const width = Math.max(1, Math.ceil(bbox[2]) - x0);
  const height = Math.max(1, Math.ceil(bbox[3]) - y0);
  return convertPng(base, ["-crop", `${width}x${height}+${x0}+${y0}`, "+repage"]);
}

type DepthSample = {
  depth_m: number;
  row: number;
  col: number;
  width: number;
  height: number;
  sample_count: number;
};

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


def load_json(path):
    return json.loads(Path(path).read_text())


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


def decode_chunk(array_dir, meta, chunk_row, chunk_col):
    from numcodecs import Blosc

    chunk_path = array_dir / f"{chunk_row}.{chunk_col}"
    if not chunk_path.exists():
        return None
    raw = chunk_path.read_bytes()
    compressor = meta.get("compressor") or {}
    if compressor.get("id") == "blosc":
        decoded = Blosc(
            cname=compressor.get("cname", "zstd"),
            clevel=int(compressor.get("clevel", 3)),
            shuffle=int(compressor.get("shuffle", 0)),
            blocksize=int(compressor.get("blocksize", 0)),
        ).decode(raw)
    elif compressor:
        raise RuntimeError(f"unsupported compressor: {compressor.get('id')}")
    else:
        decoded = raw

    dtype = np.dtype(meta["dtype"])
    shape = meta["shape"]
    chunks = meta["chunks"]
    rows = min(int(chunks[0]), int(shape[0]) - chunk_row * int(chunks[0]))
    cols = min(int(chunks[1]), int(shape[1]) - chunk_col * int(chunks[1]))
    return np.frombuffer(decoded, dtype=dtype).reshape((rows, cols), order=meta.get("order", "C"))


def sample_zarr(root, u, v, radius):
    depth_dir = root / "depth"
    mask_dir = root / "valid_mask"
    depth_meta = load_json(depth_dir / ".zarray")
    depth_attrs = load_json(depth_dir / ".zattrs") if (depth_dir / ".zattrs").exists() else {}
    mask_meta = load_json(mask_dir / ".zarray") if mask_dir.exists() else None

    height, width = [int(v) for v in depth_meta["shape"]]
    row, col = ratio_to_pixel(u, v, width, height)
    chunk_h, chunk_w = [int(v) for v in depth_meta["chunks"]]

    chunk_cache = {}
    mask_cache = {}

    def get_depth(r, c):
        chunk_key = (r // chunk_h, c // chunk_w)
        if chunk_key not in chunk_cache:
            chunk_cache[chunk_key] = decode_chunk(depth_dir, depth_meta, chunk_key[0], chunk_key[1])
        chunk = chunk_cache[chunk_key]
        if chunk is None:
            return math.nan
        return float(chunk[r % chunk_h, c % chunk_w])

    def get_valid(r, c):
        if mask_meta is None:
            return True
        chunk_key = (r // chunk_h, c // chunk_w)
        if chunk_key not in mask_cache:
            mask_cache[chunk_key] = decode_chunk(mask_dir, mask_meta, chunk_key[0], chunk_key[1])
        chunk = mask_cache[chunk_key]
        if chunk is None:
            return False
        return bool(chunk[r % chunk_h, c % chunk_w])

    def get_value(yy, xx):
        if not get_valid(yy, xx):
            return math.nan
        encoded = get_depth(yy, xx)
        if not math.isfinite(encoded):
            return math.nan
        if (
            depth_attrs.get("encoding") == "quantized_uint16"
            and "min_m" in depth_attrs
            and "step_m" in depth_attrs
        ):
            depth_m = encoded * float(depth_attrs["step_m"]) + float(depth_attrs["min_m"])
        elif "min_m" in depth_attrs and "max_m" in depth_attrs:
            dmin = float(depth_attrs["min_m"])
            dmax = float(depth_attrs["max_m"])
            scale = 65535.0 / (dmax - dmin) if dmax > dmin else 1.0
            depth_m = encoded / scale + dmin
        else:
            depth_m = encoded
        return float(depth_m)

    values = sample_window(row, col, width, height, radius, get_value)
    return {
        "depth_m": float(np.median(np.asarray(values, dtype=np.float32))),
        "row": row,
        "col": col,
        "width": width,
        "height": height,
        "sample_count": len(values),
    }


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
    depth_path = Path(req["depth_path"])
    u = float(req["u"])
    v = float(req["v"])
    radius = max(0, int(req.get("radius", 3)))
    suffix = depth_path.suffix.lower()
    if depth_path.is_dir() or suffix == ".zarr":
        result = sample_zarr(depth_path, u, v, radius)
    elif suffix in {".tif", ".tiff"}:
        result = sample_tiff(depth_path, u, v, radius)
    else:
        raise RuntimeError(f"unsupported depth map format: {depth_path}")
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


def load_json(path):
    return json.loads(Path(path).read_text())


def decode_tiff_uint16_meters(encoded):
    values = np.asarray(encoded, dtype=np.uint16)
    valid = values != 0
    raw = values.astype(np.int32) >> _SHIFT
    normed = (raw - 1).astype(np.float32) / max(_LEVELS - 1, 1)
    depth = (_SQRT_LO + normed * (_SQRT_HI - _SQRT_LO)) ** 2
    return np.where(valid, depth.astype(np.float32), np.float32(np.nan))


def decode_chunk(array_dir, meta, chunk_row, chunk_col):
    from numcodecs import Blosc

    chunk_path = array_dir / f"{chunk_row}.{chunk_col}"
    if not chunk_path.exists():
        return None
    raw = chunk_path.read_bytes()
    compressor = meta.get("compressor") or {}
    if compressor.get("id") == "blosc":
        decoded = Blosc(
            cname=compressor.get("cname", "zstd"),
            clevel=int(compressor.get("clevel", 3)),
            shuffle=int(compressor.get("shuffle", 0)),
            blocksize=int(compressor.get("blocksize", 0)),
        ).decode(raw)
    elif compressor:
        raise RuntimeError(f"unsupported compressor: {compressor.get('id')}")
    else:
        decoded = raw

    dtype = np.dtype(meta["dtype"])
    shape = meta["shape"]
    chunks = meta["chunks"]
    rows = min(int(chunks[0]), int(shape[0]) - chunk_row * int(chunks[0]))
    cols = min(int(chunks[1]), int(shape[1]) - chunk_col * int(chunks[1]))
    return np.frombuffer(decoded, dtype=dtype).reshape((rows, cols), order=meta.get("order", "C"))


def depth_from_encoded(encoded, attrs):
    encoded = encoded.astype(np.float32)
    if (
        attrs.get("encoding") == "quantized_uint16"
        and "min_m" in attrs
        and "step_m" in attrs
    ):
        return encoded * float(attrs["step_m"]) + float(attrs["min_m"])
    if "min_m" in attrs and "max_m" in attrs:
        dmin = float(attrs["min_m"])
        dmax = float(attrs["max_m"])
        scale = 65535.0 / (dmax - dmin) if dmax > dmin else 1.0
        return encoded / scale + dmin
    return encoded


def load_zarr_preview(root, max_width):
    depth_dir = root / "depth"
    mask_dir = root / "valid_mask"
    depth_meta = load_json(depth_dir / ".zarray")
    depth_attrs = load_json(depth_dir / ".zattrs") if (depth_dir / ".zattrs").exists() else {}
    mask_meta = load_json(mask_dir / ".zarray") if mask_dir.exists() else None
    height, width = [int(v) for v in depth_meta["shape"]]
    out_width = min(width, max(1, int(max_width)))
    out_height = max(1, round(height * out_width / width))
    rows = np.linspace(0, height - 1, out_height).round().astype(np.int64)
    cols = np.linspace(0, width - 1, out_width).round().astype(np.int64)
    chunk_h, chunk_w = [int(v) for v in depth_meta["chunks"]]
    chunk_cache = {}
    mask_cache = {}
    out = np.full((out_height, out_width), np.nan, dtype=np.float32)
    for out_r, src_r in enumerate(rows):
        for out_c, src_c in enumerate(cols):
            chunk_key = (int(src_r) // chunk_h, int(src_c) // chunk_w)
            if chunk_key not in chunk_cache:
                chunk_cache[chunk_key] = decode_chunk(depth_dir, depth_meta, chunk_key[0], chunk_key[1])
            chunk = chunk_cache[chunk_key]
            if chunk is None:
                continue
            if mask_meta is not None:
                if chunk_key not in mask_cache:
                    mask_cache[chunk_key] = decode_chunk(mask_dir, mask_meta, chunk_key[0], chunk_key[1])
                mask = mask_cache[chunk_key]
                if mask is None or not bool(mask[int(src_r) % chunk_h, int(src_c) % chunk_w]):
                    continue
            encoded = np.asarray([[chunk[int(src_r) % chunk_h, int(src_c) % chunk_w]]])
            out[out_r, out_c] = float(depth_from_encoded(encoded, depth_attrs)[0, 0])
    return out


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
    suffix = depth_path.suffix.lower()
    if depth_path.is_dir() or suffix == ".zarr":
        depth = load_zarr_preview(depth_path, max_width)
    elif suffix in {".tif", ".tiff"}:
        depth = load_tiff_preview(depth_path, max_width)
    else:
        raise RuntimeError(f"unsupported depth map format: {depth_path}")
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
    `Could not sample depth map. Install numcodecs for Zarr depths, tifffile for TIFF depths, or set OBJECT_SEARCH_WORKBENCH_PYTHON. ${errors.join(" | ")}`,
  );
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
      `Could not render depth map. Install numcodecs for Zarr depths, tifffile for TIFF depths, or set OBJECT_SEARCH_WORKBENCH_PYTHON. ${errors.join(" | ")}`,
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

async function resolveDepthPath(map: MapEntry, keyframeId: string): Promise<string | null> {
  const depthDir = path.join(map.path, "depths");
  const georefFilename = await lookupGeoRefKeyframeImageFilename(map, keyframeId);
  const stems = [
    georefFilename ? path.parse(georefFilename).name : null,
    keyframeId,
  ].filter((item): item is string => Boolean(item));
  const extensions = [".zarr", ".tif", ".tiff"];
  for (const stem of stems) {
    for (const extension of extensions) {
      const candidate = path.join(depthDir, `${stem}${extension}`);
      if (await pathExists(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function cubemapPixelToEquirect(
  u: number,
  v: number,
  faceName: string,
  faceSize: number,
  fovDeg: number,
): [number, number] {
  const f = 0.5 * faceSize / Math.tan(0.5 * fovDeg * Math.PI / 180);
  const c = (faceSize - 1) / 2;
  let x = (u - c) / f;
  let y = (v - c) / f;
  let z = 1;
  const norm = Math.hypot(x, y, z);
  x /= norm;
  y /= norm;
  z /= norm;
  let px = x;
  let py = y;
  let pz = z;
  switch (faceName) {
    case "back":
      px = -x;
      pz = -z;
      break;
    case "left":
      px = -z;
      pz = x;
      break;
    case "right":
      px = z;
      pz = -x;
      break;
    case "top":
      py = -z;
      pz = y;
      break;
    case "bottom":
      py = z;
      pz = -y;
      break;
  }
  const lon = Math.atan2(px, pz);
  const lat = Math.asin(Math.max(-1, Math.min(1, py)));
  return [((lon / (2 * Math.PI) + 0.5) % 1 + 1) % 1, lat / Math.PI + 0.5];
}

function bboxCorners(bbox: Bbox): Array<[number, number]> {
  return [
    [bbox[0], bbox[1]],
    [bbox[2], bbox[1]],
    [bbox[2], bbox[3]],
    [bbox[0], bbox[3]],
  ];
}

export async function indexDepthPinPayload(
  map: MapEntry,
  keyframeId: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const index = await requireIndex(map);
  if (index.manifest.geometry !== "cubemap") {
    throw new WorkbenchRouteError(404, `Unsupported index geometry for depth pin: ${String(index.manifest.geometry)}`);
  }
  const projection = typeof body.projection === "string" ? body.projection : "erp";
  const xRatio = asNumber(body.x_ratio, Number.NaN);
  const yRatio = asNumber(body.y_ratio, Number.NaN);
  if (!Number.isFinite(xRatio) || !Number.isFinite(yRatio)) {
    throw new WorkbenchRouteError(400, "Missing x_ratio/y_ratio click coordinates.");
  }

  let u = Math.max(0, Math.min(1, xRatio));
  let v = Math.max(0, Math.min(1, yRatio));
  let cutoutId: string | null = null;
  let faceName: string | null = null;
  if (projection === "cutout") {
    cutoutId = typeof body.cutout_id === "string" ? body.cutout_id : null;
    if (!cutoutId) {
      throw new WorkbenchRouteError(400, "Missing cutout_id for cutout depth pin.");
    }
    const cutout = index.cutoutsById.get(cutoutId);
    if (!cutout) {
      throw new WorkbenchRouteError(404, "Cutout id not found.");
    }
    if (cutout.keyframe_id !== keyframeId) {
      throw new WorkbenchRouteError(400, `Cutout ${cutoutId} belongs to keyframe ${cutout.keyframe_id}, not ${keyframeId}.`);
    }
    const faceSize = Math.max(1, asNumber(index.manifest.cubemap_face_size, 512));
    const fov = Math.max(1, asNumber(index.manifest.cubemap_fov_deg, 90));
    faceName = cubemapFace(cutoutId, index.manifest);
    [u, v] = cubemapPixelToEquirect(
      Math.max(0, Math.min(1, xRatio)) * (faceSize - 1),
      Math.max(0, Math.min(1, yRatio)) * (faceSize - 1),
      faceName,
      faceSize,
      fov,
    );
  } else if (projection !== "erp") {
    throw new WorkbenchRouteError(400, `Unsupported depth pin projection: ${projection}`);
  }

  const pose = await lookupGeoRefKeyframePose(map, keyframeId);
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
    cutout_id: cutoutId,
    projection,
    face: faceName,
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
  const pose = await lookupGeoRefKeyframePose(map, keyframeId);
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
  const pose = await lookupGeoRefKeyframePose(map, keyframeId);
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

export async function indexKeyframeEquirectPreviewPng(
  map: MapEntry,
  keyframeId: string,
  options: {
    selectedObjectId: string | null;
    drawBoxes: boolean;
    nmsIou: number;
    minBBoxArea: number;
    maxBBoxArea: number;
    showYolo: boolean;
    showGdino: boolean;
  },
): Promise<Buffer> {
  const index = await requireIndex(map);
  if (index.manifest.geometry !== "cubemap") {
    throw new WorkbenchRouteError(404, `Unsupported index geometry for ERP preview: ${String(index.manifest.geometry)}`);
  }
  const sourceImagePath = await resolveSourceImagePath(map, index, keyframeId);
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
  const faceSize = Math.max(1, asNumber(index.manifest.cubemap_face_size, 512));
  const fov = Math.max(1, asNumber(index.manifest.cubemap_fov_deg, 90));
  const keyframeObjects = await queryObjects(index.indexPath, `keyframe_id = ${sqlString(keyframeId)}`);
  const objects = postprocessByCutout(keyframeObjects, {
    nmsIou: options.nmsIou,
    minBBoxArea: options.minBBoxArea,
    maxBBoxArea: options.maxBBoxArea,
    showYolo: options.showYolo,
    showGdino: options.showGdino,
  });
  const drawArgs: string[] = [];
  if (scale !== 1) {
    drawArgs.push("-resize", `${MAX_PREVIEW_WIDTH}x`);
  }
  for (const item of objects) {
    const faceName = cubemapFace(item.cutout_id, index.manifest);
    const points = bboxCorners(item.bbox).map(([u, v]) => {
      const [eu, ev] = cubemapPixelToEquirect(u, v, faceName, faceSize, fov);
      return [eu * (size.width - 1) * scale, ev * (size.height - 1) * scale] as [number, number];
    });
    const color = item.object_id === options.selectedObjectId ? BOX_COLOR_SELECTED : BOX_COLOR;
    const width = item.object_id === options.selectedObjectId ? "4" : "2";
    const pointText = [...points, points[0]].map(([x, y]) => `${x},${y}`).join(" ");
    drawArgs.push("-fill", "none", "-stroke", color, "-strokewidth", width, "-draw", `polyline ${pointText}`);
  }
  return convertPng(sourceImagePath, drawArgs);
}
