/**
 * Read a map's v2 object-search metadata: `{map}/object-search/metadata.parquet`.
 *
 * This replaces `loadIndex` and the `sqlite3`-shelling helpers it needed. The v2
 * pipeline ships **no index file** — the vector index lives in pgvector — so the
 * only browsable artefact is the parquet `prepare` wrote, read here natively with
 * `hyparquet` (pure JS, no runtime deps; Snappy is built in, and pyarrow writes
 * Snappy by default).
 *
 * The old cutout → objects hierarchy is gone with it: in v2 a parquet row **is**
 * both the cutout and the detection, so this module exposes a flat row list plus a
 * per-keyframe range index rather than `cutoutIdsByKeyframe` /
 * `objectCountByCutout`.
 *
 * ## Traps this module exists to close
 *
 * - **INT64 comes back as `BigInt`.** `row_index` and `video_keyframe_id` are
 *   `Number()`-ed here, at the reader boundary. Left as BigInt they would never
 *   match a `Map<number, …>` key: empty table, no error anywhere.
 * - **A rejected promise must not poison the cache.** `loadIndex` cached the
 *   promise and never deleted it on failure, so one bad read meant "no index" until
 *   restart. The `.catch` below removes the entry, as
 *   `indexKeyframeEquirectPreviewPng` already did for image previews.
 * - **Multiple captures collide.** `prepare_postprocess` hardcodes the thumbnail
 *   prefix and `row_index` restarts at 0 per run, so two captures produce the same
 *   `thumbnail_key` and overwrite each other on disk. Refused loudly rather than
 *   silently taking the first.
 * - **A non-postprocessed parquet looks like a failed capture.** No `depth`, no
 *   `thumbnail_key`; every preview 404s. Reported as `postprocessed: false` so the
 *   UI can name the command to run instead.
 */

import { access, stat } from "node:fs/promises";
import { readdir } from "node:fs/promises";
import path from "node:path";

import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from "hyparquet";

import type { MapEntry } from "./config.js";

const METADATA_FILENAME = "metadata.parquet";
const OBJECT_SEARCH_DIRNAME = "object-search";
/** Above this the whole-file read stops being a reasonable thing to do in one heap. */
const MAX_ROW_COUNT = 500_000;

export class MetadataError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** One `metadata.parquet` row, with ids narrowed to `number` and blanks to `null`. */
export type MetadataRow = {
  rowIndex: number;
  videoKeyframeId: number;
  thetaCenter: number;
  phiCenter: number;
  angularWidth: number;
  angularHeight: number;
  detectorSource: string;
  label: string | null;
  detectionScore: number | null;
  /** Path relative to the map directory, or null when crops were not stored. */
  thumbnailKey: string | null;
  /** Metres at the view centre; null when no depth map covered it. */
  depth: number | null;
};

/** Rows of one keyframe: `[start, end)` into {@link LoadedMetadata.rows}. */
export type KeyframeRowRange = {
  keyframeId: number;
  rowStart: number;
  rowEnd: number;
  withDepth: number;
};

export type LoadedMetadata = {
  metadataPath: string;
  mtimeMs: number;
  rowCount: number;
  rows: MetadataRow[];
  keyframeIds: number[];
  rangeByKeyframe: Map<number, KeyframeRowRange>;
  /** Whether `prepare_postprocess` has run (`thumbnail_key` + `depth` present). */
  postprocessed: boolean;
  detectorSourceCounts: Record<string, number>;
  withDepthCount: number;
};

async function pathExists(target: string): Promise<boolean> {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

/**
 * Locate a map's `metadata.parquet`, mirroring `ingest_cli.discover_capture_dirs`.
 *
 * `checkedPaths` is kept (and surfaced by the explorer) because it is what
 * distinguishes "this map was never prepared" from "I looked in the wrong place".
 */
export async function resolveMetadataPath(map: MapEntry): Promise<{
  metadataPath: string | null;
  checkedPaths: string[];
  captureCount: number;
}> {
  const root = path.resolve(map.path, OBJECT_SEARCH_DIRNAME);
  const direct = path.join(root, METADATA_FILENAME);
  const checkedPaths = [direct];
  if (await pathExists(direct)) {
    return { metadataPath: direct, checkedPaths, captureCount: 1 };
  }

  let entries: string[];
  try {
    entries = (await readdir(root)).sort();
  } catch {
    return { metadataPath: null, checkedPaths, captureCount: 0 };
  }
  const captures: string[] = [];
  for (const entry of entries) {
    const candidate = path.join(root, entry, METADATA_FILENAME);
    checkedPaths.push(candidate);
    if (await pathExists(candidate)) {
      captures.push(candidate);
    }
  }
  return {
    metadataPath: captures[0] ?? null,
    checkedPaths,
    captureCount: captures.length,
  };
}

function asNumber(value: unknown): number {
  // BigInt (INT64) must be converted here or every numeric join downstream misses.
  return typeof value === "bigint" ? Number(value) : Number(value);
}

function asNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = asNumber(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asText(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

function asNullableText(value: unknown): string | null {
  const text = asText(value).trim();
  return text.length ? text : null;
}

function toMetadataRow(raw: Record<string, unknown>): MetadataRow {
  return {
    rowIndex: asNumber(raw.row_index),
    videoKeyframeId: asNumber(raw.video_keyframe_id),
    thetaCenter: asNumber(raw.theta_center),
    phiCenter: asNumber(raw.phi_center),
    angularWidth: asNumber(raw.angular_width),
    angularHeight: asNumber(raw.angular_height),
    detectorSource: asText(raw.detector_source),
    label: asNullableText(raw.label),
    detectionScore: asNullableNumber(raw.detection_score),
    thumbnailKey: asNullableText(raw.thumbnail_key),
    depth: asNullableNumber(raw.depth),
  };
}

/**
 * Group rows per keyframe, asserting that each keyframe's rows are contiguous.
 *
 * `row_index` is a global monotonic counter over `image_entries`, so contiguity
 * holds by construction. If it does not, the file did not come out of this
 * pipeline and every other assumption here (thumbnail naming, embedding alignment)
 * is suspect too — better to say so than to serve a plausible mixture.
 */
function buildRanges(rows: MetadataRow[], metadataPath: string): {
  keyframeIds: number[];
  rangeByKeyframe: Map<number, KeyframeRowRange>;
} {
  const keyframeIds: number[] = [];
  const rangeByKeyframe = new Map<number, KeyframeRowRange>();
  let current: KeyframeRowRange | null = null;
  for (let index = 0; index < rows.length; index += 1) {
    const keyframeId = rows[index].videoKeyframeId;
    if (current && current.keyframeId === keyframeId) {
      current.rowEnd = index + 1;
    } else {
      if (rangeByKeyframe.has(keyframeId)) {
        throw new MetadataError(
          500,
          `${metadataPath}: rows of keyframe ${keyframeId} are not contiguous `
          + `(they resume at row ${index}). This file was not produced by the v2 `
          + "prepare pipeline.",
        );
      }
      current = { keyframeId, rowStart: index, rowEnd: index + 1, withDepth: 0 };
      rangeByKeyframe.set(keyframeId, current);
      keyframeIds.push(keyframeId);
    }
    if (rows[index].depth !== null) {
      current.withDepth += 1;
    }
  }
  return { keyframeIds, rangeByKeyframe };
}

async function loadMetadataUncached(
  metadataPath: string,
  mtimeMs: number,
): Promise<LoadedMetadata> {
  const file = await asyncBufferFromFile(metadataPath);
  const fileMetadata = await parquetMetadataAsync(file);
  const declaredRows = Number(fileMetadata.num_rows);
  if (declaredRows > MAX_ROW_COUNT) {
    throw new MetadataError(
      507,
      `${metadataPath} holds ${declaredRows} rows; this explorer reads the whole `
      + `file into memory and refuses above ${MAX_ROW_COUNT}.`,
    );
  }

  const columns = new Set(
    fileMetadata.schema.map((element) => element.name).filter(Boolean),
  );
  const rawRows = (await parquetReadObjects({ file })) as Record<string, unknown>[];
  const rows = rawRows.map(toMetadataRow);
  if (rows.length && !Number.isFinite(rows[0].rowIndex)) {
    // The BigInt trap, caught at the one place it is still cheap to notice.
    throw new MetadataError(500, `${metadataPath}: row_index did not decode to a number.`);
  }

  const { keyframeIds, rangeByKeyframe } = buildRanges(rows, metadataPath);
  const detectorSourceCounts: Record<string, number> = {};
  let withDepthCount = 0;
  for (const row of rows) {
    const key = row.detectorSource || "unknown";
    detectorSourceCounts[key] = (detectorSourceCounts[key] ?? 0) + 1;
    if (row.depth !== null) {
      withDepthCount += 1;
    }
  }

  return {
    metadataPath,
    mtimeMs,
    rowCount: rows.length,
    rows,
    keyframeIds,
    rangeByKeyframe,
    postprocessed: columns.has("depth") && columns.has("thumbnail_key"),
    detectorSourceCounts,
    withDepthCount,
  };
}

// Keyed by path + mtime, and — unlike the SQLite cache this replaces — the entry is
// dropped when the promise rejects.
const metadataCache = new Map<string, Promise<LoadedMetadata>>();

export async function loadMetadata(metadataPath: string): Promise<LoadedMetadata> {
  const resolved = path.resolve(metadataPath);
  const mtimeMs = (await stat(resolved)).mtimeMs;
  const cacheKey = `${resolved}:${mtimeMs}`;
  const cached = metadataCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const promise = loadMetadataUncached(resolved, mtimeMs).catch((error: unknown) => {
    metadataCache.delete(cacheKey);
    throw error;
  });
  metadataCache.set(cacheKey, promise);
  if (metadataCache.size > 8) {
    metadataCache.delete(metadataCache.keys().next().value as string);
  }
  return promise;
}

/**
 * The map's metadata, or a 404/409 explaining what to do about its absence.
 *
 * Multi-capture is refused here rather than in `resolveMetadataPath` so that the
 * status route can still report `capture_count` without throwing.
 */
export async function requireMetadata(map: MapEntry): Promise<LoadedMetadata> {
  const { metadataPath, checkedPaths, captureCount } = await resolveMetadataPath(map);
  if (!metadataPath) {
    throw new MetadataError(
      404,
      "No object-search metadata for this map. Run scripts/build-index.sh to produce "
      + `${OBJECT_SEARCH_DIRNAME}/${METADATA_FILENAME}. Looked in: ${checkedPaths.join(", ")}.`,
    );
  }
  if (captureCount > 1) {
    throw new MetadataError(
      409,
      `${captureCount} capture directories under ${path.dirname(path.dirname(metadataPath))} `
      + "each hold a metadata.parquet. They cannot be browsed together: row_index restarts "
      + "at 0 per capture and prepare_postprocess writes the same thumbnail_key prefix for "
      + "all of them, so their thumbnails overwrite each other on disk. Keep one capture "
      + "directory, or re-run prepare into a single output directory.",
    );
  }
  return loadMetadata(metadataPath);
}

/** Rows of one keyframe, as a slice of the loaded table. */
export function rowsForKeyframe(
  metadata: LoadedMetadata,
  keyframeId: number,
): MetadataRow[] {
  const range = metadata.rangeByKeyframe.get(keyframeId);
  return range ? metadata.rows.slice(range.rowStart, range.rowEnd) : [];
}

/** One row by its `row_index`, or null. */
export function rowByIndex(
  metadata: LoadedMetadata,
  rowIndex: number,
): MetadataRow | null {
  // row_index is a dense 0-based counter, so try the direct hit before scanning.
  const direct = metadata.rows[rowIndex];
  if (direct && direct.rowIndex === rowIndex) {
    return direct;
  }
  return metadata.rows.find((row) => row.rowIndex === rowIndex) ?? null;
}
