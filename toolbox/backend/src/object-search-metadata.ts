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

import { asyncBufferFromFile, parquetMetadataAsync, parquetRead } from "hyparquet";
import type { ParquetParsers } from "hyparquet";

import type { MapEntry } from "./config.js";

const METADATA_FILENAME = "metadata.parquet";
const OBJECT_SEARCH_DIRNAME = "object-search";
/**
 * The fixed-width store uses 77 bytes/row: 2×int32 + 6×float64 + 1 depth-mask
 * byte + 3×dictionary int32 + 2×thumbnail int32. At two million rows that is
 * 154 MB, before the comparatively small dictionaries and fallback strings.
 */
const MAX_ROW_COUNT = 2_000_000;
const MAX_CACHED_MAPS = 2;
const INT32_UPPER_BOUND = 2 ** 31;
const INT32_LOWER_BOUND = -(2 ** 31);
const NULL_DICTIONARY_CODE = -1;
const THUMBNAIL_PATTERN = /^(.*\/)(\d{6})\.jpg$/;
const THUMBNAIL_BASENAME_BASE = 1_000_000;
const UTF8_DECODER = new TextDecoder();

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

type DictionaryColumn = {
  dictionary: string[];
  codes: Int32Array;
};

type ThumbnailColumn =
  | {
      storage: "prefix-basename";
      prefixes: string[];
      prefixCodes: Int32Array;
      basenames: Int32Array;
    }
  | {
      storage: "plain";
      values: (string | null)[];
    };

export type MetadataColumns = {
  rowIndex: Int32Array;
  videoKeyframeId: Int32Array;
  thetaCenter: Float64Array;
  phiCenter: Float64Array;
  angularWidth: Float64Array;
  angularHeight: Float64Array;
  detectorSource: DictionaryColumn;
  label: DictionaryColumn;
  detectionScore: Float64Array;
  vkImagePath: DictionaryColumn;
  thumbnailKey: ThumbnailColumn;
  depth: Float64Array;
  /** One means parquet supplied a value, including NaN; zero means parquet NULL. */
  depthPresent: Uint8Array;
};

/** Rows of one keyframe: `[start, end)` into the loaded column store. */
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
  columns: MetadataColumns;
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

function assertInt32(value: unknown, columnName: string, metadataPath: string): number {
  const parsed = asNumber(value);
  if (
    !Number.isSafeInteger(parsed)
    || parsed < INT32_LOWER_BOUND
    || parsed >= INT32_UPPER_BOUND
  ) {
    throw new MetadataError(
      500,
      `${metadataPath}: ${columnName} value ${String(value)} is not a safe int below 2^31.`,
    );
  }
  return parsed;
}

function dictionaryValue(column: DictionaryColumn, rowPosition: number): string | null {
  const code = column.codes[rowPosition];
  return code === NULL_DICTIONARY_CODE ? null : (column.dictionary[code] ?? null);
}

function thumbnailValue(column: ThumbnailColumn, rowPosition: number): string | null {
  if (column.storage === "plain") {
    return column.values[rowPosition] ?? null;
  }
  const prefixCode = column.prefixCodes[rowPosition];
  if (prefixCode === NULL_DICTIONARY_CODE) {
    return null;
  }
  const prefix = column.prefixes[prefixCode];
  return prefix === undefined
    ? null
    : `${prefix}${String(column.basenames[rowPosition]).padStart(6, "0")}.jpg`;
}

function rowAtPosition(metadata: LoadedMetadata, rowPosition: number): MetadataRow {
  const { columns } = metadata;
  const depth = columns.depthPresent[rowPosition]
    ? asNullableNumber(columns.depth[rowPosition])
    : null;
  return {
    rowIndex: columns.rowIndex[rowPosition],
    videoKeyframeId: columns.videoKeyframeId[rowPosition],
    thetaCenter: columns.thetaCenter[rowPosition],
    phiCenter: columns.phiCenter[rowPosition],
    angularWidth: columns.angularWidth[rowPosition],
    angularHeight: columns.angularHeight[rowPosition],
    detectorSource: dictionaryValue(columns.detectorSource, rowPosition) ?? "",
    label: dictionaryValue(columns.label, rowPosition),
    detectionScore: asNullableNumber(columns.detectionScore[rowPosition]),
    thumbnailKey: thumbnailValue(columns.thumbnailKey, rowPosition),
    depth,
  };
}

export type MetadataColumnFilter = {
  detectorSource: string | null;
  labelQuery: string | null;
  withDepthOnly: boolean;
};

/** Test route filters directly against columns without constructing a row object. */
export function metadataRowMatches(
  metadata: LoadedMetadata,
  rowPosition: number,
  filter: MetadataColumnFilter,
): boolean {
  const { columns } = metadata;
  if (
    filter.detectorSource
    && dictionaryValue(columns.detectorSource, rowPosition) !== filter.detectorSource
  ) {
    return false;
  }
  if (
    filter.withDepthOnly
    && (!columns.depthPresent[rowPosition] || !Number.isFinite(columns.depth[rowPosition]))
  ) {
    return false;
  }
  return !filter.labelQuery
    || (dictionaryValue(columns.label, rowPosition) ?? "")
      .toLowerCase()
      .includes(filter.labelQuery);
}

/**
 * Group rows per keyframe, asserting that each keyframe's rows are contiguous.
 *
 * `row_index` is a global monotonic counter over `image_entries`, so contiguity
 * holds by construction. If it does not, the file did not come out of this
 * pipeline and every other assumption here (thumbnail naming, embedding alignment)
 * is suspect too — better to say so than to serve a plausible mixture.
 */
function buildRanges(columns: MetadataColumns, metadataPath: string): {
  keyframeIds: number[];
  rangeByKeyframe: Map<number, KeyframeRowRange>;
} {
  const keyframeIds: number[] = [];
  const rangeByKeyframe = new Map<number, KeyframeRowRange>();
  let current: KeyframeRowRange | null = null;
  for (let index = 0; index < columns.videoKeyframeId.length; index += 1) {
    const keyframeId = columns.videoKeyframeId[index];
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
    if (columns.depthPresent[index] && Number.isFinite(columns.depth[index])) {
      current.withDepth += 1;
    }
  }
  return { keyframeIds, rangeByKeyframe };
}

async function readColumn(
  file: Awaited<ReturnType<typeof asyncBufferFromFile>>,
  fileMetadata: Awaited<ReturnType<typeof parquetMetadataAsync>>,
  columnName: string,
  onValue: (value: unknown, rowPosition: number) => void,
  stringFromBytes?: (bytes: Uint8Array) => unknown,
): Promise<void> {
  let decodedCount = 0;
  let callbackError: unknown = null;
  await parquetRead({
    file,
    metadata: fileMetadata,
    columns: [columnName],
    ...(stringFromBytes
      ? {
          parsers: { stringFromBytes } as ParquetParsers,
        }
      : {}),
    onChunk: ({ columnData, rowStart }) => {
      for (let offset = 0; offset < columnData.length; offset += 1) {
        if (callbackError === null) {
          try {
            onValue(columnData[offset] as unknown, rowStart + offset);
          } catch (error: unknown) {
            callbackError = error;
          }
        }
        decodedCount += 1;
      }
    },
  });
  if (callbackError !== null) {
    throw callbackError;
  }
  if (decodedCount !== Number(fileMetadata.num_rows)) {
    throw new MetadataError(
      500,
      `${columnName} decoded ${decodedCount} rows; expected ${fileMetadata.num_rows}.`,
    );
  }
}

async function readInt32Column(
  file: Awaited<ReturnType<typeof asyncBufferFromFile>>,
  fileMetadata: Awaited<ReturnType<typeof parquetMetadataAsync>>,
  columnName: string,
  metadataPath: string,
  rowCount: number,
): Promise<Int32Array> {
  const values = new Int32Array(rowCount);
  await readColumn(file, fileMetadata, columnName, (value, rowPosition) => {
    values[rowPosition] = assertInt32(value, columnName, metadataPath);
  });
  return values;
}

async function readFloat64Column(
  file: Awaited<ReturnType<typeof asyncBufferFromFile>>,
  fileMetadata: Awaited<ReturnType<typeof parquetMetadataAsync>>,
  columnName: string,
  rowCount: number,
  missingValue = Number.NaN,
  present?: Uint8Array,
): Promise<Float64Array> {
  const values = new Float64Array(rowCount);
  values.fill(missingValue);
  await readColumn(file, fileMetadata, columnName, (value, rowPosition) => {
    if (value !== null && value !== undefined && present) {
      present[rowPosition] = 1;
    }
    values[rowPosition] = value === null || value === undefined
      ? missingValue
      : asNumber(value);
  });
  return values;
}

async function readDictionaryColumn(
  file: Awaited<ReturnType<typeof asyncBufferFromFile>>,
  fileMetadata: Awaited<ReturnType<typeof parquetMetadataAsync>>,
  columnName: string,
  rowCount: number,
  nullable: boolean,
): Promise<DictionaryColumn> {
  const dictionary: string[] = [];
  const codeByValue = new Map<string, number>();
  const codes = new Int32Array(rowCount);
  codes.fill(NULL_DICTIONARY_CODE);
  const stringFromBytes = (bytes: Uint8Array): number => {
    const decoded = UTF8_DECODER.decode(bytes);
    const text = nullable ? asNullableText(decoded) : decoded;
    if (text === null) {
      return NULL_DICTIONARY_CODE;
    }
    let code = codeByValue.get(text);
    if (code === undefined) {
      code = dictionary.length;
      dictionary.push(text);
      codeByValue.set(text, code);
    }
    return code;
  };
  await readColumn(
    file,
    fileMetadata,
    columnName,
    (value, rowPosition) => {
      codes[rowPosition] = value === null || value === undefined
        ? NULL_DICTIONARY_CODE
        : asNumber(value);
    },
    stringFromBytes,
  );
  return { dictionary, codes };
}

async function readThumbnailColumn(
  file: Awaited<ReturnType<typeof asyncBufferFromFile>>,
  fileMetadata: Awaited<ReturnType<typeof parquetMetadataAsync>>,
  metadataPath: string,
  rowCount: number,
): Promise<ThumbnailColumn> {
  const prefixes: string[] = [];
  const codeByPrefix = new Map<string, number>();
  const prefixCodes = new Int32Array(rowCount);
  prefixCodes.fill(NULL_DICTIONARY_CODE);
  const basenames = new Int32Array(rowCount);
  let plainValues: (string | null)[] | null = null;
  let parserUsesPlainStrings = false;

  const stringFromBytes = (bytes: Uint8Array): number | string => {
    const text = UTF8_DECODER.decode(bytes).trim();
    if (!text.length) {
      return text;
    }
    if (parserUsesPlainStrings) {
      return text;
    }
    const match = THUMBNAIL_PATTERN.exec(text);
    if (!match) {
      parserUsesPlainStrings = true;
      return text;
    }
    const prefix = match[1];
    let prefixCode = codeByPrefix.get(prefix);
    if (prefixCode === undefined) {
      prefixCode = prefixes.length;
      prefixes.push(prefix);
      codeByPrefix.set(prefix, prefixCode);
    }
    return prefixCode * THUMBNAIL_BASENAME_BASE + Number(match[2]);
  };

  await readColumn(file, fileMetadata, "thumbnail_key", (value, rowPosition) => {
    if (value === null || value === undefined) {
      return;
    }
    if (plainValues) {
      if (typeof value === "number") {
        const prefixCode = Math.floor(value / THUMBNAIL_BASENAME_BASE);
        plainValues[rowPosition] = `${prefixes[prefixCode]}`
          + `${String(value % THUMBNAIL_BASENAME_BASE).padStart(6, "0")}.jpg`;
      } else {
        plainValues[rowPosition] = asNullableText(value);
      }
      return;
    }
    if (typeof value === "number") {
      const prefixCode = Math.floor(value / THUMBNAIL_BASENAME_BASE);
      prefixCodes[rowPosition] = prefixCode;
      basenames[rowPosition] = value % THUMBNAIL_BASENAME_BASE;
      return;
    }
    const text = asNullableText(value);
    if (text !== null) {
      plainValues = Array.from({ length: rowCount }, (_, priorPosition) =>
        priorPosition < rowPosition
          ? thumbnailValue(
              { storage: "prefix-basename", prefixes, prefixCodes, basenames },
              priorPosition,
            )
          : null,
      );
      plainValues[rowPosition] = asNullableText(value);
      console.warn(
        `${metadataPath}: thumbnail_key does not use <prefix>/<NNNNNN>.jpg; `
        + "storing that column as plain strings.",
      );
    }
  }, stringFromBytes);

  return plainValues
    ? { storage: "plain", values: plainValues }
    : { storage: "prefix-basename", prefixes, prefixCodes, basenames };
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
      `${metadataPath} holds ${declaredRows} rows; the columnar metadata store is `
      + `limited to ${MAX_ROW_COUNT} rows.`,
    );
  }

  const columns = new Set(
    fileMetadata.schema.map((element) => element.name).filter(Boolean),
  );
  const hasColumn = (columnName: string): boolean => columns.has(columnName);
  const emptyDictionary = (): DictionaryColumn => ({
    dictionary: [],
    codes: new Int32Array(declaredRows).fill(NULL_DICTIONARY_CODE),
  });
  const depthPresent = new Uint8Array(declaredRows);
  const metadataColumns: MetadataColumns = {
    rowIndex: await readInt32Column(
      file,
      fileMetadata,
      "row_index",
      metadataPath,
      declaredRows,
    ),
    videoKeyframeId: await readInt32Column(
      file,
      fileMetadata,
      "video_keyframe_id",
      metadataPath,
      declaredRows,
    ),
    thetaCenter: await readFloat64Column(
      file,
      fileMetadata,
      "theta_center",
      declaredRows,
    ),
    phiCenter: await readFloat64Column(
      file,
      fileMetadata,
      "phi_center",
      declaredRows,
    ),
    angularWidth: await readFloat64Column(
      file,
      fileMetadata,
      "angular_width",
      declaredRows,
    ),
    angularHeight: await readFloat64Column(
      file,
      fileMetadata,
      "angular_height",
      declaredRows,
    ),
    detectorSource: await readDictionaryColumn(
      file,
      fileMetadata,
      "detector_source",
      declaredRows,
      false,
    ),
    label: await readDictionaryColumn(
      file,
      fileMetadata,
      "label",
      declaredRows,
      true,
    ),
    detectionScore: await readFloat64Column(
      file,
      fileMetadata,
      "detection_score",
      declaredRows,
    ),
    vkImagePath: hasColumn("vk_image_path")
      ? await readDictionaryColumn(
          file,
          fileMetadata,
          "vk_image_path",
          declaredRows,
          false,
        )
      : emptyDictionary(),
    thumbnailKey: hasColumn("thumbnail_key")
      ? await readThumbnailColumn(file, fileMetadata, metadataPath, declaredRows)
      : {
          storage: "prefix-basename",
          prefixes: [],
          prefixCodes: new Int32Array(declaredRows).fill(NULL_DICTIONARY_CODE),
          basenames: new Int32Array(declaredRows),
        },
    depth: hasColumn("depth")
      ? await readFloat64Column(
          file,
          fileMetadata,
          "depth",
          declaredRows,
          Number.NaN,
          depthPresent,
        )
      : new Float64Array(declaredRows).fill(Number.NaN),
    depthPresent,
  };

  const { keyframeIds, rangeByKeyframe } = buildRanges(metadataColumns, metadataPath);
  const detectorSourceCounts: Record<string, number> = {};
  let withDepthCount = 0;
  for (let rowPosition = 0; rowPosition < declaredRows; rowPosition += 1) {
    const detectorSource = dictionaryValue(metadataColumns.detectorSource, rowPosition) ?? "";
    const key = detectorSource || "unknown";
    detectorSourceCounts[key] = (detectorSourceCounts[key] ?? 0) + 1;
    if (
      metadataColumns.depthPresent[rowPosition]
      && Number.isFinite(metadataColumns.depth[rowPosition])
    ) {
      withDepthCount += 1;
    }
  }

  return {
    metadataPath,
    mtimeMs,
    rowCount: declaredRows,
    columns: metadataColumns,
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
  // Two 935k-row maps are about 144 MB of fixed-width data. A larger entry-count
  // cache would retain hundreds of MB even though the explorer normally uses one map.
  if (metadataCache.size > MAX_CACHED_MAPS) {
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
  return range ? rowSlice(metadata, range.rowStart, range.rowEnd) : [];
}

/** Materialise only `[start, end)` from the column store. */
export function rowSlice(
  metadata: LoadedMetadata,
  start: number,
  end: number,
): MetadataRow[] {
  return Array.from(iterateMetadataRows(metadata, start, end));
}

/** Iterate rows on demand without retaining a materialised table. */
export function* iterateMetadataRows(
  metadata: LoadedMetadata,
  start = 0,
  end = metadata.rowCount,
): Generator<MetadataRow> {
  const boundedStart = Math.max(0, Math.min(metadata.rowCount, Math.trunc(start)));
  const boundedEnd = Math.max(boundedStart, Math.min(metadata.rowCount, Math.trunc(end)));
  for (let rowPosition = boundedStart; rowPosition < boundedEnd; rowPosition += 1) {
    yield rowAtPosition(metadata, rowPosition);
  }
}

/** One row by its `row_index`, or null. */
export function rowByIndex(
  metadata: LoadedMetadata,
  rowIndex: number,
): MetadataRow | null {
  // row_index is a dense 0-based counter, so try the direct hit before scanning.
  const directRowIndex = metadata.columns.rowIndex[rowIndex];
  if (directRowIndex === rowIndex) {
    return rowAtPosition(metadata, rowIndex);
  }
  const rowPosition = metadata.columns.rowIndex.findIndex((value) => value === rowIndex);
  return rowPosition >= 0 ? rowAtPosition(metadata, rowPosition) : null;
}
