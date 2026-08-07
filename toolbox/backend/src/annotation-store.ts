import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
} from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";

import type { MapEntry } from "./config.js";
import { WorkbenchRouteError } from "./workbench-index.js";

export const ANNOTATION_DB_FILENAME = "object-search-annotations.db";
const BENCHMARK_IMPORTED_KEY = "benchmark_geojson_imported";
const databaseCache = new Map<string, Database.Database>();
let cleanupRegistered = false;

const MANUAL_DETECTION_SCHEMA = `
  CREATE TABLE manual_detection (
    manual_detection_id INTEGER PRIMARY KEY,
    keyframe_id TEXT NOT NULL,
    geometry_type TEXT NOT NULL CHECK (geometry_type IN ('bbox', 'polygon')),
    geometry BLOB NOT NULL,
    class_id INTEGER REFERENCES manual_detection_classes(class_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    used_as_ground_truth INTEGER NOT NULL DEFAULT 1,
    lat REAL,
    lng REAL,
    alt REAL,
    level TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    note TEXT,
    enrichment_status TEXT NOT NULL DEFAULT 'pending'
      CHECK (enrichment_status IN ('pending', 'enriched', 'failed')),
    enriched_at TEXT
  )`;

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS detection_review (
    detection_review_id INTEGER PRIMARY KEY,
    target_type TEXT NOT NULL CHECK (target_type IN ('object')),
    target_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'false_positive'
      CHECK (status IN ('false_positive', 'true_positive')),
    created_at TEXT NOT NULL,
    created_by TEXT,
    note TEXT,
    enrichment_status TEXT NOT NULL DEFAULT 'pending'
      CHECK (enrichment_status IN ('pending', 'enriched', 'failed')),
    enriched_at TEXT
  );
  CREATE UNIQUE INDEX IF NOT EXISTS uq_detection_review_key
    ON detection_review (target_type, target_id, query);

  CREATE TABLE IF NOT EXISTS manual_detection_classes (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    class TEXT NOT NULL UNIQUE
  );

  CREATE TABLE IF NOT EXISTS manual_detection (
    manual_detection_id INTEGER PRIMARY KEY,
    keyframe_id TEXT NOT NULL,
    geometry_type TEXT NOT NULL CHECK (geometry_type IN ('bbox', 'polygon')),
    geometry BLOB NOT NULL,
    class_id INTEGER REFERENCES manual_detection_classes(class_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    used_as_ground_truth INTEGER NOT NULL DEFAULT 1,
    lat REAL,
    lng REAL,
    alt REAL,
    level TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    note TEXT,
    enrichment_status TEXT NOT NULL DEFAULT 'pending'
      CHECK (enrichment_status IN ('pending', 'enriched', 'failed')),
    enriched_at TEXT
  );

  CREATE TABLE IF NOT EXISTS enrichment_run (
    enrichment_run_id TEXT PRIMARY KEY,
    triggered_by TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    params BLOB,
    log TEXT
  );

  CREATE TABLE IF NOT EXISTS used_detection (
    enrichment_run_id TEXT NOT NULL REFERENCES enrichment_run(enrichment_run_id),
    target_type TEXT NOT NULL
      CHECK (target_type IN ('detection_review', 'manual_detection')),
    target_id INTEGER NOT NULL,
    PRIMARY KEY (enrichment_run_id, target_type, target_id)
  );

  CREATE TABLE IF NOT EXISTS ground_truth_point (
    ground_truth_point_id INTEGER PRIMARY KEY,
    lng REAL NOT NULL,
    lat REAL NOT NULL,
    alt REAL,
    level TEXT,
    class TEXT,
    prompt TEXT,
    accuracy REAL,
    extra_properties TEXT,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
`;

type ReviewStatus = "true_positive" | "false_positive";

type DetectionReviewRow = {
  detection_review_id: number;
  target_type: "object";
  target_id: number;
  query: string;
  status: ReviewStatus;
  created_at: string;
};

type ManualDetectionRow = {
  manual_detection_id: number;
  keyframe_id: string | number;
  geometry_type: "bbox" | "polygon";
  geometry: Buffer;
  class_name: string | null;
  query: string;
  used_as_ground_truth: number;
  lat: number | null;
  lng: number | null;
  alt: number | null;
  level: string | null;
  created_at: string;
};

type GroundTruthPointRow = {
  ground_truth_point_id: number;
  lng: number;
  lat: number;
  alt: number | null;
  level: string | null;
  class_name: string | null;
  prompt: string | null;
  accuracy: number | null;
  extra_properties: string | null;
};

type ReviewMutation = {
  targetId: number;
  query: string;
  status?: ReviewStatus;
};

export type FeatureCollection = {
  type: "FeatureCollection";
  features: Array<Record<string, unknown>>;
};

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function annotationDatabasePath(map: MapEntry): string {
  return path.join(map.path, ANNOTATION_DB_FILENAME);
}

function closeDatabaseAtPath(databasePath: string): void {
  const database = databaseCache.get(databasePath);
  if (!database) {
    return;
  }
  databaseCache.delete(databasePath);
  if (database.open) {
    database.close();
  }
}

/** Close one cached map connection, primarily for controlled DB replacement. */
export function closeAnnotationDatabase(map: MapEntry): void {
  closeDatabaseAtPath(path.resolve(annotationDatabasePath(map)));
}

/** Close every cached annotation connection during backend shutdown. */
export function closeAnnotationDatabases(): void {
  for (const databasePath of [...databaseCache.keys()]) {
    closeDatabaseAtPath(databasePath);
  }
}

function registerCleanup(): void {
  if (cleanupRegistered) {
    return;
  }
  cleanupRegistered = true;
  process.once("exit", closeAnnotationDatabases);
}

function tableColumns(database: Database.Database, table: string): Set<string> {
  const rows = database.prepare(`PRAGMA table_info(${table})`).all() as Array<{
    name: string;
  }>;
  return new Set(rows.map((row) => row.name));
}

function migrateLegacyManualDetections(database: Database.Database): void {
  const columns = tableColumns(database, "manual_detection");
  if (!columns.size || columns.has("class_id")) {
    return;
  }

  const rows = database.prepare("SELECT * FROM manual_detection").all() as Array<
    Record<string, unknown>
  >;
  const migrate = database.transaction(() => {
    database.exec(
      "INSERT OR IGNORE INTO manual_detection_classes (class) " +
        "SELECT DISTINCT class FROM manual_detection WHERE class IS NOT NULL",
    );
    database.exec("ALTER TABLE manual_detection RENAME TO manual_detection_old");
    database.exec(MANUAL_DETECTION_SCHEMA);
    const classId = database.prepare(
      "SELECT class_id FROM manual_detection_classes WHERE class = ?",
    );
    const insert = database.prepare(`
      INSERT INTO manual_detection (
        manual_detection_id, keyframe_id, geometry_type, geometry, class_id,
        query, used_as_ground_truth, lat, lng, alt, level, created_at,
        created_by, note, enrichment_status, enriched_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const row of rows) {
      const classRow =
        typeof row.class === "string"
          ? (classId.get(row.class) as { class_id: number } | undefined)
          : undefined;
      insert.run(
        row.manual_detection_id,
        row.keyframe_id,
        row.geometry_type,
        row.geometry,
        classRow?.class_id ?? null,
        row.query,
        row.used_as_ground_truth ?? 1,
        row.lat ?? null,
        row.lng ?? null,
        row.alt ?? null,
        row.level ?? null,
        row.created_at,
        row.created_by ?? null,
        row.note ?? null,
        row.enrichment_status ?? "pending",
        row.enriched_at ?? null,
      );
    }
    database.exec("DROP TABLE manual_detection_old");
  });
  migrate();
}

function addMissingManualDetectionColumns(database: Database.Database): void {
  const columns = tableColumns(database, "manual_detection");
  const additions: Array<[string, string]> = [
    ["used_as_ground_truth", "INTEGER NOT NULL DEFAULT 1"],
    ["lat", "REAL"],
    ["lng", "REAL"],
    ["alt", "REAL"],
    ["level", "TEXT"],
  ];
  for (const [name, declaration] of additions) {
    if (!columns.has(name)) {
      database.exec(`ALTER TABLE manual_detection ADD COLUMN ${name} ${declaration}`);
    }
  }
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function importBenchmarkGeoJsonOnce(
  database: Database.Database,
  map: MapEntry,
): void {
  const imported = database
    .prepare("SELECT value FROM meta WHERE key = ?")
    .get(BENCHMARK_IMPORTED_KEY);
  if (imported) {
    return;
  }
  const markImported = database.prepare(
    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
  );
  const annotationsPath = path.join(map.path, "benchmark", "annotations.geojson");
  if (!existsSync(annotationsPath)) {
    markImported.run(BENCHMARK_IMPORTED_KEY, nowIso());
    return;
  }
  try {
    copyFileSync(annotationsPath, `${annotationsPath}.pre-import.bak`);
  } catch {
    // The backup is best-effort; failure must not block annotation review.
  }

  try {
    const parsed = asObject(JSON.parse(readFileSync(annotationsPath, "utf8")));
    const features = Array.isArray(parsed?.features) ? parsed.features : [];
    const insert = database.prepare(`
      INSERT INTO ground_truth_point (
        lng, lat, alt, level, class, prompt, accuracy, extra_properties, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const transaction = database.transaction(() => {
      for (const rawFeature of features) {
        const feature = asObject(rawFeature);
        const properties = asObject(feature?.properties) ?? {};
        const geometry = asObject(feature?.geometry);
        const coordinates = Array.isArray(geometry?.coordinates)
          ? geometry.coordinates
          : [];
        if (
          properties.source === "annotation" ||
          geometry?.type !== "Point" ||
          typeof coordinates[0] !== "number" ||
          typeof coordinates[1] !== "number"
        ) {
          continue;
        }
        const {
          class: className,
          prompt,
          accuracy,
          level,
          altitude,
          ...extraProperties
        } = properties;
        const altitudeValue =
          typeof coordinates[2] === "number"
            ? coordinates[2]
            : typeof altitude === "number"
              ? altitude
              : null;
        insert.run(
          coordinates[0],
          coordinates[1],
          altitudeValue,
          level == null ? null : String(level),
          typeof className === "string" ? className : null,
          typeof prompt === "string" ? prompt : null,
          typeof accuracy === "number" ? accuracy : null,
          JSON.stringify(extraProperties),
          nowIso(),
        );
      }
      markImported.run(BENCHMARK_IMPORTED_KEY, nowIso());
    });
    transaction();
  } catch {
    markImported.run(BENCHMARK_IMPORTED_KEY, nowIso());
  }
}

function openAnnotationDatabase(map: MapEntry): Database.Database {
  const databasePath = path.resolve(annotationDatabasePath(map));
  const cached = databaseCache.get(databasePath);
  if (cached?.open) {
    return cached;
  }
  if (cached) {
    databaseCache.delete(databasePath);
  }

  mkdirSync(map.path, { recursive: true });
  const database = new Database(databasePath);
  try {
    database.pragma("journal_mode = WAL");
    database.pragma("foreign_keys = ON");
    database.exec(SCHEMA);
    database.prepare("DELETE FROM detection_review WHERE target_type = 'cluster'").run();
    migrateLegacyManualDetections(database);
    addMissingManualDetectionColumns(database);
    importBenchmarkGeoJsonOnce(database, map);
    databaseCache.set(databasePath, database);
    registerCleanup();
    return database;
  } catch (error) {
    database.close();
    throw error;
  }
}

function withDatabase<T>(map: MapEntry, operation: (database: Database.Database) => T): T {
  const database = openAnnotationDatabase(map);
  return operation(database);
}

function decodeGeometry(buffer: Buffer): number[][] {
  if (buffer.byteLength % 8 !== 0) {
    return [];
  }
  const points: number[][] = [];
  for (let offset = 0; offset < buffer.byteLength; offset += 8) {
    points.push([buffer.readFloatLE(offset), buffer.readFloatLE(offset + 4)]);
  }
  return points;
}

function manualDetectionRows(
  database: Database.Database,
  query: string | null,
): ManualDetectionRow[] {
  const where = query === null ? "" : " WHERE md.query = ?";
  const statement = database.prepare(`
    SELECT md.manual_detection_id, md.keyframe_id, md.geometry_type, md.geometry,
      mdc.class AS class_name, md.query, md.used_as_ground_truth,
      md.lat, md.lng, md.alt, md.level, md.created_at
    FROM manual_detection md
    LEFT JOIN manual_detection_classes mdc ON mdc.class_id = md.class_id
    ${where}
    ORDER BY md.manual_detection_id
  `);
  return (query === null ? statement.all() : statement.all(query)) as ManualDetectionRow[];
}

export function listAnnotations(map: MapEntry, query: string | null): Record<string, unknown> {
  return withDatabase(map, (database) => {
    const reviewStatement = database.prepare(`
      SELECT detection_review_id, target_type, target_id, query, status, created_at
      FROM detection_review
      ${query === null ? "" : "WHERE query = ?"}
      ORDER BY detection_review_id
    `);
    const detectionReviews = (
      query === null ? reviewStatement.all() : reviewStatement.all(query)
    ) as DetectionReviewRow[];
    const missedDetections = manualDetectionRows(database, query).map((row) => ({
      manual_detection_id: row.manual_detection_id,
      keyframe_id: row.keyframe_id,
      geometry_type: row.geometry_type,
      geometry: decodeGeometry(row.geometry),
      class_name: row.class_name,
      query: row.query,
      used_as_ground_truth: Boolean(row.used_as_ground_truth),
      lat: row.lat,
      lng: row.lng,
      alt: row.alt,
      level: row.level,
      created_at: row.created_at,
    }));
    const detectionClasses = database
      .prepare("SELECT class FROM manual_detection_classes ORDER BY class")
      .all()
      .map((row) => (row as { class: string }).class);
    return {
      detection_reviews: detectionReviews,
      missed_detections: missedDetections,
      detection_classes: detectionClasses,
    };
  });
}

export function parseReviewMutation(
  body: Record<string, unknown>,
  requireStatus: boolean,
): ReviewMutation {
  const targetId = Number(body.target_id);
  const query = typeof body.query === "string" ? body.query.trim() : "";
  const status = body.status;
  if (body.target_type !== "object" || !Number.isInteger(targetId) || !query) {
    throw new WorkbenchRouteError(422, "Invalid detection review payload.");
  }
  if (
    requireStatus &&
    status !== "true_positive" &&
    status !== "false_positive"
  ) {
    throw new WorkbenchRouteError(422, "Invalid detection review status.");
  }
  return {
    targetId,
    query,
    ...(requireStatus ? { status: status as ReviewStatus } : {}),
  };
}

export function upsertDetectionReview(map: MapEntry, review: ReviewMutation): number {
  if (!review.status) {
    throw new WorkbenchRouteError(422, "A detection review status is required.");
  }
  return withDatabase(map, (database) => {
    const result = database
      .prepare(`
        INSERT OR REPLACE INTO detection_review
          (target_type, target_id, query, status, created_at)
        VALUES ('object', ?, ?, ?, ?)
      `)
      .run(review.targetId, review.query, review.status, nowIso());
    return Number(result.lastInsertRowid);
  });
}

export function deleteDetectionReview(map: MapEntry, review: ReviewMutation): void {
  withDatabase(map, (database) => {
    database
      .prepare(`
        DELETE FROM detection_review
        WHERE target_type = 'object' AND target_id = ? AND query = ?
      `)
      .run(review.targetId, review.query);
  });
}

function parseExtraProperties(value: string | null): Record<string, unknown> {
  if (!value) {
    return {};
  }
  try {
    return asObject(JSON.parse(value)) ?? {};
  } catch {
    return {};
  }
}

export function buildGroundTruth(map: MapEntry): FeatureCollection {
  return withDatabase(map, (database) => {
    const features: Array<Record<string, unknown>> = [];
    for (const detection of manualDetectionRows(database, null)) {
      if (
        !detection.used_as_ground_truth ||
        detection.lat === null ||
        detection.lng === null
      ) {
        continue;
      }
      const className = (detection.class_name ?? "").trim() || detection.query.trim();
      if (!className) {
        continue;
      }
      const geometry = decodeGeometry(detection.geometry);
      const properties: Record<string, unknown> = {
        class: className,
        id: detection.manual_detection_id,
        source: "annotation",
        source_keyframe_id: String(detection.keyframe_id),
      };
      if (detection.query.trim()) {
        properties.prompt = detection.query.trim();
      }
      if (detection.level !== null && detection.level.trim()) {
        properties.level = detection.level;
      }
      if (detection.alt !== null) {
        properties.altitude = detection.alt;
      }
      if (geometry.length) {
        properties.source_erp_u =
          geometry.reduce((sum, point) => sum + point[0], 0) / geometry.length;
        properties.source_erp_v =
          geometry.reduce((sum, point) => sum + point[1], 0) / geometry.length;
      }
      const coordinates = [detection.lng, detection.lat];
      if (detection.alt !== null) {
        coordinates.push(detection.alt);
      }
      features.push({
        type: "Feature",
        id: `annotation-${detection.manual_detection_id}`,
        geometry: { type: "Point", coordinates },
        properties,
      });
    }

    const groundTruthPoints = database
      .prepare(`
        SELECT ground_truth_point_id, lng, lat, alt, level, class AS class_name,
          prompt, accuracy, extra_properties
        FROM ground_truth_point ORDER BY ground_truth_point_id
      `)
      .all() as GroundTruthPointRow[];
    for (const point of groundTruthPoints) {
      const properties = parseExtraProperties(point.extra_properties);
      if (point.class_name !== null) properties.class = point.class_name;
      if (point.prompt !== null) properties.prompt = point.prompt;
      if (point.accuracy !== null) properties.accuracy = point.accuracy;
      if (point.level !== null) properties.level = point.level;
      if (point.alt !== null) properties.altitude = point.alt;
      const coordinates = [point.lng, point.lat];
      if (point.alt !== null) coordinates.push(point.alt);
      features.push({
        type: "Feature",
        id: `gt-point-${point.ground_truth_point_id}`,
        geometry: { type: "Point", coordinates },
        properties,
      });
    }
    return { type: "FeatureCollection", features };
  });
}
