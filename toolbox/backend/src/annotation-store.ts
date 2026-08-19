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
const OFFICE_IMPORTED_KEY = "office_geojson_imported";
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

  /**
   * The *true* grouping of detections, as a human reads it — the reference partition
   * that says which detections are one physical object.
   *
   * Deliberately query-independent, unlike 'detection_review': "these two detections
   * are the same extinguisher" is a fact about the map, not about a prompt. The
   * unique index over the detection key is the partition constraint itself — a
   * detection belongs to exactly one group, and re-labelling moves it.
   *
   * 'group_name' NULL means "not an object" (a box on a wall, two objects at once):
   * an explicit reject, so junk does not have to be filed into some group to be
   * annotated. 'row_index' is resolved at write time when the parquet can answer,
   * because offline scripts join on it; '(keyframe_id, theta, phi)' stays
   * authoritative since pgvector cannot produce a row index.
   *
   * Not wired into the benchmark's ground truth — that stays 'ground_truth_point'.
   */
  CREATE TABLE IF NOT EXISTS detection_group_label (
    detection_group_label_id INTEGER PRIMARY KEY,
    keyframe_id TEXT NOT NULL,
    theta_center REAL NOT NULL,
    phi_center REAL NOT NULL,
    row_index INTEGER,
    group_name TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT
  );
  CREATE UNIQUE INDEX IF NOT EXISTS uq_detection_group_label_key
    ON detection_group_label (keyframe_id, theta_center, phi_center);

  /**
   * The Annotation office's class palette: name, colour, default prompt.
   *
   * It exists so an empty class survives a reload. Colour and prompt could be read
   * off the annotations that use a class, but a class created and not yet drawn on
   * would then vanish — which is exactly the moment the palette is being set up.
   *
   * Distinct from 'manual_detection_classes', which names image-space detections and
   * carries no colour. Merging the two would tie the office's palette to a table the
   * benchmark reads.
   */
  CREATE TABLE IF NOT EXISTS annotation_class (
    annotation_class_id INTEGER PRIMARY KEY,
    class TEXT NOT NULL,
    annotation_type TEXT NOT NULL CHECK (annotation_type IN ('point', 'polygon')),
    color TEXT NOT NULL,
    prompt TEXT,
    created_at TEXT NOT NULL
  );
  CREATE UNIQUE INDEX IF NOT EXISTS uq_annotation_class_key
    ON annotation_class (class, annotation_type);

  /**
   * Polygon annotations from the office.
   *
   * Points go to 'ground_truth_point' — they *are* benchmark ground truth, which is
   * the whole reason the office exists. A polygon is not: it has no single position
   * to score a localization against, and 'ground_truth_point' has no geometry
   * column. It gets its own table rather than a nullable geometry on that one, so
   * nothing reading ground truth has to learn to skip rows.
   *
   * 'ring' is a JSON array of [lng, lat] pairs, open — the first vertex is
   * implicitly reconnected to the last, matching what the livemap draws.
   */
  CREATE TABLE IF NOT EXISTS annotation_polygon (
    annotation_polygon_id INTEGER PRIMARY KEY,
    ring TEXT NOT NULL,
    class TEXT,
    prompt TEXT,
    level TEXT,
    alt REAL,
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

const SOURCE_KEY_SQL = `COALESCE(
  json_extract(extra_properties, '$.source.videoKeyframeUuid'),
  json_extract(extra_properties, '$.source.video_keyframe_uuid'),
  json_extract(extra_properties, '$.source.keyframeId'),
  json_extract(extra_properties, '$.source_keyframe_id')
)`;
const SOURCE_ERP_U_SQL = `ROUND(COALESCE(
  json_extract(extra_properties, '$.source.erpU'),
  json_extract(extra_properties, '$.source.erp_u'),
  json_extract(extra_properties, '$.source_erp_u')
), 6)`;
const SOURCE_ERP_V_SQL = `ROUND(COALESCE(
  json_extract(extra_properties, '$.source.erpV'),
  json_extract(extra_properties, '$.source.erp_v'),
  json_extract(extra_properties, '$.source_erp_v')
), 6)`;

/** Remove the historical double inserts, then make one image click idempotent. */
function enforceGroundTruthClickUniqueness(database: Database.Database): void {
  database.exec(`
    DELETE FROM ground_truth_point
    WHERE ground_truth_point_id NOT IN (
      SELECT MIN(ground_truth_point_id)
      FROM ground_truth_point
      WHERE json_valid(extra_properties)
        AND ${SOURCE_KEY_SQL} IS NOT NULL
        AND ${SOURCE_ERP_U_SQL} IS NOT NULL
        AND ${SOURCE_ERP_V_SQL} IS NOT NULL
      GROUP BY class, ${SOURCE_KEY_SQL}, ${SOURCE_ERP_U_SQL}, ${SOURCE_ERP_V_SQL}
    )
      AND json_valid(extra_properties)
      AND ${SOURCE_KEY_SQL} IS NOT NULL
      AND ${SOURCE_ERP_U_SQL} IS NOT NULL
      AND ${SOURCE_ERP_V_SQL} IS NOT NULL;

    CREATE UNIQUE INDEX IF NOT EXISTS uq_ground_truth_click
      ON ground_truth_point (
        class,
        ${SOURCE_KEY_SQL},
        ${SOURCE_ERP_U_SQL},
        ${SOURCE_ERP_V_SQL}
      )
      WHERE json_valid(extra_properties)
        AND ${SOURCE_KEY_SQL} IS NOT NULL
        AND ${SOURCE_ERP_U_SQL} IS NOT NULL
        AND ${SOURCE_ERP_V_SQL} IS NOT NULL;
  `);
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
    importOfficeGeoJsonOnce(database, map);
    enforceGroundTruthClickUniqueness(database);
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

export type DetectionGroupLabel = {
  keyframeId: string;
  thetaCenter: number;
  phiCenter: number;
  rowIndex: number | null;
  /** null = "not an object": an explicit reject, not an absence of annotation. */
  groupName: string | null;
  note: string | null;
};

/**
 * Angles are the key, so they must be compared the way they are stored. 6 decimals is
 * far below the float16 precision the angles carry, and matches what the frontend
 * rounds to before sending — an exact SQL equality on a REAL is only safe because
 * both sides round first.
 */
function roundAngle(value: number): number {
  return Number(value.toFixed(6));
}

export function parseDetectionGroupLabel(
  body: Record<string, unknown>,
  requireGroup: boolean,
): DetectionGroupLabel {
  const keyframeId = typeof body.keyframe_id === "string" ? body.keyframe_id.trim() : "";
  const thetaCenter = Number(body.theta_center);
  const phiCenter = Number(body.phi_center);
  if (!keyframeId || !Number.isFinite(thetaCenter) || !Number.isFinite(phiCenter)) {
    throw new WorkbenchRouteError(422, "Invalid detection group label payload.");
  }
  // `Number(null)` is 0, and 0 is a perfectly valid row index — the coercion would
  // file every angle-only annotation against the parquet's first row.
  const rowIndex = body.row_index == null ? Number.NaN : Number(body.row_index);
  const groupName =
    typeof body.group_name === "string" && body.group_name.trim()
      ? body.group_name.trim()
      : null;
  if (requireGroup && groupName === null && body.group_name !== null) {
    throw new WorkbenchRouteError(
      422,
      "A group name is required; send null explicitly to mark a detection as 'not an object'.",
    );
  }
  return {
    keyframeId,
    thetaCenter: roundAngle(thetaCenter),
    phiCenter: roundAngle(phiCenter),
    rowIndex: Number.isInteger(rowIndex) ? rowIndex : null,
    groupName,
    note: typeof body.note === "string" && body.note.trim() ? body.note.trim() : null,
  };
}

export function upsertDetectionGroupLabel(
  map: MapEntry,
  label: DetectionGroupLabel,
): number {
  return withDatabase(map, (database) => {
    const result = database
      .prepare(`
        INSERT INTO detection_group_label
          (keyframe_id, theta_center, phi_center, row_index, group_name, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (keyframe_id, theta_center, phi_center) DO UPDATE SET
          row_index = COALESCE(excluded.row_index, row_index),
          group_name = excluded.group_name,
          note = excluded.note,
          created_at = excluded.created_at
      `)
      .run(
        label.keyframeId,
        label.thetaCenter,
        label.phiCenter,
        label.rowIndex,
        label.groupName,
        label.note,
        nowIso(),
      );
    return Number(result.lastInsertRowid);
  });
}

export function deleteDetectionGroupLabel(
  map: MapEntry,
  label: DetectionGroupLabel,
): void {
  withDatabase(map, (database) => {
    database
      .prepare(`
        DELETE FROM detection_group_label
        WHERE keyframe_id = ? AND theta_center = ? AND phi_center = ?
      `)
      .run(label.keyframeId, label.thetaCenter, label.phiCenter);
  });
}

/**
 * The reference partition, grouped and ready to be scored against an algorithm's.
 *
 * Rejected detections come back under `not_an_object` rather than being dropped:
 * "annotated as junk" and "never looked at" are different states, and a scoring pass
 * that confuses them will count its own blind spots as agreement.
 */
export function listDetectionGroups(map: MapEntry): Record<string, unknown> {
  return withDatabase(map, (database) => {
    const rows = database
      .prepare(`
        SELECT keyframe_id, theta_center, phi_center, row_index, group_name, note, created_at
        FROM detection_group_label
        ORDER BY group_name IS NULL, group_name, keyframe_id
      `)
      .all() as Array<{
      keyframe_id: string;
      theta_center: number;
      phi_center: number;
      row_index: number | null;
      group_name: string | null;
      note: string | null;
      created_at: string;
    }>;
    const groups = new Map<string, typeof rows>();
    const notAnObject: typeof rows = [];
    for (const row of rows) {
      if (row.group_name === null) {
        notAnObject.push(row);
        continue;
      }
      const members = groups.get(row.group_name) ?? [];
      members.push(row);
      groups.set(row.group_name, members);
    }
    return {
      labelled_count: rows.length,
      groups: [...groups].map(([name, members]) => ({
        group_name: name,
        member_count: members.length,
        keyframe_count: new Set(members.map((member) => member.keyframe_id)).size,
        members,
      })),
      not_an_object: notAnObject,
    };
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

// ---------------------------------------------------------------------------
// The Annotation office
// ---------------------------------------------------------------------------

/**
 * The office's annotations, in the store the rest of the pipeline already uses.
 *
 * Points land in `ground_truth_point` **on purpose**: they are the positional
 * annotations the benchmark scores against, and they used to sit in a
 * `annotations/annotations.geojson` that nothing ever read — the office wrote it and
 * no consumer opened it. Writing them here is what connects the two.
 *
 * The consequence is that opening the office now shows the map's existing ground
 * truth (258 points on vinci, 674 on bbhotel at the time of writing) and that
 * deleting one there removes it from the benchmark. That is the point of the
 * change, and it is why the UI confirms a deletion by naming it.
 *
 * Ids are prefixed (`gt:1`, `poly:4`) so one opaque string tells the caller which
 * table a row came from. The frontend treats them as opaque.
 */
const POINT_ID_PREFIX = "gt:";
const POLYGON_ID_PREFIX = "poly:";

/**
 * Palette for classes met in `ground_truth_point` with no `annotation_class` row.
 *
 * Twelve, because colour is how classes are told apart on the map and bbhotel
 * already has twelve. Collisions past that are resolved by taking the next free
 * entry rather than repeating one — see `backfillAnnotationClasses`.
 */
const DERIVED_CLASS_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#0f766e",
  "#b91c1c",
  "#4338ca",
  "#a16207",
];

/**
 * A stable colour for a class name.
 *
 * Deterministic rather than random: the same class keeps its colour across maps and
 * across reloads, so a screenshot taken today still reads tomorrow.
 */
function derivedClassColor(name: string): string {
  let hash = 0;
  for (let index = 0; index < name.length; index += 1) {
    hash = (hash * 31 + name.charCodeAt(index)) | 0;
  }
  return DERIVED_CLASS_COLORS[Math.abs(hash) % DERIVED_CLASS_COLORS.length];
}

export type WorkspaceClass = {
  name: string;
  annotationType: "point" | "polygon";
  color: string;
  prompt: string | null;
};

export type WorkspaceAnnotation = {
  id: string;
  className: string;
  annotationType: "point" | "polygon";
  prompt: string | null;
  level: string | null;
  altitude: number | null;
  accuracyM: number | null;
  /** `[lng, lat]` for a point; an open ring of `[lng, lat]` for a polygon. */
  coordinates: number[] | number[][];
  source: Record<string, unknown> | null;
  groundTruth: WorkspaceGroundTruth;
};

export type WorkspaceGroundTruth = {
  objectId: string | null;
  extentM: number | null;
  exhaustiveZone: string | null;
  isDepiction: boolean;
  labels: {
    synonyms: string[];
    depictions: string[];
    visuallySimilar: string[];
    clutter: string[];
  };
};

type AnnotationClassRow = {
  class: string;
  annotation_type: "point" | "polygon";
  color: string;
  prompt: string | null;
};

function parseJsonObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string" || !value) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function stringList(value: unknown): string[] {
  const values = Array.isArray(value) ? value : typeof value === "string" ? [value] : [];
  return [...new Set(values.flatMap((item) => {
    if (typeof item !== "string") {
      return [];
    }
    const trimmed = item.trim();
    return trimmed ? [trimmed] : [];
  }))];
}

function groundTruthFromRecord(
  value: Record<string, unknown> | null,
): WorkspaceGroundTruth {
  const labels = asObject(value?.labels) ?? {};
  return {
    objectId: optionalString(value?.objectId ?? value?.object_id),
    extentM:
      typeof (value?.extentM ?? value?.extent_m) === "number"
        ? Number(value?.extentM ?? value?.extent_m)
        : null,
    exhaustiveZone: optionalString(
      value?.exhaustiveZone ?? value?.exhaustive_zone,
    ),
    isDepiction: (value?.isDepiction ?? value?.is_depiction) === true,
    labels: {
      synonyms: stringList(labels.synonyms ?? value?.synonyms),
      depictions: stringList(labels.depictions ?? value?.depictions),
      visuallySimilar: stringList(
        labels.visuallySimilar ?? labels.visually_similar
          ?? value?.visuallySimilar ?? value?.visually_similar,
      ),
      clutter: stringList(labels.clutter ?? value?.clutter),
    },
  };
}

function annotationExtraProperties(
  annotation: WorkspaceAnnotation,
): Record<string, unknown> {
  const properties: Record<string, unknown> = {
    is_depiction: annotation.groundTruth.isDepiction,
    labels: {
      synonyms: annotation.groundTruth.labels.synonyms,
      depictions: annotation.groundTruth.labels.depictions,
      visually_similar: annotation.groundTruth.labels.visuallySimilar,
      clutter: annotation.groundTruth.labels.clutter,
    },
  };
  if (annotation.source) properties.source = annotation.source;
  if (annotation.groundTruth.objectId) {
    properties.object_id = annotation.groundTruth.objectId;
  }
  if (annotation.groundTruth.extentM !== null) {
    properties.extent_m = annotation.groundTruth.extentM;
  }
  if (annotation.groundTruth.exhaustiveZone) {
    properties.exhaustive_zone = annotation.groundTruth.exhaustiveZone;
  }
  return properties;
}

/**
 * Make sure every class named by an annotation has a palette row.
 *
 * The 258 ground-truth points that predate the office carry a class name and no
 * colour. Without this the office would open on annotations whose class does not
 * exist in its own palette, so they could be neither hidden nor recoloured.
 */
function backfillAnnotationClasses(database: Database.Database): void {
  const insert = database.prepare(`
    INSERT OR IGNORE INTO annotation_class
      (class, annotation_type, color, prompt, created_at)
    VALUES (?, ?, ?, ?, ?)
  `);
  const rows = database
    .prepare(`
      SELECT DISTINCT class, 'point' AS annotation_type FROM ground_truth_point
        WHERE class IS NOT NULL AND TRIM(class) <> ''
      UNION
      SELECT DISTINCT class, 'polygon' AS annotation_type FROM annotation_polygon
        WHERE class IS NOT NULL AND TRIM(class) <> ''
      ORDER BY 1, 2
    `)
    .all() as Array<{ class: string; annotation_type: "point" | "polygon" }>;
  const taken = new Set(
    (
      database.prepare("SELECT color FROM annotation_class").all() as Array<{
        color: string;
      }>
    ).map((row) => row.color.toLowerCase()),
  );
  for (const row of rows) {
    // Preferred colour first, then the next free one. The row order is stable
    // (the query sorts), so a class keeps the colour it was first given.
    let color = derivedClassColor(row.class);
    if (taken.has(color)) {
      const free = DERIVED_CLASS_COLORS.find(
        (candidate) => !taken.has(candidate.toLowerCase()),
      );
      color = free ?? color;
    }
    taken.add(color.toLowerCase());
    insert.run(row.class, row.annotation_type, color, null, nowIso());
  }
}

export function listWorkspaceAnnotations(map: MapEntry): {
  classes: WorkspaceClass[];
  annotations: WorkspaceAnnotation[];
} {
  return withDatabase(map, (database) => {
    backfillAnnotationClasses(database);

    const classes = (
      database
        .prepare(
          "SELECT class, annotation_type, color, prompt FROM annotation_class ORDER BY class",
        )
        .all() as AnnotationClassRow[]
    ).map((row) => ({
      name: row.class,
      annotationType: row.annotation_type,
      color: row.color,
      prompt: row.prompt,
    }));

    const annotations: WorkspaceAnnotation[] = [];
    const points = database
      .prepare(`
        SELECT ground_truth_point_id, lng, lat, alt, level, class, prompt, accuracy,
          extra_properties
        FROM ground_truth_point ORDER BY ground_truth_point_id
      `)
      .all() as Array<{
      ground_truth_point_id: number;
      lng: number;
      lat: number;
      alt: number | null;
      level: string | null;
      class: string | null;
      prompt: string | null;
      accuracy: number | null;
      extra_properties: string | null;
    }>;
    for (const point of points) {
      const extra = parseJsonObject(point.extra_properties);
      annotations.push({
        id: `${POINT_ID_PREFIX}${point.ground_truth_point_id}`,
        className: point.class ?? "",
        annotationType: "point",
        prompt: point.prompt,
        level: point.level,
        altitude: point.alt,
        accuracyM: point.accuracy,
        coordinates: [point.lng, point.lat],
        source: parseJsonObject(extra?.source ? JSON.stringify(extra.source) : null),
        groundTruth: groundTruthFromRecord(extra),
      });
    }

    const polygons = database
      .prepare(`
        SELECT annotation_polygon_id, ring, class, prompt, level, alt, accuracy
        FROM annotation_polygon ORDER BY annotation_polygon_id
      `)
      .all() as Array<{
      annotation_polygon_id: number;
      ring: string;
      class: string | null;
      prompt: string | null;
      level: string | null;
      alt: number | null;
      accuracy: number | null;
    }>;
    for (const polygon of polygons) {
      let ring: number[][] = [];
      try {
        const parsed = JSON.parse(polygon.ring) as unknown;
        ring = Array.isArray(parsed) ? (parsed as number[][]) : [];
      } catch {
        continue;
      }
      annotations.push({
        id: `${POLYGON_ID_PREFIX}${polygon.annotation_polygon_id}`,
        className: polygon.class ?? "",
        annotationType: "polygon",
        prompt: polygon.prompt,
        level: polygon.level,
        altitude: polygon.alt,
        accuracyM: polygon.accuracy,
        coordinates: ring,
        source: null,
        groundTruth: groundTruthFromRecord(null),
      });
    }

    return { classes, annotations };
  });
}

/**
 * Parse one annotation off the wire, rejecting anything the tables cannot hold.
 *
 * Validation is here rather than in the route because a bad ring or a missing
 * coordinate would be stored as valid JSON and only fail later, when something
 * tries to draw or score it.
 */
export function parseWorkspaceAnnotation(body: Record<string, unknown>): WorkspaceAnnotation {
  const annotationType = body.annotationType === "polygon" ? "polygon" : "point";
  const className = typeof body.className === "string" ? body.className.trim() : "";
  if (!className) {
    throw new Error("An annotation needs a class name.");
  }
  const coordinates = body.coordinates;
  if (annotationType === "point") {
    if (
      !Array.isArray(coordinates) ||
      typeof coordinates[0] !== "number" ||
      typeof coordinates[1] !== "number"
    ) {
      throw new Error("A point annotation needs [lng, lat] coordinates.");
    }
  } else if (
    !Array.isArray(coordinates) ||
    coordinates.length < 3 ||
    !coordinates.every(
      (point) =>
        Array.isArray(point) &&
        typeof point[0] === "number" &&
        typeof point[1] === "number",
    )
  ) {
    throw new Error("A polygon annotation needs at least 3 [lng, lat] vertices.");
  }

  const groundTruthRecord = asObject(body.groundTruth);
  const groundTruth = groundTruthFromRecord(groundTruthRecord);
  if (annotationType === "point" && groundTruthRecord) {
    if (!groundTruth.objectId) {
      throw new Error("A ground-truth point needs an object ID.");
    }
    if (groundTruth.extentM === null || groundTruth.extentM <= 0) {
      throw new Error("A ground-truth point needs a positive horizontal extent.");
    }
    if (!groundTruth.labels.synonyms.length) {
      throw new Error("A ground-truth point needs at least one exact synonym.");
    }
  }

  return {
    id: typeof body.id === "string" ? body.id : "",
    className,
    annotationType,
    prompt: typeof body.prompt === "string" && body.prompt.trim() ? body.prompt.trim() : null,
    level: typeof body.level === "string" && body.level.trim() ? body.level.trim() : null,
    altitude: typeof body.altitude === "number" ? body.altitude : null,
    accuracyM: typeof body.accuracyM === "number" ? body.accuracyM : null,
    coordinates: coordinates as number[] | number[][],
    source:
      body.source && typeof body.source === "object" && !Array.isArray(body.source)
        ? (body.source as Record<string, unknown>)
        : null,
    groundTruth,
  };
}

function insertWorkspaceAnnotation(
  database: Database.Database,
  annotation: WorkspaceAnnotation,
): string {
  if (annotation.annotationType === "point") {
    const [lng, lat] = annotation.coordinates as number[];
    const extra = JSON.stringify(annotationExtraProperties(annotation));
    const result = database
      .prepare(`
        INSERT OR IGNORE INTO ground_truth_point
          (lng, lat, alt, level, class, prompt, accuracy, extra_properties, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `)
      .run(
        lng,
        lat,
        annotation.altitude,
        annotation.level,
        annotation.className,
        annotation.prompt,
        annotation.accuracyM,
        extra,
        nowIso(),
      );
    if (result.changes > 0) {
      return `${POINT_ID_PREFIX}${Number(result.lastInsertRowid)}`;
    }
    const source = annotation.source ?? {};
    const sourceKey = optionalString(
      source.videoKeyframeUuid
        ?? source.video_keyframe_uuid
        ?? source.keyframeId
        ?? source.legacy_keyframe_id,
    );
    const erpU = Number(source.erpU ?? source.erp_u);
    const erpV = Number(source.erpV ?? source.erp_v);
    const existing = sourceKey && Number.isFinite(erpU) && Number.isFinite(erpV)
      ? database.prepare(`
          SELECT ground_truth_point_id AS id
          FROM ground_truth_point
          WHERE class = ?
            AND ${SOURCE_KEY_SQL} = ?
            AND ${SOURCE_ERP_U_SQL} = ROUND(?, 6)
            AND ${SOURCE_ERP_V_SQL} = ROUND(?, 6)
          LIMIT 1
        `).get(annotation.className, sourceKey, erpU, erpV) as { id: number } | undefined
      : undefined;
    if (!existing) {
      throw new Error("The point annotation could not be stored.");
    }
    return `${POINT_ID_PREFIX}${existing.id}`;
  }

  const result = database
    .prepare(`
      INSERT INTO annotation_polygon
        (ring, class, prompt, level, alt, accuracy, extra_properties, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `)
    .run(
      JSON.stringify(annotation.coordinates),
      annotation.className,
      annotation.prompt,
      annotation.level,
      annotation.altitude,
      annotation.accuracyM,
      null,
      nowIso(),
    );
  return `${POLYGON_ID_PREFIX}${Number(result.lastInsertRowid)}`;
}

export function createWorkspaceAnnotation(
  map: MapEntry,
  annotation: WorkspaceAnnotation,
): string {
  return withDatabase(map, (database) => insertWorkspaceAnnotation(database, annotation));
}

/** Replace one existing point annotation while preserving its stable row id. */
export function updateWorkspaceAnnotation(
  map: MapEntry,
  annotation: WorkspaceAnnotation,
): boolean {
  if (!annotation.id.startsWith(POINT_ID_PREFIX) || annotation.annotationType !== "point") {
    return false;
  }
  const [lng, lat] = annotation.coordinates as number[];
  return withDatabase(map, (database) => {
    const result = database.prepare(`
      UPDATE ground_truth_point SET
        lng = ?, lat = ?, alt = ?, level = ?, class = ?, prompt = ?, accuracy = ?,
        extra_properties = ?
      WHERE ground_truth_point_id = ?
    `).run(
      lng,
      lat,
      annotation.altitude,
      annotation.level,
      annotation.className,
      annotation.prompt,
      annotation.accuracyM,
      JSON.stringify(annotationExtraProperties(annotation)),
      Number(annotation.id.slice(POINT_ID_PREFIX.length)),
    );
    return result.changes > 0;
  });
}

/** Delete one annotation by its prefixed id. Returns false when nothing matched. */
export function deleteWorkspaceAnnotation(map: MapEntry, id: string): boolean {
  return withDatabase(map, (database) => {
    if (id.startsWith(POINT_ID_PREFIX)) {
      const result = database
        .prepare("DELETE FROM ground_truth_point WHERE ground_truth_point_id = ?")
        .run(Number(id.slice(POINT_ID_PREFIX.length)));
      return result.changes > 0;
    }
    if (id.startsWith(POLYGON_ID_PREFIX)) {
      const result = database
        .prepare("DELETE FROM annotation_polygon WHERE annotation_polygon_id = ?")
        .run(Number(id.slice(POLYGON_ID_PREFIX.length)));
      return result.changes > 0;
    }
    return false;
  });
}

/** Delete every office annotation. Classes survive: the palette is not the data. */
export function clearWorkspaceAnnotations(map: MapEntry): number {
  return withDatabase(map, (database) => {
    const transaction = database.transaction(() => {
      const points = database.prepare("DELETE FROM ground_truth_point").run().changes;
      const polygons = database.prepare("DELETE FROM annotation_polygon").run().changes;
      return points + polygons;
    });
    return transaction();
  });
}

export function parseWorkspaceClass(body: Record<string, unknown>): WorkspaceClass {
  const name = typeof body.name === "string" ? body.name.trim() : "";
  if (!name) {
    throw new Error("A class needs a name.");
  }
  const color = typeof body.color === "string" ? body.color.trim() : "";
  if (!/^#[0-9A-Fa-f]{6}$/.test(color)) {
    throw new Error("A class colour must be a #rrggbb hex string.");
  }
  return {
    name,
    annotationType: body.annotationType === "polygon" ? "polygon" : "point",
    color,
    prompt: typeof body.prompt === "string" && body.prompt.trim() ? body.prompt.trim() : null,
  };
}

/**
 * Create a class, or rename/recolour an existing one.
 *
 * A rename cascades to the annotations that carry the old name, because the class
 * is referenced by name rather than by id — `ground_truth_point.class` is the
 * column the benchmark reads, and turning it into a foreign key would change what
 * `buildGroundTruth` emits.
 */
export function upsertWorkspaceClass(
  map: MapEntry,
  next: WorkspaceClass,
  original: { name: string; annotationType: "point" | "polygon" } | null,
): void {
  withDatabase(map, (database) => {
    const transaction = database.transaction(() => {
      if (original && original.name !== next.name) {
        database
          .prepare(
            "DELETE FROM annotation_class WHERE class = ? AND annotation_type = ?",
          )
          .run(original.name, original.annotationType);
        const table =
          original.annotationType === "point" ? "ground_truth_point" : "annotation_polygon";
        database
          .prepare(`UPDATE ${table} SET class = ? WHERE class = ?`)
          .run(next.name, original.name);
      }
      database
        .prepare(`
          INSERT INTO annotation_class (class, annotation_type, color, prompt, created_at)
          VALUES (?, ?, ?, ?, ?)
          ON CONFLICT (class, annotation_type) DO UPDATE SET
            color = excluded.color,
            prompt = excluded.prompt
        `)
        .run(next.name, next.annotationType, next.color, next.prompt, nowIso());
    });
    transaction();
  });
}

/** Delete a class and every annotation carrying it. Returns how many rows went. */
export function deleteWorkspaceClass(
  map: MapEntry,
  name: string,
  annotationType: "point" | "polygon",
): number {
  return withDatabase(map, (database) => {
    const table = annotationType === "point" ? "ground_truth_point" : "annotation_polygon";
    const transaction = database.transaction(() => {
      const removed = database
        .prepare(`DELETE FROM ${table} WHERE class = ?`)
        .run(name).changes;
      database
        .prepare("DELETE FROM annotation_class WHERE class = ? AND annotation_type = ?")
        .run(name, annotationType);
      return removed;
    });
    return transaction();
  });
}

/**
 * Merge a GeoJSON FeatureCollection into the store — the file-import path.
 *
 * Kept because the office's only way in used to be dropping a `.geojson` on it, and
 * a store that cannot read what the previous one wrote strands whatever people
 * already have on disk.
 */
export function importWorkspaceAnnotations(
  map: MapEntry,
  collection: Record<string, unknown>,
): { imported: number; skipped: number } {
  return withDatabase(map, (database) => importCollection(database, collection));
}

/**
 * The same merge against an open handle.
 *
 * Split out because the one-shot legacy import runs *inside* database open, where
 * `withDatabase` would recurse.
 */
function importCollection(
  database: Database.Database,
  collection: Record<string, unknown>,
): { imported: number; skipped: number } {
  const features = Array.isArray(collection.features) ? collection.features : [];
  {
    let imported = 0;
    let skipped = 0;
    const transaction = database.transaction(() => {
      for (const rawFeature of features) {
        const feature = asObject(rawFeature);
        const geometry = asObject(feature?.geometry);
        const properties = asObject(feature?.properties) ?? {};
        const className =
          typeof properties.class === "string" ? properties.class.trim() : "";
        if (!className) {
          skipped += 1;
          continue;
        }
        const {
          class: _class,
          prompt,
          accuracy,
          level,
          altitude,
          color,
          ...rest
        } = properties;
        const hasGroundTruthContract = [
          "objectId",
          "object_id",
          "extentM",
          "extent_m",
          "labels",
          "synonyms",
        ].some((key) => Object.hasOwn(rest, key));

        let annotation: WorkspaceAnnotation;
        try {
          if (geometry?.type === "Point") {
            const coordinates = geometry.coordinates as number[];
            annotation = parseWorkspaceAnnotation({
              className,
              annotationType: "point",
              coordinates: [coordinates?.[0], coordinates?.[1]],
              altitude:
                typeof coordinates?.[2] === "number" ? coordinates[2] : altitude,
              level,
              prompt,
              accuracyM: accuracy,
              source: rest.source,
              groundTruth: hasGroundTruthContract ? rest : undefined,
            });
          } else if (geometry?.type === "Polygon") {
            const rings = geometry.coordinates as number[][][];
            annotation = parseWorkspaceAnnotation({
              className,
              annotationType: "polygon",
              coordinates: rings?.[0],
              altitude,
              level,
              prompt,
              accuracyM: accuracy,
            });
          } else {
            skipped += 1;
            continue;
          }
        } catch {
          skipped += 1;
          continue;
        }

        insertWorkspaceAnnotation(database, annotation);
        database
          .prepare(`
            INSERT OR IGNORE INTO annotation_class
              (class, annotation_type, color, prompt, created_at)
            VALUES (?, ?, ?, ?, ?)
          `)
          .run(
            className,
            annotation.annotationType,
            typeof color === "string" && /^#[0-9A-Fa-f]{6}$/.test(color)
              ? color
              : derivedClassColor(className),
            annotation.prompt,
            nowIso(),
          );
        imported += 1;
      }
    });
    transaction();
    return { imported, skipped };
  }
}

/**
 * One-shot import of `{map}/annotations/annotations.geojson`, the file this store
 * replaces.
 *
 * That file was write-only: the office saved it and nothing ever read it back, so
 * whatever is in one is work that never reached the pipeline. Importing it once,
 * and recording that in `meta`, is what stops the migration from silently dropping
 * it. Neither map in use here has one, so this is for the ones that might.
 */
function importOfficeGeoJsonOnce(database: Database.Database, map: MapEntry): void {
  const alreadyImported = database
    .prepare("SELECT value FROM meta WHERE key = ?")
    .get(OFFICE_IMPORTED_KEY);
  if (alreadyImported) {
    return;
  }
  const markImported = database.prepare(
    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
  );
  const officePath = path.join(map.path, "annotations", "annotations.geojson");
  if (!existsSync(officePath)) {
    markImported.run(OFFICE_IMPORTED_KEY, nowIso());
    return;
  }
  try {
    copyFileSync(officePath, `${officePath}.pre-import.bak`);
  } catch {
    // Best-effort, as for the benchmark import: a missing backup must not block.
  }
  try {
    const parsed = asObject(JSON.parse(readFileSync(officePath, "utf8")));
    if (parsed) {
      importCollection(database, parsed);
    }
  } catch {
    // A malformed file is left on disk untouched rather than retried every open.
  }
  markImported.run(OFFICE_IMPORTED_KEY, nowIso());
}
