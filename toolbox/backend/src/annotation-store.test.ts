import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import Database from "better-sqlite3";

import {
  annotationDatabasePath,
  buildGroundTruth,
  closeAnnotationDatabase,
  deleteDetectionReview,
  listAnnotations,
  parseReviewMutation,
  upsertDetectionReview,
} from "./annotation-store.js";
import type { MapEntry } from "./config.js";
import { WorkbenchRouteError } from "./workbench-index.js";

async function createMap(): Promise<{
  map: MapEntry;
  cleanup: () => Promise<void>;
}> {
  const mapPath = await mkdtemp(path.join(os.tmpdir(), "toolbox-annotations-"));
  const map: MapEntry = {
    id: "test-map",
    display_name: "Test map",
    path: mapPath,
    emmid: null,
    geo_ref_id: 1,
    object_search: null,
    parent_map: null,
  };
  return {
    map,
    cleanup: async () => {
      closeAnnotationDatabase(map);
      await rm(mapPath, { recursive: true, force: true });
    },
  };
}

test("stores, filters, replaces and deletes detection reviews", async () => {
  const fixture = await createMap();
  try {
    const firstId = upsertDetectionReview(fixture.map, {
      targetId: 42,
      query: "chair",
      status: "false_positive",
    });
    const secondId = upsertDetectionReview(fixture.map, {
      targetId: 42,
      query: "chair",
      status: "true_positive",
    });
    upsertDetectionReview(fixture.map, {
      targetId: 7,
      query: "table",
      status: "true_positive",
    });

    assert.notEqual(firstId, secondId);
    assert.equal(annotationDatabasePath(fixture.map), path.join(fixture.map.path, "object-search-annotations.db"));
    assert.deepEqual(
      (listAnnotations(fixture.map, "chair").detection_reviews as Array<Record<string, unknown>>).map(
        ({ target_id, query, status }) => ({ target_id, query, status }),
      ),
      [{ target_id: 42, query: "chair", status: "true_positive" }],
    );

    deleteDetectionReview(fixture.map, { targetId: 42, query: "chair" });
    assert.deepEqual(listAnnotations(fixture.map, "chair").detection_reviews, []);
  } finally {
    await fixture.cleanup();
  }
});

test("imports existing benchmark points into the integrated database", async () => {
  const fixture = await createMap();
  try {
    const benchmarkDir = path.join(fixture.map.path, "benchmark");
    await mkdir(benchmarkDir);
    await writeFile(
      path.join(benchmarkDir, "annotations.geojson"),
      JSON.stringify({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            id: "existing",
            geometry: { type: "Point", coordinates: [2.3, 48.8, 12] },
            properties: { class: "chair", prompt: "red chair", level: 1 },
          },
        ],
      }),
      "utf8",
    );

    assert.deepEqual(buildGroundTruth(fixture.map), {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          id: "gt-point-1",
          geometry: { type: "Point", coordinates: [2.3, 48.8, 12] },
          properties: {
            class: "chair",
            prompt: "red chair",
            level: "1",
            altitude: 12,
          },
        },
      ],
    });
  } finally {
    await fixture.cleanup();
  }
});

test("migrates legacy class-name manual detections", async () => {
  const fixture = await createMap();
  try {
    const database = new Database(annotationDatabasePath(fixture.map));
    database.exec(`
      CREATE TABLE manual_detection (
        manual_detection_id INTEGER PRIMARY KEY,
        keyframe_id TEXT NOT NULL,
        geometry_type TEXT NOT NULL,
        geometry BLOB NOT NULL,
        class TEXT,
        query TEXT NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT,
        note TEXT,
        enrichment_status TEXT,
        enriched_at TEXT
      )
    `);
    const geometry = Buffer.alloc(8);
    geometry.writeFloatLE(0.25, 0);
    geometry.writeFloatLE(0.75, 4);
    database
      .prepare(`
        INSERT INTO manual_detection (
          keyframe_id, geometry_type, geometry, class, query, created_at,
          enrichment_status
        ) VALUES (?, 'bbox', ?, ?, ?, ?, 'pending')
      `)
      .run("keyframe-a", geometry, "chair", "red chair", "2026-01-01T00:00:00Z");
    database.close();

    const annotations = listAnnotations(fixture.map, null);
    assert.deepEqual(
      (annotations.missed_detections as Array<Record<string, unknown>>).map(
        ({ keyframe_id, geometry, class_name, used_as_ground_truth }) => ({
          keyframe_id,
          geometry,
          class_name,
          used_as_ground_truth,
        }),
      ),
      [
        {
          keyframe_id: "keyframe-a",
          geometry: [[0.25, 0.75]],
          class_name: "chair",
          used_as_ground_truth: true,
        },
      ],
    );
  } finally {
    await fixture.cleanup();
  }
});

test("validates review payloads at the Toolbox API boundary", () => {
  assert.deepEqual(
    parseReviewMutation(
      {
        target_type: "object",
        target_id: 9,
        query: "  chair  ",
        status: "false_positive",
      },
      true,
    ),
    { targetId: 9, query: "chair", status: "false_positive" },
  );
  assert.throws(
    () =>
      parseReviewMutation(
        { target_type: "cluster", target_id: 9, query: "chair" },
        false,
      ),
    (error: unknown) => error instanceof WorkbenchRouteError && error.status === 422,
  );
});
