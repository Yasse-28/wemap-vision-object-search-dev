import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import Database from "better-sqlite3";

import {
  annotationDatabasePath,
  buildGroundTruth,
  clearWorkspaceAnnotations,
  closeAnnotationDatabase,
  createWorkspaceAnnotation,
  deleteDetectionReview,
  deleteWorkspaceAnnotation,
  deleteWorkspaceClass,
  listAnnotations,
  listWorkspaceAnnotations,
  parseReviewMutation,
  parseWorkspaceAnnotation,
  parseWorkspaceClass,
  upsertDetectionReview,
  upsertWorkspaceClass,
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

// ---------------------------------------------------------------------------
// The Annotation office, now backed by the store instead of a GeoJSON file
// ---------------------------------------------------------------------------

test("a point annotation is stored as benchmark ground truth", async () => {
  const fixture = await createMap();
  try {
    const id = createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({
        className: "fire extinguisher",
        annotationType: "point",
        coordinates: [2.35, 48.85],
        altitude: 3.5,
        level: "1",
        prompt: "red fire extinguisher",
        accuracyM: 2,
        source: { keyframeId: "42", depthM: 4.2 },
      }),
    );
    assert.ok(id.startsWith("gt:"));

    // The point is what the benchmark scores against — that is the whole reason
    // this went into `ground_truth_point` rather than a table of its own.
    const groundTruth = buildGroundTruth(fixture.map);
    assert.equal(groundTruth.features.length, 1);
    const feature = groundTruth.features[0] as Record<string, unknown>;
    const properties = feature.properties as Record<string, unknown>;
    assert.equal(properties.class, "fire extinguisher");
    assert.equal(properties.prompt, "red fire extinguisher");
    assert.deepEqual(
      (feature.geometry as Record<string, unknown>).coordinates,
      [2.35, 48.85, 3.5],
    );

    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations.length, 1);
    assert.equal(stored.annotations[0].id, id);
    assert.deepEqual(stored.annotations[0].source, {
      keyframeId: "42",
      depthM: 4.2,
    });
  } finally {
    await fixture.cleanup();
  }
});

test("a polygon annotation stays out of the ground truth", async () => {
  const fixture = await createMap();
  try {
    createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({
        className: "zone",
        annotationType: "polygon",
        coordinates: [
          [2.35, 48.85],
          [2.36, 48.85],
          [2.36, 48.86],
        ],
      }),
    );

    // A polygon has no single position to score a localization against.
    assert.equal(buildGroundTruth(fixture.map).features.length, 0);
    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations.length, 1);
    assert.equal(stored.annotations[0].annotationType, "polygon");
    assert.equal((stored.annotations[0].coordinates as number[][]).length, 3);
  } finally {
    await fixture.cleanup();
  }
});

test("malformed geometry is refused rather than stored", async () => {
  assert.throws(
    () => parseWorkspaceAnnotation({ className: "x", coordinates: [2.35] }),
    /\[lng, lat\]/,
  );
  assert.throws(
    () =>
      parseWorkspaceAnnotation({
        className: "x",
        annotationType: "polygon",
        coordinates: [[2.35, 48.85], [2.36, 48.85]],
      }),
    /at least 3/,
  );
  assert.throws(
    () => parseWorkspaceAnnotation({ className: "  ", coordinates: [2.35, 48.85] }),
    /class name/,
  );
});

test("classes existing only in the ground truth are given a palette entry", async () => {
  const fixture = await createMap();
  try {
    // Points that predate the office: a class name, no colour. Without the
    // backfill the office would render annotations whose class it cannot list.
    listWorkspaceAnnotations(fixture.map); // creates the schema
    closeAnnotationDatabase(fixture.map);
    const database = new Database(annotationDatabasePath(fixture.map));
    database
      .prepare(
        "INSERT INTO ground_truth_point (lng, lat, class, created_at) VALUES (?, ?, ?, ?)",
      )
      .run(2.35, 48.85, "legacy class", new Date().toISOString());
    database.close();
    closeAnnotationDatabase(fixture.map);

    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations.length, 1);
    const derived = stored.classes.find((item) => item.name === "legacy class");
    assert.ok(derived, "the class should have been backfilled");
    assert.match(derived.color, /^#[0-9a-f]{6}$/);
    // Deterministic, so a class keeps its colour across reloads and maps.
    assert.equal(
      listWorkspaceAnnotations(fixture.map).classes.find(
        (item) => item.name === "legacy class",
      )?.color,
      derived.color,
    );
  } finally {
    await fixture.cleanup();
  }
});

test("renaming a class moves its annotations with it", async () => {
  const fixture = await createMap();
  try {
    createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({
        className: "extinguisher",
        coordinates: [2.35, 48.85],
      }),
    );
    upsertWorkspaceClass(
      fixture.map,
      parseWorkspaceClass({
        name: "fire extinguisher",
        color: "#123456",
        annotationType: "point",
      }),
      { name: "extinguisher", annotationType: "point" },
    );

    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations[0].className, "fire extinguisher");
    assert.deepEqual(
      stored.classes.map((item) => item.name),
      ["fire extinguisher"],
    );
    // The benchmark reads `class`, so the rename has to reach it.
    const properties = (buildGroundTruth(fixture.map).features[0] as Record<string, unknown>)
      .properties as Record<string, unknown>;
    assert.equal(properties.class, "fire extinguisher");
  } finally {
    await fixture.cleanup();
  }
});

test("deleting a class removes its annotations, and only its own", async () => {
  const fixture = await createMap();
  try {
    createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({ className: "a", coordinates: [2.35, 48.85] }),
    );
    createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({ className: "b", coordinates: [2.36, 48.86] }),
    );

    assert.equal(deleteWorkspaceClass(fixture.map, "a", "point"), 1);
    const stored = listWorkspaceAnnotations(fixture.map);
    assert.deepEqual(
      stored.annotations.map((item) => item.className),
      ["b"],
    );
    assert.deepEqual(
      stored.classes.map((item) => item.name),
      ["b"],
    );
  } finally {
    await fixture.cleanup();
  }
});

test("deleting one annotation, and clearing them all", async () => {
  const fixture = await createMap();
  try {
    const first = createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({ className: "a", coordinates: [2.35, 48.85] }),
    );
    createWorkspaceAnnotation(
      fixture.map,
      parseWorkspaceAnnotation({
        className: "a",
        annotationType: "polygon",
        coordinates: [
          [2.35, 48.85],
          [2.36, 48.85],
          [2.36, 48.86],
        ],
      }),
    );

    assert.equal(deleteWorkspaceAnnotation(fixture.map, first), true);
    assert.equal(deleteWorkspaceAnnotation(fixture.map, first), false);
    assert.equal(listWorkspaceAnnotations(fixture.map).annotations.length, 1);

    // Both tables go; the palette survives, because it is not the data.
    assert.equal(clearWorkspaceAnnotations(fixture.map), 1);
    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations.length, 0);
    assert.equal(stored.classes.length, 1);
  } finally {
    await fixture.cleanup();
  }
});

test("the legacy annotations.geojson is imported once, and backed up", async () => {
  const fixture = await createMap();
  try {
    await mkdir(path.join(fixture.map.path, "annotations"), { recursive: true });
    const legacyPath = path.join(
      fixture.map.path,
      "annotations",
      "annotations.geojson",
    );
    await writeFile(
      legacyPath,
      JSON.stringify({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: { type: "Point", coordinates: [2.35, 48.85, 3] },
            properties: { class: "extinguisher", prompt: "red", color: "#abcdef" },
          },
          {
            type: "Feature",
            geometry: {
              type: "Polygon",
              coordinates: [[[2.35, 48.85], [2.36, 48.85], [2.36, 48.86]]],
            },
            properties: { class: "zone" },
          },
          // No class: nothing to file it under, so it is skipped rather than guessed.
          {
            type: "Feature",
            geometry: { type: "Point", coordinates: [2.37, 48.87] },
            properties: {},
          },
        ],
      }),
      "utf8",
    );

    const stored = listWorkspaceAnnotations(fixture.map);
    assert.equal(stored.annotations.length, 2);
    assert.equal(
      stored.classes.find((item) => item.name === "extinguisher")?.color,
      "#abcdef",
    );
    assert.equal(stored.annotations[0].altitude, 3);

    // A second open must not import it again.
    closeAnnotationDatabase(fixture.map);
    assert.equal(listWorkspaceAnnotations(fixture.map).annotations.length, 2);
    assert.ok(existsSync(`${legacyPath}.pre-import.bak`));
  } finally {
    await fixture.cleanup();
  }
});
