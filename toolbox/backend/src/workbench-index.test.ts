import assert from "node:assert/strict";
import { copyFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import type { MapEntry } from "./config.js";
import {
  indexProjectWorldPointPayload,
  keyframeHeadingDegreesFromPose,
  legacyObjectSearchMetadataStatusPayloadForComparison,
  objectSearchMetadataStatusPayload,
  WorkbenchRouteError,
} from "./workbench-index.js";

const FIXTURES = path.resolve(import.meta.dirname, "..", "test-fixtures");

test("derives compass heading from world-to-camera extrinsics", () => {
  const identity = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  assert.equal(keyframeHeadingDegreesFromPose(identity), 180);

  const worldToCameraWest = [
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
  ];
  assert.equal(keyframeHeadingDegreesFromPose(worldToCameraWest), 270);
});

/**
 * A map directory holding a v2 manifest with three keyframes, the real one at
 * index 2 — so the literal "2" in every test below still names it, and the
 * index-is-the-id rule is actually exercised.
 *
 * The orientation is **[0, 0, 1, 0]**, not identity: 180° about the EUS up axis.
 * These tests are written against a world-to-camera matrix of identity in WDS,
 * and the adapter reaches that from `R_eus = diag(-1,-1,1) · I · diag(1,-1,-1)
 * = diag(-1,1,-1)`, whose quaternion is `w=0, y=1`. It checks out semantically
 * too: identity-in-WDS world-to-camera means heading 180° (asserted above), i.e.
 * facing South — and 180° of yaw from North is South. Do not "fix" this to
 * [1, 0, 0, 0]; that would rotate every expectation by 180°.
 */
async function createMap(
  positionEus: [number, number, number],
): Promise<{ map: MapEntry; cleanup: () => Promise<void> }> {
  const mapPath = await mkdtemp(path.join(os.tmpdir(), "object-search-projection-"));
  const filler = { x: 0, y: 0, z: 0, orientation: [1, 0, 0, 0] };
  const manifest = {
    // [lng, lat, alt] — PointField order, the reverse of the old GeoRef columns.
    local_origin: [6.0, 45.0, 100.0],
    map: { name: "test", uuid: "u", venue_type: "rail" },
    geo_levels: [
      { value: 0, min_altitude: -10.0, max_altitude: 10.0, geo_ref: 1, geometry: null },
    ],
    geo_keyframes: [
      { ...filler, image_url: "https://e/images/0.jpg", depth_url: "https://e/depths/0.tif" },
      { ...filler, image_url: "https://e/images/1.jpg", depth_url: "https://e/depths/1.tif" },
      {
        x: positionEus[0],
        y: positionEus[1],
        z: positionEus[2],
        orientation: [0, 0, 1, 0],
        image_url: "https://e/images/2.jpg",
        depth_url: "https://e/depths/2.tif",
      },
    ],
  };
  await writeFile(
    path.join(mapPath, "test_1_20260101_000000.json"),
    JSON.stringify(manifest),
    "utf8",
  );
  return {
    map: {
      id: "test",
      display_name: "Test",
      path: mapPath,
      emmid: null,
      geo_ref_id: 1,
      object_search: null,
    },
    cleanup: () => rm(mapPath, { recursive: true, force: true }),
  };
}

async function createMetadataMap(): Promise<{
  map: MapEntry;
  cleanup: () => Promise<void>;
}> {
  const fixture = await createMap([1.23456789, 2.34567891, 3.45678912]);
  const metadataDirectory = path.join(fixture.map.path, "object-search");
  await mkdir(metadataDirectory, { recursive: true });
  await copyFile(
    path.join(FIXTURES, "metadata-postprocessed.parquet"),
    path.join(metadataDirectory, "metadata.parquet"),
  );
  return fixture;
}

function asRecord(value: unknown): Record<string, unknown> {
  assert.ok(value && typeof value === "object" && !Array.isArray(value));
  return value as Record<string, unknown>;
}

function alignedColumnLength(value: unknown): number {
  const columns = asRecord(value);
  const lengths = Object.values(columns).map((column) => {
    assert.ok(Array.isArray(column));
    return column.length;
  });
  assert.ok(lengths.length > 0);
  assert.ok(lengths.every((length) => length === lengths[0]));
  return lengths[0];
}

test("status payload uses index-aligned columns without marker styles", async () => {
  const fixture = await createMetadataMap();
  try {
    const payload = await objectSearchMetadataStatusPayload(fixture.map);
    const summary = asRecord(payload.summary);
    assert.deepEqual(Object.keys(asRecord(payload.markers)), [
      "ids",
      "latitude",
      "longitude",
      "level",
      "heading_deg",
    ]);
    assert.deepEqual(Object.keys(asRecord(summary.keyframes)), [
      "ids",
      "row_start",
      "row_count",
      "with_depth",
      "ingested",
      "no_position",
    ]);
    assert.equal(alignedColumnLength(payload.markers), payload.resolved_marker_count);
    assert.equal(alignedColumnLength(summary.keyframes), 3);
    assert.doesNotMatch(JSON.stringify(payload.markers), /\"(?:color|radius)\"/);
  } finally {
    await fixture.cleanup();
  }
});

test("columnar status is semantically equivalent to the object fixture payload", async () => {
  const fixture = await createMetadataMap();
  try {
    const legacy = await legacyObjectSearchMetadataStatusPayloadForComparison(fixture.map);
    const columnar = await objectSearchMetadataStatusPayload(fixture.map);
    const markerColumns = asRecord(columnar.markers);
    const summary = asRecord(columnar.summary);
    const summaryColumns = asRecord(summary.keyframes);

    const reconstructedMarkers = (markerColumns.ids as string[]).map((id, index) => ({
      id,
      latitude: (markerColumns.latitude as number[])[index],
      longitude: (markerColumns.longitude as number[])[index],
      level: (markerColumns.level as Array<string | null>)[index],
      heading_deg: (markerColumns.heading_deg as Array<number | null>)[index],
    }));
    const reconstructedKeyframes = (summaryColumns.ids as string[]).map((id, index) => ({
      id,
      row_start: (summaryColumns.row_start as number[])[index],
      row_count: (summaryColumns.row_count as number[])[index],
      with_depth: (summaryColumns.with_depth as number[])[index],
      ingested: (summaryColumns.ingested as Array<number | null>)[index],
      no_position: (summaryColumns.no_position as Array<number | null>)[index],
    }));

    const normalizedLegacy = structuredClone(legacy);
    normalizedLegacy.markers = (normalizedLegacy.markers as Array<Record<string, unknown>>).map(
      ({ color: _color, radius: _radius, ...marker }) => ({
        ...marker,
        latitude: Number(Number(marker.latitude).toFixed(7)),
        longitude: Number(Number(marker.longitude).toFixed(7)),
        heading_deg:
          marker.heading_deg === null
            ? null
            : Number(Number(marker.heading_deg).toFixed(2)),
      }),
    );
    const normalizedSummary = asRecord(normalizedLegacy.summary);
    normalizedSummary.keyframes = (
      normalizedSummary.keyframes as Array<Record<string, unknown>>
    ).map(({ row_end: _rowEnd, ...keyframe }) => keyframe);

    const reconstructed = structuredClone(columnar);
    reconstructed.markers = reconstructedMarkers;
    asRecord(reconstructed.summary).keyframes = reconstructedKeyframes;
    assert.deepEqual(reconstructed, normalizedLegacy);
  } finally {
    await fixture.cleanup();
  }
});

test("projects a WDS point into an identity keyframe ERP", async () => {
  const fixture = await createMap([0, 0, 0]);
  try {
    const payload = await indexProjectWorldPointPayload(
      fixture.map,
      "2",
      { point_world_wds: [0, 0, 5] },
    );
    assert.equal(payload.erp_u, 0.5);
    assert.equal(payload.erp_v, 0.5);
    assert.equal(payload.distance_m, 5);
    assert.deepEqual(payload.point_keyframe, [0, 0, 5]);
  } finally {
    await fixture.cleanup();
  }
});

test("applies destination world-to-camera extrinsics before ERP projection", async () => {
  const fixture = await createMap([-1, 0, 0]);
  try {
    const payload = await indexProjectWorldPointPayload(
      fixture.map,
      "2",
      { point_world_wds: [0, 0, 5] },
    );
    const expectedU = 0.5 + Math.atan2(-1, 5) / (2 * Math.PI);
    assert.ok(Math.abs(Number(payload.erp_u) - expectedU) < 1e-12);
    assert.equal(payload.erp_v, 0.5);
    assert.ok(Math.abs(Number(payload.distance_m) - Math.sqrt(26)) < 1e-12);
    assert.deepEqual(payload.point_keyframe, [-1, 0, 5]);
  } finally {
    await fixture.cleanup();
  }
});

test("preserves arbitrary ERP directions through identity projection", async () => {
  const fixture = await createMap([0, 0, 0]);
  try {
    const expectedU = 0.93;
    const expectedV = 0.21;
    const lon = (expectedU - 0.5) * 2 * Math.PI;
    const lat = (expectedV - 0.5) * Math.PI;
    const distance = 7;
    const point = [
      Math.sin(lon) * Math.cos(lat) * distance,
      Math.sin(lat) * distance,
      Math.cos(lon) * Math.cos(lat) * distance,
    ];
    const payload = await indexProjectWorldPointPayload(
      fixture.map,
      "2",
      { point_world_wds: point },
    );
    assert.ok(Math.abs(Number(payload.erp_u) - expectedU) < 1e-12);
    assert.ok(Math.abs(Number(payload.erp_v) - expectedV) < 1e-12);
    assert.ok(Math.abs(Number(payload.distance_m) - distance) < 1e-12);
  } finally {
    await fixture.cleanup();
  }
});

test("rejects malformed WDS points", async () => {
  const fixture = await createMap([0, 0, 0]);
  try {
    await assert.rejects(
      indexProjectWorldPointPayload(fixture.map, "2", { point_world_wds: [1, 2] }),
      (error: unknown) => error instanceof WorkbenchRouteError && error.status === 400,
    );
  } finally {
    await fixture.cleanup();
  }
});

test("rejects a point at the destination camera origin", async () => {
  const fixture = await createMap([0, 0, 0]);
  try {
    await assert.rejects(
      indexProjectWorldPointPayload(fixture.map, "2", { point_world_wds: [0, 0, 0] }),
      (error: unknown) => error instanceof WorkbenchRouteError && error.status === 422,
    );
  } finally {
    await fixture.cleanup();
  }
});
