import assert from "node:assert/strict";
import { copyFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import type { MapEntry } from "./config.js";
import {
  columnarKeyframeMarkers,
  indexProjectWorldPointPayload,
  keyframeHeadingDegreesFromPose,
  metadataKeyframesPayload,
  objectSearchMetadataMarkersPayload,
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

test("status leaves marker and keyframe tables on their own routes", async () => {
  const fixture = await createMetadataMap();
  try {
    const payload = await objectSearchMetadataStatusPayload(fixture.map);
    const summary = asRecord(payload.summary);
    assert.equal("markers" in payload, false);
    assert.equal("keyframes" in summary, false);
    assert.equal(payload.marker_count, 3);
    assert.equal(payload.first_keyframe_id, "0");
  } finally {
    await fixture.cleanup();
  }
});

test("marker payload omits dense ids and dictionary-encodes levels", async () => {
  const fixture = await createMetadataMap();
  try {
    const columns = await objectSearchMetadataMarkersPayload(fixture.map);
    assert.equal(columns.ids_are_dense, true);
    assert.equal("ids" in columns, false);
    assert.deepEqual(columns.levels, ["0"]);
    assert.deepEqual(columns.level_codes, [0, 0, 0]);
  } finally {
    await fixture.cleanup();
  }
});

test("marker payload sends non-dense ids and reserves -1 for null levels", () => {
  const columns = columnarKeyframeMarkers([
    { id: "2", latitude: 1, longitude: 2, level: "L1", heading_deg: 90 },
    { id: "7", latitude: 3, longitude: 4, level: null, heading_deg: null },
    { id: "9", latitude: 5, longitude: 6, level: "L1", heading_deg: 180 },
  ]);
  assert.equal(columns.ids_are_dense, false);
  assert.deepEqual(columns.ids, ["2", "7", "9"]);
  assert.deepEqual(columns.levels, ["L1"]);
  assert.deepEqual(columns.level_codes, [0, -1, 0]);
});

test("keyframe pages match the former client-side sort and filters", async () => {
  const fixture = await createMetadataMap();
  const parquetOrder = [
    { id: "0", row_count: 3 },
    { id: "3", row_count: 2 },
    { id: "7", row_count: 2 },
  ];
  try {
    for (const sort of ["parquet", "objects-desc", "objects-asc"] as const) {
      for (const includeEmpty of [true, false]) {
        for (const keyframeId of [null, "3", "missing"]) {
          let expected = parquetOrder
            .filter((item) => keyframeId === null || item.id === keyframeId)
            .filter((item) => includeEmpty || item.row_count > 0);
          if (sort === "objects-desc") {
            expected = [...expected].sort(
              (a, b) => b.row_count - a.row_count || a.id.localeCompare(b.id),
            );
          } else if (sort === "objects-asc") {
            expected = [...expected].sort(
              (a, b) => a.row_count - b.row_count || a.id.localeCompare(b.id),
            );
          }
          const payload = await metadataKeyframesPayload(fixture.map, {
            offset: 0,
            limit: 2,
            sort,
            includeEmpty,
            keyframeId,
          });
          assert.equal(payload.total, expected.length);
          assert.deepEqual(payload.ids, expected.slice(0, 2).map((item) => item.id));
          assert.deepEqual(
            payload.row_count,
            expected.slice(0, 2).map((item) => item.row_count),
          );
        }
      }
    }
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
