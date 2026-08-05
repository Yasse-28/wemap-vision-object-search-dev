import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";

import type { MapEntry } from "./config.js";
import {
  indexProjectWorldPointPayload,
  keyframeHeadingDegreesFromPose,
  WorkbenchRouteError,
} from "./workbench-index.js";

const execFileAsync = promisify(execFile);

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

function storedPoseHex(pose: number[][]): string {
  const buffer = Buffer.alloc(16 * 8);
  let offset = 0;
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      buffer.writeDoubleLE(pose[col][row], offset);
      offset += 8;
    }
  }
  return buffer.toString("hex");
}

async function createMap(pose: number[][]): Promise<{ map: MapEntry; cleanup: () => Promise<void> }> {
  const mapPath = await mkdtemp(path.join(os.tmpdir(), "object-search-projection-"));
  const dbPath = path.join(mapPath, "georef.db");
  const sql = `
    CREATE TABLE GeoRef(latitude REAL, longitude REAL, altitude REAL);
    INSERT INTO GeoRef VALUES(45.0, 6.0, 100.0);
    CREATE TABLE Level(id INTEGER, min_altitude REAL, max_altitude REAL);
    INSERT INTO Level VALUES(0, -10.0, 10.0);
    CREATE TABLE GeoRefKeyframe(id INTEGER PRIMARY KEY, pose BLOB);
    INSERT INTO GeoRefKeyframe VALUES(2, X'${storedPoseHex(pose)}');
  `;
  await execFileAsync(process.env.OBJECT_SEARCH_WORKBENCH_SQLITE ?? "sqlite3", [dbPath, sql]);
  return {
    map: {
      id: "test",
      display_name: "Test",
      path: mapPath,
      emmid: null,
      geo_ref_id: 1,
      object_search_index_path: null,
      object_search: null,
    },
    cleanup: () => rm(mapPath, { recursive: true, force: true }),
  };
}

test("projects a WDS point into an identity keyframe ERP", async () => {
  const fixture = await createMap([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
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
  const fixture = await createMap([
    [1, 0, 0, -1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
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
  const fixture = await createMap([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
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
  const fixture = await createMap([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
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
  const fixture = await createMap([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]);
  try {
    await assert.rejects(
      indexProjectWorldPointPayload(fixture.map, "2", { point_world_wds: [0, 0, 0] }),
      (error: unknown) => error instanceof WorkbenchRouteError && error.status === 422,
    );
  } finally {
    await fixture.cleanup();
  }
});
