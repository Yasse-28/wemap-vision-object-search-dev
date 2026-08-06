import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import type { MapEntry } from "./config.js";
import {
  indexProjectWorldPointPayload,
  keyframeHeadingDegreesFromPose,
  WorkbenchRouteError,
} from "./workbench-index.js";

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
