/**
 * Parser and frame-adapter tests for the v2 manifest reader.
 *
 * The frame battery is ported from the Python `test_frame_conventions.py`, which
 * guarded the same composition from the other direction. Everything it covered
 * fails *silently* when wrong — a dropped flip mirrors the world, an unnormalised
 * quaternion drifts with distance — so the guards matter more than the happy path.
 */

import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  assetBasename,
  findManifest,
  formatLevel,
  loadMapManifest,
  manifestKeyframeToWorldToCameraWds,
  parseManifest,
  quaternionToMatrix3,
  type ManifestKeyframe,
} from "./map-manifest.js";
// The real consumer of the adapter — testing against it, rather than a local copy
// of the heading math, is what makes these assertions about the integration.
import { keyframeHeadingDegreesFromPose } from "./workbench-index.js";

// ------------------------------------------------------------------------ helpers

function keyframe(
  orientationWxyz: [number, number, number, number],
  positionEus: [number, number, number] = [0, 0, 0],
): ManifestKeyframe {
  return {
    keyframeId: 0,
    imageFilename: "a.jpg",
    depthFilename: "a.tif",
    positionEus,
    orientationWxyz,
  };
}

function angularDiffDeg(a: number, b: number): number {
  return Math.abs((((a - b + 180) % 360) + 360) % 360 - 180);
}

/** `[w, x, y, z]` for a clockwise-from-North yaw about the EUS up axis (+Y). */
function yawQuaternion(yawDeg: number): [number, number, number, number] {
  const half = ((-yawDeg * Math.PI) / 180) / 2;
  return [Math.cos(half), 0, Math.sin(half), 0];
}

function manifestJson(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    local_origin: [2.3522, 48.8566, 35],
    map: { name: "m", uuid: "u", venue_type: "rail" },
    geo_levels: [{ value: 0, min_altitude: -2, max_altitude: 3, geo_ref: 149, geometry: null }],
    geo_keyframes: [
      {
        x: 0,
        y: 0,
        z: 0,
        orientation: [1, 0, 0, 0],
        image_url: "https://example/images/a.jpg",
        depth_url: "https://example/depths/a.tif",
      },
    ],
    ...overrides,
  });
}

// --------------------------------------------------------------------- frames

test("identity orientation is heading North", () => {
  // In EUS the OpenGL camera forward (-Z) points North, so identity means heading 0.
  const pose = manifestKeyframeToWorldToCameraWds(keyframe([1, 0, 0, 0]));
  assert.ok(angularDiffDeg(keyframeHeadingDegreesFromPose(pose), 0) < 1e-9);
});

for (const yawDeg of [0, 90, 180, 270]) {
  test(`yaw ${yawDeg}° maps to compass heading ${yawDeg}`, () => {
    const pose = manifestKeyframeToWorldToCameraWds(keyframe(yawQuaternion(yawDeg)));
    assert.ok(
      angularDiffDeg(keyframeHeadingDegreesFromPose(pose), yawDeg) < 1e-9,
      `got ${keyframeHeadingDegreesFromPose(pose)}`,
    );
  });
}

test("position round-trips through the WDS flip", () => {
  const pose = manifestKeyframeToWorldToCameraWds(keyframe([1, 0, 0, 0], [12.5, 3, -7.25]));
  // world-to-camera translation of an axis-aligned pose: -R·t_c2w, with
  // t_c2w = diag(-1,-1,1)·p_eus = (-12.5, -3, -7.25).
  const cameraToWorldTranslation = [
    -(pose[0][0] * pose[0][3] + pose[1][0] * pose[1][3] + pose[2][0] * pose[2][3]),
    -(pose[0][1] * pose[0][3] + pose[1][1] * pose[1][3] + pose[2][1] * pose[2][3]),
    -(pose[0][2] * pose[0][3] + pose[1][2] * pose[1][3] + pose[2][2] * pose[2][3]),
  ];
  assert.deepEqual(
    cameraToWorldTranslation.map((v) => Number(v.toFixed(9))),
    [-12.5, -3, -7.25],
  );
});

test("the conversion is not accidentally symmetric", () => {
  // Guard the guards. If either flip were the identity, or the two cancelled, the
  // heading tests above would pass for a degenerate reason.
  const identityPose = manifestKeyframeToWorldToCameraWds(keyframe([1, 0, 0, 0]));
  const rotation = [identityPose[0].slice(0, 3), identityPose[1].slice(0, 3), identityPose[2].slice(0, 3)];
  assert.notDeepEqual(rotation, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  // A proper rotation, not a reflection — a reflection would mirror the world and
  // still look plausible on a map.
  const det =
    rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
    - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
    + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
  assert.ok(Math.abs(det - 1) < 1e-12, `det = ${det}`);
});

test("a non-unit quaternion still yields an orthonormal rotation", () => {
  // invertRigid4 inverts by transposing, which is only correct when R is exactly
  // orthonormal. Without normalisation a 1.001-norm quaternion folds a scale into
  // R and depth-pinned points drift in proportion to distance.
  const scaled: [number, number, number, number] = [1.001, 0, 0, 0];
  const r = quaternionToMatrix3(scaled);
  const rTr = [0, 1, 2].map((i) =>
    [0, 1, 2].map((j) => r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j]),
  );
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      assert.ok(Math.abs(rTr[i][j] - (i === j ? 1 : 0)) < 1e-12);
    }
  }
});

test("a degenerate quaternion is rejected rather than silently normalised", () => {
  assert.throws(() => quaternionToMatrix3([0, 0, 0, 0]), /degenerate quaternion/);
});

// --------------------------------------------------------------------- parsing

test("asset basenames ignore query strings and fragments", () => {
  assert.equal(assetBasename("https://e/uuid/images/a.jpg"), "a.jpg");
  assert.equal(assetBasename("https://e/uuid/images/a.jpg?X-Amz-Signature=deadbeef"), "a.jpg");
  assert.equal(assetBasename("https://e/uuid/images/a.jpg#frag"), "a.jpg");
  assert.equal(assetBasename(undefined), "");
});

test("level values are truncated, matching what the bricks emit", () => {
  assert.equal(formatLevel(0), "0");
  assert.equal(formatLevel(3), "3");
  assert.equal(formatLevel(-1), "-1");
});

test("parses origin, levels, geo_ref and keyframes", () => {
  const manifest = parseManifest("/m/m_2_20260101_000000.json", manifestJson());
  assert.deepEqual(manifest.origin, { lng: 2.3522, lat: 48.8566, alt: 35 });
  assert.equal(manifest.geoRefId, 149);
  assert.equal(manifest.venueType, "rail");
  assert.equal(manifest.mapVersion, 2);
  assert.equal(manifest.keyframes.length, 1);
  assert.equal(manifest.keyframes[0].imageFilename, "a.jpg");
  assert.equal(manifest.keyframes[0].depthFilename, "a.tif");
  assert.equal(manifest.keyframeIdByImageFilename.get("a.jpg"), 0);
});

test("null altitude bounds are unbounded, not zero", () => {
  // asNumber(null) === 0 would collapse the band to [0, 0]: every point falls
  // outside every level, `level` becomes null everywhere, and floor filtering
  // silently degrades to "show everything".
  const manifest = parseManifest(
    "/m/m_1_20260101_000000.json",
    manifestJson({
      geo_levels: [{ value: 0, min_altitude: null, max_altitude: null, geo_ref: 1 }],
    }),
  );
  assert.equal(manifest.levels[0].minAltitude, Number.NEGATIVE_INFINITY);
  assert.equal(manifest.levels[0].maxAltitude, Number.POSITIVE_INFINITY);
});

test("levels are sorted by value", () => {
  const manifest = parseManifest(
    "/m/m_1_20260101_000000.json",
    manifestJson({
      geo_levels: [
        { value: 2, min_altitude: 6, max_altitude: 9, geo_ref: 1 },
        { value: 0, min_altitude: -2, max_altitude: 3, geo_ref: 1 },
      ],
    }),
  );
  assert.deepEqual(manifest.levels.map((level) => level.value), [0, 2]);
});

test("keyframe ids are geo_keyframes indices, not sort order", () => {
  const manifest = parseManifest(
    "/m/m_1_20260101_000000.json",
    manifestJson({
      geo_keyframes: ["c.jpg", "a.jpg", "b.jpg"].map((name) => ({
        x: 0,
        y: 0,
        z: 0,
        orientation: [1, 0, 0, 0],
        image_url: `https://e/images/${name}`,
        depth_url: `https://e/depths/${name}`,
      })),
    }),
  );
  assert.equal(manifest.keyframeIdByImageFilename.get("c.jpg"), 0);
  assert.equal(manifest.keyframeIdByImageFilename.get("a.jpg"), 1);
  assert.equal(manifest.keyframeIdByImageFilename.get("b.jpg"), 2);
});

// The Python reader raises on each of these; a more tolerant TS reader would serve
// a map the index build refuses, with keyframes numbered differently on each side.
const rejected: Array<[string, Record<string, unknown>, RegExp]> = [
  ["a short local_origin", { local_origin: [1, 2] }, /local_origin/],
  ["no keyframes", { geo_keyframes: [] }, /no geo_keyframes/],
  [
    "a keyframe with no image_url",
    { geo_keyframes: [{ x: 0, y: 0, z: 0, orientation: [1, 0, 0, 0] }] },
    /has no image_url/,
  ],
  [
    "a three-element orientation",
    {
      geo_keyframes: [
        { x: 0, y: 0, z: 0, orientation: [1, 0, 0], image_url: "https://e/i/a.jpg" },
      ],
    },
    /orientation must be/,
  ],
  [
    "duplicate image filenames",
    {
      geo_keyframes: [0, 1].map(() => ({
        x: 0,
        y: 0,
        z: 0,
        orientation: [1, 0, 0, 0],
        image_url: "https://e/images/same.jpg",
      })),
    },
    /share an image filename/,
  ],
  [
    "several geo_refs",
    {
      geo_levels: [
        { value: 0, min_altitude: 0, max_altitude: 1, geo_ref: 1 },
        { value: 1, min_altitude: 1, max_altitude: 2, geo_ref: 2 },
      ],
    },
    /several geo_refs/,
  ],
];

for (const [label, overrides, pattern] of rejected) {
  test(`rejects ${label}`, () => {
    assert.throws(
      () => parseManifest("/m/m_1_20260101_000000.json", manifestJson(overrides)),
      pattern,
    );
  });
}

// ------------------------------------------------------------------- discovery

test("the newest manifest wins by filename stamp", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "manifest-"));
  await writeFile(path.join(dir, "m_2_20260101_000000.json"), manifestJson());
  await writeFile(path.join(dir, "m_2_20260607_120000.json"), manifestJson());
  await writeFile(path.join(dir, "unrelated.json"), "{}");

  const found = await findManifest(dir);
  assert.equal(path.basename(found ?? ""), "m_2_20260607_120000.json");
});

test("a directory with no manifest reports it by name", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "manifest-"));
  assert.equal(await findManifest(dir), null);
  await assert.rejects(loadMapManifest(dir), /No v2 map manifest/);
});

test("loadMapManifest re-reads an edited manifest", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "manifest-"));
  const file = path.join(dir, "m_1_20260101_000000.json");
  await writeFile(file, manifestJson());
  assert.equal((await loadMapManifest(dir)).keyframes.length, 1);

  await writeFile(
    file,
    manifestJson({
      geo_keyframes: ["a.jpg", "b.jpg"].map((name) => ({
        x: 0,
        y: 0,
        z: 0,
        orientation: [1, 0, 0, 0],
        image_url: `https://e/images/${name}`,
      })),
    }),
  );
  assert.equal((await loadMapManifest(dir)).keyframes.length, 2);
});
