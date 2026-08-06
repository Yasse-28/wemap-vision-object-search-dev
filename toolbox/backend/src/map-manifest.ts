/**
 * Read a v2 map manifest — the workbench's pose source.
 *
 * This is the TypeScript counterpart of `toolbox/bricks/map_manifest.py`, and the
 * two must agree exactly. Both halves of the build read the same file: the Python
 * side maps image filename → keyframe id when it indexes, this side does the same
 * when it serves. A disagreement would number keyframes differently in the index
 * and in the workbench — wrong image, wrong pose, no error anywhere. So the parse
 * rules, the errors raised, and the basename semantics below are all deliberately
 * copied rather than reinvented.
 *
 * Keyframe ids are **indices into `geo_keyframes`**. The manifest carries no
 * integer id, and production's are opaque anyway. The consequence to keep in mind
 * is that re-exporting a manifest renumbers everything.
 */

import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

/** `{map_id}_{map_version}_{date}_{time}.json` — e.g. `bbhotel-choisy_2_20260805_083206.json`. */
export const MANIFEST_PATTERN = /^(?<mapId>.+)_(?<version>\d+)_(?<stamp>\d{8}_\d{6})\.json$/;

export type ManifestKeyframe = {
  /** Index in `geo_keyframes`. Doubles as the video-keyframe id, as in the bricks. */
  keyframeId: number;
  imageFilename: string;
  depthFilename: string;
  /** EUS metres: +X East, +Y Up, +Z South. */
  positionEus: [number, number, number];
  /** `[w, x, y, z]`, CameraFrame(OpenGL) → LocalFrame(EUS). */
  orientationWxyz: [number, number, number, number];
};

export type ManifestLevel = {
  value: number;
  /**
   * Heights above the map origin, **not** WGS84 altitudes. `null` in the manifest
   * means unbounded, so it becomes ±Infinity here — mapping it to 0 would collapse
   * the band and silently drop every keyframe out of every level.
   */
  minAltitude: number;
  maxAltitude: number;
};

export type MapManifest = {
  path: string;
  mapId: string;
  mapVersion: number;
  mapUuid: string;
  venueType: string | null;
  geoRefId: number | null;
  /** `[lng, lat, alt]` — Django `PointField` order, longitude first. */
  origin: { lng: number; lat: number; alt: number };
  /** Sorted by `value`; first matching band wins, as in the Python. */
  levels: ManifestLevel[];
  keyframes: ManifestKeyframe[];
  keyframeById: Map<number, ManifestKeyframe>;
  keyframeIdByImageFilename: Map<string, number>;
};

/**
 * Filename an asset URL points at, with any fragment and query discarded.
 *
 * Mirrors Python's `Path(urlparse(url).path).name`. Today's manifests point at a
 * public bucket with no query string, so splitting on "/" would also work — but a
 * presigned URL would yield `abc.jpg?X-Amz-Signature=…`, matching nothing on disk:
 * no keyframe id resolved and no depth map found, both silent. `new URL()` is not
 * used because it throws on relative inputs.
 */
export function assetBasename(url: unknown): string {
  const raw = typeof url === "string" ? url : "";
  const withoutFragment = raw.split("#", 1)[0];
  const withoutQuery = withoutFragment.split("?", 1)[0];
  return withoutQuery.slice(withoutQuery.lastIndexOf("/") + 1);
}

/** Level identifier as a string, matching what the bricks emit (`int(value)`). */
export function formatLevel(value: number): string {
  return String(Math.trunc(value));
}

function asFiniteNumber(value: unknown, context: string): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${context} must be a finite number, got ${JSON.stringify(value)}.`);
  }
  return parsed;
}

/** `null`/missing altitude bounds are unbounded, as `levels_for_altitudes` treats them. */
function asBound(value: unknown, fallback: number): number {
  if (value === null || value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function parseManifest(manifestPath: string, text: string): MapManifest {
  const data = JSON.parse(text) as Record<string, unknown>;
  const label = path.basename(manifestPath);

  const originRaw = data.local_origin;
  if (!Array.isArray(originRaw) || originRaw.length !== 3) {
    throw new Error(`${label}: 'local_origin' must be [lng, lat, alt].`);
  }
  // Longitude first, as in Django's PointField. Getting this backwards puts the
  // whole map on the wrong continent, so it is spelled out rather than unpacked.
  const origin = {
    lng: asFiniteNumber(originRaw[0], `${label}: local_origin[0]`),
    lat: asFiniteNumber(originRaw[1], `${label}: local_origin[1]`),
    alt: asFiniteNumber(originRaw[2], `${label}: local_origin[2]`),
  };

  const levelsRaw = Array.isArray(data.geo_levels) ? data.geo_levels : [];
  const levels: ManifestLevel[] = levelsRaw.map((entry) => {
    const level = (entry ?? {}) as Record<string, unknown>;
    return {
      value: asFiniteNumber(level.value, `${label}: geo_levels[].value`),
      minAltitude: asBound(level.min_altitude, Number.NEGATIVE_INFINITY),
      maxAltitude: asBound(level.max_altitude, Number.POSITIVE_INFINITY),
    };
  });
  levels.sort((a, b) => a.value - b.value);

  const geoRefIds = new Set(
    levelsRaw
      .map((entry) => (entry as Record<string, unknown> | null)?.geo_ref)
      .filter((value) => value !== null && value !== undefined)
      .map((value) => Number(value)),
  );
  if (geoRefIds.size > 1) {
    throw new Error(
      `${label}: geo_levels reference several geo_refs: ${[...geoRefIds].join(", ")}.`,
    );
  }
  const geoRefId = geoRefIds.size === 1 ? [...geoRefIds][0] : null;

  const keyframesRaw = Array.isArray(data.geo_keyframes) ? data.geo_keyframes : [];
  const keyframes: ManifestKeyframe[] = keyframesRaw.map((entry, index) => {
    const keyframe = (entry ?? {}) as Record<string, unknown>;
    const imageFilename = assetBasename(keyframe.image_url);
    if (!imageFilename) {
      throw new Error(`${label}: geo_keyframes[${index}] has no image_url.`);
    }
    const orientation = keyframe.orientation;
    if (!Array.isArray(orientation) || orientation.length !== 4) {
      throw new Error(`${label}: geo_keyframes[${index}].orientation must be [w, x, y, z].`);
    }
    return {
      keyframeId: index,
      imageFilename,
      depthFilename: assetBasename(keyframe.depth_url),
      positionEus: [
        asFiniteNumber(keyframe.x, `${label}: geo_keyframes[${index}].x`),
        asFiniteNumber(keyframe.y, `${label}: geo_keyframes[${index}].y`),
        asFiniteNumber(keyframe.z, `${label}: geo_keyframes[${index}].z`),
      ],
      orientationWxyz: orientation.map((value, axis) =>
        asFiniteNumber(value, `${label}: geo_keyframes[${index}].orientation[${axis}]`),
      ) as [number, number, number, number],
    };
  });
  if (keyframes.length === 0) {
    throw new Error(`${label}: no geo_keyframes.`);
  }

  const keyframeIdByImageFilename = new Map(
    keyframes.map((keyframe) => [keyframe.imageFilename, keyframe.keyframeId]),
  );
  const duplicates = keyframes.length - keyframeIdByImageFilename.size;
  if (duplicates > 0) {
    throw new Error(
      `${label}: ${duplicates} keyframe(s) share an image filename, so an image cannot `
      + "be mapped to one pose.",
    );
  }

  const match = MANIFEST_PATTERN.exec(label);
  const mapBlock = (data.map ?? {}) as Record<string, unknown>;
  return {
    path: manifestPath,
    mapId: String(mapBlock.name || match?.groups?.mapId || path.parse(label).name),
    mapVersion: match ? Number(match.groups?.version) : 0,
    mapUuid: String(mapBlock.uuid ?? ""),
    venueType: mapBlock.venue_type ? String(mapBlock.venue_type) : null,
    geoRefId,
    origin,
    levels,
    keyframes,
    keyframeById: new Map(keyframes.map((keyframe) => [keyframe.keyframeId, keyframe])),
    keyframeIdByImageFilename,
  };
}

/**
 * The v2 manifest in a map directory, or null.
 *
 * When several exist the newest wins by the `{date}_{time}` stamp in the name —
 * that stamp is why the filename carries it. Stale manifests renumber keyframes,
 * so this matches the Python's choice deliberately.
 */
export async function findManifest(mapPath: string): Promise<string | null> {
  let entries: string[];
  try {
    entries = await readdir(mapPath);
  } catch {
    return null;
  }
  const matches = entries
    .map((name) => ({ name, match: MANIFEST_PATTERN.exec(name) }))
    .filter((item): item is { name: string; match: RegExpExecArray } => item.match !== null)
    .sort((a, b) => (a.match.groups!.stamp < b.match.groups!.stamp ? -1 : 1));
  const newest = matches.at(-1);
  return newest ? path.join(mapPath, newest.name) : null;
}

// Keyed on path + mtime + size so an edited manifest is re-read. The promise is
// cached rather than the value: /ui/api/maps polls, and concurrent keyframe
// requests would otherwise each parse the file.
const manifestCache = new Map<string, Promise<MapManifest>>();

/** Load and cache the manifest for a map directory. Throws if there is none. */
export async function loadMapManifest(mapPath: string): Promise<MapManifest> {
  const manifestPath = await findManifest(mapPath);
  if (!manifestPath) {
    throw new Error(
      `No v2 map manifest in '${mapPath}'. Expected a file named `
      + "'{map_id}_{version}_{date}_{time}.json'.",
    );
  }
  const info = await stat(manifestPath);
  const cacheKey = `${manifestPath}:${info.mtimeMs}:${info.size}`;
  const cached = manifestCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const promise = readFile(manifestPath, "utf8")
    .then((text) => parseManifest(manifestPath, text))
    .catch((error: unknown) => {
      manifestCache.delete(cacheKey);
      throw error;
    });
  manifestCache.set(cacheKey, promise);
  return promise;
}

// ------------------------------------------------------------------ frame adapter

// EUS (East, Up, South) ← WDS (West, Down, South): East = -West, Up = -Down.
const ROT_WDS_TO_EUS = [
  [-1, 0, 0],
  [0, -1, 0],
  [0, 0, 1],
];

// OpenGL (+Y up, -Z forward) ← OpenCV (+Y down, +Z forward).
const ROT_OPENGL_TO_OPENCV = [
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
];

function matMul3(a: number[][], b: number[][]): number[][] {
  return [0, 1, 2].map((row) =>
    [0, 1, 2].map((col) => a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]),
  );
}

/**
 * Rotation matrix for a `[w, x, y, z]` quaternion.
 *
 * Mirrors `toolbox/bricks/vendored/maths/quaternion.py::to_matrix3` — row-major,
 * active rotation. The quaternion is normalised first: `invertRigid4` inverts by
 * transposing the rotation block, which is only the inverse when the block is
 * exactly orthonormal. A quaternion arriving through JSON at 1.0001 in norm would
 * otherwise fold a scale factor into the matrix, and every depth-pinned point
 * would drift in proportion to its distance, with no error raised.
 */
export function quaternionToMatrix3(q: readonly number[]): number[][] {
  const norm = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(norm) || norm < 1e-12) {
    throw new Error(`Cannot build a rotation from a degenerate quaternion [${q.join(", ")}].`);
  }
  const [w, x, y, z] = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * A manifest keyframe as the 4x4 world-to-camera matrix in WDS this module's
 * consumers expect.
 *
 * Everything downstream in `workbench-index.ts` — headings, the ERP projection,
 * the view cone, `point_world_wds` on the wire — works in the WDS world-to-camera
 * frame that `georef.db` used to store directly. Rather than convert all of it to
 * EUS (and change the meaning of a field the frontend round-trips), this adapter
 * puts a manifest pose back into that frame.
 *
 * It is the bricks' v1 conversion run backwards. Forward, `georef_source` had:
 *
 *     R_eus = ROT_WDS_TO_EUS · R_c2w_wds · ROT_OPENGL_TO_OPENCV
 *     p_eus = ROT_WDS_TO_EUS · t_c2w_wds
 *
 * Both matrices are diagonal ±1, hence self-inverse, so the inverse is the same
 * expression. The Python test suite carries this exact composition as a fixture
 * builder (`_camera_to_world_wds_for_eus`), which is where it was verified.
 */
export function manifestKeyframeToWorldToCameraWds(keyframe: ManifestKeyframe): number[][] {
  const rotationC2w = matMul3(
    matMul3(ROT_WDS_TO_EUS, quaternionToMatrix3(keyframe.orientationWxyz)),
    ROT_OPENGL_TO_OPENCV,
  );
  const [ex, ey, ez] = keyframe.positionEus;
  const translationC2w = [
    ROT_WDS_TO_EUS[0][0] * ex,
    ROT_WDS_TO_EUS[1][1] * ey,
    ROT_WDS_TO_EUS[2][2] * ez,
  ];

  // Invert the rigid transform: R⁻¹ = Rᵀ (safe — the quaternion was normalised).
  const rT = [0, 1, 2].map((row) => [0, 1, 2].map((col) => rotationC2w[col][row]));
  const invT = [0, 1, 2].map(
    (row) =>
      -(
        rT[row][0] * translationC2w[0]
        + rT[row][1] * translationC2w[1]
        + rT[row][2] * translationC2w[2]
      ),
  );
  return [
    [rT[0][0], rT[0][1], rT[0][2], invT[0]],
    [rT[1][0], rT[1][1], rT[1][2], invT[1]],
    [rT[2][0], rT[2][1], rT[2][2], invT[2]],
    [0, 0, 0, 1],
  ];
}
