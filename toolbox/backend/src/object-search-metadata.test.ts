import assert from "node:assert/strict";
import { copyFile, mkdir, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import type { MapEntry } from "./config.js";
import {
  loadMetadata,
  MetadataError,
  requireMetadata,
  resolveMetadataPath,
  rowByIndex,
  rowSlice,
  rowsForKeyframe,
} from "./object-search-metadata.js";

const FIXTURES = path.resolve(import.meta.dirname, "..", "test-fixtures");

function mapEntry(mapPath: string): MapEntry {
  return {
    id: "test-map",
    display_name: "test-map",
    path: mapPath,
    emmid: null,
    geo_ref_id: 1,
    object_search: null,
    parent_map: null,
  };
}

/**
 * A map directory holding one or more prepare outputs.
 *
 * `captures` maps a subdirectory (`""` for the flat layout) to the fixture to drop
 * in as its `metadata.parquet`.
 */
async function createMap(
  captures: Record<string, string>,
): Promise<{ map: MapEntry; cleanup: () => Promise<void> }> {
  const mapPath = await mkdtemp(path.join(os.tmpdir(), "object-search-metadata-"));
  for (const [capture, fixture] of Object.entries(captures)) {
    const dir = path.join(mapPath, "object-search", capture);
    await mkdir(dir, { recursive: true });
    await copyFile(path.join(FIXTURES, fixture), path.join(dir, "metadata.parquet"));
  }
  return {
    map: mapEntry(mapPath),
    cleanup: () => rm(mapPath, { recursive: true, force: true }),
  };
}

test("decodes INT64 ids to numbers, not BigInt", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    // The single most silent failure in this migration: a BigInt id looks fine in a
    // JSON dump and never matches a `Map<number, …>` key.
    const first = rowByIndex(metadata, 0);
    assert.equal(typeof first?.rowIndex, "number");
    assert.equal(typeof first?.videoKeyframeId, "number");
    assert.deepEqual(metadata.keyframeIds, [0, 3, 7]);
    assert.ok(metadata.rangeByKeyframe.get(3), "keyframe 3 must be reachable by number");
  } finally {
    await cleanup();
  }
});

test("maps blank thumbnail_key and NaN depth to null", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.equal(rowByIndex(metadata, 0)?.thumbnailKey, "object-search/thumbnails/000000.jpg");
    assert.equal(rowByIndex(metadata, 6)?.thumbnailKey, null);
    assert.equal(rowByIndex(metadata, 0)?.depth, 1.5);
    // depth = NaN is the per-row invisibility class: NULL object_position, invisible
    // to localize. It must not read as "0 m".
    assert.equal(rowByIndex(metadata, 2)?.depth, null);
    assert.equal(metadata.withDepthCount, 5);
    assert.equal(metadata.rangeByKeyframe.get(0)?.withDepth, 2);
  } finally {
    await cleanup();
  }
});

test("keeps a nullable label null rather than the string \"null\"", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.equal(rowByIndex(metadata, 0)?.label, "class-0");
    assert.equal(rowByIndex(metadata, 1)?.label, null);
  } finally {
    await cleanup();
  }
});

test("indexes rows by keyframe and by row_index", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.equal(rowsForKeyframe(metadata, 0).length, 3);
    assert.equal(rowsForKeyframe(metadata, 7).length, 2);
    assert.equal(rowsForKeyframe(metadata, 42).length, 0);
    assert.equal(rowByIndex(metadata, 4)?.videoKeyframeId, 3);
    assert.equal(rowByIndex(metadata, 99), null);
  } finally {
    await cleanup();
  }
});

test("reports a prepare output that was never post-processed", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-raw.parquet" });
  try {
    const metadata = await requireMetadata(map);
    // Reported rather than guessed: without this flag the UI shows 404 previews and
    // "no depth" everywhere, which looks exactly like a bad capture.
    assert.equal(metadata.postprocessed, false);
    assert.equal(rowByIndex(metadata, 0)?.thumbnailKey, null);
    assert.equal(rowByIndex(metadata, 0)?.depth, null);
    assert.equal(metadata.rowCount, 3);
  } finally {
    await cleanup();
  }
});

test("refuses a parquet whose keyframe rows are not contiguous", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-noncontiguous.parquet" });
  try {
    await assert.rejects(requireMetadata(map), /not contiguous/);
  } finally {
    await cleanup();
  }
});

test("finds a per-capture layout and refuses more than one capture", async () => {
  const { map, cleanup } = await createMap({ "capture-a": "metadata-postprocessed.parquet" });
  try {
    const single = await resolveMetadataPath(map);
    assert.equal(single.captureCount, 1);
    assert.ok(single.metadataPath?.endsWith(path.join("capture-a", "metadata.parquet")));
    await assert.doesNotReject(requireMetadata(map));
  } finally {
    await cleanup();
  }

  const two = await createMap({
    "capture-a": "metadata-postprocessed.parquet",
    "capture-b": "metadata-raw.parquet",
  });
  try {
    const resolved = await resolveMetadataPath(two.map);
    assert.equal(resolved.captureCount, 2);
    // Two captures write the same thumbnail_key prefix with row_index restarting at
    // 0, so their thumbnails overwrite each other: browsing them together would show
    // one capture's crop for the other's proposal.
    await assert.rejects(requireMetadata(two.map), /capture directories/);
  } finally {
    await two.cleanup();
  }
});

test("reports where it looked when there is no metadata", async () => {
  const { map, cleanup } = await createMap({});
  try {
    const resolved = await resolveMetadataPath(map);
    assert.equal(resolved.metadataPath, null);
    assert.deepEqual(resolved.checkedPaths, [
      path.join(map.path, "object-search", "metadata.parquet"),
    ]);
    await assert.rejects(requireMetadata(map), (error: unknown) => {
      assert.ok(error instanceof MetadataError);
      assert.equal(error.status, 404);
      assert.match(error.message, /build-index\.sh/);
      return true;
    });
  } finally {
    await cleanup();
  }
});

test("does not cache a failed read", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-noncontiguous.parquet" });
  const metadataPath = path.join(map.path, "object-search", "metadata.parquet");
  try {
    await assert.rejects(loadMetadata(metadataPath), /not contiguous/);
    // The bug this inherits from `loadIndex`: a rejected promise left in the cache
    // meant "no metadata" until the process restarted, even after the file was fixed.
    await copyFile(
      path.join(FIXTURES, "metadata-postprocessed.parquet"),
      metadataPath,
    );
    const metadata = await loadMetadata(metadataPath);
    assert.equal(metadata.rowCount, 7);
  } finally {
    await cleanup();
  }
});

test("counts rows per detector source", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.deepEqual(metadata.detectorSourceCounts, { yolo: 4, gdino: 3 });
  } finally {
    await cleanup();
  }
});

test("round-trips dictionaries and thumbnail prefixes", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.deepEqual(metadata.columns.detectorSource.dictionary, ["yolo", "gdino"]);
    assert.deepEqual(metadata.columns.vkImagePath.dictionary, [
      "images/keyframe-0.jpg",
      "images/keyframe-1.jpg",
    ]);
    assert.equal(metadata.columns.thumbnailKey.storage, "prefix-basename");
    if (metadata.columns.thumbnailKey.storage === "prefix-basename") {
      assert.deepEqual(metadata.columns.thumbnailKey.prefixes, [
        "object-search/thumbnails/",
        "capture-b/thumbnails/",
      ]);
    }
    assert.equal(rowByIndex(metadata, 4)?.thumbnailKey, "capture-b/thumbnails/000004.jpg");
    assert.deepEqual(
      rowSlice(metadata, 0, 2).map((row) => row.label),
      ["class-0", null],
    );
  } finally {
    await cleanup();
  }
});

test("falls back to plain thumbnail strings for a non-conforming key", async () => {
  const { map, cleanup } = await createMap({
    "": "metadata-thumbnail-fallback.parquet",
  });
  try {
    const metadata = await requireMetadata(map);
    assert.equal(metadata.columns.thumbnailKey.storage, "plain");
    assert.deepEqual(
      rowSlice(metadata, 0, 3).map((row) => row.thumbnailKey),
      [
        "capture-a/thumbnails/000042.jpg",
        "legacy-thumb.jpg",
        "capture-b/thumbnails/000007.jpg",
      ],
    );
  } finally {
    await cleanup();
  }
});

test("keeps parquet null separate from a present NaN depth", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-postprocessed.parquet" });
  try {
    const metadata = await requireMetadata(map);
    assert.equal(metadata.columns.depthPresent[2], 1);
    assert.ok(Number.isNaN(metadata.columns.depth[2]));
    assert.equal(metadata.columns.depthPresent[5], 0);
    assert.equal(rowByIndex(metadata, 2)?.depth, null);
    assert.equal(rowByIndex(metadata, 5)?.depth, null);
  } finally {
    await cleanup();
  }
});

test("refuses an INT64 id outside the Int32Array range", async () => {
  const { map, cleanup } = await createMap({ "": "metadata-int32-overflow.parquet" });
  try {
    await assert.rejects(requireMetadata(map), /not a safe int below 2\^31/);
  } finally {
    await cleanup();
  }
});
