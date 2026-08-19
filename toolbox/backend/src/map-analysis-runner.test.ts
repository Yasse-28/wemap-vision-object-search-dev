/**
 * The disk side of the analysis runner: which cached run the panel opens on.
 *
 * The python call itself is not exercised here — it reads a whole parquet and takes
 * minutes. What is worth pinning is the cache contract, because every one of its
 * failure modes is silent: a half-written run must not be offered as the latest one,
 * run ids must not escape the map directory, and a missing file must be a 404 rather
 * than an empty payload the panel would render as a measured zero.
 */

import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import type { MapEntry } from "./config.js";
import {
  analysisLayerPayload,
  analysisRunPayload,
  analysisStatusPayload,
} from "./map-analysis-runner.js";
import { WorkbenchRouteError } from "./workbench-index.js";

async function scratchMap(): Promise<MapEntry> {
  const dir = await mkdtemp(path.join(tmpdir(), "map-analysis-"));
  return {
    id: "scratch",
    display_name: "scratch",
    path: dir,
    emmid: null,
    geo_ref_id: 1,
    object_search: null,
    parent_map: null,
  };
}

async function writeLayer(
  map: MapEntry,
  runId: string,
  name: string,
  collection: unknown,
): Promise<void> {
  const dir = path.join(map.path, "analysis", runId, "layers");
  await mkdir(dir, { recursive: true });
  await writeFile(
    path.join(dir, `${name}.geojson`),
    JSON.stringify(collection),
    "utf8",
  );
}

async function writeRun(
  map: MapEntry,
  runId: string,
  payload: unknown,
  report = "===== S0 — inventaire\n",
): Promise<void> {
  const dir = path.join(map.path, "analysis", runId);
  await mkdir(dir, { recursive: true });
  if (payload !== undefined) {
    await writeFile(path.join(dir, "analysis.json"), JSON.stringify(payload), "utf8");
  }
  await writeFile(path.join(dir, "report.txt"), report, "utf8");
}

test("a map never analysed reports no run rather than failing", async () => {
  const map = await scratchMap();
  const status = await analysisStatusPayload(map);
  assert.deepEqual(status.runs, []);
  assert.equal(status.latest_run_id, null);
  assert.equal(status.analysis, null);
  assert.equal(status.job, null);
});

test("the newest run is the one offered, ordered by id", async () => {
  const map = await scratchMap();
  await writeRun(map, "2026-08-18_10-00-00", { s0: { map: "old" } });
  await writeRun(map, "2026-08-19_09-00-00", { s0: { map: "new" } });
  const status = await analysisStatusPayload(map);
  assert.equal(status.latest_run_id, "2026-08-19_09-00-00");
  assert.deepEqual(status.runs, ["2026-08-19_09-00-00", "2026-08-18_10-00-00"]);
  assert.deepEqual(status.analysis, { s0: { map: "new" } });
});

test("a run without a payload is skipped, so a crashed run is not the latest", async () => {
  const map = await scratchMap();
  await writeRun(map, "2026-08-18_10-00-00", { s0: { map: "old" } });
  // In flight or crashed: the directory and the report exist, the JSON does not.
  await writeRun(map, "2026-08-19_09-00-00", undefined);
  const status = await analysisStatusPayload(map);
  assert.equal(status.latest_run_id, "2026-08-18_10-00-00");
  assert.deepEqual(status.runs, ["2026-08-18_10-00-00"]);
});

test("an unparseable payload leaves the analysis null instead of throwing", async () => {
  const map = await scratchMap();
  const dir = path.join(map.path, "analysis", "2026-08-19_09-00-00");
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, "analysis.json"), "{ truncated", "utf8");
  const status = await analysisStatusPayload(map);
  assert.equal(status.latest_run_id, "2026-08-19_09-00-00");
  assert.equal(status.analysis, null);
});

test("one run's payload and its report text are both readable", async () => {
  const map = await scratchMap();
  await writeRun(map, "run-1", { s3: { covered_in_own_keyframe: 0.63 } }, "hello\n");
  assert.deepEqual(await analysisRunPayload(map, "run-1", false), {
    s3: { covered_in_own_keyframe: 0.63 },
  });
  assert.deepEqual(await analysisRunPayload(map, "run-1", true), {
    run_id: "run-1",
    report: "hello\n",
  });
});

test("a missing run is a 404, not an empty payload", async () => {
  const map = await scratchMap();
  await assert.rejects(
    () => analysisRunPayload(map, "nope", false),
    (error: unknown) =>
      error instanceof WorkbenchRouteError && error.status === 404,
  );
});

test("a run id cannot walk out of the map directory", async () => {
  const map = await scratchMap();
  await assert.rejects(
    () => analysisRunPayload(map, "../../etc", false),
    (error: unknown) =>
      error instanceof WorkbenchRouteError && error.status === 400,
  );
});

test("the status lists the layers the latest run actually wrote", async () => {
  const map = await scratchMap();
  await writeRun(map, "run-1", { s0: {} });
  await writeLayer(map, "run-1", "depth-range", { type: "FeatureCollection" });
  await writeLayer(map, "run-1", "parallax", { type: "FeatureCollection" });

  const status = await analysisStatusPayload(map);

  assert.deepEqual(status.layers, ["depth-range", "parallax"]);
});

test("a run with no layers directory reports no layers, not an error", async () => {
  const map = await scratchMap();
  await writeRun(map, "run-1", { s0: {} });

  assert.deepEqual((await analysisStatusPayload(map)).layers, []);
});

test("only the newest run's layers are offered", async () => {
  // The picker is driven by this list, and a name that exists in an older run would
  // 404 against the run the panel is actually reading.
  const map = await scratchMap();
  await writeRun(map, "2026-08-18_10-00-00", { s0: {} });
  await writeLayer(map, "2026-08-18_10-00-00", "retired-layer", {});
  await writeRun(map, "2026-08-19_09-00-00", { s0: {} });
  await writeLayer(map, "2026-08-19_09-00-00", "depth-range", {});

  assert.deepEqual((await analysisStatusPayload(map)).layers, ["depth-range"]);
});

test("a layer is served as the collection the tool wrote", async () => {
  const map = await scratchMap();
  const collection = { type: "FeatureCollection", features: [{ id: 1 }] };
  await writeLayer(map, "run-1", "depth-scatter", collection);

  assert.deepEqual(
    await analysisLayerPayload(map, "run-1", "depth-scatter"),
    collection,
  );
});

test("an empty layer is served as empty rather than as a 404", async () => {
  // An empty collection is a measured state — the layer's input is missing for this
  // map — and the panel says so. A 404 would read as a broken run instead.
  const map = await scratchMap();
  await writeLayer(map, "run-1", "depth-scatter", {
    type: "FeatureCollection",
    features: [],
  });

  const collection = (await analysisLayerPayload(map, "run-1", "depth-scatter")) as {
    features: unknown[];
  };

  assert.deepEqual(collection.features, []);
});

test("a layer name cannot walk out of the layers directory", async () => {
  const map = await scratchMap();
  await writeLayer(map, "run-1", "depth-range", {});

  await assert.rejects(
    () => analysisLayerPayload(map, "run-1", "../analysis"),
    (error: unknown) => error instanceof WorkbenchRouteError && error.status === 400,
  );
});

test("a layer the run did not write is a 404", async () => {
  const map = await scratchMap();
  await writeLayer(map, "run-1", "depth-range", {});

  await assert.rejects(
    () => analysisLayerPayload(map, "run-1", "parallax"),
    (error: unknown) => error instanceof WorkbenchRouteError && error.status === 404,
  );
});
