import assert from "node:assert/strict";
import test from "node:test";

import { parseKeyframeGraph } from "./keyframe-graph.js";

test("parses graph edges and ignores point features", () => {
  const parsed = parseKeyframeGraph({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { id: 12, level: 3 },
        geometry: { type: "Point", coordinates: [2.1, 48.8] },
      },
      {
        type: "Feature",
        properties: {
          keyframeId1: 12,
          keyframeId2: 13,
          level: 3,
        },
        geometry: {
          type: "LineString",
          coordinates: [
            [2.1, 48.8],
            [2.2, 48.9],
          ],
        },
      },
    ],
  });

  assert.equal(parsed.skippedFeatureCount, 0);
  assert.deepEqual(parsed.edges, [
    {
      id: "graph-edge-12-13-1",
      keyframe_id_1: "12",
      keyframe_id_2: "13",
      from: [2.1, 48.8],
      to: [2.2, 48.9],
      levels: ["3"],
    },
  ]);
});

test("preserves both levels for cross-floor edges", () => {
  const parsed = parseKeyframeGraph({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { keyframeId1: "a", keyframeId2: "b", level: [1, 2] },
        geometry: {
          type: "LineString",
          coordinates: [
            [2.1, 48.8],
            [2.15, 48.85],
            [2.2, 48.9],
          ],
        },
      },
    ],
  });

  assert.deepEqual(parsed.edges[0]?.levels, ["1", "2"]);
  assert.deepEqual(parsed.edges[0]?.from, [2.1, 48.8]);
  assert.deepEqual(parsed.edges[0]?.to, [2.2, 48.9]);
});

test("skips malformed line features", () => {
  const parsed = parseKeyframeGraph({
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: {},
        geometry: { type: "LineString", coordinates: [[2.1]] },
      },
    ],
  });

  assert.deepEqual(parsed.edges, []);
  assert.equal(parsed.skippedFeatureCount, 1);
});

test("rejects non-FeatureCollection input", () => {
  assert.throws(
    () => parseKeyframeGraph({ type: "Feature", features: [] }),
    /FeatureCollection/,
  );
});
