/**
 * The two pure helpers behind the export UI.
 *
 * `keyframesInRois` is what the panel's live count means, and it has to match the
 * floor rule the rest of the annotation surface already uses. `buildMapTree` is
 * only interesting in the cases the flat list never had: a parent that is gone, and
 * a `parent_map` cycle a hand-edited config can produce.
 */

import assert from "node:assert/strict";
import test from "node:test";

import type { MapSummary } from "../api";
import type { KeyframeMarker } from "../index-explorer/types";
import { buildMapTree } from "./mapTree";
import { keyframesInRois, roisForRequest } from "./selection";
import type { RoiEntry } from "./types";

function marker(
  id: string,
  longitude: number,
  latitude: number,
  level: string | null,
): KeyframeMarker {
  return { id, longitude, latitude, level, heading_deg: null };
}

/** A square region, given as the open ring the livemap produces. */
function square(
  id: string,
  level: string | null,
  west: number,
  south: number,
  east: number,
  north: number,
): RoiEntry {
  return {
    id,
    level,
    createdAt: 0,
    ring: [
      { longitude: west, latitude: south },
      { longitude: east, latitude: south },
      { longitude: east, latitude: north },
      { longitude: west, latitude: north },
    ],
  };
}

test("a region only takes keyframes on its own floor", () => {
  const markers = [
    marker("0", 1, 1, "0"),
    marker("1", 1, 1, "1"),
    marker("2", 50, 50, "0"),
  ];

  const ground = keyframesInRois(markers, [square("a", "0", 0, 0, 2, 2)]);
  assert.deepEqual(ground.keyframeIds, ["0"]);
  assert.deepEqual(ground.perLevel, [{ level: "0", count: 1 }]);

  const first = keyframesInRois(markers, [square("a", "1", 0, 0, 2, 2)]);
  assert.deepEqual(first.keyframeIds, ["1"]);
});

test("floors compare canonically, so 0 and 0.0 are one floor", () => {
  const selection = keyframesInRois(
    [marker("0", 1, 1, "0.0")],
    [square("a", "0", 0, 0, 2, 2)],
  );
  assert.equal(selection.total, 1);
});

test("overlapping regions credit both but count the keyframe once", () => {
  const selection = keyframesInRois(
    [marker("0", 1, 1, "0")],
    [square("a", "0", 0, 0, 2, 2), square("b", "0", 0, 0, 3, 3)],
  );

  assert.equal(selection.total, 1);
  assert.equal(selection.countByRoi.a, 1);
  assert.equal(selection.countByRoi.b, 1);
});

test("regions on several floors union", () => {
  const selection = keyframesInRois(
    [marker("0", 1, 1, "0"), marker("1", 1, 1, "1"), marker("2", 9, 9, "1")],
    [square("a", "0", 0, 0, 2, 2), square("b", "1", 0, 0, 2, 2)],
  );

  assert.deepEqual(selection.keyframeIds, ["0", "1"]);
  assert.deepEqual(selection.perLevel, [
    { level: "0", count: 1 },
    { level: "1", count: 1 },
  ]);
});

test("a region with no floor matches only floorless keyframes", () => {
  const markers = [marker("0", 1, 1, null), marker("1", 1, 1, "0")];
  const selection = keyframesInRois(markers, [square("a", null, 0, 0, 2, 2)]);
  assert.deepEqual(selection.keyframeIds, ["0"]);
});

test("the request shape is [lng, lat] pairs with a numeric level", () => {
  const [roi] = roisForRequest([square("a", "1", 0, 0, 2, 2)]);
  assert.deepEqual(roi.ring[0], [0, 0]);
  assert.deepEqual(roi.ring[1], [2, 0]);
  assert.equal(roi.level, 1);

  const [floorless] = roisForRequest([square("b", null, 0, 0, 2, 2)]);
  assert.equal(floorless.level, null);
});

// --- the map tree -----------------------------------------------------------

function summary(id: string, parent: string | null): MapSummary {
  return {
    id,
    display_name: id,
    path: `/maps/${id}`,
    emmid: null,
    geo_ref_id: 1,
    object_search_available: true,
    unavailable_reason: null,
    parent_map: parent,
    child_map_ids: [],
  };
}

test("sub-maps nest under their parent, to any depth", () => {
  const tree = buildMapTree([
    summary("vinci", null),
    summary("vinci-roi", "vinci"),
    summary("vinci-roi-north", "vinci-roi"),
  ]);

  assert.equal(tree.length, 1);
  assert.equal(tree[0].map.id, "vinci");
  assert.equal(tree[0].children[0].map.id, "vinci-roi");
  assert.equal(tree[0].children[0].depth, 1);
  assert.equal(tree[0].children[0].children[0].map.id, "vinci-roi-north");
  assert.equal(tree[0].children[0].children[0].depth, 2);
});

test("a sub-map whose parent is gone stays visible at the root", () => {
  const tree = buildMapTree([summary("vinci-roi", "vinci")]);

  assert.deepEqual(
    tree.map((node) => node.map.id),
    ["vinci-roi"],
  );
});

test("a parent_map cycle is broken instead of recursed into", () => {
  const tree = buildMapTree([summary("a", "b"), summary("b", "a")]);

  // Neither is reachable from a root, so both surface flat rather than vanish.
  assert.deepEqual(
    tree.map((node) => node.map.id).sort(),
    ["a", "b"],
  );
  assert.ok(tree.every((node) => node.children.length === 0));
});

test("a map naming itself as parent is a root", () => {
  const tree = buildMapTree([summary("a", "a")]);
  assert.equal(tree.length, 1);
  assert.equal(tree[0].children.length, 0);
});
