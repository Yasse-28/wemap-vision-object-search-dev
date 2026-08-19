/**
 * The handoff is the only thing that survives the tab switch, and every failure
 * here is silent: a dropped row index looks like the user never clicked, and a
 * queue that never advances looks like the Save button is broken.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_HANDOFF,
  parseHandoff,
  withKeyframe,
  withPanoramaPoint,
  withProposal,
  withSelection,
  withSkipped,
  withValidated,
} from "./handoff";

test("a malformed entry reads as empty rather than throwing on arrival", () => {
  assert.deepEqual(parseHandoff(null), EMPTY_HANDOFF);
  assert.deepEqual(parseHandoff("not json"), EMPTY_HANDOFF);
  assert.deepEqual(parseHandoff("[]"), EMPTY_HANDOFF);
  assert.deepEqual(parseHandoff('{"queue":["7",-1,3.5,4]}').queue, [4]);
});

test("a pin missing a ratio is dropped, not half-restored", () => {
  assert.equal(parseHandoff('{"pin":{"projection":"erp","xRatio":0.5}}').pin, null);
  assert.equal(parseHandoff('{"pin":{"projection":"other","xRatio":0.5,"yRatio":0.5}}').pin, null);
  assert.deepEqual(parseHandoff('{"pin":{"projection":"cutout","xRatio":0.5,"yRatio":0.25}}').pin, {
    projection: "cutout",
    xRatio: 0.5,
    yRatio: 0.25,
  });
});

test("re-sending a queued proposal re-selects it instead of duplicating it", () => {
  let handoff = withProposal(EMPTY_HANDOFF, { keyframeId: "1287", rowIndex: 44812 });
  handoff = withProposal(handoff, { keyframeId: "1287", rowIndex: 44815 });
  handoff = withProposal(handoff, { keyframeId: "1287", rowIndex: 44812 });
  assert.deepEqual(handoff.queue, [44812, 44815]);
  assert.equal(handoff.rowIndex, 44812);
});

test("changing keyframe starts a new queue", () => {
  let handoff = withProposal(EMPTY_HANDOFF, { keyframeId: "1287", rowIndex: 44812 });
  handoff = withValidated(handoff);
  handoff = withProposal(handoff, { keyframeId: "1290", rowIndex: 51004 });
  assert.deepEqual(handoff.queue, [51004]);
  assert.deepEqual(handoff.done, []);
});

test("a panorama point keeps the queue of its own keyframe", () => {
  let handoff = withProposal(EMPTY_HANDOFF, { keyframeId: "1287", rowIndex: 44812 });
  handoff = withPanoramaPoint(handoff, {
    keyframeId: "1287",
    pin: { projection: "erp", xRatio: 0.5, yRatio: 0.5 },
  });
  assert.equal(handoff.rowIndex, null);
  assert.deepEqual(handoff.queue, [44812]);
});

test("validating advances to the first row still to do", () => {
  let handoff = EMPTY_HANDOFF;
  for (const rowIndex of [1, 2, 3]) {
    handoff = withProposal(handoff, { keyframeId: "1287", rowIndex });
  }
  handoff = withSelection(handoff, 1);
  handoff = withValidated(handoff);
  assert.equal(handoff.rowIndex, 2);
  handoff = withValidated(handoff);
  assert.equal(handoff.rowIndex, 3);
  handoff = withValidated(handoff);
  // Nothing is left: the last row stays selected rather than emptying the capsule.
  assert.equal(handoff.rowIndex, 3);
  assert.deepEqual(handoff.done, [1, 2, 3]);
});

test("skipping moves on without marking the row done", () => {
  let handoff = EMPTY_HANDOFF;
  for (const rowIndex of [1, 2]) {
    handoff = withProposal(handoff, { keyframeId: "1287", rowIndex });
  }
  handoff = withSelection(handoff, 1);
  handoff = withSkipped(handoff);
  assert.equal(handoff.rowIndex, 2);
  assert.deepEqual(handoff.done, []);
  // Skipping the last one wraps back to the row still to do.
  handoff = withSkipped(handoff);
  assert.equal(handoff.rowIndex, 1);
});

test("a pin is consumed by the move, not carried to the next row", () => {
  let handoff = withProposal(EMPTY_HANDOFF, {
    keyframeId: "1287",
    rowIndex: 1,
    pin: { projection: "cutout", xRatio: 0.5, yRatio: 0.5 },
  });
  handoff = withProposal(handoff, { keyframeId: "1287", rowIndex: 2 });
  handoff = withSelection(handoff, 1);
  assert.equal(handoff.pin, null);
});

test("navigating to another keyframe drops the queue of the old turn", () => {
  let handoff = withProposal(EMPTY_HANDOFF, { keyframeId: "1287", rowIndex: 44812 });
  handoff = withValidated(handoff);
  const moved = withKeyframe(handoff, "1290");
  assert.equal(moved.keyframeId, "1290");
  assert.equal(moved.rowIndex, null);
  assert.deepEqual(moved.queue, []);
  assert.deepEqual(moved.done, []);
});

test("re-selecting the keyframe already open keeps the queue untouched", () => {
  const handoff = withProposal(EMPTY_HANDOFF, { keyframeId: "1287", rowIndex: 44812 });
  assert.equal(withKeyframe(handoff, "1287"), handoff);
});
