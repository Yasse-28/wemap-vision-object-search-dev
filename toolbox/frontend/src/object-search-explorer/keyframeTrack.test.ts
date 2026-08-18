/**
 * The two cuts in the capture path.
 *
 * Both failures are silent and look plausible: a line drawn across a floor change
 * or across a break between captures is indistinguishable, on a floor plan, from a
 * corridor that was really walked. Nothing downstream would flag it.
 */

import assert from "node:assert/strict";
import test from "node:test";

import type { KeyframeMarker } from "../index-explorer/types";
import { buildKeyframeTrack } from "./keyframeTrack";

const LNG = 2.3522;
const LAT = 48.8566;
/** Metres per degree of latitude here, near enough for placing test points. */
const M_PER_DEG_LAT = 111_320;

function marker(id: string, northM: number, level: string | null): KeyframeMarker {
  return {
    id,
    longitude: LNG,
    latitude: LAT + northM / M_PER_DEG_LAT,
    level,
    heading_deg: null,
  };
}

test("consecutive keyframes on one floor are joined in order", () => {
  const track = buildKeyframeTrack(
    [marker("0", 0, "1"), marker("1", 2, "1"), marker("2", 4, "1")],
    "#000",
  );

  assert.deepEqual(
    track.map((segment) => segment.id),
    ["keyframe-track-0-1", "keyframe-track-1-2"],
  );
  assert.equal(track[0].level, "1");
  assert.equal(track[0].interactive, true);
});

test("a floor change is not joined", () => {
  // Same spot, different floor: a stairwell. The segment would carry one level and
  // be drawn on that floor's plan, crossing a wall.
  const track = buildKeyframeTrack(
    [marker("0", 0, "1"), marker("1", 0, "2"), marker("2", 2, "2")],
    "#000",
  );

  assert.deepEqual(
    track.map((segment) => segment.id),
    ["keyframe-track-1-2"],
  );
});

test("a gap wider than the threshold breaks the track", () => {
  const track = buildKeyframeTrack(
    [marker("0", 0, "1"), marker("1", 100, "1"), marker("2", 102, "1")],
    "#000",
  );

  assert.deepEqual(
    track.map((segment) => segment.id),
    ["keyframe-track-1-2"],
  );
});

test("the gap threshold is the boundary, and it is configurable", () => {
  const pair = [marker("0", 0, "1"), marker("1", 10, "1")];

  assert.equal(buildKeyframeTrack(pair, "#000", 11).length, 1);
  assert.equal(buildKeyframeTrack(pair, "#000", 9).length, 0);
});

test("an empty or single-keyframe map draws nothing", () => {
  assert.deepEqual(buildKeyframeTrack(null, "#000"), []);
  assert.deepEqual(buildKeyframeTrack([], "#000"), []);
  assert.deepEqual(buildKeyframeTrack([marker("0", 0, "1")], "#000"), []);
});

test("keyframes with no resolved floor still join each other", () => {
  // A null level is a floor of its own, not a wildcard: it matches only null.
  const track = buildKeyframeTrack(
    [marker("0", 0, null), marker("1", 2, null), marker("2", 4, "1")],
    "#000",
  );

  assert.deepEqual(
    track.map((segment) => segment.id),
    ["keyframe-track-0-1"],
  );
  assert.equal(track[0].level, null);
});
