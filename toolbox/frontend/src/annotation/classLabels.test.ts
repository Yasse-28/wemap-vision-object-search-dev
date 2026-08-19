import assert from "node:assert/strict";
import test from "node:test";

import type { AnnotationFeature } from "../annotations/types";
import { classLabelDefaults } from "./classLabels";

let counter = 0;

function point(
  className: string,
  synonyms: string[],
  visuallySimilar: string[] = [],
): AnnotationFeature {
  counter += 1;
  return {
    id: `point-${counter}`,
    className,
    prompt: null,
    classColor: "#000000",
    annotationType: "point",
    geometryType: "Point",
    coordinates: [0, 0],
    altitude: null,
    level: null,
    accuracyM: 5,
    source: null,
    groundTruth: {
      objectId: "x",
      extentM: 0.5,
      exhaustiveZone: null,
      isDepiction: false,
      labels: { synonyms, depictions: [], visuallySimilar, clutter: [] },
    },
  };
}

test("a class nobody has annotated yet suggests nothing", () => {
  assert.equal(classLabelDefaults([], "chair"), null);
  assert.equal(classLabelDefaults([point("door", ["door"])], "chair"), null);
  assert.equal(classLabelDefaults([point("chair", [])], "chair"), null);
  assert.equal(classLabelDefaults([point("chair", ["chair"])], null), null);
});

test("the set most of the class agrees on wins", () => {
  const annotations = [
    point("chair", ["chair", "seat"]),
    point("chair", ["chair", "seat"]),
    point("chair", ["chair"]),
  ];
  assert.deepEqual(classLabelDefaults(annotations, "chair")?.synonyms, ["chair", "seat"]);
});

test("on a tie the most recent wins, so a correction spreads", () => {
  const annotations = [
    point("chair", ["chair"]),
    point("chair", ["chair", "seat", "chaise"]),
  ];
  assert.deepEqual(classLabelDefaults(annotations, "chair")?.synonyms, [
    "chair",
    "seat",
    "chaise",
  ]);
});

test("one stray point does not rewrite the class", () => {
  const annotations = [
    point("chair", ["chair", "seat"]),
    point("chair", ["chair", "seat"]),
    point("chair", ["chiar"]),
  ];
  assert.deepEqual(classLabelDefaults(annotations, "chair")?.synonyms, ["chair", "seat"]);
});

test("the two sets are read independently", () => {
  const annotations = [
    point("chair", ["chair"], ["stool", "bench"]),
    point("chair", ["chair", "seat"], ["stool", "bench"]),
  ];
  const defaults = classLabelDefaults(annotations, "chair");
  assert.deepEqual(defaults?.synonyms, ["chair", "seat"]);
  assert.deepEqual(defaults?.visuallySimilar, ["stool", "bench"]);
});

test("polygons carry no ground truth and are ignored", () => {
  const polygon = { ...point("chair", ["ring"]), annotationType: "polygon" as const };
  assert.equal(classLabelDefaults([polygon], "chair"), null);
});
