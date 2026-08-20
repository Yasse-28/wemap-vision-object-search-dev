/**
 * The measurement fields drawn on the livemap: what each one shows, and how to read it.
 *
 * Same contract as `metricCatalogue.ts` and for the same reason — the picker label, the
 * hover overlay and the legend all read one entry, so a layer cannot be described one
 * way in the list and another way over the map.
 *
 * Names match the file names in `toolbox/benchmark/map_layers.py`. A layer the run did
 * not write is simply absent from the picker; a layer this catalogue does not know is
 * still offered, with its raw name, rather than hidden.
 */
import type { MetricExplanation } from "./metricCatalogue";

export type LayerDescription = MetricExplanation & {
  /** Cells are filled squares of ground; points are single measurements. */
  geometry: "cell" | "point";
  /** What the colour ramp means, green end first. */
  legend: string;
};

export const LAYERS: Record<string, LayerDescription> = {
  "depth-range": {
    label: "Depth range",
    geometry: "cell",
    legend: "green: near · red: at or past the 15 m trusted range",
    measures:
      "Per 2 m ground cell, the mean distance from a keyframe to the detections placed"
      + " there.",
    reading:
      "The depth magnitude field. A red cell holds positions built by extrapolating the"
      + " depth map, so every distance decision there — the clustering radius included"
      + " — is guesswork. Check beyond_trusted_share on a cell before trusting its"
      + " colour: a moderate mean held down by near rows can still hide a far minority.",
    why:
      "Depth is the pipeline's single largest source of position error, and it is not"
      + " uniform over a building. Knowing WHERE it degrades is what separates a fix"
      + " from a global cap.",
    direction: "lower",
  },
  "depth-blowups": {
    label: "Depth blow-ups",
    geometry: "point",
    legend: "orange: 30 m · dark red: 90 m and beyond",
    measures:
      "One point per detection placed further than 30 m — where the depth map"
      + " saturated rather than measured.",
    reading:
      "Not a field, a set of rays. The diagnostic is in the alignment: these line up"
      + " along window bays, mirrors and glass balustrades, which is what names the"
      + " cause. Worst first, and capped — the tail of a line adds nothing.",
    why:
      "A handful of these is enough to drag a cluster centroid across a room, so they"
      + " cost far more than their count suggests.",
    direction: "lower",
  },
  "depth-scatter": {
    label: "Depth scatter",
    geometry: "cell",
    legend: "green: tight · red: spread of 2 m, the clustering radius",
    measures:
      "Per cell, how far apart the detections of one annotated object land — the mean"
      + " distance of its observations to their own centroid.",
    reading:
      "The fragmentation field. An object seen from several keyframes should collapse"
      + " into one cluster; it does not when its observations' depths disagree, and it"
      + " is that spread, not the object's size, that decides. Read the colour against"
      + " the clustering radius: at or past it, one cluster cannot hold the object.",
    why:
      "Fragmentation is what makes one object become three results, and it follows"
      + " depth scatter rather than object extent — so this is the field to look at"
      + " before trying to estimate extents.",
    remedy:
      "Compare with depth-range on the same cells: scatter concentrated where the range"
      + " is long is a depth problem, scatter at close range is a detector-box problem.",
    direction: "lower",
  },
  "detection-coverage": {
    label: "Detector coverage",
    geometry: "cell",
    legend: "green: every annotation covered · red: none",
    measures:
      "Per cell, the share of annotations there that some detection box actually"
      + " covered, measured in the panorama the annotator clicked.",
    reading:
      "The depth-free measurement, spatialised: it owes nothing to the depth map, so a"
      + " red cell is the detector failing in that part of the building rather than"
      + " geometry failing. Cells hold few annotations each — check the count first.",
    why:
      "It separates a where problem from a what problem. A detector that misses one"
      + " corridor needs different work from one that misses one class everywhere.",
    direction: "higher",
  },
  parallax: {
    label: "Available parallax",
    geometry: "cell",
    legend: "green: wide baseline available · red: none",
    measures:
      "Per cell, the widest angle any two keyframes within trusted range could ever"
      + " have seen that cell under — plus how much the capture turned around it.",
    reading:
      "A capture ceiling, and the only field here that needs neither annotations nor"
      + " labels to be right. Where it is red, no association will place an object"
      + " correctly however good the algorithm. A wide parallax with a low anisotropy"
      + " is a corridor: the capture went past, not around.",
    why:
      "It is the field that says whether a failure is worth debugging or worth"
      + " re-capturing — the one distinction no amount of tuning can change.",
    direction: "higher",
  },
  "detection-grid": {
    label: "Detection density",
    geometry: "point",
    legend: "green: sparse · red: at the 95th percentile of cell counts",
    measures: "Per cell, how many detections the index holds there.",
    reading:
      "The only layer here that is a density, so the only one a viewer's own heat-map"
      + " style would render honestly. Useful as the denominator of every other field:"
      + " a dramatic colour on a cell holding four rows is not a finding.",
    why:
      "Every other layer aggregates over these rows, so where they are thin, the"
      + " aggregates are noise.",
    direction: "none",
  },
  "ground-truth": {
    label: "Ground truth",
    geometry: "point",
    legend: "green: reached · orange: detected in 2D, placed elsewhere · red: missed",
    measures:
      "One point per annotation, coloured by whether the pipeline reached it at all.",
    reading:
      "The layer to open first when something is wrong. Each point's failure property"
      + " says which of the two unrelated failures it is — the detector missing the box,"
      + " or the box existing and the position landing somewhere else.",
    why:
      "The two have unrelated remedies, and a table cannot show that the red ones are"
      + " all on one floor or all along one wall.",
    direction: "none",
  },
  keyframes: {
    label: "Keyframes",
    geometry: "point",
    legend: "green: few detections · red: 120 or more",
    measures: "One point per keyframe: how much it produced and how varied it was.",
    reading:
      "The capture itself. Gaps in this layer explain gaps in every other one, and it"
      + " is the fastest way to see that a wing was walked once and quickly.",
    why: "A measurement's absence is usually a capture fact, not a pipeline fact.",
    direction: "none",
  },
  "capture-distance": {
    label: "Capture distance",
    geometry: "point",
    legend: "green: the capture came within a metre · red: 8 m or more",
    measures: "One point per annotation, coloured by how close the capture ever came.",
    reading:
      "The simplest capture ceiling. An object only ever seen from far away has a poor"
      + " angular resolution in every panorama that saw it, whatever the detector does.",
    why:
      "It sets a floor on box precision that no threshold change can lift.",
    direction: "lower",
  },
  "embedding-agreement": {
    label: "Embedding agreement",
    geometry: "point",
    legend: "green: the cutouts look alike · red: they do not",
    measures:
      "Per annotation, how alike the cutouts gathered on it look in embedding space.",
    reading:
      "The semantic counterpart of depth scatter: geometry says the observations belong"
      + " together, this says whether appearance agrees. Low means an embedding-based"
      + " merge would have to overcome its own evidence.",
    why:
      "It says whether appearance can be used to repair what geometry fragmented — on"
      + " this repo's maps, mostly not.",
    direction: "higher",
  },
};

/** The description for a layer, or a bare one so an unknown layer is still offered. */
export function describeLayer(name: string): LayerDescription {
  return (
    LAYERS[name] ?? {
      label: name,
      geometry: "point",
      legend: "colour resolved by the analysis tool",
      measures: `The '${name}' layer, which this catalogue has no description for.`,
      reading:
        "Written by map_layers.py after this UI was; read its docstring for what it"
        + " means.",
      why: "Offered rather than hidden: a layer the tool writes is a layer to look at.",
      direction: "none",
    }
  );
}
