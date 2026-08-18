/**
 * Score a partition of detections against the annotated objects, two ways.
 *
 * Pair counting (`pairScore.ts`) compares partitions without matching identities, but
 * it weights an error by the *square* of the blocks it mixes: a 33-detection object is
 * worth 528 pairs and a 4-detection one 6, so destroying the small object barely moves
 * the number. That is the wrong emphasis when the thing being tuned is a merge/split
 * brick, hence the two metrics here — both linear in what they count.
 *
 * - **object level**: an annotated object is recovered when some hypothesis carries its
 *   label. Answers "how many real objects came back", which is the reading you get off
 *   the map; extra hypotheses for an object already represented cost precision.
 * - **detection level**: each hypothesis takes the label its members vote for; a
 *   detection is correct when it sits in a hypothesis carrying its own label. Answers
 *   "how many detections were filed under the right object".
 *
 * They fail in different directions on purpose: splitting one object in two leaves the
 * detection score perfect (every detection is still with its own kind) and costs object
 * precision, which is exactly why neither is shown alone.
 */

/** `hypothesis` null = claimed by no model; `group` null = annotated "not an object". */
export type Assignment = {
  key: string;
  hypothesis: number | null;
  group: string | null;
};

/** Pseudo-label so rejects take part in the vote instead of being invisible. */
const REJECT = "\u0000not-an-object";

export type ObjectScore = {
  predictedObjects: number;
  trueObjects: number;
  matched: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

export type DetectionScore = {
  annotated: number;
  assigned: number;
  correct: number;
  /** Of the detections placed in some object, the share filed correctly. */
  precision: number | null;
  /** Of every annotated detection, the share both placed and filed correctly. */
  recall: number | null;
  f1: number | null;
};

export type GroupOutcome = {
  group: string;
  size: number;
  /** Hypotheses holding at least one member, largest share first. */
  hypotheses: Array<{ index: number; count: number }>;
  bestIou: number;
  verdict: "recovered" | "split" | "merged" | "lost";
};

function groupBy(
  assignments: Assignment[],
  pick: (assignment: Assignment) => string | number | null,
): Map<string | number, Assignment[]> {
  const out = new Map<string | number, Assignment[]>();
  for (const assignment of assignments) {
    const bucket = pick(assignment);
    if (bucket === null) continue;
    const list = out.get(bucket) ?? [];
    list.push(assignment);
    out.set(bucket, list);
  }
  return out;
}

function intersectionOverUnion(left: Assignment[], right: Assignment[]): number {
  const rightKeys = new Set(right.map((item) => item.key));
  const shared = left.filter((item) => rightKeys.has(item.key)).length;
  const union = left.length + right.length - shared;
  return union ? shared / union : 0;
}

/**
 * Objects recovered, counted through the labels the hypotheses carry.
 *
 * Each hypothesis takes the label its members voted for — the same rule the detection
 * score uses. An annotated object is a true positive when *some* hypothesis carries
 * its name; every other hypothesis is a false positive, being either a duplicate of an
 * object already represented or a body labelled "not an object".
 *
 * Fragmentation therefore lands on precision, not on recall: an object split five ways
 * is still recovered once and leaves four spurious objects behind. That is the right
 * charge for a merge brick — the object exists, the partition just invented company
 * for it.
 *
 * An IoU >= 0.5 gate was tried first and read differently: an object split 2/2 across
 * two hypotheses counted as *lost*, so fragmentation was paid twice. Overlap is still
 * reported per object in `groupOutcomes`, as a diagnostic rather than a gate.
 */
export function objectLevelScore(assignments: Assignment[]): ObjectScore {
  const byHypothesis = groupBy(assignments, (item) => item.hypothesis);
  const byGroup = groupBy(assignments, (item) => item.group);

  const represented = new Set<string>();
  for (const members of byHypothesis.values()) {
    const label = majorityLabel(members);
    if (label !== null && label !== REJECT && byGroup.has(label)) {
      represented.add(label);
    }
  }

  const matched = represented.size;
  const predictedObjects = byHypothesis.size;
  const trueObjects = byGroup.size;
  const precision = predictedObjects ? matched / predictedObjects : null;
  const recall = trueObjects ? matched / trueObjects : null;
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;
  return { predictedObjects, trueObjects, matched, precision, recall, f1 };
}

/** The label a hypothesis carries: the one most of its members were annotated with. */
function majorityLabel(members: Assignment[]): string | null {
  const counts = new Map<string, number>();
  for (const member of members) {
    const label = member.group ?? REJECT;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  let best: string | null = null;
  let bestCount = 0;
  for (const [label, count] of counts) {
    // Ties broken by name, so the same basket always scores the same.
    if (count > bestCount || (count === bestCount && best !== null && label < best)) {
      best = label;
      bestCount = count;
    }
  }
  return best;
}

export function detectionLevelScore(assignments: Assignment[]): DetectionScore {
  const byHypothesis = groupBy(assignments, (item) => item.hypothesis);
  const labels = new Map<string | number, string | null>();
  for (const [hypothesis, members] of byHypothesis) {
    labels.set(hypothesis, majorityLabel(members));
  }

  let assigned = 0;
  let correct = 0;
  for (const assignment of assignments) {
    if (assignment.hypothesis === null) continue;
    assigned += 1;
    if ((assignment.group ?? REJECT) === labels.get(assignment.hypothesis)) {
      correct += 1;
    }
  }
  const annotated = assignments.length;
  const precision = assigned ? correct / assigned : null;
  const recall = annotated ? correct / annotated : null;
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;
  return { annotated, assigned, correct, precision, recall, f1 };
}

/**
 * What happened to each annotated object — the table that says what to fix.
 *
 * A scalar tells you the partition is wrong; this says whether it is wrong by
 * splitting or by merging, and with what.
 */
export function groupOutcomes(
  assignments: Assignment[],
  iouThreshold = 0.5,
): GroupOutcome[] {
  const byHypothesis = groupBy(assignments, (item) => item.hypothesis);
  const byGroup = groupBy(assignments, (item) => item.group);

  const outcomes: GroupOutcome[] = [];
  for (const [group, members] of byGroup) {
    const counts = new Map<number, number>();
    for (const member of members) {
      if (member.hypothesis === null) continue;
      counts.set(member.hypothesis, (counts.get(member.hypothesis) ?? 0) + 1);
    }
    const hypotheses = [...counts]
      .map(([index, count]) => ({ index, count }))
      .sort((a, b) => b.count - a.count);
    let bestIou = 0;
    for (const { index } of hypotheses) {
      bestIou = Math.max(
        bestIou,
        intersectionOverUnion(byHypothesis.get(index) ?? [], members),
      );
    }
    const dominant = hypotheses[0];
    const foreign = dominant
      ? (byHypothesis.get(dominant.index)?.length ?? 0) - dominant.count
      : 0;
    const verdict: GroupOutcome["verdict"] = !hypotheses.length
      ? "lost"
      : hypotheses.length > 1
        ? "split"
        : foreign > 0 && bestIou < iouThreshold
          ? "merged"
          : "recovered";
    outcomes.push({
      group: String(group),
      size: members.length,
      hypotheses,
      bestIou,
      verdict,
    });
  }
  return outcomes.sort((a, b) => b.size - a.size);
}

// ── Partition-comparison metrics ─────────────────────────────────────────────
//
// The object and detection scores above each hide one failure mode: the detection
// score barely charges fragmentation (every detection stays with its own kind when an
// object is cut in two) and a majority vote hides a cluster that is 4 of one object
// and 2 of another. These three are the counter-weights, and they are what the
// literature uses to compare partitions.

export type PairScore = {
  predictedPairs: number;
  truthPairs: number;
  agreedPairs: number;
  /** Predicted co-clustered pairs whose detections belong to different objects. */
  falseMergePairs: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

export type VariationOfInformation = {
  /** H(prediction | truth), bits — one true object spread over several clusters. */
  splitBits: number;
  /** H(truth | prediction), bits — one cluster holding several true objects. */
  mergeBits: number;
  total: number;
};

export type ImpurityStats = {
  /** Hypotheses holding detections of more than one annotated object. */
  impureClusters: number;
  clusters: number;
  /** Annotated objects spread over more than one hypothesis. */
  splitObjects: number;
  objects: number;
};

/**
 * Cluster key that makes every unassigned detection its own singleton.
 *
 * Keyed on the detection, not on its position in whatever array is being walked:
 * `impurityStats` iterates per group, where a positional key would make the first
 * member of every group collide into one phantom cluster.
 */
function predictedKey(assignment: Assignment): string {
  return assignment.hypothesis === null
    ? `solo:${assignment.key}`
    : `h:${assignment.hypothesis}`;
}

function truthKey(assignment: Assignment): string {
  return assignment.group ?? REJECT;
}

function pairsWithin(sizes: number[]): number {
  return sizes.reduce((total, size) => total + (size * (size - 1)) / 2, 0);
}

function countBy<T>(values: T[]): Map<T, number> {
  const counts = new Map<T, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return counts;
}

/**
 * Rand-family counting. Quadratic weighting is its known bias — a 33-detection object
 * is worth 528 pairs against 6 for a 4-detection one — which is why it sits beside the
 * two linear scores rather than replacing them. What it alone gives: `falseMergePairs`,
 * the count to minimise first when choosing between two partitions.
 */
export function pairLevelScore(assignments: Assignment[]): PairScore {
  const predicted = assignments.map(predictedKey);
  const truth = assignments.map(truthKey);

  const predictedPairs = pairsWithin([...countBy(predicted).values()]);
  const truthPairs = pairsWithin([...countBy(truth).values()]);

  let agreedPairs = 0;
  const byPredicted = new Map<string, string[]>();
  predicted.forEach((key, index) => {
    const list = byPredicted.get(key) ?? [];
    list.push(truth[index]);
    byPredicted.set(key, list);
  });
  for (const labels of byPredicted.values()) {
    agreedPairs += pairsWithin([...countBy(labels).values()]);
  }

  const precision = predictedPairs ? agreedPairs / predictedPairs : null;
  const recall = truthPairs ? agreedPairs / truthPairs : null;
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;
  return {
    predictedPairs,
    truthPairs,
    agreedPairs,
    falseMergePairs: predictedPairs - agreedPairs,
    precision,
    recall,
    f1,
  };
}

/**
 * Variation of information, split into its two halves.
 *
 * Unlike a single F1 it says *which way* the partition is wrong, in bits: `splitBits`
 * rises when one object is cut up, `mergeBits` when one cluster holds several objects.
 * Both are 0 for a perfect partition, and neither is normalised — compare them across
 * methods on the same basket, not across baskets.
 */
export function variationOfInformation(
  assignments: Assignment[],
): VariationOfInformation {
  const total = assignments.length;
  if (!total) return { splitBits: 0, mergeBits: 0, total: 0 };

  const predicted = assignments.map(predictedKey);
  const truth = assignments.map(truthKey);
  const predictedCounts = countBy(predicted);
  const truthCounts = countBy(truth);
  const joint = countBy(predicted.map((key, index) => `${key}|${truth[index]}`));

  let splitBits = 0;
  let mergeBits = 0;
  for (const [key, count] of joint) {
    const [predictedPart, truthPart] = key.split("|");
    const p = count / total;
    const pPredicted = (predictedCounts.get(predictedPart) ?? 0) / total;
    const pTruth = (truthCounts.get(truthPart) ?? 0) / total;
    splitBits -= p * Math.log2(p / pTruth);
    mergeBits -= p * Math.log2(p / pPredicted);
  }
  return { splitBits, mergeBits, total: splitBits + mergeBits };
}

export function impurityStats(assignments: Assignment[]): ImpurityStats {
  const byHypothesis = groupBy(assignments, (item) => item.hypothesis);
  const byGroup = groupBy(assignments, (item) => item.group);

  let impureClusters = 0;
  for (const members of byHypothesis.values()) {
    if (new Set(members.map((member) => member.group ?? REJECT)).size > 1) {
      impureClusters += 1;
    }
  }
  let splitObjects = 0;
  for (const members of byGroup.values()) {
    const holders = new Set(members.map(predictedKey));
    if (holders.size > 1) splitObjects += 1;
  }
  return {
    impureClusters,
    clusters: byHypothesis.size,
    splitObjects,
    objects: byGroup.size,
  };
}
