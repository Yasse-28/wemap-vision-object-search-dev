/**
 * Score the clustering's partition against the annotated one, on pairs.
 *
 * Pairs, not mAP: two partitions of the same detections differ in *which detections
 * sit together*, and a ranking metric cannot express that — the same set of objects
 * split two ways scores identically. Pair precision/recall says exactly what went
 * wrong: low precision is over-merging, low recall is fragmentation.
 *
 * Only annotated detections count. Anything else would let an un-inspected corner of
 * the map read as agreement.
 */

import type { GroupAnnotations } from "./groupAnnotations";

export type PairScore = {
  annotatedCount: number;
  truthPairs: number;
  predictedPairs: number;
  agreedPairs: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
};

type ScoredDetection = {
  /** Index of the cluster the algorithm put it in. */
  clusterIndex: number;
  /** Annotated group, or null for "not an object". */
  group: string | null;
  key: string;
};

function pairCount(sizes: number[]): number {
  return sizes.reduce((total, size) => total + (size * (size - 1)) / 2, 0);
}

function groupSizes(values: string[]): number[] {
  const counts = new Map<string, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.values()];
}

/**
 * A detection annotated "not an object" is scored as a group of its own rather than
 * dropped: it belongs with nothing, so every cluster pairing it with something is a
 * real error and must be counted as one.
 */
export function pairScore(detections: ScoredDetection[]): PairScore {
  const byCluster = new Map<number, ScoredDetection[]>();
  for (const detection of detections) {
    const members = byCluster.get(detection.clusterIndex) ?? [];
    members.push(detection);
    byCluster.set(detection.clusterIndex, members);
  }

  const truthLabel = (detection: ScoredDetection): string =>
    detection.group === null ? `__reject__${detection.key}` : `g:${detection.group}`;

  const truthPairs = pairCount(groupSizes(detections.map(truthLabel)));
  const predictedPairs = pairCount([...byCluster.values()].map((list) => list.length));
  let agreedPairs = 0;
  for (const members of byCluster.values()) {
    agreedPairs += pairCount(groupSizes(members.map(truthLabel)));
  }

  const precision = predictedPairs ? agreedPairs / predictedPairs : null;
  const recall = truthPairs ? agreedPairs / truthPairs : null;
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;

  return {
    annotatedCount: detections.length,
    truthPairs,
    predictedPairs,
    agreedPairs,
    precision,
    recall,
    f1,
  };
}

/**
 * Collect the annotated detections of the displayed clusters.
 *
 * A detection appearing in two clusters cannot happen — `localize` partitions — so
 * the first occurrence wins without ambiguity; the guard is against the same
 * observation being listed twice inside one cluster.
 */
export function scoreLocalizations(
  clusters: Array<{
    observations: Array<{
      keyframeId: string;
      thetaCenter: number | null;
      phiCenter: number | null;
    }>;
  }>,
  groups: GroupAnnotations,
): PairScore {
  const seen = new Set<string>();
  const detections: ScoredDetection[] = [];
  clusters.forEach((cluster, clusterIndex) => {
    for (const obs of cluster.observations) {
      if (obs.thetaCenter === null || obs.phiCenter === null) continue;
      const target = {
        keyframeId: obs.keyframeId,
        thetaCenter: obs.thetaCenter,
        phiCenter: obs.phiCenter,
      };
      const group = groups.labelFor(target);
      if (group === undefined) continue;
      const key = `${obs.keyframeId}:${obs.thetaCenter}:${obs.phiCenter}`;
      if (seen.has(key)) continue;
      seen.add(key);
      detections.push({ clusterIndex, group, key });
    }
  });
  return pairScore(detections);
}
