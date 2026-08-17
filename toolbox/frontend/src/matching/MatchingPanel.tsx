/**
 * The Matching tab: judge a hand-picked set of detections.
 *
 * Two readings of the same basket, side by side. The cosine matrix answers "does
 * MetaCLIP think these are the same thing?", the triangulation answers "do the rays
 * meet?" — and the interesting cases are the ones where they disagree, which is why
 * neither is allowed to hide the other behind a single verdict.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import type { MapSummary } from "../api";
import LivemapAnnotation from "../annotations/LivemapAnnotation";
import type { LivemapMarker, LivemapSegment } from "../annotations/types";
import { cutoutPreviewUrl } from "../object-search/utils";
import { runMatching, type MatchingResponse, type PartitionMethod } from "./api";
import { basketKey, useMatchingBasket } from "./basket";
import { GroupBrowser, type GroupBrowserSelection } from "./GroupBrowser";
import { GroupPicker } from "./GroupPicker";
import { useGroupAnnotations } from "./groupAnnotations";
import {
  detectionLevelScore,
  groupOutcomes,
  impurityStats,
  objectLevelScore,
  pairLevelScore,
  variationOfInformation,
  type Assignment,
} from "./partitionScore";
import { matchColor, objectColor, similarityRange } from "./matchColor";
import "./matching.css";

type Props = {
  map: MapSummary | null;
  mapId: string;
  isMapKnown: boolean;
};

/**
 * Degrees, because the residual is angular. 8 deg is half the measured p90 aiming
 * error of a bbox centre (12.8 deg), so it is generous where the noise actually is
 * without accepting a ray pointing somewhere else.
 */
const DEFAULT_INLIER_THRESHOLD_DEG = 8;
/** Depth beyond this is where the depth map stops being trustworthy. */
const DEFAULT_MAX_DEPTH_M = 15;

/** The line-up of the comparison, in the order the table shows them. */
const COMPARED: Array<{
  label: string;
  method: PartitionMethod;
  cannotLink: boolean;
  covis?: number;
}> = [
  { label: "sequential RANSAC", method: "sequential", cannotLink: false },
  { label: "J-linkage", method: "jlinkage", cannotLink: false },
  { label: "T-linkage", method: "tlinkage", cannotLink: false },
  { label: "GASP average", method: "gasp", cannotLink: false },
  { label: "GASP + cannot-link", method: "gasp", cannotLink: true },
  { label: "GASP 1v2 score", method: "gasp1v2", cannotLink: false },
  { label: "GASP 1v2 + co-visibility", method: "gasp1v2", cannotLink: false, covis: 1 },
];

type MethodComparison = {
  label: string;
  objects: string;
  detections: string;
  falseMerge: number;
  voi: string;
  clusters: number;
  /** Same-keyframe pairs the rule blocked, and how many were truly different objects. */
  cue: string;
  /** Set when this method failed; the other rows are still worth showing. */
  error?: string;
};

function MatchingPanel(props: Props) {
  const basket = useMatchingBasket(props.mapId);
  const groups = useGroupAnnotations(props.mapId);
  const [result, setResult] = useState<MatchingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [threshold, setThreshold] = useState(DEFAULT_INLIER_THRESHOLD_DEG);
  const [maxDepthOn, setMaxDepthOn] = useState(false);
  const [maxDepth, setMaxDepth] = useState(DEFAULT_MAX_DEPTH_M);
  const [cannotLink, setCannotLink] = useState(false);
  const [covisibility, setCovisibility] = useState(false);
  const [comparison, setComparison] = useState<MethodComparison[] | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [zoomed, setZoomed] = useState<{ src: string; caption: string } | null>(null);
  const [stretchScale, setStretchScale] = useState(false);
  const [showRays, setShowRays] = useState(true);
  const [showClusters, setShowClusters] = useState(false);
  const [browsedGroup, setBrowsedGroup] = useState<string | null>(null);
  const [hoveredMarker, setHoveredMarker] = useState<string | null>(null);
  const [colorBy, setColorBy] = useState<"match" | "object">("match");
  const [partitionMethod, setPartitionMethod] =
    useState<PartitionMethod>("sequential");

  // Escape closes the viewer. Bound on the window rather than the overlay so it works
  // without the overlay having to hold focus.
  useEffect(() => {
    if (!zoomed) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setZoomed(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [zoomed]);

  const items = basket.items;

  const run = useCallback(async () => {
    if (!items.length) return;
    setIsRunning(true);
    setError(null);
    try {
      setResult(
        await runMatching(
          props.mapId,
          items.map((item) => ({
            keyframeId: item.keyframeId,
            thetaCenter: item.rays[0].thetaCenter,
            phiCenter: item.rays[0].phiCenter,
          })),
          {
            inlierThresholdDeg: threshold,
            partitionMethod,
            maxDepthM: maxDepthOn ? maxDepth : null,
            cannotLinkSameKeyframe: cannotLink,
            covisibilityWeight: covisibility ? 1 : 0,
          },
        ),
      );
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setIsRunning(false);
    }
  }, [
    cannotLink,
    covisibility,
    items,
    maxDepth,
    maxDepthOn,
    partitionMethod,
    props.mapId,
    threshold,
  ]);

  // A group loaded from the annotations has no previews — the annotation stores a
  // direction, not a crop. Resolving them is the same lookup `Match` does, so it runs
  // on its own rather than making the user press a button to see pictures.
  const missingThumbnails = items.some((item) => !item.thumbnail);
  useEffect(() => {
    if (!items.length || !missingThumbnails) return;
    let cancelled = false;
    runMatching(
      props.mapId,
      items.map((item) => ({
        keyframeId: item.keyframeId,
        thetaCenter: item.rays[0].thetaCenter,
        phiCenter: item.rays[0].phiCenter,
      })),
      { inlierThresholdDeg: threshold },
    )
      .then((payload) => {
        if (cancelled) return;
        const thumbnails = new Map<string, string | null>();
        payload.items.forEach((resolved, index) => {
          const item = items[index];
          if (item) thumbnails.set(item.key, resolved.thumbnail);
        });
        basket.setThumbnails(thumbnails);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
    // `threshold` is deliberately out: it changes the verdict, not the previews, and
    // re-resolving on every keystroke in that field would be pure noise.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, missingThumbnails, props.mapId, basket.setThumbnails]);

  /**
   * Run every method on the same basket and score them all against the annotations.
   *
   * Sequentially rather than in parallel: five requests that each read the same
   * parquet and hit the same connection gain nothing from concurrency, and a serial
   * run keeps the failure attributable to one method.
   */
  const compareMethods = useCallback(async () => {
    if (!items.length) return;
    setIsComparing(true);
    setError(null);
    const rows: MethodComparison[] = [];
    // Per-method try/catch: one method failing must not discard the rows that
    // succeeded — that is exactly what made the table come back empty.
    for (const entry of COMPARED) {
      try {
        const payload = await runMatching(
          props.mapId,
          items.map((item) => ({
            keyframeId: item.keyframeId,
            thetaCenter: item.rays[0].thetaCenter,
            phiCenter: item.rays[0].phiCenter,
          })),
          {
            inlierThresholdDeg: threshold,
            partitionMethod: entry.method,
            maxDepthM: maxDepthOn ? maxDepth : null,
            cannotLinkSameKeyframe: entry.cannotLink,
            covisibilityWeight: entry.covis ?? 0,
          },
        );
        const byItem = new Map<number, number>();
        (payload.triangulation.hypotheses ?? []).forEach((hypothesis, index) => {
          for (const item of hypothesis.items) byItem.set(item, index);
        });
        const scored: Assignment[] = items.flatMap((item, index) => {
          const group = groups.labelFor({
            keyframeId: item.keyframeId,
            thetaCenter: item.rays[0].thetaCenter,
            phiCenter: item.rays[0].phiCenter,
          });
          return group === undefined
            ? []
            : [{ key: item.key, hypothesis: byItem.get(index) ?? null, group }];
        });
        if (scored.length < 2) continue;
        const objects = objectLevelScore(scored);
        const detections = detectionLevelScore(scored);
        const pairs = pairLevelScore(scored);
        const information = variationOfInformation(scored);
        const blocked = payload.triangulation.cannot_link_pairs ?? [];
        const byIndex = new Map(scored.map((entry) => [entry.key, entry.group]));
        const judged = blocked.filter(
          ([i, j]) =>
            byIndex.has(items[i]?.key ?? "") && byIndex.has(items[j]?.key ?? ""),
        );
        const correct = judged.filter(
          ([i, j]) => byIndex.get(items[i].key) !== byIndex.get(items[j].key),
        ).length;
        rows.push({
          label: entry.label,
          objects: `${objects.precision?.toFixed(2) ?? "—"} / ${objects.recall?.toFixed(2) ?? "—"}`,
          detections: `${detections.precision?.toFixed(2) ?? "—"} / ${detections.recall?.toFixed(2) ?? "—"}`,
          falseMerge: pairs.falseMergePairs,
          voi: `${information.splitBits.toFixed(2)} / ${information.mergeBits.toFixed(2)}`,
          clusters: objects.predictedObjects,
          cue: judged.length ? `${correct}/${judged.length}` : "—",
        });
      } catch (err) {
        rows.push({
          label: entry.label,
          objects: "—",
          detections: "—",
          falseMerge: 0,
          voi: "—",
          clusters: 0,
          cue: "—",
          error: (err as Error).message,
        });
      }
    }
    setComparison(rows);
    setIsComparing(false);
  }, [groups, items, maxDepth, maxDepthOn, props.mapId, threshold]);

  const inliers = useMemo(
    () => new Set(result?.triangulation.inlier_items ?? []),
    [result],
  );

  /**
   * Three counts, not two: a detection that pgvector could not resolve contributes no
   * ray at all, and folding it into "outlier" would blame the geometry for a lookup
   * miss.
   */
  const counts = useMemo(() => {
    const rays = result?.triangulation.ray_index?.length ?? 0;
    return {
      inliers: inliers.size,
      outliers: Math.max(0, rays - inliers.size),
      withoutRay: Math.max(0, items.length - rays),
    };
  }, [inliers, items.length, result]);

  /** Which object each detection was assigned to, null when no hypothesis claimed it. */
  const hypothesisOf = useMemo(() => {
    const byItem = new Map<number, number>();
    (result?.triangulation.hypotheses ?? []).forEach((hypothesis, index) => {
      for (const item of hypothesis.items) byItem.set(item, index);
    });
    return byItem;
  }, [result]);

  /**
   * How well each detection agrees with the rest of the basket: the mean cosine to
   * every other member. One number per ray, which is what a segment can carry — a
   * pairwise matrix cannot be drawn on a map.
   */
  const meanSimilarity = useMemo(() => {
    if (!result) return [] as Array<number | null>;
    return result.similarity.map((row, index) => {
      const others = row.filter(
        (value, column): value is number => column !== index && value !== null,
      );
      return others.length
        ? others.reduce((total, value) => total + value, 0) / others.length
        : null;
    });
  }, [result]);

  const colorRange = useMemo(
    () =>
      stretchScale && result
        ? similarityRange(
            result.similarity.flatMap((row, index) =>
              row.filter((_, column) => column !== index),
            ),
          )
        : null,
    [result, stretchScale],
  );

  /**
   * Keyframes, their rays, and the triangulated point.
   *
   * This is where parallax stops being a number and becomes readable: rays leaving
   * from nearly the same place, at nearly the same angle, look like one line — and
   * that is exactly the case where the agreement means nothing.
   */
  const { mapMarkers, mapSegments } = useMemo(() => {
    const markers: LivemapMarker[] = [];
    const segments: LivemapSegment[] = [];
    if (!result) {
      return { mapMarkers: markers, mapSegments: segments };
    }
    result.items.forEach((item, index) => {
      const isInlier = !result.triangulation.available || inliers.has(index);
      // Colour now carries the match strength, so the geometric verdict moves to the
      // outline and the line weight — two readings, two channels, no collision.
      const color =
        colorBy === "object"
          ? objectColor(hypothesisOf.get(index) ?? null)
          : matchColor(meanSimilarity[index] ?? null, colorRange);
      if (item.keyframe_wgs84) {
        // Small disc, no label: sixteen numbered pins on a floor plan is a wall of
        // text, and the ray leaving each disc already says which detection it is.
        // `scaleWithZoom` keeps them discs rather than growing blobs.
        markers.push({
          id: `kf-${index}`,
          latitude: item.keyframe_wgs84[0],
          longitude: item.keyframe_wgs84[1],
          color,
          level: null,
          radius: 4,
          scaleWithZoom: true,
          strokeColor: isInlier ? "#0f766e" : "#111827",
          strokeWidth: isInlier ? 1 : 2,
        });
      }
      if (showRays && item.keyframe_wgs84 && item.ray_end_wgs84) {
        segments.push({
          id: `ray-${index}`,
          fromLatitude: item.keyframe_wgs84[0],
          fromLongitude: item.keyframe_wgs84[1],
          toLatitude: item.ray_end_wgs84[0],
          toLongitude: item.ray_end_wgs84[1],
          color,
          width: isInlier ? 3 : 1,
          opacity: isInlier ? 0.95 : 0.45,
        });
      }
      if (item.stored_wgs84) {
        // The pipeline's own answer for this detection, kept visible next to the
        // triangulated one: disagreement between the two is the thing to look at.
        markers.push({
          id: `depth-${index}`,
          latitude: item.stored_wgs84[0],
          longitude: item.stored_wgs84[1],
          color: "#94a3b8",
          level: null,
          radius: 3,
          scaleWithZoom: true,
          opacity: 0.75,
        });
      }
    });
    if (showClusters) {
      // One marker per distinct cluster, not per detection: fifteen members of one
      // cluster share one localization, and stacking fifteen discs on it says nothing.
      const seen = new Set<string>();
      items.forEach((item) => {
        const position = item.clusterPosition;
        if (!position || item.clusterRank === null) return;
        const key = `${item.clusterRank}`;
        if (seen.has(key)) return;
        seen.add(key);
        markers.push({
          id: `cluster-${key}`,
          latitude: position.lat,
          longitude: position.lng,
          color: "#0ea5e9",
          level: position.level,
          radius: 6,
          scaleWithZoom: true,
          strokeColor: "#0c4a6e",
          strokeWidth: 2,
          label: `C${item.clusterRank}`,
        });
      });
    }
    const hypotheses = result.triangulation.hypotheses ?? [];
    if (hypotheses.length) {
      hypotheses.forEach((hypothesis, index) => {
        markers.push({
          id: `hypothesis-${index}`,
          latitude: hypothesis.lat,
          longitude: hypothesis.lng,
          color: colorBy === "object" ? objectColor(index) : "#f59e0b",
          level: null,
          radius: 7,
          scaleWithZoom: true,
          strokeColor: "#0f172a",
          strokeWidth: 2,
          label: hypotheses.length > 1 ? `O${index + 1}` : undefined,
        });
      });
    } else if (result.triangulation.available && result.triangulation.lat != null) {
      markers.push({
        id: "triangulated",
        latitude: result.triangulation.lat,
        longitude: result.triangulation.lng as number,
        color: "#f59e0b",
        level: null,
        radius: 7,
        scaleWithZoom: true,
        strokeColor: "#7c2d12",
        strokeWidth: 2,
      });
    }
    return { mapMarkers: markers, mapSegments: segments };
  }, [
    colorBy,
    colorRange,
    hypothesisOf,
    inliers,
    items,
    meanSimilarity,
    result,
    showClusters,
    showRays,
  ]);

  // What the basket was filled from, so a merge test states its own premise.
  const composition = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of items) {
      const bucket =
        item.groupName != null
          ? `Group ${item.groupName}`
          : item.clusterRank === null
            ? "panorama"
            : `Cluster ${item.clusterRank}`;
      counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
    }
    return [...counts];
  }, [items]);

  const objectCount = result?.triangulation.hypotheses?.length ?? 0;

  /**
   * The partition this method produced, scored against the annotated objects.
   *
   * Pairs, not per-object matching: two partitions of the same detections differ in
   * which detections sit together, and pair precision/recall names the failure —
   * precision drops when two objects were merged, recall when one was split. Scored
   * over annotated detections only; anything else counts an un-inspected corner of
   * the basket as agreement.
   *
   * A ray no hypothesis claimed is its own singleton rather than dropped: "belongs
   * with nothing" is a prediction, and a wrong one must cost recall.
   */
  /**
   * How many basket items carry an annotation, independent of any run.
   *
   * The comparison needs ground truth, not a previous partition — gating it on the
   * latter meant a fully annotated basket could not be compared until something else
   * had been computed first.
   */
  const annotatedCount = useMemo(
    () =>
      items.filter(
        (item) =>
          groups.labelFor({
            keyframeId: item.keyframeId,
            thetaCenter: item.rays[0].thetaCenter,
            phiCenter: item.rays[0].phiCenter,
          }) !== undefined,
      ).length,
    [groups, items],
  );

  const assignments = useMemo<Assignment[] | null>(() => {
    if (!result) return null;
    const built = items.flatMap((item, index) => {
      const group = groups.labelFor({
        keyframeId: item.keyframeId,
        thetaCenter: item.rays[0].thetaCenter,
        phiCenter: item.rays[0].phiCenter,
      });
      // Unannotated detections are not evidence either way and would otherwise
      // inflate whichever object they landed in.
      if (group === undefined) return [];
      return [
        { key: item.key, hypothesis: hypothesisOf.get(index) ?? null, group },
      ];
    });
    return built.length > 1 ? built : null;
  }, [groups, hypothesisOf, items, result]);

  const objectScore = useMemo(
    () => (assignments ? objectLevelScore(assignments) : null),
    [assignments],
  );
  const detectionScore = useMemo(
    () => (assignments ? detectionLevelScore(assignments) : null),
    [assignments],
  );
  const outcomes = useMemo(
    () => (assignments ? groupOutcomes(assignments) : []),
    [assignments],
  );
  const pairScore = useMemo(
    () => (assignments ? pairLevelScore(assignments) : null),
    [assignments],
  );
  const voi = useMemo(
    () => (assignments ? variationOfInformation(assignments) : null),
    [assignments],
  );
  const impurity = useMemo(
    () => (assignments ? impurityStats(assignments) : null),
    [assignments],
  );

  const clusterPositionCount = useMemo(
    () =>
      new Set(
        items
          .filter((item) => item.clusterPosition && item.clusterRank !== null)
          .map((item) => item.clusterRank),
      ).size,
    [items],
  );

  /**
   * What the marker under the cursor is, in words.
   *
   * Each marker type answers a different question, so each gets a different read-out:
   * a keyframe disc says who it agrees with, the triangulated point says which rays
   * built it, and a projected point says where it came from.
   */
  const markerPopover = useMemo(() => {
    if (!hoveredMarker || !result) return null;
    const marker = mapMarkers.find((candidate) => candidate.id === hoveredMarker);
    if (!marker) return null;
    const base = { latitude: marker.latitude, longitude: marker.longitude };
    const residualFor = (index: number) => {
      const position = result.triangulation.ray_index?.indexOf(index) ?? -1;
      const value =
        position < 0 ? undefined : result.triangulation.residuals_deg?.[position];
      return value !== undefined && Number.isFinite(value) ? value : null;
    };

    const keyframeMatch = /^kf-(\d+)$/.exec(hoveredMarker);
    if (keyframeMatch) {
      const index = Number(keyframeMatch[1]);
      const item = items[index];
      const row = result.similarity[index] ?? [];
      const ranked = row
        .map((value, column) => ({ value, column }))
        .filter((entry) => entry.column !== index && entry.value !== null)
        .sort((a, b) => (b.value as number) - (a.value as number));
      const residual = residualFor(index);
      return {
        ...base,
        content: (
          <div className="matching-popover">
            <strong>
              #{index + 1} · Keyframe {item?.keyframeId}
            </strong>
            <span>
              {inliers.has(index) ? "inlier" : "outlier"}
              {residual === null ? "" : ` · ${residual.toFixed(1)}° off`}
              {meanSimilarity[index] != null
                ? ` · mean ${(meanSimilarity[index] as number).toFixed(3)}`
                : ""}
            </span>
            <ol>
              {ranked.slice(0, 8).map((entry) => (
                <li key={entry.column}>
                  <span>#{entry.column + 1} kf {items[entry.column]?.keyframeId}</span>
                  <span>{(entry.value as number).toFixed(3)}</span>
                </li>
              ))}
            </ol>
            {ranked.length > 8 ? <small>+{ranked.length - 8} more</small> : null}
          </div>
        ),
      };
    }

    const depthMatch = /^depth-(\d+)$/.exec(hoveredMarker);
    if (depthMatch) {
      const index = Number(depthMatch[1]);
      return {
        ...base,
        content: (
          <div className="matching-popover">
            <strong>Depth-projected position</strong>
            <span>
              from #{index + 1} · keyframe {items[index]?.keyframeId}
            </span>
            <span>the pipeline's own answer for this detection</span>
          </div>
        ),
      };
    }

    if (hoveredMarker === "triangulated") {
      const contributors = (result.triangulation.inlier_items ?? [])
        .map((index) => ({ index, residual: residualFor(index) }))
        .sort((a, b) => (a.residual ?? Infinity) - (b.residual ?? Infinity));
      return {
        ...base,
        content: (
          <div className="matching-popover">
            <strong>Triangulated point</strong>
            <span>
              {contributors.length}/{items.length} rays · parallax{" "}
              {result.triangulation.max_parallax_deg?.toFixed(1)}° · level{" "}
              {result.triangulation.level ?? "unresolved"}
            </span>
            <ol>
              {contributors.slice(0, 8).map((entry) => (
                <li key={entry.index}>
                  <span>
                    #{entry.index + 1} kf {items[entry.index]?.keyframeId}
                  </span>
                  <span>
                    {entry.residual === null ? "—" : `${entry.residual.toFixed(1)}°`}
                  </span>
                </li>
              ))}
            </ol>
            {contributors.length > 8 ? (
              <small>+{contributors.length - 8} more</small>
            ) : null}
          </div>
        ),
      };
    }

    const clusterMatch = /^cluster-(\d+)$/.exec(hoveredMarker);
    if (clusterMatch) {
      const rank = Number(clusterMatch[1]);
      const members = items.filter((item) => item.clusterRank === rank);
      return {
        ...base,
        content: (
          <div className="matching-popover">
            <strong>Cluster {rank}</strong>
            <span>{members.length} detection(s) in the basket</span>
            <span>level {members[0]?.clusterPosition?.level ?? "unresolved"}</span>
          </div>
        ),
      };
    }
    return null;
  }, [hoveredMarker, inliers, items, mapMarkers, meanSimilarity, result]);

  const mapFocus = useMemo(
    () =>
      result?.triangulation.available && result.triangulation.lat != null
        ? {
            latitude: result.triangulation.lat,
            longitude: result.triangulation.lng as number,
          }
        : null,
    [result],
  );

  if (!props.isMapKnown) {
    return (
      <section className="matching-panel">
        <p className="error-box">Map not found in the current config.</p>
      </section>
    );
  }

  return (
    <section className="matching-panel">
      <header className="matching-header">
        <div>
          <h2>Matching</h2>
          <p className="muted">
            {items.length
              ? composition.map(([bucket, count]) => `${bucket}: ${count}`).join(" · ")
              : "Right-click a cluster header, a detection card or a panorama box in Object Search to fill the basket."}
          </p>
        </div>
        <div className="matching-actions">
          <label title="How the rays are split into objects: sequential RANSAC takes them one model at a time; the linkage methods cluster preference sets and do not depend on the order">
            Partition
            <select
              value={partitionMethod}
              onChange={(event) =>
                setPartitionMethod(event.target.value as PartitionMethod)
              }
            >
              <option value="sequential">sequential RANSAC</option>
              <option value="jlinkage">J-linkage</option>
              <option value="tlinkage">T-linkage</option>
              <option value="gasp">GASP average-linkage</option>
              <option value="gasp1v2">GASP · 1-vs-2 score</option>
            </select>
          </label>
          <label title="Angular residual accepted between a ray and the solution. Angular, not metric: the noise is the bbox centre wobbling (median 5.3 deg), and a threshold in degrees does not have to be retuned per venue.">
            Inlier threshold
            <input
              type="number"
              min={0.5}
              max={45}
              step={0.5}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
            °
          </label>
          <label title="Leave out detections whose depth-map point sits beyond this range — far depth is where the estimate stops being trustworthy. Detections with no depth at all are kept.">
            <input
              type="checkbox"
              checked={maxDepthOn}
              onChange={(event) => setMaxDepthOn(event.target.checked)}
            />
            Max depth
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              disabled={!maxDepthOn}
              value={maxDepth}
              onChange={(event) => setMaxDepth(Number(event.target.value))}
            />
            m
          </label>
          <button
            type="button"
            className="object-search-primary-button"
            disabled={!items.length || isRunning}
            onClick={() => void run()}
          >
            {isRunning ? "Matching…" : "Match"}
          </button>
          <label title="Two angularly disjoint boxes in one panorama are two objects — a hard cannot-link that no chain of attractions can route around. Overlapping boxes are left alone: those are the duplicate proposals the association is meant to merge. GASP only.">
            <input
              type="checkbox"
              checked={cannotLink}
              onChange={(event) => setCannotLink(event.target.checked)}
            />
            intra-keyframe cannot-link
          </label>
          <label title="Charge a merge when a keyframe that detected one cluster looks somewhere else entirely for the other. A cost, not a veto: there is no occlusion test, so a wall between the camera and the object reads as a conflict. GASP 1-vs-2 only.">
            <input
              type="checkbox"
              checked={covisibility}
              onChange={(event) => setCovisibility(event.target.checked)}
            />
            co-visibility conflict
          </label>
          <button
            type="button"
            className="object-search-primary-button"
            disabled={!items.length || isComparing || annotatedCount < 2}
            onClick={() => void compareMethods()}
            title={
              annotatedCount >= 2
                ? `Run every partition method and score them against your ${annotatedCount} annotated detections`
                : "Annotate at least two detections of this basket first — there is nothing to score against"
            }
          >
            {isComparing ? "Comparing…" : `Compare ${COMPARED.length} methods`}
          </button>
          <button
            type="button"
            className="object-search-secondary-button"
            disabled={!items.length}
            onClick={() => {
              basket.clear();
              setResult(null);
              setComparison(null);
            }}
          >
            Empty basket
          </button>
        </div>
      </header>

      {zoomed ? (
        <div
          className="matching-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={zoomed.caption}
          onClick={() => setZoomed(null)}
        >
          <img src={zoomed.src} alt={zoomed.caption} />
          <p>
            {zoomed.caption} — click anywhere or press <kbd>Esc</kbd> to close
          </p>
        </div>
      ) : null}

      {browsedGroup ? (
        <GroupBrowser
          mapId={props.mapId}
          groupName={browsedGroup}
          members={groups.members.get(browsedGroup) ?? []}
          onClose={() => setBrowsedGroup(null)}
          onAdd={(selection: GroupBrowserSelection[]) => {
            basket.add(
              selection.map(({ target, thumbnail }) => ({
                key: basketKey(target.keyframeId, {
                  thetaCenter: target.thetaCenter,
                  phiCenter: target.phiCenter,
                }),
                keyframeId: target.keyframeId,
                rays: [
                  { thetaCenter: target.thetaCenter, phiCenter: target.phiCenter },
                ],
                source: "group" as const,
                clusterRank: null,
                groupName: browsedGroup,
                label: browsedGroup,
                thumbnail,
                similarity: null,
              })),
            );
            setBrowsedGroup(null);
            setResult(null);
          }}
        />
      ) : null}

      {error ? <p className="error-box">{error}</p> : null}
      {groups.error ? <p className="error-box">{groups.error}</p> : null}

      {result ? (
        <section className="matching-verdict">
          {result.triangulation.available ? (
            <>
              <span className="matching-verdict-main">
                {objectCount > 1
                  ? `${objectCount} objects`
                  : `${counts.inliers} inlier${counts.inliers === 1 ? "" : "s"}`}
                {objectCount > 1
                  ? ""
                  : ` · ${counts.outliers} outlier${counts.outliers === 1 ? "" : "s"}`}
                {counts.withoutRay ? ` · ${counts.withoutRay} without a ray` : ""}
              </span>
              {objectCount > 1
                ? (result.triangulation.hypotheses ?? []).map((hypothesis, index) => (
                    <span key={index} className="matching-object-tag">
                      <span
                        className="matching-object-dot"
                        style={{ background: objectColor(index) }}
                      />
                      O{index + 1}: {hypothesis.items.length} rays ·{" "}
                      {hypothesis.mean_residual_deg?.toFixed(1)}° ·{" "}
                      {hypothesis.max_parallax_deg?.toFixed(0)}° parallax
                    </span>
                  ))
                : null}
              <span title="Mean angular residual of the inlier rays; the metric equivalent is shown beside it because a map is read in metres.">
                residual {result.triangulation.mean_inlier_residual_deg?.toFixed(1)}° (
                {result.triangulation.mean_inlier_residual_m?.toFixed(2)} m)
              </span>
              <span
                className={
                  (result.triangulation.max_parallax_deg ?? 0) < 10
                    ? "matching-warning"
                    : undefined
                }
              >
                parallax {result.triangulation.max_parallax_deg?.toFixed(1)}°
                {(result.triangulation.max_parallax_deg ?? 0) < 10
                  ? " — too small to prove anything"
                  : ""}
              </span>
              <span>
                {result.triangulation.lat?.toFixed(6)}, {result.triangulation.lng?.toFixed(6)}
              </span>
              {objectScore ? (
                <span
                  className="matching-partition-score"
                  title={
                    `${objectScore.matched} of ${objectScore.trueObjects} annotated objects are carried by ` +
                    `one of ${objectScore.predictedObjects} hypotheses (majority vote of its members). ` +
                    "Precision falls when the partition invents objects — fragmentation lands here — " +
                    "recall when a real object has no hypothesis at all."
                  }
                >
                  objects · P{" "}
                  {objectScore.precision === null ? "—" : objectScore.precision.toFixed(2)}
                  {" · R "}
                  {objectScore.recall === null ? "—" : objectScore.recall.toFixed(2)}
                  {` (${objectScore.matched}/${objectScore.trueObjects})`}
                </span>
              ) : null}
              {detectionScore ? (
                <span
                  className="matching-partition-score matching-partition-score--secondary"
                  title={
                    `${detectionScore.correct} of ${detectionScore.assigned} assigned detections carry the ` +
                    `label their hypothesis voted for, out of ${detectionScore.annotated} annotated. ` +
                    "Blind to fragmentation on purpose — that is what the object score is for."
                  }
                >
                  detections · P{" "}
                  {detectionScore.precision === null
                    ? "—"
                    : detectionScore.precision.toFixed(2)}
                  {" · R "}
                  {detectionScore.recall === null
                    ? "—"
                    : detectionScore.recall.toFixed(2)}
                </span>
              ) : null}
              <span>
                alt {result.triangulation.alt?.toFixed(2)} m
                {" · level "}
                {result.triangulation.level ?? "unresolved"}
              </span>
            </>
          ) : (
            <>
              <span className="matching-verdict-main">
                No triangulation — {result.triangulation.reason}
              </span>
              <span>
                {result.triangulation.ray_index?.length ?? 0} ray(s) of {items.length}{" "}
                detections
              </span>
            </>
          )}
        </section>
      ) : null}

      {props.map?.emmid ? (
        <section className="matching-map">
          <div className="matching-map-header">
            <h3>
              Keyframes and rays
              {result ? null : " — run a match to draw them"}
            </h3>
            <label>
              <input
                type="checkbox"
                checked={showRays}
                onChange={(event) => setShowRays(event.target.checked)}
              />
              rays
            </label>
            <label title="Colour by pairwise similarity, or by the object each ray was assigned to">
              colour by
              <select
                value={colorBy}
                onChange={(event) =>
                  setColorBy(event.target.value as "match" | "object")
                }
              >
                <option value="match">match</option>
                <option value="object">object</option>
              </select>
            </label>
            <label
              title={
                clusterPositionCount
                  ? "The localization of each cluster the basket was filled from"
                  : "No cluster position in the basket — items added before this existed, or loaded from a group"
              }
            >
              <input
                type="checkbox"
                checked={showClusters}
                disabled={!clusterPositionCount}
                onChange={(event) => setShowClusters(event.target.checked)}
              />
              cluster localizations{clusterPositionCount ? ` (${clusterPositionCount})` : ""}
            </label>
          </div>
          <LivemapAnnotation
            emmid={props.map.emmid}
            markers={mapMarkers}
            segments={mapSegments}
            polygons={[]}
            draftVertices={[]}
            draftClosed={false}
            pendingPoint={null}
            activeColor={null}
            mode="inspect"
            focusTarget={mapFocus}
            onMapClick={() => {}}
            onMarkerHover={setHoveredMarker}
            markerPopover={markerPopover}
            height={320}
          />
        </section>
      ) : null}

      {pairScore && voi && impurity ? (
        <section className="matching-verdict matching-verdict--quality">
          <span
            className="matching-verdict-main"
            title="Detections wrongly put together, counted in pairs. Choose between two partitions on this first: a false merge destroys an object, a split only postpones it."
          >
            {pairScore.falseMergePairs} false-merge pair
            {pairScore.falseMergePairs === 1 ? "" : "s"}
          </span>
          <span title="Clusters holding detections of more than one annotated object, and objects spread over more than one cluster.">
            {impurity.impureClusters}/{impurity.clusters} impure clusters ·{" "}
            {impurity.splitObjects}/{impurity.objects} split objects
          </span>
          <span title="Pair precision and recall (Rand family). Quadratically weighted, so a big object dominates — kept as a cross-check, not as the headline.">
            pairs · P {pairScore.precision === null ? "—" : pairScore.precision.toFixed(2)}
            {" · R "}
            {pairScore.recall === null ? "—" : pairScore.recall.toFixed(2)}
          </span>
          <span title="Variation of information, in bits, split into its two halves: split = one object cut across clusters, merge = one cluster holding several objects. Both 0 is perfect.">
            VI · split {voi.splitBits.toFixed(2)} · merge {voi.mergeBits.toFixed(2)}
          </span>
        </section>
      ) : null}

      {comparison?.length ? (
        <section className="matching-compare">
          <h3>Methods on this basket</h3>
            <table className="matching-outcomes matching-comparison">
              <thead>
                <tr>
                  <th>method</th>
                  <th title="Object precision / recall">obj P/R</th>
                  <th title="Detection precision / recall">det P/R</th>
                  <th title="Detections wrongly put together, in pairs — the first criterion">
                    false-merge
                  </th>
                  <th title="Variation of information: split / merge, in bits">VI s/m</th>
                  <th>clusters</th>
                  <th title="Same-keyframe pairs blocked by the rule, and how many were genuinely different objects">
                    cue ok
                  </th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((row) => (
                  <tr key={row.label} className={row.error ? "is-failed" : undefined}>
                    <td title={row.error}>
                      {row.label}
                      {row.error ? " ⚠" : ""}
                    </td>
                    <td>{row.objects}</td>
                    <td>{row.detections}</td>
                    <td>{row.falseMerge}</td>
                    <td>{row.voi}</td>
                    <td>{row.clusters}</td>
                    <td>{row.cue}</td>
                  </tr>
                ))}
              </tbody>
            </table>
        </section>
      ) : null}

      <div className="matching-body">
        <section className="matching-items">
          <h3>Annotated groups</h3>
          {groups.names.length ? (
            <ul className="matching-groups">
              {groups.names.map((name) => (
                <li key={name}>
                  <button
                    type="button"
                    className="matching-group-open"
                    onClick={() => setBrowsedGroup(name)}
                  >
                    {name}
                    <small>{groups.members.get(name)?.length ?? 0} detections</small>
                  </button>
                  <button
                    type="button"
                    className="matching-group-delete"
                    aria-label={`Delete the group ${name}`}
                    title="Delete this group's annotations"
                    onClick={() => {
                      const count = groups.members.get(name)?.length ?? 0;
                      if (
                        window.confirm(
                          `Delete the ${count} annotation(s) of "${name}"? The detections stay, only the grouping goes.`,
                        )
                      ) {
                        void groups.deleteGroup(name);
                        if (browsedGroup === name) setBrowsedGroup(null);
                      }
                    }}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No annotated group on this map yet.</p>
          )}

          <h3>Basket</h3>
          {items.length ? (
            <ul>
              {items.map((item, index) => {
                const resolved = result?.items[index];
                const preview = cutoutPreviewUrl(
                  props.mapId,
                  resolved?.thumbnail ?? item.thumbnail,
                );
                const residual = result?.triangulation.residuals_deg?.[
                  result.triangulation.ray_index?.indexOf(index) ?? -1
                ];
                return (
                  <li
                    key={item.key}
                    className={
                      result && result.triangulation.available
                        ? inliers.has(index)
                          ? "is-inlier"
                          : "is-outlier"
                        : undefined
                    }
                  >
                    {preview ? (
                      <img
                        className="matching-thumb"
                        src={preview}
                        alt={`Detection in keyframe ${item.keyframeId}`}
                        loading="lazy"
                        onClick={() =>
                          setZoomed({
                            src: preview,
                            caption: `#${index + 1} · Keyframe ${item.keyframeId}`,
                          })
                        }
                      />
                    ) : (
                      <span className="matching-thumb matching-thumb--empty" aria-hidden />
                    )}
                    <div className="matching-item-meta">
                      <strong>
                        #{index + 1} · Keyframe {item.keyframeId}
                      </strong>
                      <small>
                        {item.groupName != null
                          ? `from group ${item.groupName}`
                          : item.clusterRank === null
                            ? "from the panorama"
                            : `from cluster ${item.clusterRank}`}
                        {resolved && !resolved.resolved ? " · not in the index" : ""}
                        {residual !== undefined && Number.isFinite(residual)
                          ? ` · ${residual.toFixed(1)}° off`
                          : ""}
                      </small>
                      <GroupPicker
                        compact
                        ariaLabel={`Group of detection ${index + 1}`}
                        value={groups.labelFor({
                          keyframeId: item.keyframeId,
                          thetaCenter: item.rays[0].thetaCenter,
                          phiCenter: item.rays[0].phiCenter,
                        })}
                        names={groups.names}
                        onAssign={(groupName) =>
                          void groups.setLabel(
                            [
                              {
                                keyframeId: item.keyframeId,
                                thetaCenter: item.rays[0].thetaCenter,
                                phiCenter: item.rays[0].phiCenter,
                              },
                            ],
                            groupName,
                          )
                        }
                        onClear={() =>
                          void groups.clearLabel([
                            {
                              keyframeId: item.keyframeId,
                              thetaCenter: item.rays[0].thetaCenter,
                              phiCenter: item.rays[0].phiCenter,
                            },
                          ])
                        }
                      />
                    </div>
                    <button
                      type="button"
                      aria-label={`Remove detection ${index + 1} from the basket`}
                      onClick={() => basket.remove(item.key)}
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="muted">The basket is empty.</p>
          )}
        </section>

        <section className="matching-matrix">
          {outcomes.length ? (
            <>
              <h3>Annotated objects</h3>
              <table className="matching-outcomes">
                <tbody>
                  {outcomes.map((outcome) => (
                    <tr key={outcome.group}>
                      <td>{outcome.group}</td>
                      <td>{outcome.size}</td>
                      <td className={`is-${outcome.verdict}`}>{outcome.verdict}</td>
                      <td>
                        {outcome.hypotheses
                          .map((entry) => `O${entry.index + 1}:${entry.count}`)
                          .join(" ") || "—"}
                      </td>
                      <td>IoU {outcome.bestIou.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : null}

          <h3>MetaCLIP cosine</h3>
          <div className="matching-scale">
            <span className="matching-scale-ramp" aria-hidden />
            <span>
              {colorRange
                ? `${colorRange[0].toFixed(2)} → ${colorRange[1].toFixed(2)} (stretched)`
                : "0 → 1 (absolute)"}
            </span>
            <label>
              <input
                type="checkbox"
                checked={stretchScale}
                onChange={(event) => setStretchScale(event.target.checked)}
              />
              stretch to range
            </label>
          </div>
          {result ? (
            <table>
              <thead>
                <tr>
                  <th />
                  {items.map((_, index) => (
                    <th key={index}>{index + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.similarity.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <th>{rowIndex + 1}</th>
                    {row.map((value, columnIndex) => (
                      <td
                        key={columnIndex}
                        style={{
                          background: matchColor(value, colorRange),
                          color: value === null ? "inherit" : "#ffffff",
                        }}
                        title={
                          value === null
                            ? "No embedding for one of the two detections"
                            : `${items[rowIndex].keyframeId} ↔ ${items[columnIndex].keyframeId}: ${value.toFixed(4)}`
                        }
                      >
                        {value === null ? "—" : value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">Run a match to see the pairwise similarities.</p>
          )}
        </section>
      </div>
    </section>
  );
}

export default MatchingPanel;
