import { useEffect, useRef, useState } from "react";

import { GroupPicker } from "../matching/GroupPicker";
import type { GroupAnnotations, GroupName } from "../matching/groupAnnotations";
import { scoreLocalizations } from "../matching/pairScore";
import { ReviewButtons } from "../object-search-review/ReviewControls";
import type { ObjectSearchReviews } from "../object-search-review/useObjectSearchReviews";
import type { ObjectLocalization, ObjectObservation } from "./types";
import {
  cutoutPreviewUrl,
  formatNumber,
  localizationScoreSymbol,
} from "./utils";

/** Sentinel for a cluster whose detections do not all sit in the same group. */
const MIXED = Symbol("mixed");

function LocalizeInspector(props: {
  mapId: string;
  localizations: ObjectLocalization[];
  matchThreshold: number;
  selectedIdx: number;
  onSelectIdx: (idx: number) => void;
  selectedObsIdx: number;
  onSelectObservation: (locIdx: number, obsIdx: number) => void;
  reviews: ObjectSearchReviews | null;
  /**
   * Right-click adds to the matching basket — a whole cluster from its header, one
   * detection from its card. Asking "are these two clusters one object?" starts by
   * putting both clusters in, so the header is the primary gesture, not the card.
   */
  onAddToBasket: (
    observations: ObjectObservation[],
    clusterRank: number,
    cluster: ObjectLocalization,
  ) => void;
  /**
   * The reference partition. Null while the map has none loaded; observations with no
   * angles cannot be annotated at all, and their picker is disabled rather than
   * hidden — an annotation you cannot make is worth saying out loud.
   */
  groups: GroupAnnotations | null;
}) {
  // Recomputed on every annotation, which is the point: the score is meant to move
  // as you label, so you can see a re-grouping land.
  const partitionScore = props.groups
    ? scoreLocalizations(props.localizations, props.groups)
    : null;

  const annotationTargets = (observations: ObjectObservation[]) =>
    observations.flatMap((obs) =>
      obs.thetaCenter === null || obs.phiCenter === null
        ? []
        : [
            {
              keyframeId: obs.keyframeId,
              thetaCenter: obs.thetaCenter,
              phiCenter: obs.phiCenter,
            },
          ],
    );
  const [collapsedClusters, setCollapsedClusters] = useState<Set<number>>(() => new Set());
  const selectedClusterRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setCollapsedClusters(new Set());
  }, [props.localizations]);

  // Bring the selected cluster into view — the selection often comes from a click on
  // the livemap marker, where the matching card can be anywhere in a 76-cluster list.
  useEffect(() => {
    const item = selectedClusterRef.current;
    // The scroller is the sidebar <section> this component renders *into* (its
    // `overflow: auto` lives on .object-search-cluster-sidebar), so there is no ref
    // to it here.
    const scroller = item?.closest(".object-search-cluster-sidebar");
    if (!item || !(scroller instanceof HTMLElement)) {
      return;
    }
    const itemBox = item.getBoundingClientRect();
    const scrollerBox = scroller.getBoundingClientRect();
    // Adjust the sidebar's own scrollTop rather than calling scrollIntoView, which
    // would also scroll the results pane and the page. No-op when already visible.
    //
    // A card with many detections is taller than the sidebar; aligning its bottom
    // would then push the header — label, match score, review buttons, i.e. the only
    // part that identifies the cluster — off the top. Align the top in that case.
    if (itemBox.top < scrollerBox.top || itemBox.height > scrollerBox.height) {
      scroller.scrollTop -= scrollerBox.top - itemBox.top;
    } else if (itemBox.bottom > scrollerBox.bottom) {
      scroller.scrollTop += itemBox.bottom - scrollerBox.bottom;
    }
  }, [props.selectedIdx, props.localizations]);

  const allClustersCollapsed =
    props.localizations.length > 0 &&
    collapsedClusters.size === props.localizations.length;

  if (!props.localizations.length) {
    return (
      <>
        <div className="object-search-cluster-sidebar-header">
          <h3>Clusters</h3>
        </div>
        <p className="muted">No localizations.</p>
      </>
    );
  }
  return (
    <>
      <div className="object-search-cluster-sidebar-header">
        <h3>Clusters</h3>
        {partitionScore && partitionScore.annotatedCount > 1 ? (
          <span
            className="partition-score"
            title={
              `Pair score over the ${partitionScore.annotatedCount} annotated detections: ` +
              `${partitionScore.agreedPairs} agreed of ${partitionScore.predictedPairs} clustered ` +
              `and ${partitionScore.truthPairs} true pairs. Low precision = over-merging, ` +
              "low recall = fragmentation."
            }
          >
            pair F1 {partitionScore.f1 === null ? "—" : partitionScore.f1.toFixed(2)}
            {" · P "}
            {partitionScore.precision === null ? "—" : partitionScore.precision.toFixed(2)}
            {" · R "}
            {partitionScore.recall === null ? "—" : partitionScore.recall.toFixed(2)}
          </span>
        ) : null}
        <button
          type="button"
          className="object-search-secondary-button object-search-collapse-all-button"
          disabled={allClustersCollapsed}
          onClick={() =>
            setCollapsedClusters(
              new Set(props.localizations.map((_, locIndex) => locIndex)),
            )
          }
        >
          Collapse all
        </button>
      </div>
      {props.groups?.error ? (
        // Loud on purpose: a failed write leaves the picker showing "unannotated"
        // again, which reads as "it did not save" only if you are told why.
        <p className="error-box">Annotation not saved — {props.groups.error}</p>
      ) : null}
      <div className="object-search-cluster-list">
        {props.localizations.map((loc, locIndex) => {
          const isSelectedCluster = locIndex === props.selectedIdx;
          const isCollapsed = collapsedClusters.has(locIndex);
          const clusterReviewStatus = props.reviews?.clusterStatus(loc) ?? null;
          const clusterTargets = annotationTargets(loc.observations);
          // A cluster shows one group only when its members agree. Disagreement is
          // the interesting state — it *is* the fragmentation being recorded — so it
          // must not be hidden behind whichever member happens to come first.
          const clusterGroup = props.groups
            ? clusterTargets.reduce<GroupName | undefined | typeof MIXED>(
                (accumulator, target, index) => {
                  const label = props.groups?.labelFor(target);
                  if (accumulator === MIXED) return MIXED;
                  return index === 0 || accumulator === label ? label : MIXED;
                },
                undefined,
              )
            : undefined;
          return (
            <article
              className={`object-search-cluster-group${
                isSelectedCluster ? " is-selected" : ""
              }${clusterReviewStatus ? ` is-review-${clusterReviewStatus}` : ""}`}
              key={`cluster-${locIndex}`}
              ref={isSelectedCluster ? selectedClusterRef : null}
            >
              <div
                className="object-search-cluster-header"
                title="Right-click: add this cluster to the matching basket"
                onContextMenu={(event) => {
                  event.preventDefault();
                  props.onAddToBasket(loc.observations, locIndex + 1, loc);
                }}
              >
                <button
                  type="button"
                  className="object-search-cluster-collapse-button"
                  aria-label={`${isCollapsed ? "Expand" : "Collapse"} cluster ${locIndex + 1}`}
                  aria-expanded={!isCollapsed}
                  onClick={() => {
                    setCollapsedClusters((current) => {
                      const next = new Set(current);
                      if (next.has(locIndex)) {
                        next.delete(locIndex);
                      } else {
                        next.add(locIndex);
                      }
                      return next;
                    });
                  }}
                >
                  {isCollapsed ? "+" : "-"}
                </button>
                <button
                  type="button"
                  className="object-search-cluster-select-button"
                  onClick={() => {
                    props.onSelectIdx(locIndex);
                  }}
                >
                  <span className="object-search-cluster-label">
                    {localizationScoreSymbol(loc, props.matchThreshold)} Cluster {locIndex + 1}
                  </span>
                  <span className="object-search-cluster-score">
                    Match {formatNumber(loc.matchScore)}
                  </span>
                  <span className="object-search-cluster-count">
                    {loc.observations.length || loc.observationCount} detections
                  </span>
                </button>
                {props.reviews ? (
                  <ReviewButtons
                    label={`Cluster ${locIndex + 1}`}
                    status={clusterReviewStatus}
                    onChange={(status) => props.reviews?.setClusterStatus(loc, status)}
                  />
                ) : null}
                {props.groups ? (
                  <GroupPicker
                    ariaLabel={
                      clusterGroup === MIXED
                        ? `Cluster ${locIndex + 1}: members are in different groups`
                        : `Group of every detection in cluster ${locIndex + 1}`
                    }
                    value={clusterGroup === MIXED ? undefined : clusterGroup}
                    names={props.groups.names}
                    disabled={!clusterTargets.length}
                    onAssign={(groupName) =>
                      void props.groups?.setLabel(clusterTargets, groupName)
                    }
                    onClear={() => void props.groups?.clearLabel(clusterTargets)}
                  />
                ) : null}
              </div>
              <div className="object-search-cluster-meta">
                <span>sim {formatNumber(loc.similarityScore)}</span>
                <span>conf {formatNumber(loc.confidence)}</span>
                {loc.level !== null ? <span>level {loc.level}</span> : null}
                <span>
                  {formatNumber(loc.lat)}, {formatNumber(loc.lng)}
                </span>
              </div>
              {isCollapsed ? null : loc.observations.length ? (
                <div className="object-search-cluster-detections">
                  {loc.observations.map((obs, obsIndex) => {
                    const isSelectedDetection =
                      isSelectedCluster && obsIndex === props.selectedObsIdx;
                    const reviewStatus = props.reviews?.observationStatus(obs) ?? null;
                    const detectionTargets = annotationTargets([obs]);
                    return (
                      <div
                        key={`${locIndex}-${obs.objectIdx}-${obsIndex}`}
                        className={`object-search-detection-card${
                          isSelectedDetection ? " is-selected" : ""
                        }${reviewStatus ? ` is-review-${reviewStatus}` : ""}`}
                        title="Right-click: add this detection to the matching basket"
                        onContextMenu={(event) => {
                          event.preventDefault();
                          props.onAddToBasket([obs], locIndex + 1, loc);
                        }}
                      >
                        <button
                          type="button"
                          className="object-search-detection-select-button"
                          onClick={() => props.onSelectObservation(locIndex, obsIndex)}
                        >
                          <span className="object-search-detection-image-wrap">
                            <img
                              src={cutoutPreviewUrl(props.mapId, obs.thumbnail)}
                              alt={`Detection ${obsIndex + 1} in cluster ${locIndex + 1}`}
                              loading="lazy"
                            />
                            <span className="object-search-detection-index">
                              {obsIndex + 1}
                            </span>
                            <span className="object-search-detection-score">
                              {formatNumber(obs.similarityScore)}
                            </span>
                          </span>
                          <span className="object-search-detection-meta">
                            <strong>Keyframe {obs.keyframeId}</strong>
                            <small>cutout {obs.cutoutId}</small>
                          </span>
                        </button>
                        {props.reviews ? (
                          <ReviewButtons
                            label={`Detection ${obsIndex + 1} in cluster ${locIndex + 1}`}
                            status={reviewStatus}
                            onChange={(status) =>
                              props.reviews?.setObservationStatus(obs, status)
                            }
                          />
                        ) : null}
                        {props.groups ? (
                          <GroupPicker
                            compact
                            ariaLabel={
                              detectionTargets.length
                                ? `Group of detection ${obsIndex + 1} in cluster ${locIndex + 1}`
                                : "This detection has no angles, so it cannot be annotated"
                            }
                            value={
                              detectionTargets.length
                                ? props.groups.labelFor(detectionTargets[0])
                                : undefined
                            }
                            names={props.groups.names}
                            disabled={!detectionTargets.length}
                            onAssign={(groupName) =>
                              void props.groups?.setLabel(detectionTargets, groupName)
                            }
                            onClear={() => void props.groups?.clearLabel(detectionTargets)}
                          />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="muted">No detections returned for this cluster.</p>
              )}
            </article>
          );
        })}
      </div>
    </>
  );
}

export default LocalizeInspector;

