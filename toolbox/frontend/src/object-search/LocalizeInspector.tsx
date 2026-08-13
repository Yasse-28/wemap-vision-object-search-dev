import { useEffect, useRef, useState } from "react";

import { ReviewButtons } from "../object-search-review/ReviewControls";
import type { ObjectSearchReviews } from "../object-search-review/useObjectSearchReviews";
import type { ObjectLocalization } from "./types";
import {
  cutoutPreviewUrl,
  formatNumber,
  localizationScoreSymbol,
} from "./utils";

function LocalizeInspector(props: {
  mapId: string;
  localizations: ObjectLocalization[];
  matchThreshold: number;
  selectedIdx: number;
  onSelectIdx: (idx: number) => void;
  selectedObsIdx: number;
  onSelectObservation: (locIdx: number, obsIdx: number) => void;
  reviews: ObjectSearchReviews | null;
}) {
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
      <div className="object-search-cluster-list">
        {props.localizations.map((loc, locIndex) => {
          const isSelectedCluster = locIndex === props.selectedIdx;
          const isCollapsed = collapsedClusters.has(locIndex);
          const clusterReviewStatus = props.reviews?.clusterStatus(loc) ?? null;
          return (
            <article
              className={`object-search-cluster-group${
                isSelectedCluster ? " is-selected" : ""
              }${clusterReviewStatus ? ` is-review-${clusterReviewStatus}` : ""}`}
              key={`cluster-${locIndex}`}
              ref={isSelectedCluster ? selectedClusterRef : null}
            >
              <div className="object-search-cluster-header">
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
                    return (
                      <div
                        key={`${locIndex}-${obs.objectIdx}-${obsIndex}`}
                        className={`object-search-detection-card${
                          isSelectedDetection ? " is-selected" : ""
                        }${reviewStatus ? ` is-review-${reviewStatus}` : ""}`}
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

