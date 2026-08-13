import type { ObjectLocalization, ObjectObservation } from "./types";
import { cutoutPreviewUrl, formatNumber } from "./utils";

function SelectedObservationPanel(props: {
  mapId: string;
  localization: ObjectLocalization | null;
  localizationIndex: number;
  observation: ObjectObservation | null;
  observationIndex: number;
}) {
  if (!props.localization) {
    return (
      <section className="object-search-selected-observation">
        <h3>Selected cutout</h3>
        <p className="muted">Select a cluster to inspect its detections.</p>
      </section>
    );
  }
  if (!props.observation) {
    return (
      <section className="object-search-selected-observation">
        <h3>Selected cutout</h3>
        <p className="muted">No detection is available for this cluster.</p>
      </section>
    );
  }

  const previewUrl = cutoutPreviewUrl(props.mapId, props.observation.thumbnail);

  return (
    <section className="object-search-selected-observation">
      <div className="object-search-selected-observation-header">
        <h3>Selected cutout</h3>
        <span>
          Cluster {props.localizationIndex + 1} · Detection {props.observationIndex + 1}
        </span>
      </div>
      <div className="object-search-selected-observation-body">
        <img
          className="object-search-selected-observation-image"
          src={previewUrl}
          alt={`Selected cutout ${props.observation.cutoutId}`}
        />
        <div className="object-search-selected-observation-meta">
          <div>
            <strong>Keyframe</strong>
            <span>{props.observation.keyframeId}</span>
          </div>
          <div>
            <strong>Cutout</strong>
            <span>{props.observation.cutoutId}</span>
          </div>
          <div>
            <strong>Similarity</strong>
            <span>{formatNumber(props.observation.similarityScore)}</span>
          </div>
          <div>
            <strong>Bbox</strong>
            <span>{props.observation.bbox.map((value) => Math.round(value)).join(", ")}</span>
          </div>
          {props.observation.heading !== null ? (
            <div>
              <strong>Heading</strong>
              <span>{formatNumber(props.observation.heading)} deg</span>
            </div>
          ) : null}
          {props.observation.quaternion !== null ? (
            <div>
              <strong>Quaternion</strong>
              <span>{props.observation.quaternion.map(formatNumber).join(", ")}</span>
            </div>
          ) : null}
          {props.observation.coordinates ? (
            <div>
              <strong>Coordinates</strong>
              <span>
                {formatNumber(props.observation.coordinates[0])},{" "}
                {formatNumber(props.observation.coordinates[1])}
              </span>
            </div>
          ) : null}
          <div>
            <strong>Cluster position</strong>
            <span>
              {formatNumber(props.localization.lat)}, {formatNumber(props.localization.lng)}
            </span>
          </div>
          {props.localization.level !== null ? (
            <div>
              <strong>Cluster level</strong>
              <span>{props.localization.level}</span>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default SelectedObservationPanel;

