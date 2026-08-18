import type {
  KeyframeGroup,
  ObjectLocalization,
  ObjectSearchRunResult,
} from "./types";
import { formatNumber } from "./utils";

function OverviewSection(props: {
  result: ObjectSearchRunResult | null;
  keyframeGroups: Record<string, KeyframeGroup>;
  localizations?: ObjectLocalization[];
}) {
  const endpointLabel = props.result?.debug.url ?? null;
  const elapsedLabel =
    props.result?.debug.elapsedMs == null
      ? null
      : `${Math.round(props.result.debug.elapsedMs)} ms`;
  return (
    <div className="object-search-overview-body">
      {!props.result ? (
        <p className="muted">No results yet. Run a query.</p>
      ) : (
        <>
          <div className="object-search-endpoint-row">
            {endpointLabel ? <span title={endpointLabel}>{endpointLabel}</span> : null}
            {elapsedLabel ? <strong>{elapsedLabel}</strong> : null}
          </div>
          {props.result.mode === "text" ? (
            <>
              <div className="metrics-row">
                <span>Keyframes: {Object.keys(props.keyframeGroups).length}</span>
                <span>Results: {props.result.enriched.length}</span>
              </div>
              <details>
                <summary>Ranked results</summary>
                <ul className="ranked-list">
                  {props.result.enriched.map((item) => (
                    <li key={`${item.rank}-${item.id}`}>
                      <strong>
                        #{item.rank} {item.id}
                      </strong>{" "}
                      score {formatNumber(item.score)}
                      {item.keyframe_id ? ` · keyframe ${item.keyframe_id}` : ""}
                      {item.resolution_error ? (
                        <small className="block-muted">{item.resolution_error}</small>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </details>
            </>
          ) : (
            <>
              <div className="metrics-row">
                <span>Clusters: {(props.localizations ?? props.result.localizations).length}</span>
                <span>
                  Detections:{" "}
                  {(props.localizations ?? props.result.localizations).reduce(
                    (sum, loc) => sum + loc.observationCount,
                    0,
                  )}
                </span>
              </div>
              <details>
                <summary>Ranked localizations</summary>
                <ul className="ranked-list">
                  {(props.localizations ?? props.result.localizations).map((loc, index) => (
                    <li key={`loc-${index}`}>
                      <strong>
                        #{index + 1} match {formatNumber(loc.matchScore)}
                      </strong>{" "}
                      sim {formatNumber(loc.similarityScore)}
                      <small className="block-muted">
                        {formatNumber(loc.lat)}, {formatNumber(loc.lng)} ·{" "}
                        {loc.observationCount} detections
                      </small>
                    </li>
                  ))}
                </ul>
              </details>
            </>
          )}
        </>
      )}
    </div>
  );
}

export default OverviewSection;

