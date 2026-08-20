import { useCallback, useEffect, useState } from "react";

import LivemapAnnotation, {
  type AnalysisOverlay,
} from "../annotations/LivemapAnnotation";
import type { GeoJsonFeatureCollection } from "../annotations/livemapHost";
import { fetchAnalysisLayer } from "./api";
import { describeLayer } from "./layerCatalogue";
import MetricHint from "./MetricHint";

type Props = {
  mapId: string;
  emmid: number | null;
  runId: string | null;
  /** Layer names the run actually wrote, from the status payload. */
  available: string[];
};

/** Preferred first pick: the depth field, which is why this map exists. */
const DEFAULT_LAYER = "depth-range";

/**
 * The analysis layers on the livemap: one field at a time, over the real building.
 *
 * One at a time on purpose. These fields are all coloured on the same green-to-red
 * scale, so two of them stacked would be unreadable — and the useful comparison is
 * sequential anyway ("is the scatter where the range is long?"), which switching does
 * better than superposition.
 *
 * Layers are fetched on demand and kept, because a run's GeoJSON does not change and
 * the depth field alone is tens of thousands of coordinates.
 */
function AnalysisMap(props: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [cache, setCache] = useState<Record<string, GeoJsonFeatureCollection>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // A new run invalidates every layer: same names, different numbers.
  useEffect(() => {
    setCache({});
    setError(null);
  }, [props.runId]);

  useEffect(() => {
    if (selected && props.available.includes(selected)) {
      return;
    }
    setSelected(
      props.available.includes(DEFAULT_LAYER)
        ? DEFAULT_LAYER
        : (props.available[0] ?? null),
    );
  }, [props.available, selected]);

  const runId = props.runId;
  useEffect(() => {
    if (!runId || !selected || cache[selected]) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    void fetchAnalysisLayer(props.mapId, runId, selected)
      .then((collection) => {
        if (cancelled) {
          return;
        }
        setCache((current) => ({ ...current, [selected]: collection }));
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, runId, selected, cache]);

  const onSelect = useCallback((name: string) => setSelected(name), []);

  if (!props.emmid) {
    return (
      <p className="analysis-inert">
        No emmid configured for this map, so there is no livemap to draw the layers on.
      </p>
    );
  }
  if (!runId || props.available.length === 0) {
    return (
      <p className="analysis-inert">
        This run wrote no layers. Re-run the analysis to produce them.
      </p>
    );
  }

  const collection = selected ? cache[selected] : undefined;
  const description = selected ? describeLayer(selected) : null;
  const overlay: AnalysisOverlay | null =
    collection && description
      ? { data: collection, geometry: description.geometry }
      : null;
  const featureCount = collection?.features.length ?? 0;

  return (
    <div className="analysis-map">
      <div className="analysis-layer-picker">
        {props.available.map((name) => {
          const layer = describeLayer(name);
          return (
            <MetricHint key={name} metric={layer}>
              <button
                className={`analysis-layer-button${
                  name === selected ? " is-active" : ""
                }`}
                type="button"
                onClick={() => onSelect(name)}
              >
                {layer.label}
              </button>
            </MetricHint>
          );
        })}
      </div>

      {description ? (
        <p className="analysis-legend-text">
          <span className="analysis-legend-ramp" aria-hidden="true" />
          {description.legend}
          {loading
            ? " · loading…"
            : featureCount === 0
              ? " · this layer is empty: its input is missing for this map"
              : ` · ${featureCount.toLocaleString("en-US")} features`}
        </p>
      ) : null}

      {error ? <p className="error-box">{error}</p> : null}

      <div className="analysis-map-canvas">
        <LivemapAnnotation
          emmid={props.emmid}
          markers={[]}
          polygons={[]}
          draftVertices={[]}
          draftClosed={false}
          pendingPoint={null}
          activeColor={null}
          mode="inspect"
          focusTarget={null}
          overlay={overlay}
          onMapClick={() => {}}
          height="100%"
        />
      </div>
      <p className="muted analysis-map-note">
        Cells are drawn on every floor: the layers carry an altitude per cell but no
        indoor level, so on a multi-storey map read them one floor at a time.
      </p>
    </div>
  );
}

export default AnalysisMap;
