import CollapsibleSection from "../annotations/CollapsibleSection";
import type { BboxPostProcessParams } from "./bboxPostProcess";

type Props = {
  params: BboxPostProcessParams;
  /** Upper bound of the area sliders, in square degrees. */
  areaSliderMax: number;
  rawCount: number;
  filteredCount: number;
  onChange: (params: BboxPostProcessParams) => void;
  onReset: () => void;
};

/** Square degrees — v2 proposals have an angular extent, not a pixel bbox. */
function formatArea(value: number): string {
  if (value <= 0) {
    return "no limit";
  }
  return `${value < 10 ? value.toFixed(2) : Math.round(value)} deg²`;
}

function BboxPostProcessControls(props: Props) {
  const summary = props.params.enabled
    ? `${props.filteredCount} / ${props.rawCount} boxes`
    : `Off | ${props.rawCount} boxes`;

  return (
    <CollapsibleSection title="BBox post-processing" summary={summary} defaultOpen={false}>
      <div className="bbox-postprocess-controls">
        <label className="bbox-postprocess-enabled">
          <input
            type="checkbox"
            checked={props.params.enabled}
            onChange={(event) =>
              props.onChange({
                ...props.params,
                enabled: event.target.checked,
              })
            }
          />
          Apply bbox post-processing
        </label>

        <fieldset className="bbox-postprocess-settings" disabled={!props.params.enabled}>
          <label className="bbox-postprocess-slider">
            <span className="bbox-postprocess-slider-label">
              NMS IoU <strong>{props.params.nmsIou.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={props.params.nmsIou}
              onChange={(event) =>
                props.onChange({
                  ...props.params,
                  nmsIou: Number(event.target.value),
                })
              }
            />
          </label>

          <label className="bbox-postprocess-slider">
            <span className="bbox-postprocess-slider-label">
              Min angular area <strong>{formatArea(props.params.minBboxArea)}</strong>
            </span>
            <input
              type="range"
              min={0}
              max={props.areaSliderMax}
              step={props.areaSliderMax / 200}
              value={Math.min(props.params.minBboxArea, props.areaSliderMax)}
              onChange={(event) =>
                props.onChange({
                  ...props.params,
                  minBboxArea: Number(event.target.value),
                })
              }
            />
          </label>

          <label className="bbox-postprocess-slider">
            <span className="bbox-postprocess-slider-label">
              Max angular area <strong>{formatArea(props.params.maxBboxArea)}</strong>
            </span>
            <input
              type="range"
              min={0}
              max={props.areaSliderMax}
              step={props.areaSliderMax / 200}
              value={
                props.params.maxBboxArea <= 0
                  ? props.areaSliderMax
                  : Math.min(props.params.maxBboxArea, props.areaSliderMax)
              }
              onChange={(event) => {
                const value = Number(event.target.value);
                props.onChange({
                  ...props.params,
                  maxBboxArea: value >= props.areaSliderMax ? 0 : value,
                });
              }}
            />
          </label>

          <p className="map-caption">
            Greedy NMS uses the detector score (or angular area when it is NULL), across the
            whole keyframe. Detector visibility is controlled from the toolbar above. Areas
            are in square degrees; max area at the slider end means no upper limit.
          </p>

          <button type="button" className="link-button" onClick={props.onReset}>
            Reset post-processing
          </button>
        </fieldset>
      </div>
    </CollapsibleSection>
  );
}

export default BboxPostProcessControls;
