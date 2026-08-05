import CollapsibleSection from "../annotations/CollapsibleSection";
import type { BboxPostProcessParams } from "./bboxPostProcess";

type Props = {
  params: BboxPostProcessParams;
  areaSliderMax: number;
  rawCount: number;
  filteredCount: number;
  onChange: (params: BboxPostProcessParams) => void;
  onReset: () => void;
};

function formatArea(value: number): string {
  if (value <= 0) {
    return "no limit";
  }
  return String(Math.round(value));
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

          <div className="bbox-postprocess-source-toggles">
            <label>
              <input
                type="checkbox"
                checked={props.params.showYolo}
                onChange={(event) =>
                  props.onChange({
                    ...props.params,
                    showYolo: event.target.checked,
                  })
                }
              />
              YOLO boxes
            </label>
            <label>
              <input
                type="checkbox"
                checked={props.params.showGdino}
                onChange={(event) =>
                  props.onChange({
                    ...props.params,
                    showGdino: event.target.checked,
                  })
                }
              />
              G-DINO boxes
            </label>
          </div>

          <label className="bbox-postprocess-slider">
            <span className="bbox-postprocess-slider-label">
              Min bbox area <strong>{formatArea(props.params.minBboxArea)}</strong>
            </span>
            <input
              type="range"
              min={0}
              max={props.areaSliderMax}
              step={Math.max(1, Math.round(props.areaSliderMax / 200))}
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
              Max bbox area <strong>{formatArea(props.params.maxBboxArea)}</strong>
            </span>
            <input
              type="range"
              min={0}
              max={props.areaSliderMax}
              step={Math.max(1, Math.round(props.areaSliderMax / 200))}
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
            Greedy NMS uses textness (or bbox area) as score. Max area at the slider end means no
            upper limit.
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
