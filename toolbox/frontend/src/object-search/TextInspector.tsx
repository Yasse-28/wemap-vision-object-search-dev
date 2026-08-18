import type { EnrichedResult, KeyframeGroup } from "./types";
import { formatNumber } from "./utils";

function TextInspector(props: {
  group: KeyframeGroup | null;
  selectedCutout: EnrichedResult | null;
  selectedCutoutId: string | null;
  onSelectCutout: (id: string) => void;
  previewUrl: string | null;
}) {
  if (!props.group) {
    return <p className="muted">Select a keyframe on the map.</p>;
  }
  const cutoutOptions = props.group.results.map((item) => item.cutout_id ?? item.id);
  return (
    <>
      <p>
        <strong>Keyframe {props.group.keyframeId}</strong>
        <br />
        <span className="muted">
          {props.group.results.length} cutouts · best {formatNumber(props.group.bestScore)}
        </span>
      </p>
      <label>
        Cutout
        <select
          value={props.selectedCutoutId ?? cutoutOptions[0] ?? ""}
          onChange={(event) => props.onSelectCutout(event.target.value)}
        >
          {cutoutOptions.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>
      {props.previewUrl ? (
        <img className="inspector-preview" src={props.previewUrl} alt="Cutout preview" />
      ) : (
        <p className="muted">Preview unavailable for this cutout.</p>
      )}
    </>
  );
}

export default TextInspector;

