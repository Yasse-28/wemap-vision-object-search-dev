import { FEEDBACK_NORMALIZATIONS } from "../benchmark/types";
import type { OnlineLocalizeOverrides } from "./types";

function CollapsibleOnlineOverrides(props: {
  overrides: OnlineLocalizeOverrides;
  onChange: (value: OnlineLocalizeOverrides) => void;
}) {
  return (
    <div className="object-search-online-overrides">
      <label className="object-search-online-input">
        <span>min_keyframes_per_cluster</span>
        <input
          type="number"
          value={props.overrides.min_keyframes_per_cluster}
          onChange={(event) =>
            props.onChange({
              ...props.overrides,
              min_keyframes_per_cluster: Number(event.target.value),
            })
          }
        />
      </label>
      <div className="object-search-online-feedback">
        <span className="object-search-online-group-title">Feedback gains</span>
        <label className="object-search-online-input">
          <span>feedback_alpha</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={props.overrides.feedback_alpha}
            onChange={(event) =>
              props.onChange({
                ...props.overrides,
                feedback_alpha: Number(event.target.value),
              })
            }
          />
        </label>
        <label className="object-search-online-input">
          <span>feedback_beta</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={props.overrides.feedback_beta}
            onChange={(event) =>
              props.onChange({
                ...props.overrides,
                feedback_beta: Number(event.target.value),
              })
            }
          />
        </label>
        <label
          className="object-search-online-input object-search-online-normalization"
          title={
            "Rescales pos_sim/neg_sim across the retrieved candidates before the "
            + "gains apply. They are image↔image similarities (~0.7–0.9), so raw "
            + "they are mostly a constant offset, and a constant offset flattens "
            + "the cluster ranking rather than sharpening it. Inert while both "
            + "gains are 0."
          }
        >
          <span>feedback_normalization</span>
          <select
            value={props.overrides.feedback_normalization}
            disabled={
              props.overrides.feedback_alpha === 0
              && props.overrides.feedback_beta === 0
            }
            onChange={(event) =>
              props.onChange({
                ...props.overrides,
                feedback_normalization: event.target
                  .value as OnlineLocalizeOverrides["feedback_normalization"],
              })
            }
          >
            {FEEDBACK_NORMALIZATIONS.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label
        className="object-search-online-input"
        title={
          "Absolute cluster eligibility floor. match_score is a ratio to the "
          + "query's best cluster and no longer depends on this value. Sent "
          + "explicitly so 'Score this prompt' measures the list you see — the "
          + "benchmark script's own default is 0.15, the service's is 0.2."
        }
      >
        <span>min_similarity</span>
        <input
          type="number"
          min={-1}
          max={1}
          step={0.05}
          value={props.overrides.min_similarity}
          onChange={(event) =>
            props.onChange({
              ...props.overrides,
              min_similarity: Number(event.target.value),
            })
          }
        />
      </label>
      <label className="object-search-online-input">
        <span>level_strategy (dev-only)</span>
        <select
          value={props.overrides.level_strategy}
          onChange={(event) =>
            props.onChange({
              ...props.overrides,
              level_strategy: event.target.value === "median" ? "median" : "seed",
            })
          }
        >
          <option value="seed">seed (production)</option>
          <option value="median">median</option>
        </select>
      </label>
      <label className="object-search-online-input">
        <span>candidate_count</span>
        <input
          type="number"
          value={props.overrides.candidate_count}
          onChange={(event) =>
            props.onChange({
              ...props.overrides,
              candidate_count: Number(event.target.value),
            })
          }
        />
      </label>
    </div>
  );
}

export default CollapsibleOnlineOverrides;
