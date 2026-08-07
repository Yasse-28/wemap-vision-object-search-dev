import type { JSX } from "react";

import type { ReviewStatus } from "./api";

export function ReviewButtons(props: {
  label: string;
  status: ReviewStatus | null;
  onChange: (status: ReviewStatus) => void;
}): JSX.Element {
  return (
    <span className="object-search-review-actions" role="group" aria-label={props.label}>
      <button
        type="button"
        className={`object-search-review-button is-false${
          props.status === "false_positive" ? " is-active" : ""
        }`}
        aria-label={`${props.label}: mark as incorrect`}
        aria-pressed={props.status === "false_positive"}
        title="Mark as incorrect"
        onClick={(event) => {
          event.stopPropagation();
          props.onChange("false_positive");
        }}
      >
        ×
      </button>
      <button
        type="button"
        className={`object-search-review-button is-true${
          props.status === "true_positive" ? " is-active" : ""
        }`}
        aria-label={`${props.label}: mark as correct`}
        aria-pressed={props.status === "true_positive"}
        title="Mark as correct"
        onClick={(event) => {
          event.stopPropagation();
          props.onChange("true_positive");
        }}
      >
        ✓
      </button>
    </span>
  );
}
