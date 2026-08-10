import { type JSX, useState } from "react";

import type { ReviewStatus } from "./api";
import type { ReviewAnnotation } from "./useObjectSearchReviews";

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

export function ReviewAnnotationList(props: {
  annotations: ReviewAnnotation[];
  onClear: (targetId: number) => void;
}): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false);

  if (!props.annotations.length) {
    return null;
  }

  return (
    <section className="object-search-review-annotation-list">
      <button
        type="button"
        className="object-search-review-annotation-summary"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((current) => !current)}
      >
        <span>Annotations ({props.annotations.length})</span>
        <span aria-hidden="true">{isOpen ? "−" : "+"}</span>
      </button>
      {isOpen ? (
        <div className="object-search-review-annotation-rows">
          {props.annotations.map((annotation) => {
            const isCorrect = annotation.status === "true_positive";
            return (
              <div
                className="object-search-review-annotation-row"
                key={annotation.targetId}
              >
                <span className="object-search-review-annotation-target">
                  #{annotation.targetId}
                </span>
                <span
                  className={`object-search-review-button object-search-review-annotation-badge ${
                    isCorrect ? "is-true" : "is-false"
                  } is-active`}
                >
                  {isCorrect ? "correct" : "incorrect"}
                </span>
                {!annotation.inResults ? (
                  <span className="object-search-review-annotation-hidden">hidden</span>
                ) : null}
                <button
                  type="button"
                  className="object-search-review-annotation-clear"
                  aria-label={`Remove annotation for candidate ${annotation.targetId}`}
                  onClick={() => props.onClear(annotation.targetId)}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
