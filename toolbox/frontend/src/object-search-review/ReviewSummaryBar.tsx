import type { JSX } from "react";

import { configSummary as scoreConfigSummary } from "../benchmark/config-summary";
import type { ByPromptRow, PromptScore } from "../benchmark/types";
import { ReviewAnnotationList } from "./ReviewControls";
import type { ObjectSearchReviews } from "./useObjectSearchReviews";

type PromptBaseline = {
  runId: string;
  row: ByPromptRow;
  config: Record<string, unknown>;
};

type Props = {
  reviews: ObjectSearchReviews;
  promptScore: PromptScore | null;
  promptBaseline: PromptBaseline | null;
  promptScoreError: string | null;
  promptScoreMissing: boolean;
  isPromptScoring: boolean;
  resultQuery: string;
  onScorePrompt: () => void | Promise<void>;
  open: boolean;
  onToggle: () => void;
};

function promptMetricSummary(row: ByPromptRow): string {
  // A failed prompt still gets a row, with `error` set and every count at zero. Left
  // formatted as metrics it reads as a legitimate "found nothing" — the one reading
  // the exit-2 path exists to prevent. Same handling as BenchmarkPanel's tables.
  if (row.error) {
    return `not measured: ${row.error}`;
  }
  return (
    `P ${row.precision.toFixed(2)} · R ${row.recall.toFixed(2)} · F1 ${row.f1.toFixed(2)}`
    + ` · ${row.true_positives}/${row.false_positives}/${row.false_negatives} (TP/FP/FN)`
  );
}

/** The threshold-free half: what to read when comparing two feedback settings.
 *
 * P/R/F1 above are one operating point, and the review-feedback gains *shift the score
 * distribution* — so a change there conflates "the ranking improved" with "the fixed
 * threshold now sits somewhere else on the same curve". AP does not move under a
 * monotone rescaling of the scores, so it isolates the ranking.
 */
function promptCurveSummary(row: ByPromptRow): string | null {
  if (row.average_precision == null) {
    return null;
  }
  const best = row.best_f1 == null ? "" : ` · best F1 ${row.best_f1.toFixed(2)}`;
  const at =
    row.best_f1_threshold == null ? "" : ` @ ${row.best_f1_threshold.toFixed(3)}`;
  return `AP ${row.average_precision.toFixed(3)}${best}${at}`;
}

/** The populations behind TP/FP/FN, which the three numbers alone do not reveal.
 *
 * TP+FP counts *predictions* the benchmark kept, TP+FN counts *ground-truth
 * annotations*: adding all three together totals nothing, and `kept` is well below the
 * cluster count on screen because the benchmark accepts only `match_score >
 * acceptance_threshold` (0.9) while the list is filtered by the Sensitivity slider.
 * `kept + rejected` is the identity that does hold — it equals the clusters returned.
 */
function promptCountsSummary(row: ByPromptRow): string {
  return (
    `${row.accepted_predictions} kept · ${row.rejected_predictions} rejected`
    + ` · ${row.ground_truth} ground truth`
  );
}

export function ReviewSummaryBar(props: Props): JSX.Element {
  const curveSummary = props.promptScore
    ? promptCurveSummary(props.promptScore.row)
    : null;

  return (
    <section className={`object-search-review-toolbar${props.open ? " is-open" : ""}`}>
      <button
        type="button"
        className="object-search-review-summary"
        aria-expanded={props.open}
        onClick={props.onToggle}
      >
        <span aria-hidden="true">{props.open ? "⌃" : "⌄"}</span>
        <span>Revue</span>
        <span>· {props.reviews.reviewedCount} revus</span>
        {props.promptScore && !props.promptScore.row.error ? (
          <span>· F1 {props.promptScore.row.f1.toFixed(2)}</span>
        ) : null}
      </button>

      {props.open ? (
        <div className="object-search-review-content">
          <div className="object-search-review-toolbar-content">
            <span>
              {props.reviews.isLoading
                ? "Loading reviews…"
                : `${props.reviews.reviewedCount} reviewed · ${props.reviews.truePositiveCount} correct · ${props.reviews.falsePositiveCount} incorrect`}
            </span>
            <span className="object-search-review-score">
              <button
                type="button"
                className="object-search-secondary-button"
                disabled={
                  props.reviews.isLoading
                  || props.isPromptScoring
                  || !props.resultQuery
                }
                onClick={() => void props.onScorePrompt()}
              >
                {props.isPromptScoring ? "Scoring…" : "Score this prompt"}
              </button>
              {props.promptScore ? (
                <span
                  className={
                    props.promptScore.row.error
                      ? "object-search-review-score-error"
                      : undefined
                  }
                  title={
                    `Scored with ${scoreConfigSummary(props.promptScore.config)}. The `
                    + "acceptance threshold follows the Sensitivity slider and "
                    + "min_similarity is sent from these controls, so the scored "
                    + "clusters are the ones listed."
                  }
                >
                  {promptMetricSummary(props.promptScore.row)}
                </span>
              ) : null}
              {props.promptScore && curveSummary ? (
                <span
                  className="object-search-review-score-curve"
                  title={
                    "Threshold-free. Compare two feedback settings on AP, not on "
                    + "F1: the gains shift the score distribution, so the fixed "
                    + "threshold lands elsewhere on the curve even when the "
                    + "ranking is unchanged. The full sweep is in the run's "
                    + "raw_results.json."
                  }
                >
                  {curveSummary}
                </span>
              ) : null}
              {props.promptScore && !props.promptScore.row.error ? (
                <span
                  className="object-search-review-score-muted"
                  title={
                    "TP+FP counts kept predictions, TP+FN counts ground-truth "
                    + "annotations — the three do not sum to anything. "
                    + "kept + rejected equals the clusters returned."
                  }
                >
                  {promptCountsSummary(props.promptScore.row)}
                </span>
              ) : null}
              {props.promptScoreMissing ? (
                <span className="object-search-review-score-muted">
                  no benchmark ground truth for &quot;{props.resultQuery}&quot; — ✓/×
                  reviews only tune feedback; they do not create benchmark ground truth
                </span>
              ) : null}
              {props.promptScoreError ? (
                <span className="object-search-review-score-error" role="alert">
                  Score: {props.promptScoreError}
                </span>
              ) : null}
              {props.promptBaseline ? (
                <span
                  className="object-search-review-score-muted"
                  title={
                    "The newest benchmark run that covers this prompt — not "
                    + "necessarily an unboosted one, and not necessarily measured "
                    + "at the current sensitivity. Compare F1 only against a run "
                    + `whose parameters match. Run parameters: ${scoreConfigSummary(
                      props.promptBaseline.config,
                    )}.`
                  }
                >
                  last run {props.promptBaseline.runId}: F1{" "}
                  {props.promptBaseline.row.f1.toFixed(2)} ({
                    scoreConfigSummary(props.promptBaseline.config)
                  })
                </span>
              ) : null}
            </span>
            <span className="object-search-review-history-actions">
              <button
                type="button"
                className="object-search-secondary-button"
                disabled={!props.reviews.canUndo}
                onClick={props.reviews.undo}
                title={
                  props.reviews.annotations[0]
                    ? `Undo review of #${props.reviews.annotations[0].targetId} (Ctrl/Cmd+Z)`
                    : "Undo review (Ctrl/Cmd+Z)"
                }
              >
                Undo
              </button>
              <button
                type="button"
                className="object-search-secondary-button"
                disabled={!props.reviews.canRedo}
                onClick={props.reviews.redo}
                title="Redo review (Ctrl/Cmd+Shift+Z)"
              >
                Redo
              </button>
            </span>
          </div>
          {props.reviews.error ? (
            <div className="object-search-error-banner" role="alert">
              <span>Annotations: {props.reviews.error}</span>
            </div>
          ) : null}
          <ReviewAnnotationList
            annotations={props.reviews.annotations}
            onClear={props.reviews.clearAnnotation}
          />
        </div>
      ) : null}
    </section>
  );
}
