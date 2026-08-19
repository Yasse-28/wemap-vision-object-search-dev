import { useMemo } from "react";

import type { MetadataRowRecord } from "../index-explorer/types";
import { rulerTicks, thetaToRatio } from "./ribbonGeometry";

type Props = {
  rows: MetadataRowRecord[];
  selectedRowIndex: number | null;
  doneRowIndices: number[];
  queuedRowIndices: number[];
  onSelect: (rowIndex: number) => void;
};

type RowState = "current" | "done" | "queued" | "todo";

const TICKS = rulerTicks();

function formatDegrees(degrees: number): string {
  if (degrees === 0) {
    return "0°";
  }
  return `${degrees > 0 ? "+" : "−"}${Math.abs(degrees)}°`;
}

/**
 * The keyframe's whole turn as an axis: −180° to +180°, one notch per proposal.
 *
 * It sits under the panorama viewer and does what the viewer cannot: show the turn
 * all at once. A viewer answers "what is this object?"; the ruler answers "what is
 * left on this keyframe, and where is it relative to where I am looking?" — the
 * ribbon of a flattened panorama was tried here first and was unreadable at this
 * height, so the image navigation lives in the viewer and this stays the index.
 */
function AngularRibbon(props: Props) {
  const { rows, selectedRowIndex, onSelect } = props;
  const done = useMemo(() => new Set(props.doneRowIndices), [props.doneRowIndices]);
  const queued = useMemo(() => new Set(props.queuedRowIndices), [props.queuedRowIndices]);

  const placed = useMemo(
    () =>
      rows.map((row) => {
        const state: RowState =
          row.row_index === selectedRowIndex
            ? "current"
            : done.has(row.row_index)
              ? "done"
              : queued.has(row.row_index)
                ? "queued"
                : "todo";
        return {
          row,
          state,
          degrees: Math.round((row.theta_center * 180) / Math.PI),
          notchRatio: thetaToRatio(row.theta_center),
        };
      }),
    [rows, selectedRowIndex, done, queued],
  );

  const annotatedCount = rows.filter((row) => done.has(row.row_index)).length;

  return (
    <div className="annotation-ribbon">
      <div className="annotation-ribbon-ruler">
        {TICKS.map((tick) => (
          <span
            key={tick.degrees}
            className="annotation-ribbon-tick"
            style={{ left: `${tick.ratio * 100}%` }}
          >
            {formatDegrees(tick.degrees)}
          </span>
        ))}
        {placed.map((item) => (
          <button
            key={item.row.row_index}
            type="button"
            className={`annotation-ribbon-notch is-${item.state}`}
            style={{ left: `${item.notchRatio * 100}%` }}
            title={`Row ${item.row.row_index} · ${item.row.label ?? "unlabelled"} · θ ${formatDegrees(
              item.degrees,
            )}`}
            aria-label={`Proposal ${item.row.row_index} at ${formatDegrees(item.degrees)}`}
            aria-pressed={item.state === "current"}
            onClick={() => onSelect(item.row.row_index)}
          />
        ))}
      </div>

      <div className="annotation-ribbon-legend">
        <span className="annotation-ribbon-key is-done">annotated</span>
        <span className="annotation-ribbon-key is-current">current</span>
        <span className="annotation-ribbon-key is-todo">to do</span>
        <span className="annotation-ribbon-coverage">
          turn coverage: {annotatedCount} / {rows.length}
        </span>
      </div>
    </div>
  );
}

export default AngularRibbon;
