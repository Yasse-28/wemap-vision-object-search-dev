/**
 * The selection session's panel: start, draw across floors, review, export.
 *
 * The panel deliberately shows three numbers rather than one — the per-region
 * count, the per-floor breakdown and the deduplicated total. Overlapping regions
 * on one floor are the normal case, and a single total would hide whether a region
 * is contributing anything at all.
 */

import type { ExportSession } from "./session";
import type { RoiSelection } from "./types";

function levelLabel(level: string | null): string {
  return level === null ? "no floor" : `Level ${level}`;
}

export default function ExportRoiPanel(props: {
  session: ExportSession;
  selection: RoiSelection;
  currentLevel: string | null;
  onExport: () => void;
}) {
  const { session, selection } = props;
  const allSelected =
    session.rois.length > 0 && session.selectedIds.size === session.rois.length;

  return (
    <section className="export-roi-panel">
      <header className="export-roi-header">
        <h3>Export selection</h3>
        {session.active ? (
          <button
            type="button"
            className="secondary-button"
            onClick={session.stop}
          >
            Pause
          </button>
        ) : (
          <button
            type="button"
            className="secondary-button"
            onClick={session.start}
          >
            {session.rois.length ? "Resume" : "Start"}
          </button>
        )}
      </header>

      {session.active ? (
        <p className="muted">
          Draw a region on the map; it is added on{" "}
          {levelLabel(props.currentLevel)}. Change floor and draw again to
          extend the selection.
        </p>
      ) : session.rois.length === 0 ? (
        <p className="muted">
          Start a session, then draw one or more regions to pick the keyframes
          to export as a new map.
        </p>
      ) : null}

      {session.rois.length > 0 ? (
        <>
          <div className="export-roi-list-toolbar">
            <label className="inline-check">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={(event) =>
                  session.setAllSelected(event.target.checked)
                }
              />
              Select all
            </label>
            <button
              type="button"
              className="icon-button is-danger"
              title="Delete the selected regions"
              aria-label="Delete the selected regions"
              disabled={session.selectedIds.size === 0}
              onClick={session.removeSelected}
            >
              🗑
            </button>
          </div>

          <ul className="export-roi-list">
            {session.rois.map((roi, index) => (
              <li key={roi.id} className="export-roi-row">
                <label className="inline-check">
                  <input
                    type="checkbox"
                    checked={session.selectedIds.has(roi.id)}
                    onChange={() => session.toggleSelected(roi.id)}
                  />
                  <span className="export-roi-name">Region {index + 1}</span>
                </label>
                <span className="export-roi-meta">
                  {levelLabel(roi.level)} · {selection.countByRoi[roi.id] ?? 0}{" "}
                  keyframes
                </span>
                <button
                  type="button"
                  className="icon-button"
                  title="Delete this region"
                  aria-label={`Delete region ${index + 1}`}
                  onClick={() => session.removeRoi(roi.id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>

          <div className="export-roi-summary">
            <ul className="export-roi-per-level">
              {selection.perLevel.map((entry) => (
                <li key={String(entry.level)}>
                  <span>{levelLabel(entry.level)}</span>
                  <strong>{entry.count}</strong>
                </li>
              ))}
            </ul>
            <p className="export-roi-total">
              Total keyframes: <strong>{selection.total}</strong>
            </p>
          </div>

          <div className="export-roi-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={session.clear}
            >
              Clear session
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={selection.total === 0}
              onClick={props.onExport}
            >
              Finish and export…
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}
