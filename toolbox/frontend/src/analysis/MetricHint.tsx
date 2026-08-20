import { useCallback, useRef, useState, type ReactNode } from "react";

import { CALIBRATION_NOTE, type MetricExplanation } from "./metricCatalogue";

type Props = {
  metric: MetricExplanation;
  children: ReactNode;
};

/**
 * The hover overlay that explains one metric: what it measures, how to read it, which
 * values are good, and why.
 *
 * The card is positioned `fixed` from the trigger's measured rect rather than absolutely
 * inside it. A pure-CSS tooltip is shorter, but this panel nests cards inside scrollable
 * sections, and an absolutely positioned card gets clipped by the first ancestor with
 * `overflow` set — exactly where the longest explanations live.
 */
function MetricHint(props: Props) {
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  const show = useCallback(() => {
    const rect = anchorRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const width = 340;
    // Clamp to the viewport so a tile at the right edge does not push the card off it.
    const left = Math.min(Math.max(8, rect.left), window.innerWidth - width - 8);
    setPosition({ left, top: rect.bottom + 6 });
  }, []);

  const hide = useCallback(() => setPosition(null), []);

  const metric = props.metric;
  return (
    <span
      className="metric-hint"
      ref={anchorRef}
      tabIndex={0}
      onMouseEnter={show}
      onFocus={show}
      onMouseLeave={hide}
      onBlur={hide}
    >
      {props.children}
      {position ? (
        <span
          className="metric-hint-card"
          role="tooltip"
          style={{ left: `${position.left}px`, top: `${position.top}px` }}
        >
          <span className="metric-hint-title">{metric.label}</span>
          <span className="metric-hint-row">{metric.measures}</span>
          <span className="metric-hint-row">{metric.reading}</span>
          {metric.bands ? (
            <span className="metric-hint-bands">{metric.bands}</span>
          ) : null}
          <span className="metric-hint-why">
            <em>Why:</em> {metric.why}
          </span>
          {metric.remedy ? (
            <span className="metric-hint-why">
              <em>If it is bad:</em> {metric.remedy}
            </span>
          ) : null}
          {metric.bands ? (
            <span className="metric-hint-note">{CALIBRATION_NOTE}</span>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}

export default MetricHint;
