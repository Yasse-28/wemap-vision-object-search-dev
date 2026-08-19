import type { KeyframeThresholdPoint } from "./types";

type Props = {
  points: KeyframeThresholdPoint[];
  /** The value the online path actually uses, drawn as a marker. */
  defaultThreshold?: number;
};

const WIDTH = 520;
const HEIGHT = 210;
const PAD = { top: 14, right: 14, bottom: 30, left: 44 };

/**
 * What `min_keyframes_per_cluster` costs, as a curve.
 *
 * Two series over the same x axis, because the threshold has to be judged on both at
 * once: the share of annotations that still have enough observing keyframes to survive
 * it (a ceiling it imposes before any ranking runs), and the depth-free covered share
 * among those it keeps (whether it is selecting for quality or only losing recall).
 * A flat second line under a falling first one means the filter costs and buys nothing.
 */
function KeyframeThresholdChart(props: Props) {
  const points = props.points;
  if (points.length === 0) {
    return null;
  }
  const minX = points[0].min_keyframes;
  const maxX = points[points.length - 1].min_keyframes;
  const spanX = Math.max(1, maxX - minX);

  const x = (value: number) =>
    PAD.left + ((value - minX) / spanX) * (WIDTH - PAD.left - PAD.right);
  const y = (share: number) =>
    HEIGHT - PAD.bottom - share * (HEIGHT - PAD.top - PAD.bottom);

  const line = (pick: (point: KeyframeThresholdPoint) => number) =>
    points
      .filter((point) => Number.isFinite(pick(point)))
      .map((point, index) => {
        const command = index === 0 ? "M" : "L";
        return `${command}${x(point.min_keyframes).toFixed(1)} ${y(pick(point)).toFixed(1)}`;
      })
      .join(" ");

  const marker = props.defaultThreshold ?? 2;

  return (
    <svg
      className="analysis-chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Retained and covered annotation share against min_keyframes_per_cluster"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((share) => (
        <g key={share}>
          <line
            className="analysis-chart-grid"
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={y(share)}
            y2={y(share)}
          />
          <text className="analysis-chart-tick" x={PAD.left - 6} y={y(share) + 3.5}>
            {`${Math.round(share * 100)}%`}
          </text>
        </g>
      ))}

      {marker >= minX && marker <= maxX ? (
        <g>
          <line
            className="analysis-chart-marker"
            x1={x(marker)}
            x2={x(marker)}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
          />
          <text className="analysis-chart-marker-label" x={x(marker) + 4} y={PAD.top + 9}>
            online default
          </text>
        </g>
      ) : null}

      <path className="analysis-chart-line" d={line((point) => point.retained_share)} />
      <path
        className="analysis-chart-line is-secondary"
        d={line((point) => point.covered_share)}
      />

      {points.map((point) => (
        <g key={point.min_keyframes}>
          <circle
            className="analysis-chart-dot"
            cx={x(point.min_keyframes)}
            cy={y(point.retained_share)}
            r={3}
          >
            <title>
              {`≥ ${point.min_keyframes} keyframes: ${point.retained} annotations retained`
                + ` (${(point.retained_share * 100).toFixed(1)}%)`}
            </title>
          </circle>
          {Number.isFinite(point.covered_share) ? (
            <circle
              className="analysis-chart-dot is-secondary"
              cx={x(point.min_keyframes)}
              cy={y(point.covered_share)}
              r={3}
            >
              <title>
                {`≥ ${point.min_keyframes} keyframes: ${(point.covered_share * 100).toFixed(1)}%`
                  + ` of the ${point.measurable} measurable ones covered depth-free`}
              </title>
            </circle>
          ) : null}
          <text
            className="analysis-chart-tick"
            x={x(point.min_keyframes)}
            y={HEIGHT - PAD.bottom + 14}
            textAnchor="middle"
          >
            {point.min_keyframes}
          </text>
        </g>
      ))}

      <text
        className="analysis-chart-axis"
        x={PAD.left + (WIDTH - PAD.left - PAD.right) / 2}
        y={HEIGHT - 2}
        textAnchor="middle"
      >
        min_keyframes_per_cluster
      </text>
    </svg>
  );
}

export default KeyframeThresholdChart;
