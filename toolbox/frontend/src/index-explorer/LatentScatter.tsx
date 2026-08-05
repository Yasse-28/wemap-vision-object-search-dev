import { useMemo } from "react";

import type { LatentPoint } from "./types";

type Props = {
  points: LatentPoint[];
  width?: number;
  height?: number;
};

function LatentScatter(props: Props) {
  const width = props.width ?? 520;
  const height = props.height ?? 360;

  const layout = useMemo(() => {
    if (!props.points.length) {
      return null;
    }
    const xs = props.points.map((point) => point.x);
    const ys = props.points.map((point) => point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padX = (maxX - minX) * 0.08 || 1;
    const padY = (maxY - minY) * 0.08 || 1;
    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    };
  }, [props.points]);

  if (!layout) {
    return <p className="muted">No projection points.</p>;
  }

  const spanX = layout.maxX - layout.minX || 1;
  const spanY = layout.maxY - layout.minY || 1;

  return (
    <svg
      className="latent-scatter"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      role="img"
      aria-label="Object embedding projection"
    >
      <rect x={0} y={0} width={width} height={height} fill="#f7faf6" rx={8} />
      {props.points.map((point) => {
        const cx =
          ((point.x - layout.minX) / spanX) * (width - 24) + 12;
        const cy =
          height - (((point.y - layout.minY) / spanY) * (height - 24) + 12);
        const radius = Math.max(3, Math.min(14, point.size));
        return (
          <circle
            key={point.object_id}
            cx={cx}
            cy={cy}
            r={radius}
            fill={point.color}
            fillOpacity={point.active ? 0.95 : 0.35}
            stroke={point.active ? "#1f2937" : "#d1d5db"}
            strokeWidth={point.active ? 1.5 : 1}
          >
            <title>
              {point.object_id} · sim {point.similarity}
            </title>
          </circle>
        );
      })}
    </svg>
  );
}

export default LatentScatter;
