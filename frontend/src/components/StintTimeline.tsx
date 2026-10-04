"use client";

import { COMPOUND_COLOR, COMPOUND_LETTER } from "./Tyre";

export interface TimelineRow {
  key: string;
  label: string;
  sublabel?: string;
  color?: string;
  stints: { compound: string; start_lap: number; end_lap: number }[];
  windows?: [number, number][];
  interval?: [number, number] | null;
  muted?: boolean;
}

/**
 * Horizontal stint bars on a shared lap axis (one row per plan or driver).
 * Pit windows are drawn as translucent bands; a conformal interval as a bracket.
 */
export function StintTimeline({
  rows,
  totalLaps,
  neutralised,
  currentLap,
  rowHeight = 26,
}: {
  rows: TimelineRow[];
  totalLaps: number;
  neutralised?: { sc: [number, number][]; vsc: [number, number][] };
  currentLap?: number;
  rowHeight?: number;
}) {
  const labelW = 132;
  const width = 900;
  const plotW = width - labelW - 8;
  const x = (lap: number) => labelW + ((lap - 1) / Math.max(totalLaps, 1)) * plotW;
  const barH = Math.min(14, rowHeight - 10);
  const height = rows.length * rowHeight + 26;
  const ticks = Array.from({ length: Math.floor(totalLaps / 10) + 1 }, (_, i) => i * 10).filter((t) => t > 0 && t <= totalLaps);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" role="img" aria-label="Stint timeline">
      {neutralised?.sc.map(([a, b], i) => (
        <rect key={`sc${i}`} x={x(a)} y={0} width={Math.max(x(b + 1) - x(a), 2)} height={height - 22} fill="var(--warn)" opacity={0.12} />
      ))}
      {neutralised?.vsc.map(([a, b], i) => (
        <rect key={`vsc${i}`} x={x(a)} y={0} width={Math.max(x(b + 1) - x(a), 2)} height={height - 22} fill="var(--warn)" opacity={0.06} />
      ))}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={height - 22} stroke="var(--line)" strokeWidth={1} />
          <text x={x(t)} y={height - 8} fill="var(--text-3)" fontSize={10} textAnchor="middle" className="tabular">
            {t}
          </text>
        </g>
      ))}
      <text x={labelW} y={height - 8} fill="var(--text-3)" fontSize={10} className="tabular">
        Lap 1
      </text>
      {rows.map((row, r) => {
        const y = r * rowHeight + (rowHeight - barH) / 2;
        return (
          <g key={row.key} opacity={row.muted ? 0.55 : 1}>
            {row.color && <rect x={0} y={y + 1} width={3} height={barH - 2} rx={1.5} fill={row.color} />}
            <text x={10} y={y + barH / 2 + 4} fill="var(--text)" fontSize={12} fontWeight={600}>
              {row.label}
            </text>
            {row.sublabel && (
              <text x={labelW - 6} y={y + barH / 2 + 4} fill="var(--text-3)" fontSize={10} textAnchor="end">
                {row.sublabel}
              </text>
            )}
            {row.windows?.map(([a, b], i) => (
              <rect key={`w${i}`} x={x(a)} y={y - 4} width={Math.max(x(b + 1) - x(a), 3)} height={barH + 8} rx={3} fill="var(--info)" opacity={0.16} />
            ))}
            {row.stints.map((s, i) => {
              const x0 = x(s.start_lap) + (i > 0 ? 1 : 0);
              const x1 = x(s.end_lap + 1) - 1;
              const w = Math.max(x1 - x0, 2);
              const color = COMPOUND_COLOR[s.compound] ?? "var(--text-3)";
              return (
                <g key={i}>
                  <rect x={x0} y={y} width={w} height={barH} rx={4} fill={color} opacity={0.92}>
                    <title>{`${s.compound} · laps ${s.start_lap}–${s.end_lap} (${s.end_lap - s.start_lap + 1})`}</title>
                  </rect>
                  {w > 26 && (
                    <text x={x0 + 6} y={y + barH / 2 + 4} fontSize={10} fontWeight={700} fill="#0b0b0b">
                      {COMPOUND_LETTER[s.compound] ?? "?"}
                      {w > 60 ? ` ${s.end_lap - s.start_lap + 1}` : ""}
                    </text>
                  )}
                </g>
              );
            })}
            {row.interval && (
              <g stroke="var(--text-2)" strokeWidth={1.5}>
                <line x1={x(row.interval[0])} x2={x(row.interval[1])} y1={y + barH + 3} y2={y + barH + 3} />
                <line x1={x(row.interval[0])} x2={x(row.interval[0])} y1={y + barH} y2={y + barH + 6} />
                <line x1={x(row.interval[1])} x2={x(row.interval[1])} y1={y + barH} y2={y + barH + 6} />
              </g>
            )}
          </g>
        );
      })}
      {currentLap != null && (
        <g>
          <line x1={x(currentLap)} x2={x(currentLap)} y1={0} y2={height - 22} stroke="var(--accent)" strokeWidth={2} />
          <text x={x(currentLap) + 4} y={10} fill="var(--accent)" fontSize={10} fontWeight={700}>
            NOW
          </text>
        </g>
      )}
    </svg>
  );
}
