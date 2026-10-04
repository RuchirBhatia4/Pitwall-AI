"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TyreModel } from "@/lib/types";
import { COMPOUND_COLOR, compoundName } from "./Tyre";

const axis = { stroke: "var(--line-strong)", tick: { fill: "var(--text-3)", fontSize: 11 }, tickLine: false };

function TipBox({ title, rows }: { title: string; rows: { color: string; label: string; value: string }[] }) {
  return (
    <div className="rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-semibold text-text">{title}</div>
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-2 text-text-2">
          <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
          <span className="flex-1">{r.label}</span>
          <span className="tabular text-text">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * Lap-time cost of each compound vs tyre age, relative to a new Medium.
 * Shaded band = +/-1 sd of the degradation posterior (uncertainty, not noise).
 */
export function DegradationChart({ model, mult = 1, maxAge = 40, height = 260 }: { model: TyreModel; mult?: number; maxAge?: number; height?: number }) {
  // Same wear the optimiser uses for this driver: compound rate x personal multiplier.
  const rate = (c: string) => Math.max(model.compounds[c].deg * mult, 0.005);
  const comps = ["SOFT", "MEDIUM", "HARD"].filter((c) => model.compounds[c]);
  const data = Array.from({ length: maxAge + 1 }, (_, a) => {
    const row: Record<string, number | [number, number] | null> = { age: a };
    for (const c of comps) {
      const p = model.compounds[c];
      const k = rate(c);
      const within = a <= p.max_stint;
      const mean = p.offset + k * a + model.quad * a * a;
      row[c] = within ? +mean.toFixed(3) : null;
      row[`${c}_band`] = within ? [+(mean - p.deg_sd * a).toFixed(3), +(mean + p.deg_sd * a).toFixed(3)] : null;
    }
    return row;
  });
  return (
    <div>
      <div className="flex flex-wrap gap-4 mb-2 text-xs text-text-2">
        {comps.map((c) => (
          <span key={c} className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded" style={{ background: COMPOUND_COLOR[c] }} />
            {compoundName(c)}: {rate(c).toFixed(3)} s/lap ± {(model.compounds[c].deg_sd * mult).toFixed(3)}
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="age" {...axis} label={{ value: "Tyre age (laps)", position: "insideBottom", offset: -2, fill: "var(--text-3)", fontSize: 11 }} height={36} />
          <YAxis {...axis} tickFormatter={(v) => `${v > 0 ? "+" : ""}${v}s`} width={52} />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TipBox
                  title={`Tyre age ${label} laps`}
                  rows={comps
                    .map((c) => ({ c, v: payload.find((p) => p.dataKey === c)?.value as number | undefined }))
                    .filter((r) => r.v != null)
                    .map((r) => ({ color: COMPOUND_COLOR[r.c], label: compoundName(r.c), value: `${(r.v as number) > 0 ? "+" : ""}${(r.v as number).toFixed(2)}s` }))}
                />
              ) : null
            }
          />
          {comps.map((c) => (
            <Area key={`${c}b`} dataKey={`${c}_band`} stroke="none" fill={COMPOUND_COLOR[c]} fillOpacity={0.1} isAnimationActive={false} activeDot={false} />
          ))}
          {comps.map((c) => (
            <Line key={c} dataKey={c} stroke={COMPOUND_COLOR[c]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Live: model curve for the current stint with an 80% band and the car's observed laps. */
export function LiveTyreCurve({ curve, height = 230 }: { curve: { compound: string; ages: number[]; predicted: number[]; band: number; observed: { age: number; delta: number }[] }; height?: number }) {
  const obs = new Map(curve.observed.map((o) => [o.age, o.delta]));
  const data = curve.ages.map((a, i) => ({
    age: a,
    model: curve.predicted[i],
    band: [curve.predicted[i] - curve.band, curve.predicted[i] + curve.band] as [number, number],
    observed: obs.get(a) ?? null,
  }));
  const color = COMPOUND_COLOR[curve.compound] ?? "var(--text-2)";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="age" {...axis} height={30} />
        <YAxis {...axis} tickFormatter={(v) => `${v > 0 ? "+" : ""}${Number(v).toFixed(1)}`} width={44} domain={["auto", "auto"]} />
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <TipBox
                title={`Tyre age ${label}`}
                rows={[
                  { color, label: "Model", value: `${Number(payload.find((p) => p.dataKey === "model")?.value ?? 0).toFixed(2)}s` },
                  ...(payload.find((p) => p.dataKey === "observed")?.value != null
                    ? [{ color: "var(--text)", label: "Observed", value: `${Number(payload.find((p) => p.dataKey === "observed")?.value).toFixed(2)}s` }]
                    : []),
                ]}
              />
            ) : null
          }
        />
        <Area dataKey="band" stroke="none" fill={color} fillOpacity={0.1} isAnimationActive={false} activeDot={false} />
        <Line dataKey="model" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Scatter dataKey="observed" fill="var(--text)" stroke="var(--surface)" strokeWidth={2} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
