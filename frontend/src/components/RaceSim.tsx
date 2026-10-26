"use client";

import type { Driver, RaceForecast, SimOption } from "@/lib/types";
import { pct } from "@/lib/api";
import { SequencePills } from "./Tyre";
import { YourTeamTag, useTeam } from "./TeamTheme";

export function SimTable({ options }: { options: SimOption[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular">
        <thead className="text-text-3 text-left">
          <tr className="border-b border-line">
            <th className="py-2 pr-3 font-medium">Plan</th>
            <th className="py-2 pr-3 font-medium">Stops</th>
            <th className="py-2 pr-3 font-medium">Pit laps</th>
            <th className="py-2 pr-3 font-medium" title="Mean finishing position over the simulated races (± one standard deviation)">Exp. finish</th>
            <th className="py-2 pr-3 font-medium" title="Share of simulated races finishing in the top 10">P(points)</th>
            <th className="py-2 pr-3 font-medium" title="Mean championship points">Exp. points</th>
            <th className="py-2 pr-3 font-medium" title="Share of simulated races in which this plan gave the best finishing position of these options (ties count for each)">P(best)</th>
          </tr>
        </thead>
        <tbody>
          {options.map((o, i) => (
            <tr key={o.name + o.pit_laps.join()} className={`border-b border-line/60 ${i === 0 ? "bg-surface-2" : ""}`}>
              <td className="py-2 pr-3">
                <SequencePills sequence={o.sequence} />
              </td>
              <td className="py-2 pr-3">{o.stops}</td>
              <td className="py-2 pr-3 text-text-2">{o.pit_laps.map((l) => `L${l}`).join(" · ")}</td>
              <td className="py-2 pr-3">
                {i === 0 ? <span className="text-good font-semibold">P{o.expected_position.toFixed(1)}</span> : `P${o.expected_position.toFixed(1)}`}
                <span className="text-text-3"> ± {o.position_sd.toFixed(1)}</span>
              </td>
              <td className="py-2 pr-3">{pct(o.p_points)}</td>
              <td className="py-2 pr-3">{o.expected_points.toFixed(1)}</td>
              <td className="py-2 pr-3 text-text-2">{pct(o.p_best)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function FinishForecast({ forecast, drivers, finished, selected }: { forecast: RaceForecast; drivers: Driver[]; finished: boolean; selected?: string }) {
  const rows = drivers
    .filter((d) => forecast.drivers[d.driver])
    .map((d) => ({ d, f: forecast.drivers[d.driver] }))
    .sort((a, b) => a.f.expected_position - b.f.expected_position);
  const { isMine } = useTeam();
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular">
        <thead className="text-text-3 text-left">
          <tr className="border-b border-line">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Driver</th>
            <th className="py-2 pr-3 font-medium">Grid</th>
            <th className="py-2 pr-3 font-medium" title="Mean finishing position over the simulated races">Exp. finish</th>
            <th className="py-2 pr-3 font-medium">Win</th>
            <th className="py-2 pr-3 font-medium">Podium</th>
            <th className="py-2 pr-3 font-medium">Points</th>
            {finished && <th className="py-2 pr-3 font-medium">Actual</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ d, f }, i) => (
            <tr key={d.driver} className={`border-b border-line/60 ${d.driver === selected ? "bg-surface-2" : isMine(d.team) ? "bg-mark/10" : ""}`}>
              <td className="py-1.5 pr-3 text-text-3">{i + 1}</td>
              <td className="py-1.5 pr-3">
                <span className="inline-block h-3 w-1 rounded-sm mr-2 align-middle" style={{ background: d.color }} />
                <span className="font-semibold">{d.driver}</span> <span className="text-text-3 hidden sm:inline">{d.team}</span>{" "}
                {isMine(d.team) && <YourTeamTag />}
              </td>
              <td className="py-1.5 pr-3 text-text-2">{d.grid ? `P${d.grid}` : "–"}</td>
              <td className="py-1.5 pr-3">P{f.expected_position.toFixed(1)}</td>
              <td className="py-1.5 pr-3">
                <Bar v={f.p_win} />
              </td>
              <td className="py-1.5 pr-3">
                <Bar v={f.p_podium} />
              </td>
              <td className="py-1.5 pr-3">
                <Bar v={f.p_points} />
              </td>
              {finished && <td className="py-1.5 pr-3 text-text-2">{d.position ? `P${d.position}` : "DNF"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Bar({ v }: { v: number }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="relative inline-block h-1.5 w-12 rounded-full bg-surface-3 overflow-hidden">
        <span className="absolute inset-y-0 left-0 rounded-full bg-info" style={{ width: `${Math.round(v * 100)}%` }} />
      </span>
      <span className="text-text-2 w-9">{pct(v)}</span>
    </span>
  );
}
