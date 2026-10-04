import type { Strategy } from "@/lib/types";
import { pct, secs } from "@/lib/api";
import { SequencePills } from "./Tyre";

export function PlanTable({ plans, chosen }: { plans: Strategy[]; chosen?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular">
        <thead className="text-text-3 text-left">
          <tr className="border-b border-line">
            <th className="py-2 pr-3 font-medium">Plan</th>
            <th className="py-2 pr-3 font-medium">Stops</th>
            <th className="py-2 pr-3 font-medium">Pit laps (window)</th>
            <th className="py-2 pr-3 font-medium" title="Deterministic model time vs the fastest plan">Δ time</th>
            <th className="py-2 pr-3 font-medium" title="Share of Monte Carlo races (SC/VSC + parameter uncertainty) in which this plan was fastest">P(best)</th>
            <th className="py-2 pr-3 font-medium" title="Average time lost vs the best plan in the same simulated race">Exp. regret</th>
            <th className="py-2 pr-3 font-medium" title="Mean regret in the worst 10% of simulated races">CVaR 90</th>
          </tr>
        </thead>
        <tbody>
          {plans.map((p) => (
            <tr key={p.name + p.pit_laps.join()} className={`border-b border-line/60 ${p.name === chosen ? "bg-surface-2" : ""}`}>
              <td className="py-2 pr-3">
                <SequencePills sequence={p.sequence} />
              </td>
              <td className="py-2 pr-3">{p.stops}</td>
              <td className="py-2 pr-3 text-text-2">
                {p.pit_laps.map((l, i) => (
                  <span key={i} className="mr-2">
                    L{l}
                    {p.windows[i] && <span className="text-text-3"> ({p.windows[i][0]}–{p.windows[i][1]})</span>}
                  </span>
                ))}
              </td>
              <td className="py-2 pr-3">{p.delta === 0 ? <span className="text-good font-semibold">fastest</span> : secs(p.delta, 1, true)}</td>
              <td className="py-2 pr-3">{pct(p.risk.win_probability)}</td>
              <td className="py-2 pr-3">{secs(p.risk.expected_delta)}</td>
              <td className="py-2 pr-3 text-text-2">{secs(p.risk.cvar90)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

