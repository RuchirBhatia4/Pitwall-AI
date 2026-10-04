"use client";

import { pct, useApi } from "@/lib/api";
import type { Season } from "@/lib/types";

const PIPELINE = [
  { t: "Ingest", d: "FastF1 timing for every 2026 session (practice, sprint, qualifying, race). Live: OpenF1 or F1's live-timing feed." },
  { t: "Season prior", d: "Fit the lap-time model to every earlier race: driver pace, compound offset, wear rate, fuel burn. Between-race spread = prior uncertainty." },
  { t: "Weekend evidence", d: "Detect long runs in FP/sprint (robust Theil–Sen slopes), short-run compound gaps, then map practice → race with transfer ratios learned on earlier weekends." },
  { t: "Bayesian blend", d: "Conjugate Gaussian update per compound; team effects shrunk toward the field (hierarchical); compound order enforced by isotonic projection." },
  { t: "Revealed preference", d: "Inverse optimisation: learn soft-tyre penalty, fuel-wear sensitivity and per-stop track-position cost that best reproduce teams' past choices." },
  { t: "Optimise", d: "Exact dynamic programming over every 0–3 stop compound sequence and every pit lap, with FIA two-compound rule and stint-life limits." },
  { t: "Simulate risk", d: "Monte Carlo over Safety Cars / VSCs (per-lap hazards) and posterior parameter draws, with a reactive pit-wall policy. Reports P(best), regret, CVaR." },
  { t: "Live loop", d: "Every lap: update wear from all cars' green laps, re-estimate pit loss, re-optimise from the car's tyre state, check undercuts, rejoin slot and model drift." },
];

const TECH = [
  ["Hierarchical Bayesian updating", "Practice, season and live evidence are combined with explicit uncertainty; sparse teams borrow strength from the field instead of overfitting."],
  ["Walk-forward evaluation", "Every round is predicted with models trained only on earlier rounds. No random splits, no leakage — the original project's random-split R² of 0.66 vs future-race R² < 0 is exactly why."],
  ["Split-conformal prediction intervals", "First-stop intervals come from calibrated residual quantiles on earlier races, with the finite-sample correction; coverage is reported, not assumed."],
  ["Learned transfer functions", "Practice wear is not race wear. The practice→race ratio and its error are regressed from earlier weekends, so the model knows how much to trust FP2."],
  ["Hybrid physics + ML", "A walk-forward multinomial logistic regression learns how teams actually choose stop count and starting tyre from physics features (1- vs 2-stop gap, optimal stop lap, wear, grid slot); the optimiser then times the plan inside that class. 'Fastest' and 'likely' are reported separately."],
  ["Personal wear (partial pooling)", "Every driver has their own wear multiplier. Variance components were measured on 2026 races; the per-stint weight learned on rounds 1–8 beat the field average on rounds 9–15 (MAE 0.531 vs 0.557), while raw per-driver numbers did worse. Pre-race history/practice did not beat the field, so personalisation comes from the driver's own race stints."],
  ["Weekend severity", "Compounds without practice long runs inherit the weekend's overall severity (× typical compound ratio) instead of a low-wear season average."],
  ["Stop-count odds", "Monte Carlo reports how often 1, 2 or 3 stops is fastest, so knife-edge decisions are shown as odds rather than a single confident plan."],
  ["Inverse optimisation", "Teams' decisions are treated as demonstrations; behaviour parameters are fit so the optimiser reproduces them (inverse-RL in spirit, but tractable)."],
  ["Exact DP instead of RL", "The decision problem is small enough to solve optimally in ~10 ms, so no policy-gradient approximation is needed. Uncertainty lives in the parameters, handled by Monte Carlo."],
  ["Physics-informed constraints", "Isotonic projection forces softer compounds to be faster when new and to wear faster — fixes inverted estimates caused by selection effects."],
  ["Drift monitoring", "Live residuals of the last laps vs the predictive band raise an alert when the tyre leaves the model (cliff, graining, damage)."],
  ["Tool-grounded LLM", "The chat assistant can only cite numbers it obtained by calling the engine (function calling), so it cannot invent strategy figures."],
  ["Experiment tracking", "Backtests are logged to MLflow (kept from the original pipeline) for reproducible comparisons."],
];

const LIMITS = [
  "Track position is modelled only as a per-stop cost plus live rejoin/undercut checks — there is no full multi-car race simulation with overtaking probabilities.",
  "Pre-race prediction of team behaviour is only modestly better than a naive baseline on some metrics and worse on others (see the table) — e.g. the first-stop lap is still predicted less accurately than the season median. Team orders, incidents and strategic covering are not observable in public timing data.",
  "Wet races are excluded from dry-strategy metrics; intermediate/wet tyre strategy is not modelled.",
  "Tyre wear is linear-plus-quadratic with a fuel-load interaction; real thermal degradation and cliffs are approximated through stint-life limits and the soft-tyre penalty.",
  "Sepang (Round 16) has no recent F1 data; pit loss falls back to the 2026 season median until stops are observed live.",
];

export default function Methodology() {
  const { data: season } = useApi<Season>("/api/season");
  const bt = season?.backtest;
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 py-10 space-y-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">How PitWall AI works</h1>
        <p className="mt-2 text-text-2 max-w-3xl">
          A strategy engine built the way a team&apos;s would be — a transparent physical lap-time model, calibrated with uncertainty, solved exactly, stress-tested by simulation — and evaluated
          honestly on races it had not seen.
        </p>
      </div>

      <section>
        <h2 className="text-lg font-semibold mb-3">Lap-time model</h2>
        <div className="card p-5 font-mono text-sm overflow-x-auto">
          t<sub>lap</sub> = base<sub>driver</sub> + offset<sub>compound</sub> + wear<sub>compound,team</sub> · age · (1 + φ · fuel<sub>remaining</sub>) + q · age² − fuel<sub>burn</sub> · lap + ε
        </div>
        <p className="mt-2 text-sm text-text-3">Fuel burn is identified from race data because tyre age resets at each stop while lap number does not (fitted ≈ 0.04–0.07 s/lap in 2026).</p>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Pipeline</h2>
        <ol className="grid gap-3 sm:grid-cols-2">
          {PIPELINE.map((p, i) => (
            <li key={p.t} className="card p-4">
              <div className="flex items-center gap-2">
                <span className="grid place-items-center h-6 w-6 rounded-full bg-surface-3 text-xs tabular">{i + 1}</span>
                <span className="font-semibold">{p.t}</span>
              </div>
              <p className="mt-2 text-sm text-text-2">{p.d}</p>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Machine-learning techniques, and why each is here</h2>
        <div className="card divide-y divide-line">
          {TECH.map(([k, v]) => (
            <div key={k} className="grid sm:grid-cols-3 gap-2 p-4">
              <div className="font-semibold">{k}</div>
              <div className="sm:col-span-2 text-sm text-text-2">{v}</div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Validation</h2>
        {bt ? (
          <div className="card p-5">
            <p className="text-sm text-text-3">{bt.scope}. {bt.driver_races} driver-races.</p>
            <table className="mt-3 w-full text-sm tabular">
              <thead className="text-left text-text-3">
                <tr className="border-b border-line">
                  <th className="py-2 font-medium">Metric</th>
                  <th className="py-2 font-medium">Hybrid</th>
                  <th className="py-2 font-medium">Physics only</th>
                  <th className="py-2 font-medium">Baseline</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Stops correct</td><td>{pct(bt.hybrid?.stops_accuracy)}</td><td>{pct(bt.stops_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.stops_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Compound set correct</td><td>{pct(bt.hybrid?.compound_set_accuracy)}</td><td>{pct(bt.compound_set_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.compound_set_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Starting tyre correct</td><td>{pct(bt.hybrid?.start_compound_accuracy)}</td><td>{pct(bt.start_compound_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.start_compound_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">First-stop lap MAE</td><td>{bt.hybrid ? `${bt.hybrid.first_stop_mae.toFixed(1)} laps` : "—"}</td><td>{bt.first_stop_mae.toFixed(1)} laps</td><td className="text-text-2">{bt.baseline.first_stop_mae.toFixed(1)} laps</td></tr>
                <tr><td className="py-2 text-text-2">80% conformal interval coverage (physics first stop)</td><td>—</td><td>{pct(bt.conformal_coverage)}</td><td className="text-text-2">target {pct(bt.conformal_target)}</td></tr>
              </tbody>
            </table>
            <p className="mt-3 text-xs text-text-3">Baseline: {bt.baseline.description}.</p>
          </div>
        ) : (
          <div className="text-text-3">Loading…</div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Known limitations</h2>
        <ul className="space-y-2 text-sm text-text-2 list-disc pl-5">
          {LIMITS.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
