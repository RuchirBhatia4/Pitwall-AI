"use client";

import { pct, useApi } from "@/lib/api";
import type { Season } from "@/lib/types";
import { StrategyLoader } from "@/components/Brand";

const PIPELINE = [
  { t: "Ingest", d: "FastF1 timing for every 2026 session (practice, sprint, qualifying, race). Live: OpenF1 or F1's live-timing feed." },
  { t: "Season prior", d: "Fit the lap-time model to every earlier race: driver pace, compound offset, fuel burn, and wear measured as the drop-off within stints. Between-race spread = prior uncertainty." },
  { t: "Weekend evidence", d: "Classify every practice stint (race sim, lighter-fuel run, qualifying simulation, sprint stint, other); measure the drop-off of race sims only, then map practice → race with a ratio learned on earlier weekends." },
  { t: "Tyre sets", d: "Count each driver's new and used sets left for the race from every session before it; a stop takes a new set while any are left, then the freshest used one." },
  { t: "Bayesian blend", d: "Conjugate Gaussian update per compound; compound order enforced by isotonic projection." },
  { t: "Revealed preference", d: "Inverse optimisation: learn fuel-wear sensitivity, per-stop track-position cost and a hard-start cost that best reproduce teams' past choices. Tyre wear is measured, never tuned." },
  { t: "Optimise", d: "Exact dynamic programming over every 0–3 stop compound sequence and every pit lap, with FIA two-compound rule and stint-life limits." },
  { t: "Simulate risk", d: "Monte Carlo over Safety Cars / VSCs (per-lap hazards, blended with each circuit's 2024–25 record) and posterior parameter draws, with a reactive pit-wall policy. Reports P(best), regret, CVaR." },
  { t: "Race simulation", d: "Simulate the whole field lap by lap: race pace per driver, tyres, traffic, overtaking learned per circuit, pit-stop times, SC/VSC. Each driver's candidate plans are ranked by expected finishing position against rivals' predicted plans." },
  { t: "Live loop", d: "Every lap: update wear from all cars' green laps, re-estimate pit loss, re-optimise from the car's tyre state, check undercuts, rejoin slot and model drift." },
];

const TECH = [
  ["Full-field race simulation", "The core of team strategy tools: every car is simulated lap by lap, so a stop that rejoins in traffic or a faster car that cannot pass costs places. Vectorised Monte Carlo with common random numbers (every plan faces the same Safety Cars, lap noise and pit stops), so differences between plans are the plan, not noise. Live, it restarts from the current gaps and tyres."],
  ["Learned overtaking model", "Logistic model of P(pass per lap | within 1 s, pace advantage) fitted on every close-following lap of earlier 2026 races, with one intercept per circuit. A new circuit's difficulty comes from its 2024–25 races, shrunk toward the 2026 average (Monaco: ~0% per lap at equal pace; Monza: ~20%)."],
  ["Race-pace model", "Ridge regression of each driver's race pace on starting position and recency-weighted form, walk-forward; its error is propagated into the simulation as per-race pace draws. The grid is the official starting grid (penalties, pit-lane starts) once published."],
  ["Shortlist, then simulate", "Like a pit wall, the race simulation only ranks strategies teams realistically consider: stop count and starting tyre with at least 10% probability under the behaviour model. Added after Austin together with the practice feature and the official grid, walk-forward it lifted the starting tyre from 34% to 56% and the stop count from 45% to 51%, and cut the first-stop error from 9.6 to 8.7 laps."],
  ["Practice as revealed preference", "Teams run race simulations in practice on the tyres they plan to race, so each weekend's share of long-run laps per compound is a behaviour-model feature (stop count 31% → 44%, starting tyre 52% → 56%, walk-forward). It is kept out of first-stop timing, where it made errors worse."],
  ["Circuit history", "Each circuit's 2024–25 races give its overtaking difficulty, Safety Car / VSC record and pit-lane loss, shrunk toward the 2026 season values (pit-loss error on 2026 races 2.7 → 2.2 s)."],
  ["Practice run classification", "Each practice stint is labelled from its lap times relative to the driver's best: race sims on race-start fuel, lighter-fuel runs, qualifying simulations (push laps between cool-downs), sprint stints and the rest. Push laps, cool-down and traffic laps, and the quick lap after one are dropped before the drop-off is measured. Walk-forward on rounds 4–18, race sims alone predicted each race's within-stint wear with 0.025 s/lap mean error (rank correlation 0.57 across weekends), vs 0.037 / 0.32 for the previous long-run detector and 0.032 for the season average. A team's practice drop-off relative to the field did not predict its race wear relative to the field (rank correlation 0.03), so practice sets the field's wear, not each team's."],
  ["Within-stint wear", "Race wear is the slope of lap time against tyre age inside each stint (one intercept per driver-stint, fuel burn added back), so pushing after a stop or engine-mode changes cannot pass for tyre wear. Measured this way the 2026 soft wears like the medium (soft − medium −0.010 ± 0.002 s/lap over 17 races), so the old soft-wear multiplier tuned to team choices (×1.6–2.4) was removed. With measured wear and tyre sets, stop count and timing held (race simulation 51.4% → 51.9% stops), pre-race wear error fell from 0.037 to 0.025 s/lap, live false 'BOX, BOX' calls fell from 58 to 45 on seven replays, and plans lose less time to each race's hindsight optimum (race simulation median 11.0 → 8.3 s); matching which compounds teams used fell (race simulation 40.0% → 33.8%, p = 0.06), because the old model mostly called the commonest medium–hard."],
  ["Tyre-set inventory", "Pirelli allocates 8 soft, 3 medium and 2 hard sets (6 / 4 / 2 on sprint weekends), and qualifying burns most softs: the median 2026 driver started a normal race with 1 new soft and 1 new medium. Each driver's sets are counted from the weekend's sessions and used by the optimiser, the race simulation, the behaviour model and the live engine."],
  ["Hierarchical Bayesian updating", "Practice, season and live evidence are combined with explicit uncertainty; sparse teams borrow strength from the field instead of overfitting."],
  ["Walk-forward evaluation", "Every round is predicted with models trained only on earlier rounds. No random splits, no leakage — the original project's random-split R² of 0.66 vs future-race R² < 0 is exactly why."],
  ["Split-conformal prediction intervals", "First-stop intervals come from calibrated residual quantiles on earlier races, with the finite-sample correction; coverage is reported, not assumed."],
  ["Learned transfer functions", "Practice wear is not race wear: drivers do not manage tyres in practice, and race sims drop off 2–2.5× faster than the same compounds in the race. The practice→race ratio and its error are regressed from earlier weekends, so the model knows how much to trust FP2."],
  ["Hybrid physics + ML", "A walk-forward multinomial logistic regression learns how teams actually choose stop count and starting tyre from physics features (1- vs 2-stop gap, optimal stop lap, wear, grid slot), computed with the driver's own tyre sets; the optimiser then times the plan inside that class. 'Fastest' and 'likely' are reported separately."],
  ["Walk-forward regularisation", "How far to trust 15 race-level features learned from a dozen races is itself learned: for each round, the regularisation strength that best predicted every earlier round from the rounds before it. It picks heavy shrinkage toward the field's habits for stop count and timing; a hand-set value had let the classifier call three stops for the whole field at Monaco, Spielberg and Spa. Race simulation first-stop error 7.7 → 7.1 laps (p = 0.03), hybrid compounds 35.6% → 40.0%."],
  ["Scoring what teams actually did", "Strategies are read from timing data, which needs care: a pit-lane trip that keeps the same set (the Safety Car leading the field through the pit lane at Baku) is not a stop, a tyre change under a red flag is not a pit stop (Monza's lap-2 red flag), and cars that raced intermediates leave the dry scope. 50 of 210 driver-races had been mislabelled; the fix changes every method's score, the naive baseline's included, and is reported separately from model changes."],
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
  "The race simulation picks the starting tyre far better than the physics optimiser and the stop count somewhat better; on compounds the two are level and the optimiser times the first stop slightly better (see the table). Rivals do not react to each other's green-flag stops (no undercut-covering game), and a second round of best responses did not improve the backtest.",
  "The race simulation does not model retirements, lapped-car traffic or DRS trains beyond the overtaking model; overtaking depends only on pace difference and the circuit.",
  "Which compounds a driver uses is the weakest prediction (about one in three right for the race simulation, like the naive baseline; 40% for the hybrid). Most 2026 races were run on one compound pair by nearly the whole field, but neither the pre-race physics, the circuit's 2024–25 strategies nor Pirelli's nominations tell which pair; the soft's race pace varies from 0.9 s/lap faster than the medium to 0.7 s/lap slower between races.",
  "Pre-race prediction of team behaviour is only modestly better than a naive baseline on some metrics and worse on others (see the table) — e.g. the starting tyre is still predicted less accurately than 'start on what most teams started on'. Team orders, incidents and strategic covering are not observable in public timing data.",
  "Tyre sets are counted from timing data, which does not identify individual sets: a used set's age is taken as the freshest used set of that compound, and OpenF1 (the source from Singapore on) sometimes flags a used set as new, which can only under-count what a driver has left.",
  "Wet races, and cars that raced intermediates in a dry race, are excluded from dry-strategy metrics; intermediate/wet tyre strategy is not modelled.",
  "Tyre wear is linear-plus-quadratic with a fuel-load interaction; real thermal degradation and cliffs are approximated through stint-life limits. Wear is linear within a stint, so a fresh-tyre peak in the first laps of a stint is not modelled separately.",
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
                  <th className="py-2 font-medium">Race sim</th>
                  <th className="py-2 font-medium">Hybrid</th>
                  <th className="py-2 font-medium">Physics only</th>
                  <th className="py-2 font-medium">Baseline</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Stops correct</td><td>{pct(bt.racesim?.stops_accuracy)}</td><td>{pct(bt.hybrid?.stops_accuracy)}</td><td>{pct(bt.stops_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.stops_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Compound set correct</td><td>{pct(bt.racesim?.compound_set_accuracy)}</td><td>{pct(bt.hybrid?.compound_set_accuracy)}</td><td>{pct(bt.compound_set_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.compound_set_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">Starting tyre correct</td><td>{pct(bt.racesim?.start_compound_accuracy)}</td><td>{pct(bt.hybrid?.start_compound_accuracy)}</td><td>{pct(bt.start_compound_accuracy)}</td><td className="text-text-2">{pct(bt.baseline.start_compound_accuracy)}</td></tr>
                <tr className="border-b border-line/60"><td className="py-2 text-text-2">First-stop lap MAE</td><td>{bt.racesim?.first_stop_mae != null ? `${bt.racesim.first_stop_mae.toFixed(1)} laps` : "—"}</td><td>{bt.hybrid ? `${bt.hybrid.first_stop_mae.toFixed(1)} laps` : "—"}</td><td>{bt.first_stop_mae.toFixed(1)} laps</td><td className="text-text-2">{bt.baseline.first_stop_mae.toFixed(1)} laps</td></tr>
                {bt.racesim?.finishing_order_rho != null && (
                  <tr className="border-b border-line/60"><td className="py-2 text-text-2">Finishing order (rank correlation with the result)</td><td>{bt.racesim.finishing_order_rho.toFixed(2)}</td><td>—</td><td>—</td><td className="text-text-2">grid order {bt.racesim.grid_order_rho?.toFixed(2)}</td></tr>
                )}
                <tr><td className="py-2 text-text-2">80% conformal interval coverage (physics first stop)</td><td>—</td><td>—</td><td>{pct(bt.conformal_coverage)}</td><td className="text-text-2">target {pct(bt.conformal_target)}</td></tr>
              </tbody>
            </table>
            <p className="mt-3 text-xs text-text-3">Race sim: {bt.racesim?.description ?? "not built"}. Baseline: {bt.baseline.description}.</p>
          </div>
        ) : (
          <StrategyLoader label="Loading…" />
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
