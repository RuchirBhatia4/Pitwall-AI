"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowLeft, Info, Radio, Timer } from "lucide-react";
import { api, pct, secs, useApi } from "@/lib/api";
import type { RaceFile } from "@/lib/types";
import { DriverPicker } from "@/components/DriverPicker";
import { Section, Stat } from "@/components/Stat";
import { StintTimeline, type TimelineRow } from "@/components/StintTimeline";
import { SequencePills, TyreBadge, compoundName } from "@/components/Tyre";
import { DegradationChart } from "@/components/Charts";
import { PlanTable } from "@/components/PlanTable";
import { StopOdds } from "@/components/StopOdds";

function planRow(key: string, label: string, s: { sequence: string[]; pit_laps: number[] }, total: number, extra?: Partial<TimelineRow>): TimelineRow {
  const bounds = [1, ...s.pit_laps.map((p) => p + 1), total + 1];
  return {
    key,
    label,
    stints: s.sequence.map((c, i) => ({ compound: c, start_lap: bounds[i], end_lap: bounds[i + 1] - 1 })),
    ...extra,
  };
}

function WhatIf({ race, driver }: { race: RaceFile; driver: string }) {
  const L = race.weekend.total_laps;
  const [seq, setSeq] = useState<string[]>(["MEDIUM", "HARD"]);
  const [pits, setPits] = useState<number[]>([Math.round(L * 0.4)]);
  const [result, setResult] = useState<{ delta_to_optimal: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const setStops = (n: number) => {
    const s = ["MEDIUM", "HARD", "HARD", "MEDIUM"].slice(0, n + 1);
    setSeq(s);
    setPits(Array.from({ length: n }, (_, i) => Math.round((L * (i + 1)) / (n + 1))));
    setResult(null);
  };
  const run = async () => {
    setErr(null);
    try {
      setResult(await api("/api/strategy/evaluate", { method: "POST", body: JSON.stringify({ round: race.round, driver, sequence: seq, pit_laps: pits }) }));
    } catch (e) {
      setErr((e as Error).message);
      setResult(null);
    }
  };
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-text-2 mr-1">Stops</span>
        {[1, 2, 3].map((n) => (
          <button key={n} onClick={() => setStops(n)} className={`rounded-md border px-3 py-1 text-sm ${pits.length === n ? "border-text-2 bg-surface-3" : "border-line hover:bg-surface-2"}`}>
            {n}
          </button>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {seq.map((c, i) => (
          <div key={i} className="rounded-lg border border-line p-3">
            <div className="label mb-2">Stint {i + 1}</div>
            <div className="flex gap-1.5">
              {["SOFT", "MEDIUM", "HARD"].map((opt) => (
                <button
                  key={opt}
                  onClick={() => {
                    const s = [...seq];
                    s[i] = opt;
                    setSeq(s);
                    setResult(null);
                  }}
                  className={`rounded-md border px-2 py-1 ${c === opt ? "border-text-2 bg-surface-3" : "border-line"}`}
                  aria-label={compoundName(opt)}
                >
                  <TyreBadge compound={opt} size={20} />
                </button>
              ))}
            </div>
            {i < pits.length && (
              <label className="mt-3 block text-sm text-text-2">
                Pit at end of lap <span className="tabular font-semibold text-text">{pits[i]}</span>
                <input
                  type="range"
                  min={2}
                  max={L - 2}
                  value={pits[i]}
                  onChange={(e) => {
                    const p = [...pits];
                    p[i] = Number(e.target.value);
                    setPits(p);
                    setResult(null);
                  }}
                  className="mt-1 w-full accent-[var(--accent)]"
                />
              </label>
            )}
          </div>
        ))}
      </div>
      <StintTimeline rows={[planRow("wi", "Your plan", { sequence: seq, pit_laps: [...pits].sort((a, b) => a - b) }, L)]} totalLaps={L} />
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={run} className="rounded-lg bg-accent px-4 py-2 font-semibold hover:brightness-110">
          Evaluate plan
        </button>
        {result && (
          <span className="text-sm">
            {result.delta_to_optimal < 0.05 ? (
              <span className="text-good font-semibold">This is the model optimum.</span>
            ) : (
              <>
                <span className="font-semibold tabular">{secs(result.delta_to_optimal, 1, true)}</span> <span className="text-text-2">slower than the optimal plan for {driver} (green-flag race)</span>
              </>
            )}
          </span>
        )}
        {err && <span className="text-sm text-bad">{err}</span>}
      </div>
    </div>
  );
}

export function RaceView({ round }: { round: number }) {
  const { data: race, error } = useApi<RaceFile>(`/api/races/${round}`);
  const [driver, setDriver] = useState<string | null>(null);

  const drivers = useMemo(() => {
    if (!race) return [];
    const order = race.overview ? [...race.drivers].sort((a, b) => (a.position ?? 99) - (b.position ?? 99)) : race.drivers;
    return order.filter((d) => race.predictions[d.driver]?.predicted);
  }, [race]);
  const sel = driver ?? drivers[0]?.driver ?? null;

  if (error) return <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10 text-bad">{error}</div>;
  if (!race || !sel) return <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10 text-text-3">Loading race…</div>;

  const w = race.weekend;
  const L = race.overview?.total_laps ?? w.total_laps;
  const pred = race.predictions[sel];
  const d = race.drivers.find((x) => x.driver === sel)!;
  const finished = race.status === "finished";
  const m = race.metrics;

  const rows: TimelineRow[] = [];
  if (pred.likely) rows.push(planRow("likely", "Likely call", pred.likely, L, { sublabel: pred.likely.name, color: "var(--info)" }));
  if (pred.predicted)
    rows.push(planRow("pred", "Fastest", pred.predicted, L, { sublabel: pred.predicted.name, windows: pred.predicted.windows, interval: pred.first_stop_interval ?? null }));
  if (pred.actual) rows.push(planRow("act", "Actual", pred.actual, L, { sublabel: pred.actual.name, color: d.color }));
  if (pred.hindsight) rows.push(planRow("hind", "Hindsight best", pred.hindsight, L, { sublabel: pred.hindsight.name, muted: true }));

  const fieldRows: TimelineRow[] = race.overview
    ? [...race.overview.drivers]
        .sort((a, b) => (a.position ?? 99) - (b.position ?? 99))
        .filter((x) => race.overview!.stints[x.driver]?.length)
        .map((x) => ({
          key: x.driver,
          label: `${x.position ?? "–"}  ${x.driver}`,
          sublabel: x.status === "Finished" || x.status?.startsWith("+") || x.status === "Lapped" ? "" : "DNF",
          color: x.color,
          stints: race.overview!.stints[x.driver],
          muted: x.driver !== sel,
        }))
    : race.drivers.map((x) => {
        const p = race.predictions[x.driver]?.predicted;
        return { ...planRow(x.driver, `P${x.grid ?? "–"}  ${x.driver}`, p ?? { sequence: [], pit_laps: [] }, L), color: x.color, muted: x.driver !== sel };
      });

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link href="/" className="text-sm text-text-3 hover:text-text inline-flex items-center gap-1">
            <ArrowLeft size={14} /> Season
          </Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            <span className="text-text-3 tabular mr-2">R{String(race.round).padStart(2, "0")}</span>
            {race.event}
          </h1>
          <p className="text-text-2 mt-1">
            {race.location} · {race.date} · {L} laps · {finished ? "Finished" : "Pre-race prediction"}
          </p>
        </div>
        {!finished && (
          <Link href="/live" className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 font-semibold hover:brightness-110">
            <Radio size={16} /> Live pit wall
          </Link>
        )}
      </div>

      <div className="grid gap-3 grid-cols-2 lg:grid-cols-5">
        <Stat label="Pit-lane loss" value={secs(w.pit_loss)} sub={race.overview?.pit_loss ? "measured from green-flag stops" : "season median (no data at this track)"} />
        <Stat label="SC / VSC hazard" value={`${(w.sc_rate * 100).toFixed(1)}% / ${(w.vsc_rate * 100).toFixed(1)}%`} sub="per lap, from earlier 2026 races" />
        <Stat label="Practice → race wear" value={`× ${w.transfer.beta.toFixed(2)}`} sub={`learned on ${w.transfer.n} compound-weekends`} />
        <Stat label="Long runs analysed" value={race.practice_runs.length} sub="FP / sprint stints ≥ 5 laps" />
        <Stat label="Prior races" value={w.prior_races} sub="walk-forward: no future data" />
      </div>

      <Section title="Choose a driver" subtitle={finished ? "Ordered by finishing position" : "Ordered by qualifying position"}>
        <DriverPicker
          drivers={drivers.map((x) => ({ driver: x.driver, team: x.team, color: x.color, sub: finished ? (x.position ? `P${x.position}` : null) : x.grid ? `P${x.grid}` : null }))}
          value={sel}
          onChange={setDriver}
        />
      </Section>

      <div className="grid gap-6 lg:grid-cols-3">
        <Section
          className="lg:col-span-2"
          title={`${d.name} · ${d.team}`}
          subtitle={
            <>
              Likely call = behaviour model (what the team will probably do). Fastest = physics optimum; blue band = its pit window (within 2 s), bracket = 80% conformal interval for the first stop.
              {race.overview && race.overview.neutralised.sc.length > 0 && " Amber = Safety Car laps."}
            </>
          }
        >
          <StintTimeline rows={rows} totalLaps={L} neutralised={race.overview?.neutralised} rowHeight={34} />
          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {pred.likely && (
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="label">Likely team call</div>
                <div className="mt-1 flex items-center gap-2">
                  <SequencePills sequence={pred.likely.sequence} />
                  <span className="text-sm text-text-2 tabular">{pred.likely.pit_laps.map((l) => `L${l}`).join(" · ")}</span>
                </div>
                <div className="mt-1 text-xs text-text-3 tabular">
                  {Object.entries(pred.likely.stop_probabilities)
                    .sort()
                    .map(([k, v]) => `${k}-stop ${pct(v)}`)
                    .join(" · ")}
                </div>
              </div>
            )}
            {pred.stop_probabilities && (
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="label mb-2">Stop-count odds (fastest)</div>
                <StopOdds probs={pred.stop_probabilities} />
                <div className="mt-1.5 text-[11px] text-text-3">Share of 400 simulated races (SC/VSC + tyre-wear uncertainty) in which each stop count was fastest</div>
              </div>
            )}
            {pred.wear_profile && (
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="label">{sel} personal tyre wear</div>
                <div className="mt-1 text-xl font-semibold tabular">
                  ×{pred.wear_profile.mult.toFixed(2)} <span className="text-sm text-text-3">± {pred.wear_profile.sd.toFixed(2)} vs field</span>
                </div>
                <div className="mt-1 text-xs text-text-3">
                  {pred.wear_profile.season_races} earlier races, team ×{pred.wear_profile.prior_mult.toFixed(2)}
                  {pred.wear_profile.practice_mult != null && ` · practice ×${pred.wear_profile.practice_mult.toFixed(2)} (${pred.wear_profile.practice_runs} runs, shown only)`}. Updated from {sel}&apos;s own stints live.
                </div>
              </div>
            )}
            <div className="rounded-lg bg-surface-2 p-3">
              <div className="label">Fastest (physics)</div>
              <div className="mt-1 flex items-center gap-2">
                {pred.predicted && <SequencePills sequence={pred.predicted.sequence} />}
                <span className="text-sm text-text-2 tabular">{pred.predicted?.pit_laps.map((l) => `L${l}`).join(" · ")}</span>
              </div>
              <div className="mt-1 text-xs text-text-3">P(best) {pct(pred.predicted?.risk.win_probability)} in Monte Carlo</div>
            </div>
            {pred.actual && (
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="label">What the team did</div>
                <div className="mt-1 flex items-center gap-2">
                  <SequencePills sequence={pred.actual.sequence} />
                  <span className="text-sm text-text-2 tabular">{pred.actual.pit_laps.map((l) => `L${l}`).join(" · ")}</span>
                </div>
                {(pred.likely ?? pred.predicted)?.pit_laps[0] && pred.actual.pit_laps[0] && (
                  <div className="mt-1 text-xs text-text-3">
                    First stop {pred.actual.pit_laps[0] - (pred.likely ?? pred.predicted)!.pit_laps[0] >= 0 ? "+" : ""}
                    {pred.actual.pit_laps[0] - (pred.likely ?? pred.predicted)!.pit_laps[0]} laps vs {pred.likely ? "likely call" : "prediction"}
                  </div>
                )}
              </div>
            )}
            {pred.actual_vs_hindsight != null && (
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="label flex items-center gap-1">
                  <Timer size={12} /> vs hindsight optimum
                </div>
                <div className="mt-1 text-xl font-semibold tabular">{secs(pred.actual_vs_hindsight, 1, true)}</div>
                <div className="text-xs text-text-3">model time lost on a green-flag race using the race&apos;s own fitted tyre model</div>
              </div>
            )}
          </div>
        </Section>

        <Section title="Tyre model" subtitle={`${sel}'s wear (×${(pred.wear_profile?.mult ?? 1).toFixed(2)} of the field), relative to a new Medium`}>
          <DegradationChart model={w.model} mult={pred.wear_profile?.mult ?? 1} maxAge={Math.min(45, L)} height={240} />
          <p className="mt-2 text-xs text-text-3 flex gap-1.5">
            <Info size={14} className="shrink-0 mt-0.5" />
            Season prior + this weekend&apos;s long runs (Bayesian update), compounds without long runs scaled by the weekend&apos;s severity, times this driver&apos;s personal multiplier.
          </p>
        </Section>
      </div>

      {pred.alternatives?.length > 0 && (
        <Section title="Strategy options" subtitle="Ranked by expected time after simulating Safety Cars, VSCs and tyre-model uncertainty (300 simulated races)">
          <PlanTable plans={pred.alternatives} chosen={pred.predicted?.name} />
        </Section>
      )}

      <Section title={race.overview ? "How the field actually raced" : "Predicted strategy for the whole grid"} subtitle={race.overview ? "Every driver's stints, finishing order" : "One row per driver, qualifying order"}>
        <StintTimeline rows={fieldRows} totalLaps={L} neutralised={race.overview?.neutralised} rowHeight={22} />
      </Section>

      <Section title="What-if: build your own strategy" subtitle={`Model time vs the optimum for ${sel}, green-flag race`}>
        <WhatIf race={race} driver={sel} />
      </Section>

      {m && (
        <Section title="How good was the prediction here?" subtitle={m.wet ? "Wet race — dry-tyre metrics shown for reference only" : `Classified finishers (${m.drivers_evaluated})`}>
          <div className="grid gap-3 grid-cols-2 lg:grid-cols-5">
            <Stat label="Stops correct" value={pct(m.likely_stops_accuracy ?? m.stops_accuracy)} sub={`physics ${pct(m.stops_accuracy)} · baseline ${pct(m.baseline_stops_accuracy)}`} />
            <Stat label="Compound set correct" value={pct(m.likely_compound_set_accuracy ?? m.compound_set_accuracy)} sub={`physics ${pct(m.compound_set_accuracy)} · baseline ${pct(m.baseline_compound_set_accuracy)}`} />
            <Stat label="First stop error" value={m.first_stop_mae != null ? `${m.first_stop_mae.toFixed(1)} laps` : "—"} sub={m.first_stop_bias != null ? `teams stopped ${Math.abs(m.first_stop_bias).toFixed(1)} laps ${m.first_stop_bias > 0 ? "later" : "earlier"} on average` : undefined} />
            <Stat label="Conformal coverage" value={pct(m.conformal_coverage)} sub="target 80%" />
            <Stat label="Median loss vs hindsight" value={secs(m.median_time_lost_vs_hindsight, 1)} sub="what teams left on the table" />
          </div>
        </Section>
      )}
    </div>
  );
}
