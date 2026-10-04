"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, CalendarClock, CheckCircle2, CircleDashed, Radio, Trophy } from "lucide-react";
import { countdown, pct, useApi } from "@/lib/api";
import type { Season, SeasonRound } from "@/lib/types";
import { Stat } from "@/components/Stat";

function useNow(ms = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}

function RaceDayHero({ season }: { season: Season }) {
  const now = useNow();
  const rnd = season.rounds.find((r) => r.round === (season.live_round ?? season.next_round));
  if (!rnd) return null;
  const start = new Date(rnd.race_start_utc.replace(" ", "T") + "Z").getTime();
  const secsTo = (start - now) / 1000;
  const live = season.live_round === rnd.round;
  return (
    <div className="relative overflow-hidden card p-6 sm:p-8">
      <div className="grid-bg absolute inset-0 opacity-60 pointer-events-none" />
      <div className="relative flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 text-sm">
            {live ? (
              <span className="flex items-center gap-1.5 rounded-full bg-accent/15 px-2.5 py-1 text-accent font-semibold">
                <span className="live-dot h-2 w-2 rounded-full bg-accent" /> LIVE NOW
              </span>
            ) : (
              <span className="flex items-center gap-1.5 rounded-full bg-surface-3 px-2.5 py-1 text-text-2">
                <CalendarClock size={14} /> Next race · Round {rnd.round}
              </span>
            )}
            <span className="text-text-3">{rnd.date}</span>
          </div>
          <h1 className="mt-3 text-3xl sm:text-5xl font-semibold tracking-tight">{rnd.event}</h1>
          <p className="mt-2 text-text-2">
            {rnd.location}
            {rnd.country && rnd.location !== rnd.country ? ` · held for ${rnd.country}` : ""}
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/live" className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 font-semibold hover:brightness-110">
              <Radio size={16} /> Open live pit wall
            </Link>
            <Link href={`/race/${rnd.round}`} className="inline-flex items-center gap-2 rounded-lg border border-line-strong px-4 py-2.5 hover:bg-surface-2">
              Pre-race strategy predictions <ArrowRight size={16} />
            </Link>
          </div>
        </div>
        <div className="lg:text-right">
          <div className="label">{secsTo > 0 ? "Lights out in" : "Race status"}</div>
          <div className="mt-1 font-mono text-4xl sm:text-5xl tabular font-semibold">{secsTo > 0 ? countdown(secsTo) : live ? "Racing" : "Finished"}</div>
          <div className="mt-1 text-sm text-text-3">{new Date(start).toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short" })}</div>
        </div>
      </div>
    </div>
  );
}

function RoundCard({ r }: { r: SeasonRound }) {
  const finished = r.status === "finished";
  const m = r.metrics;
  const body = (
    <div className={`card h-full p-4 transition-colors ${r.available ? "hover:border-line-strong" : "opacity-50"}`}>
      <div className="flex items-center justify-between text-xs text-text-3">
        <span className="tabular">R{String(r.round).padStart(2, "0")} · {r.date}</span>
        {finished ? <CheckCircle2 size={14} className="text-text-3" /> : <CircleDashed size={14} />}
      </div>
      <div className="mt-2 font-semibold leading-tight">{r.event.replace(" Grand Prix", " GP")}</div>
      <div className="text-sm text-text-3">{r.location}</div>
      {r.winner && (
        <div className="mt-3 flex items-center gap-2 text-sm">
          <Trophy size={13} className="text-text-3" />
          <span className="h-3 w-1 rounded-full" style={{ background: r.winner.color }} />
          <span className="font-semibold">{r.winner.driver}</span>
          <span className="text-text-3 truncate">{r.winner.team}</span>
        </div>
      )}
      {m && !m.wet && (
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <div>
            <div className="text-text-3">Stops right</div>
            <div className="tabular font-semibold">{pct(m.likely_stops_accuracy ?? m.stops_accuracy)}</div>
          </div>
          <div>
            <div className="text-text-3">1st stop error</div>
            <div className="tabular font-semibold">{(m.likely_first_stop_mae ?? m.first_stop_mae) != null ? `${(m.likely_first_stop_mae ?? m.first_stop_mae)!.toFixed(1)} laps` : "—"}</div>
          </div>
        </div>
      )}
      {m?.wet && <div className="mt-3 text-xs text-info">Wet race · excluded from dry-strategy metrics</div>}
      {!finished && r.available && <div className="mt-3 text-xs text-accent font-semibold">Predictions ready →</div>}
    </div>
  );
  return r.available ? <Link href={`/race/${r.round}`}>{body}</Link> : body;
}

export default function Home() {
  const { data: season, error } = useApi<Season>("/api/season");
  if (error) return <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10 text-bad">Backend unavailable: {error}. Start it with <code>uvicorn src.api.main:app</code>.</div>;
  if (!season) return <div className="mx-auto max-w-7xl px-4 sm:px-6 py-10 text-text-3">Loading season…</div>;
  const bt = season.backtest;
  const done = season.rounds.filter((r) => r.status === "finished").length;
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8 space-y-8">
      <RaceDayHero season={season} />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="2026 races modelled" value={`${done} / ${season.rounds.length}`} sub="every finished round, plus the next one" />
        <Stat label="Optimiser" value="Exact DP" sub="0–3 stops × every pit lap, ~10 ms" />
        <Stat label="Risk model" value="Monte Carlo" sub="SC/VSC + parameter uncertainty" />
        <Stat
          label="Prediction interval coverage"
          value={bt ? pct(bt.conformal_coverage) : "—"}
          sub={bt ? `target ${pct(bt.conformal_target)} (split-conformal, first stop lap)` : undefined}
          tone={bt && Math.abs(bt.conformal_coverage - bt.conformal_target) < 0.08 ? "good" : "warn"}
        />
      </div>

      <div>
        <div className="flex items-end justify-between mb-3">
          <h2 className="text-xl font-semibold tracking-tight">2026 season</h2>
          <span className="text-sm text-text-3">Each race was predicted using only data from before it</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {season.rounds.map((r) => (
            <RoundCard key={r.round} r={r} />
          ))}
        </div>
      </div>

      {bt && (
        <section className="card p-5">
          <div className="flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 className="font-semibold">Honest scorecard: can we predict what teams will do?</h2>
              <p className="text-sm text-text-3 mt-1">{bt.scope}. {bt.driver_races} driver-races.</p>
            </div>
            <Link href="/methodology" className="text-sm text-text-2 hover:text-text inline-flex items-center gap-1">
              How this is measured <ArrowRight size={14} />
            </Link>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm tabular">
              <thead className="text-text-3 text-left">
                <tr className="border-b border-line">
                  <th className="py-2 pr-4 font-medium">Metric</th>
                  <th className="py-2 pr-4 font-medium">Hybrid (physics + behaviour ML)</th>
                  <th className="py-2 pr-4 font-medium">Physics optimiser only</th>
                  <th className="py-2 pr-4 font-medium">Naive baseline</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Number of stops correct", "stops_accuracy", true],
                    ["Compounds used correct", "compound_set_accuracy", true],
                    ["Starting tyre correct", "start_compound_accuracy", true],
                    ["First-stop lap error (mean)", "first_stop_mae", false],
                  ] as const
                ).map(([label, key, higher]) => {
                  const vals = [bt.hybrid?.[key], bt[key], bt.baseline[key]];
                  const fmt = (v?: number) => (v == null ? "—" : higher ? pct(v) : `${v.toFixed(1)} laps`);
                  const known = vals.filter((v): v is number => v != null);
                  const best = higher ? Math.max(...known) : Math.min(...known);
                  return (
                    <tr key={label} className="border-b border-line/60">
                      <td className="py-2 pr-4 text-text-2">{label}</td>
                      {vals.map((v, i) => (
                        <td key={i} className={`py-2 pr-4 ${v === best ? "font-semibold text-good" : i === 0 ? "font-semibold" : "text-text-2"}`}>
                          {fmt(v)}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-text-3">
            Hybrid: {bt.hybrid?.description ?? "not built"}. Baseline: {bt.baseline.description}. Best value per row in green. Predicting team behaviour from public data is hard — race incidents, team orders and track position drive many calls. The engine&apos;s main job is
            decision support (what is fastest from here, and how risky), which the live pit wall and what-if tools expose.
          </p>
        </section>
      )}
    </div>
  );
}
