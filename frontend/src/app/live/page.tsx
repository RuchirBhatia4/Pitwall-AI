"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, CircleSlash, Flag, Gauge, Plug, Radio, RefreshCcw, ShieldAlert, Wifi, WifiOff } from "lucide-react";
import { api, pct, secs, useApi, usePoll } from "@/lib/api";
import type { Analysis, LiveStatus, RaceFile, RaceState, Season } from "@/lib/types";
import { DriverPicker } from "@/components/DriverPicker";
import { Section } from "@/components/Stat";
import { StintTimeline, type TimelineRow } from "@/components/StintTimeline";
import { TyreBadge, compoundName } from "@/components/Tyre";
import { LiveTyreCurve } from "@/components/Charts";
import { PlanTable } from "@/components/PlanTable";
import { StopOdds } from "@/components/StopOdds";

type SourceKind = "openf1" | "signalr" | "replay" | "manual";

const SOURCES: { id: SourceKind; label: string; hint: string }[] = [
  { id: "signalr", label: "F1 live timing (free)", hint: "Formula 1's own live-timing feed: positions, gaps, tyres, pit stops, race control. Works without login; an F1TV sign-in (`python -m src.pitwall.f1tv_login`) adds the authenticated feed. Auto-reconnects." },
  { id: "openf1", label: "OpenF1", hint: "OpenF1 REST API. Live sessions need an OpenF1 sponsor account (OPENF1_USERNAME / OPENF1_PASSWORD on the API server); free from ~30 min after the session." },
  { id: "replay", label: "Replay a 2026 race", hint: "Re-runs a finished race as if live (lap clock), using the exact same engine." },
  { id: "manual", label: "Manual input", hint: "Type the car's state from the TV graphics. Always works." },
];

const STATUS_STYLE: Record<string, string> = {
  GREEN: "bg-good/15 text-good",
  YELLOW: "bg-warn/15 text-warn",
  SC: "bg-warn text-black",
  VSC: "bg-warn/80 text-black",
  VSC_ENDING: "bg-warn/40 text-text",
  RED: "bg-bad text-white",
  FINISHED: "bg-surface-3 text-text",
};

function CallCard({ a }: { a: Analysis }) {
  const c = a.call;
  const tone =
    c.action === "BOX_NOW" ? "border-accent bg-accent/10"
    : c.action === "BOX_SOON" || c.action === "WINDOW_OPEN" ? "border-warn bg-warn/10"
    : c.action === "HOLD" ? "border-line bg-surface-2"
    : "border-good/50 bg-good/5";
  const cond = a.conditions;
  return (
    <div className={clsx("rounded-2xl border-2 p-5 sm:p-6", tone)}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="label">Pit wall call · {a.driver} · lap {a.lap}/{a.total_laps}</div>
        {c.confidence != null && (
          <div className="text-xs text-text-2">
            Best plan in <span className="font-semibold tabular text-text">{pct(c.confidence)}</span> of simulated races
          </div>
        )}
      </div>
      <div className={clsx("mt-2 text-3xl sm:text-4xl font-bold tracking-tight", c.action === "BOX_NOW" && "text-accent")}>{c.headline}</div>
      {c.reason && <div className="mt-2 text-sm text-text-2 max-w-2xl">{c.reason}</div>}
      {cond?.changing && (
        <div className="mt-3 inline-flex flex-wrap items-center gap-2 rounded-lg border border-line px-2.5 py-1 text-xs text-text-2">
          <span aria-hidden>🌧</span>
          <span className="font-medium text-text">{cond.unsettled ? "Conditions changing" : "Rain reported"}</span>
          <span>{cond.reasons.join(" · ")}</span>
          {cond.excluded_laps.length > 0 && <span className="text-text-3">· {cond.excluded_laps.length} laps kept out of the wear model</span>}
        </div>
      )}
      {c.stop_probabilities && Object.keys(c.stop_probabilities).length > 1 && (
        <div className="mt-4 max-w-md">
          <div className="text-xs text-text-3 mb-1.5">More stops needed from here (simulated)</div>
          <StopOdds probs={c.stop_probabilities} label="more stops" />
        </div>
      )}
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <div>
          <div className="text-text-3">Current tyre</div>
          <TyreBadge compound={a.car.compound} age={a.car.tyre_age} size={24} />
        </div>
        <div>
          <div className="text-text-3">Next tyre</div>
          {c.next_compound ? <TyreBadge compound={c.next_compound} size={24} /> : <span className="text-text-2">none</span>}
        </div>
        <div>
          <div className="text-text-3">Window</div>
          <div className="tabular font-semibold">{c.window ? `L${c.window[0]}–${c.window[1]}` : "—"}</div>
        </div>
        <div>
          <div className="text-text-3">Box this lap instead</div>
          <div className="tabular font-semibold">{c.box_now_cost != null ? secs(c.box_now_cost, 1, true) : c.action === "BOX_NOW" ? "this is the call" : "—"}</div>
        </div>
      </div>
    </div>
  );
}

function TimingTower({ state, selected, onSelect }: { state: RaceState; selected: string | null; onSelect: (d: string) => void }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <table className="w-full text-sm tabular">
        <thead className="bg-surface-2 text-text-3 text-xs">
          <tr>
            <th className="py-1.5 pl-3 text-left font-medium">P</th>
            <th className="py-1.5 text-left font-medium">Driver</th>
            <th className="py-1.5 text-right font-medium">Gap</th>
            <th className="py-1.5 text-right font-medium">Int</th>
            <th className="py-1.5 pl-3 text-left font-medium">Tyre</th>
            <th className="py-1.5 pr-3 text-right font-medium">Stops</th>
          </tr>
        </thead>
        <tbody>
          {state.cars.map((c) => (
            <tr
              key={c.driver}
              onClick={() => onSelect(c.driver)}
              className={clsx("cursor-pointer border-t border-line/60 hover:bg-surface-2", c.driver === selected && "bg-surface-3", c.retired && "opacity-40")}
            >
              <td className="py-1.5 pl-3 w-8 text-text-2">{c.retired ? "–" : c.position}</td>
              <td className="py-1.5">
                <span className="inline-flex items-center gap-2">
                  <span className="h-3.5 w-1 rounded-full" style={{ background: c.color }} />
                  <span className="font-semibold">{c.driver}</span>
                  {c.in_pit && <span className="rounded bg-info/20 px-1 text-[10px] text-info">PIT</span>}
                  {c.retired && <span className="text-[10px] text-text-3">OUT</span>}
                </span>
              </td>
              <td className="py-1.5 text-right text-text-2">{c.position === 1 ? "Leader" : c.gap_to_leader != null ? `+${c.gap_to_leader.toFixed(1)}` : "—"}</td>
              <td className="py-1.5 text-right text-text-2">{c.interval != null && c.position !== 1 ? `+${c.interval.toFixed(1)}` : ""}</td>
              <td className="py-1.5 pl-3">
                <TyreBadge compound={c.compound} age={c.tyre_age} size={18} />
              </td>
              <td className="py-1.5 pr-3 text-right text-text-2">{c.stops}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RaceCraft({ a }: { a: Analysis }) {
  return (
    <div className="space-y-3 text-sm">
      {a.rejoin && (
        <div className="rounded-lg bg-surface-2 p-3">
          <div className="label mb-1">If {a.driver} boxes this lap</div>
          <div>
            Rejoins <span className="font-semibold">P{a.rejoin.position}</span> after a <span className="tabular">{a.rejoin.stop_cost}s</span> stop
            {a.rejoin.traffic && <span className="ml-2 rounded bg-warn/20 px-1.5 py-0.5 text-xs text-warn">traffic</span>}
          </div>
          <div className="mt-1 text-text-2 text-xs space-y-0.5">
            {a.rejoin.ahead && (
              <div>
                behind {a.rejoin.ahead.driver} by {a.rejoin.ahead.gap.toFixed(1)}s ({compoundName(a.rejoin.ahead.compound)}, {a.rejoin.ahead.tyre_age} laps)
              </div>
            )}
            {a.rejoin.behind && (
              <div>
                ahead of {a.rejoin.behind.driver} by {a.rejoin.behind.gap.toFixed(1)}s
              </div>
            )}
          </div>
        </div>
      )}
      {a.threats?.map((t) => (
        <div key={t.type} className={clsx("rounded-lg p-3", t.live ? (t.type === "undercut_threat" ? "bg-bad/10 border border-bad/40" : "bg-good/10 border border-good/40") : "bg-surface-2")}>
          <div className="flex items-center gap-1.5 font-semibold">
            {t.type === "undercut_threat" ? <ArrowUpRight size={15} /> : <ArrowDownRight size={15} />}
            {t.type === "undercut_threat" ? `Undercut threat from ${t.driver}` : `Undercut on ${t.driver}`}
            <span className={clsx("ml-auto text-xs", t.live ? (t.type === "undercut_threat" ? "text-bad" : "text-good") : "text-text-3")}>{t.live ? "LIVE" : "not on"}</span>
          </div>
          <div className="mt-1 text-xs text-text-2 tabular">
            gap {t.gap != null ? `${t.gap.toFixed(1)}s` : "—"} · fresh-tyre gain over 2 laps ≈ {t.gain.toFixed(1)}s
          </div>
        </div>
      ))}
    </div>
  );
}

function ModelPanel({ a }: { a: Analysis }) {
  if (!a.model) return null;
  const cd = a.model.car_deg;
  return (
    <div className="space-y-3">
      {a.tyre_curve && <LiveTyreCurve curve={a.tyre_curve} />}
      {a.drift && (
        <div className={clsx("rounded-lg p-3 text-sm flex items-start gap-2", a.drift.alert ? "bg-warn/10 border border-warn/40" : "bg-surface-2")}>
          {a.drift.alert ? <AlertTriangle size={16} className="text-warn mt-0.5 shrink-0" /> : <Gauge size={16} className="text-text-3 mt-0.5 shrink-0" />}
          <div>
            <div className="font-semibold">{a.drift.message}</div>
            <div className="text-xs text-text-3 tabular">drift z = {a.drift.z} (last 3 laps vs model, alert at |z| &gt; 2.5)</div>
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 text-xs">
        {cd && (
          <div className="rounded-lg bg-surface-2 p-3 col-span-2">
            <div className="label mb-1">{a.driver} personal tyre wear</div>
            <div className="tabular">
              <b>×{cd.mult.toFixed(2)}</b> ± {cd.sd.toFixed(2)} vs field (pre-race ×{cd.prior_mult.toFixed(2)}
              {cd.observed_mult != null ? `, own stints ×${cd.observed_mult.toFixed(2)} from ${cd.stints} stint${cd.stints > 1 ? "s" : ""} / ${cd.laps} laps` : ", no long green stint yet"})
            </div>
            {cd.compound && cd.observed != null && cd.field != null && (
              <div className="mt-0.5 text-text-3 tabular">
                current {cd.compound.toLowerCase()} stint: {cd.observed.toFixed(3)} s/lap observed vs field {cd.field.toFixed(3)} · used for plan {(cd.field * cd.mult).toFixed(3)}
              </div>
            )}
            <div className="mt-1 text-text-3">Each green stint moves the estimate ~15–30% toward the driver&apos;s own number (weight learned from 2026 races).</div>
          </div>
        )}
        {Object.entries(a.model.compounds).map(([c, p]) => {
          const ev = a.model!.live_evidence[c];
          return (
            <div key={c} className="rounded-lg bg-surface-2 p-3">
              <TyreBadge compound={c} size={16} />
              <div className="mt-1 tabular">
                {p.deg.toFixed(3)} ± {p.deg_sd.toFixed(3)} s/lap
              </div>
              <div className="text-text-3">{ev ? `prior ${ev.prior.toFixed(3)} · ${ev.stints} live stints` : "no race laps yet"}</div>
            </div>
          );
        })}
        <div className="rounded-lg bg-surface-2 p-3 col-span-2 tabular">
          pit loss {a.model.pit_loss.toFixed(1)}s ({a.model.pit_losses_observed} stops observed) · track-position cost {a.model.stop_penalty}s/stop
        </div>
      </div>
    </div>
  );
}

function AnalysisView({ a, state, onSelect }: { a: Analysis; state: RaceState | null; onSelect: (d: string) => void }) {
  if (a.error) return <div className="card p-5 text-bad">{a.error}</div>;
  const plans = a.plans ?? [];
  const L = a.total_laps;
  const rows: TimelineRow[] = plans.slice(0, 3).map((p, i) => ({
    key: p.name + i,
    label: i === 0 ? "Recommended" : `Option ${i + 1}`,
    sublabel: p.name,
    stints: p.stints,
    windows: i === 0 ? p.windows : undefined,
    muted: i > 0,
  }));
  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <div className="xl:col-span-2 space-y-6">
        <CallCard a={a} />
        {rows.length > 0 && (
          <Section title="Rest-of-race plan" subtitle="From the current lap; blue = pit window within 2 s of the optimum">
            <StintTimeline rows={rows} totalLaps={L} currentLap={a.lap} rowHeight={30} />
          </Section>
        )}
        {plans.length > 0 && (
          <Section title="All options from here" subtitle="Monte Carlo over future Safety Cars / VSCs and tyre-model uncertainty">
            <PlanTable plans={plans} chosen={plans[0]?.name} />
          </Section>
        )}
        <Section title="Tyre model, live" subtitle="Current stint vs the model (80% band). Bayesian-updated from every car's green-flag laps.">
          <ModelPanel a={a} />
        </Section>
      </div>
      <div className="space-y-6">
        <Section title="Race craft">
          <RaceCraft a={a} />
        </Section>
        {state && state.cars.length > 2 && (
          <Section title="Timing" subtitle="Click a driver to switch">
            <TimingTower state={state} selected={a.driver} onSelect={onSelect} />
          </Section>
        )}
        {a.rivals && a.rivals.length > 0 && (
          <Section title="Rivals: predicted next stop">
            <div className="space-y-1 text-sm tabular">
              {a.rivals.slice(0, 12).map((r) => (
                <div key={r.driver} className="flex items-center gap-2 py-0.5">
                  <span className="w-7 text-text-3">P{r.position}</span>
                  <span className="w-10 font-semibold">{r.driver}</span>
                  <TyreBadge compound={r.compound} age={r.tyre_age} size={16} />
                  <span className="ml-auto text-text-2">{r.next_stop ? `L${r.next_stop}` : "no stop"}</span>
                  {r.next_compound && <TyreBadge compound={r.next_compound} size={16} />}
                </div>
              ))}
            </div>
          </Section>
        )}
        {state && state.race_control.length > 0 && (
          <Section title="Race control">
            <ul className="space-y-2 text-xs">
              {state.race_control.slice(0, 8).map((m, i) => (
                <li key={i} className="flex gap-2">
                  <span className="tabular text-text-3 w-9 shrink-0">{m.lap ? `L${m.lap}` : ""}</span>
                  <span className="text-text-2">{m.message}</span>
                </li>
              ))}
            </ul>
          </Section>
        )}
      </div>
    </div>
  );
}

function ManualForm({ race, onResult }: { race: RaceFile; onResult: (a: Analysis) => void }) {
  const [f, setF] = useState({
    driver: race.drivers[0]?.driver ?? "RUS",
    laps_completed: 15,
    compound: "MEDIUM",
    tyre_age: 15,
    compounds_used: [] as string[],
    track_status: "GREEN",
    position: race.drivers[0]?.grid ?? 1,
    gap_ahead: "",
    gap_behind: "",
  });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      const team = race.drivers.find((d) => d.driver === f.driver)?.team;
      const res = await api<{ analysis: Analysis }>("/api/live/manual", {
        method: "POST",
        body: JSON.stringify({
          round: race.round,
          driver: f.driver,
          team,
          laps_completed: f.laps_completed,
          compound: f.compound,
          tyre_age: f.tyre_age,
          compounds_used: f.compounds_used,
          track_status: f.track_status,
          position: f.position,
          gap_ahead: f.gap_ahead === "" ? null : Number(f.gap_ahead),
          gap_behind: f.gap_behind === "" ? null : Number(f.gap_behind),
        }),
      });
      onResult(res.analysis);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const input = "mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 tabular";
  return (
    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6 items-end text-sm">
      <label>
        Driver
        <select className={input} value={f.driver} onChange={(e) => setF({ ...f, driver: e.target.value })}>
          {race.drivers.map((d) => (
            <option key={d.driver} value={d.driver}>
              {d.driver} · {d.team}
            </option>
          ))}
        </select>
      </label>
      <label>
        Laps completed
        <input type="number" className={input} min={0} max={race.weekend.total_laps} value={f.laps_completed} onChange={(e) => setF({ ...f, laps_completed: Number(e.target.value) })} />
      </label>
      <label>
        Current tyre
        <select className={input} value={f.compound} onChange={(e) => setF({ ...f, compound: e.target.value })}>
          {["SOFT", "MEDIUM", "HARD"].map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>
      <label>
        Tyre age (laps)
        <input type="number" className={input} min={0} value={f.tyre_age} onChange={(e) => setF({ ...f, tyre_age: Number(e.target.value) })} />
      </label>
      <label>
        Track status
        <select className={input} value={f.track_status} onChange={(e) => setF({ ...f, track_status: e.target.value })}>
          {["GREEN", "SC", "VSC", "YELLOW"].map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>
      <label>
        Position
        <input type="number" className={input} min={1} max={22} value={f.position} onChange={(e) => setF({ ...f, position: Number(e.target.value) })} />
      </label>
      <label>
        Gap to car ahead (s)
        <input className={input} placeholder="optional" value={f.gap_ahead} onChange={(e) => setF({ ...f, gap_ahead: e.target.value })} />
      </label>
      <label>
        Gap to car behind (s)
        <input className={input} placeholder="optional" value={f.gap_behind} onChange={(e) => setF({ ...f, gap_behind: e.target.value })} />
      </label>
      <div className="sm:col-span-2">
        <div>Compounds already used (earlier stints)</div>
        <div className="mt-1 flex gap-1.5">
          {["SOFT", "MEDIUM", "HARD"].map((c) => {
            const on = f.compounds_used.includes(c);
            return (
              <button
                key={c}
                onClick={() => setF({ ...f, compounds_used: on ? f.compounds_used.filter((x) => x !== c) : [...f.compounds_used, c] })}
                className={clsx("rounded-md border px-2 py-1", on ? "border-text-2 bg-surface-3" : "border-line")}
              >
                <TyreBadge compound={c} size={18} />
              </button>
            );
          })}
        </div>
      </div>
      <button onClick={submit} disabled={busy} className="rounded-lg bg-accent px-4 py-2 font-semibold hover:brightness-110 disabled:opacity-60">
        {busy ? "Computing…" : "Get the call"}
      </button>
      {err && <div className="sm:col-span-6 text-bad">{err}</div>}
    </div>
  );
}

export default function LivePage() {
  const { data: season } = useApi<Season>("/api/season");
  const defaultRound = season?.live_round ?? season?.next_round ?? null;
  const [round, setRound] = useState<number | null>(null);
  const rnd = round ?? defaultRound;
  const { data: race } = useApi<RaceFile>(rnd ? `/api/races/${rnd}` : null);
  const [sourceChoice, setSource] = useState<SourceKind | null>(null);
  // During a race weekend's race window default to the live feed, otherwise to a replay.
  const source: SourceKind = sourceChoice ?? (season?.live_round ? "signalr" : "replay");
  const [replayRound, setReplayRound] = useState<number>(15);
  const [speed, setSpeed] = useState(6);
  const [startLap, setStartLap] = useState(10);
  const [driverChoice, setDriver] = useState<string | null>(null);
  const [connectErr, setConnectErr] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [manual, setManual] = useState<Analysis | null>(null);
  const { data: status, refresh: refreshStatus } = usePoll<LiveStatus>("/api/live/status", 5000);

  const connected = !!status?.connected && source !== "manual";
  const [firstCar, setFirstCar] = useState<string | null>(null);
  const driver = driverChoice ?? firstCar;
  const live = usePoll<{ state: RaceState; analysis?: Analysis; status: LiveStatus }>(
    connected ? `/api/live/state${driver ? `?driver=${driver}` : ""}` : null,
    4000,
    connected,
  );
  const state = live.data?.state ?? null;
  const analysis = live.data?.analysis && live.data.analysis.driver === driver ? live.data.analysis : null;
  // First payload has no analysis yet; default to the race leader.
  if (!driver && state?.cars?.length && firstCar !== state.cars[0].driver) setFirstCar(state.cars[0].driver);

  const finishedRounds = useMemo(() => season?.rounds.filter((r) => r.status === "finished") ?? [], [season]);

  const connect = async () => {
    setConnecting(true);
    setConnectErr(null);
    try {
      const body =
        source === "replay"
          ? { source, round: replayRound, seconds_per_lap: speed, start_lap: startLap }
          : { source, round: rnd };
      await api("/api/live/connect", { method: "POST", body: JSON.stringify(body) });
      setDriver(null);
      setFirstCar(null);
      await refreshStatus();
      await live.refresh();
    } catch (e) {
      setConnectErr((e as Error).message);
    } finally {
      setConnecting(false);
    }
  };

  const hint = SOURCES.find((s) => s.id === source)?.hint;
  const sourceReady = source === "openf1" ? status?.openf1_credentials : true;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-text-3">
            <Radio size={15} className={connected ? "text-accent live-dot" : ""} /> Live pit wall
          </div>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{state?.event ?? race?.event ?? "Race strategy, live"}</h1>
          <p className="text-text-2 mt-1">
            {state ? `${state.location} · lap ${state.lap}${state.total_laps ? ` / ${state.total_laps}` : ""}` : "Pick a data source, connect, then choose the driver to strategise for."}
          </p>
        </div>
        {state && (
          <div className="flex items-center gap-3">
            <span className={clsx("rounded-md px-3 py-1.5 text-sm font-bold tracking-wide", STATUS_STYLE[state.track_status] ?? "bg-surface-3")}>
              <Flag size={14} className="inline mr-1.5 -mt-0.5" />
              {state.track_status.replace("_", " ")}
            </span>
            <span className="font-mono text-3xl tabular font-semibold">
              {state.lap}
              <span className="text-text-3 text-xl">/{state.total_laps ?? "?"}</span>
            </span>
          </div>
        )}
      </div>

      <Section
        title="Data source"
        right={
          <span className={clsx("flex items-center gap-1.5 text-sm", connected && !status?.error ? "text-good" : "text-text-3")}>
            {connected ? <Wifi size={15} /> : <WifiOff size={15} />}
            {connected
              ? `${status?.source} · R${status?.round}${status?.last_update_age != null ? ` · updated ${status.last_update_age}s ago` : ""}${status?.shared_feed && (status?.viewers ?? 0) > 1 ? ` · ${status.viewers} watching` : ""}`
              : "not connected"}
          </span>
        }
      >
        <div className="flex flex-wrap gap-2">
          {SOURCES.map((s) => (
            <button
              key={s.id}
              onClick={() => setSource(s.id)}
              className={clsx("rounded-lg border px-3 py-2 text-sm", source === s.id ? "border-text-2 bg-surface-3" : "border-line hover:bg-surface-2")}
            >
              {s.label}
            </button>
          ))}
        </div>
        <p className="mt-3 text-sm text-text-3">{hint}</p>
        {source !== "manual" && (
          <div className="mt-4 flex flex-wrap items-end gap-3 text-sm">
            {source === "replay" ? (
              <>
                <label>
                  Race
                  <select className="mt-1 block rounded-md border border-line bg-surface-2 px-2 py-1.5" value={replayRound} onChange={(e) => setReplayRound(Number(e.target.value))}>
                    {finishedRounds.map((r) => (
                      <option key={r.round} value={r.round}>
                        R{r.round} {r.event}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Start at lap
                  <input type="number" min={1} value={startLap} onChange={(e) => setStartLap(Number(e.target.value))} className="mt-1 block w-20 rounded-md border border-line bg-surface-2 px-2 py-1.5 tabular" />
                </label>
                <label>
                  Seconds per lap
                  <input type="number" min={1} max={120} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="mt-1 block w-20 rounded-md border border-line bg-surface-2 px-2 py-1.5 tabular" />
                </label>
              </>
            ) : (
              <label>
                Round
                <select className="mt-1 block rounded-md border border-line bg-surface-2 px-2 py-1.5" value={rnd ?? ""} onChange={(e) => setRound(Number(e.target.value))}>
                  {season?.rounds
                    .filter((r) => r.available)
                    .map((r) => (
                      <option key={r.round} value={r.round}>
                        R{r.round} {r.event}
                      </option>
                    ))}
                </select>
              </label>
            )}
            <button onClick={connect} disabled={connecting} className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 font-semibold hover:brightness-110 disabled:opacity-60">
              {connecting ? <RefreshCcw size={15} className="animate-spin" /> : <Plug size={15} />}
              {connected ? "Reconnect" : "Connect"}
            </button>
            {!sourceReady && (
              <span className="inline-flex items-center gap-1.5 text-warn">
                <ShieldAlert size={15} /> credentials not configured on the API server
              </span>
            )}
          </div>
        )}
        {(connectErr || (connected && (status?.error || live.error))) && (
          <div className="mt-3 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad flex gap-2">
            <CircleSlash size={16} className="shrink-0 mt-0.5" />
            <span>{connectErr ?? status?.error ?? live.error}</span>
          </div>
        )}
        {source === "manual" && race && (
          <div className="mt-4">
            <ManualForm race={race} onResult={setManual} />
          </div>
        )}
      </Section>

      {source === "manual" && manual && <AnalysisView a={manual} state={null} onSelect={() => undefined} />}

      {source !== "manual" && state && (
        <>
          <Section title="Strategise for">
            <DriverPicker
              drivers={state.cars.filter((c) => !c.retired).map((c) => ({ driver: c.driver, team: c.team, color: c.color, sub: `P${c.position}` }))}
              value={driver}
              onChange={setDriver}
            />
          </Section>
          {analysis ? <AnalysisView a={analysis} state={state} onSelect={setDriver} /> : <div className="text-text-3">Computing strategy…</div>}
        </>
      )}
    </div>
  );
}
