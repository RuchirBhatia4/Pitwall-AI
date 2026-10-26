"""Walk-forward backtest + artefact build for every 2026 round.

    python -m src.pitwall.build_season            # all completed rounds + next race
    python -m src.pitwall.build_season --round 16 # just one round

For each round R it writes data/season/2026/rXX.json containing
  * the pre-race model and per-driver predicted strategies (built only from
    information available before R's race),
  * the actual strategies and result (if the race has run),
  * the hindsight optimum (model fitted to R's own race laps),
  * evaluation metrics.
data/season/2026/season.json indexes all rounds; backtest.json aggregates
the metrics and is logged to MLflow when available.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.pitwall import track_history
from src.pitwall.optimizer import SHORT, Optimizer, RaceContext, TyreState
from src.pitwall.race_data import load_session, neutralised_laps, normalize_laps, race_overview
from src.pitwall.race_sim import Plan, RaceSim, SimParams, candidate_plans, fit_pace_model, fit_pass_model, pass_pairs, race_pace
from src.pitwall.season import SEASON_DIR, get_schedule
from src.pitwall.tyre_model import TyreModel, apply_behaviour
from src.pitwall.weekend import YEAR, build_weekend, completed_rounds, race_fit, race_is_finished, event_sessions

OUT_DIR = SEASON_DIR / str(YEAR)
CONFORMAL_LEVEL = 0.8
# Revealed-preference search space (see calibrate_behaviour).
BEHAVIOUR_GRID = [
    {"soft_mult": sm, "fuel_wear": fw, "stop_penalty": sp, "hard_start": hs}
    for sm in (1.0, 1.6, 2.4, 3.5)
    for fw in (0.25, 1.0, 2.0)
    for sp in (0.0, 4.0, 8.0)
    for hs in (0.0, 3.0, 8.0)
]
DEFAULT_BEHAVIOUR = {"soft_mult": 1.6, "fuel_wear": 1.0, "stop_penalty": 4.0, "hard_start": 3.0}
RACE_SIMS = 160
# A circuit's overtaking difficulty from earlier seasons is shrunk toward the
# 2026 average by this many pseudo-races (the 2026 cars race differently).
VENUE_SHRINK_RACES = 1.0
# The race simulation ranks only strategies the behaviour model gives at least
# this probability (its stop count x starting tyre), like a pit wall
# simulating a shortlist. Fixed in advance; walk-forward it lifted the start
# tyre from 27% to 54% and cut the first-stop error from 9.8 to 7.9 laps.
PLAUSIBLE_CLASS_P = 0.10


def _ctx(w: dict, team: str | None = None, behaviour: dict | None = None, driver: str | None = None) -> RaceContext:
    b = behaviour or DEFAULT_BEHAVIOUR
    return RaceContext(
        total_laps=w["total_laps"], pit_loss=w["pit_loss"], sc_rate=w["sc_rate"],
        vsc_rate=w["vsc_rate"], team=team, driver=driver, stop_penalty=b["stop_penalty"], fuel_wear=b["fuel_wear"],
        start_penalty={"HARD": b.get("hard_start", 0.0)},
    )


@lru_cache(maxsize=4096)
def field_prediction(rnd: int, idx: int) -> dict | None:
    w = build_weekend(rnd)
    b = BEHAVIOUR_GRID[idx]
    model = apply_behaviour(TyreModel.from_dict(w["model"]), b)
    res = Optimizer(model, _ctx(w, behaviour=b)).optimise(top=1)
    return res[0].to_dict() if res else None


@lru_cache(maxsize=64)
def round_actuals(rnd: int) -> tuple[int, tuple[tuple, ...]] | None:
    """Actual plans of classified finishers that made 1-3 stops (no incident chaos)."""
    if not race_is_finished(rnd):
        return None
    ov = race_overview(YEAR, rnd)
    fit = race_fit(rnd)
    if ov.get("rain") or (fit or {}).get("wet"):
        return None
    plans = []
    for d in ov["drivers"]:
        if d["status"] in ("Finished", "Lapped") or d["status"].startswith("+"):
            st = ov["stints"].get(d["driver"], [])
            a = actual_plan(st) if st else None
            if a and 1 <= a["stops"] <= 3 and all(c in ("SOFT", "MEDIUM", "HARD") for c in a["sequence"]):
                plans.append((tuple(a["sequence"]), tuple(a["pit_laps"])))
    return (ov["total_laps"], tuple(plans)) if plans else None


def plan_loss(pred: dict | None, actuals: tuple, total_laps: int) -> float:
    if not pred:
        return 4.0
    losses = []
    for seq, pits in actuals:
        loss = 3.0
        loss -= len(pred["pit_laps"]) == len(pits)
        loss -= set(pred["sequence"]) == set(seq)
        loss -= pred["sequence"][0] == seq[0]
        if pred["pit_laps"] and pits:
            loss += 2.0 * abs(pred["pit_laps"][0] - pits[0]) / total_laps
        losses.append(loss)
    return float(np.mean(losses))


def calibrate_behaviour(rnd: int) -> dict:
    """Inverse optimisation: choose the behaviour parameters under which the
    optimiser best reproduces what teams actually did in rounds < rnd."""
    usable = [(r, a) for r in range(1, rnd) if (a := round_actuals(r))]
    if len(usable) < 3:
        return dict(DEFAULT_BEHAVIOUR) | {"calibrated_on": len(usable)}
    losses = [sum(plan_loss(field_prediction(r, i), a[1], a[0]) for r, a in usable) / len(usable) for i in range(len(BEHAVIOUR_GRID))]
    best = int(np.argmin(losses))
    return dict(BEHAVIOUR_GRID[best]) | {"calibrated_on": len(usable), "loss": float(losses[best]),
                                          "loss_uncalibrated": float(losses[BEHAVIOUR_GRID.index({"soft_mult": 1.0, "fuel_wear": 0.25, "stop_penalty": 0.0, "hard_start": 0.0})])}


def baseline_plan(rnd: int, total_laps: int) -> dict | None:
    """Naive baseline: the most common strategy in earlier dry races, with the
    median first-stop fraction of race distance."""
    names, fracs, seqs = Counter(), [], {}
    for r in range(1, rnd):
        a = round_actuals(r)
        if not a:
            continue
        for seq, pits in a[1]:
            names[seq] += 1
            fracs.append(pits[0] / a[0])
    if not names:
        return None
    seq = names.most_common(1)[0][0]
    first = int(round(np.median(fracs) * total_laps))
    pits = [first] + [int(round(first + (total_laps - first) * (i + 1) / len(seq))) for i in range(len(seq) - 2)]
    return {"sequence": list(seq), "pit_laps": pits, "stops": len(seq) - 1, "name": "-".join(SHORT.get(c, "?") for c in seq)}


def actual_plan(stints: list[dict]) -> dict:
    seq = [s["compound"] for s in stints]
    pits = [s["end_lap"] for s in stints[:-1]]
    return {"sequence": seq, "name": "-".join(SHORT.get(c, "?") for c in seq), "pit_laps": pits, "stops": len(pits)}


def modal_stops(overview: dict) -> int | None:
    counts = Counter()
    for d in overview["drivers"]:
        if d["status"] in ("Finished", "Lapped") or d["status"].startswith("+"):
            st = overview["stints"].get(d["driver"], [])
            if st:
                counts[len(st) - 1] += 1
    return counts.most_common(1)[0][0] if counts else None


def conformal_q(residuals: list[float], level: float = CONFORMAL_LEVEL) -> float | None:
    """Split-conformal quantile with the finite-sample correction."""
    n = len(residuals)
    if n < 10:
        return None
    k = math.ceil((n + 1) * level)
    return float(np.sort(np.abs(residuals))[min(k, n) - 1])


@lru_cache(maxsize=64)
def race_pairs(rnd: int):
    """Close-following laps of a finished dry race (see race_sim.pass_pairs)."""
    fit = race_fit(rnd)
    if not fit or fit.get("wet") or not fit.get("model"):
        return None
    from src.pitwall.live_sources import replay_path

    path = replay_path(rnd)
    if path.exists():
        saved = pd.read_pickle(path)
        laps, neutral = saved["laps"], saved["neutral_laps"]
    else:
        laps = normalize_laps(load_session(YEAR, rnd, "Race"))
        nl = neutralised_laps(laps)
        neutral = set(nl["sc"]) | set(nl["vsc"]) | set(nl["red"])
    return pass_pairs(laps, neutral, fit["model"])


def sim_params(rnd: int, drivers: list[dict], location: str) -> SimParams | None:
    """Race-simulation inputs for round rnd, from earlier rounds only: race pace
    from qualifying position + form, overtaking from 2026 races + this circuit's history."""
    history = []
    for r in range(1, rnd):
        fit = race_fit(r) if race_is_finished(r) else None
        if fit and fit.get("model") and not fit.get("wet"):
            history.append((r, race_pace(fit["model"]), {d["driver"]: d.get("quali") or d.get("grid") for d in build_weekend(r)["drivers"]}))
    pace_model = fit_pace_model(history, rnd)
    passes = fit_pass_model({r: p for r in range(1, rnd) if race_is_finished(r) and (p := race_pairs(r)) is not None and len(p)})
    # Pace from qualifying (how fast the car is); the grid, with penalties,
    # only sets where it starts.
    quali = {d["driver"]: d.get("quali") or d["grid"] for d in drivers if d.get("grid")}
    if pace_model is None or passes is None or not quali:
        return None
    pace, pace_sd = pace_model(quali)
    a, b, _ = passes
    venue = track_history.venue_history(location)
    offset = 0.0
    if venue["pass_offset"] is not None:
        offset = venue["pass_offset"] * venue["pass_races"] / (venue["pass_races"] + VENUE_SHRINK_RACES)
    return SimParams(pace=pace, pace_sd=pace_sd, pass_a=a + offset, pass_b=b,
                     info={"pass_a_2026": a, "pass_b": b, "venue_offset": offset, "venue_races": venue["pass_races"], "pace_coef": pace_model.coef,
                           "p_pass_equal_pace": float(1 / (1 + np.exp(-(a + offset)))), "p_pass_1s_faster": float(1 / (1 + np.exp(-(a + offset + b))))})


def _sim_option(o: dict) -> dict:
    p = o["plan"]
    return {"sequence": p.sequence, "pit_laps": p.pit_laps, "stops": len(p.pit_laps), "name": "-".join(SHORT.get(c, c[0]) for c in p.sequence)} | {
        k: o[k] for k in ("expected_position", "position_sd", "p_points", "expected_points", "p_best")}


def race_simulation(w: dict, drivers: list[dict], model: TyreModel, preds: dict, params: SimParams) -> dict:
    """Best plan for every driver by expected finishing position, with the rest
    of the field on its predicted plans, then the finishing-order forecast with
    everyone on those best plans. A driver's candidates are cut to the
    plausible strategies (PLAUSIBLE_CLASS_P) once the behaviour model has run.

    `model` keeps the learned soft-tyre wear correction (thermal degradation the
    linear wear curve misses; without it the simulation picks far fewer of the
    compounds teams actually used), but the per-stop track-position cost and the
    start-tyre penalty are left out: track position is simulated here."""
    cars = [d for d in drivers if d.get("grid")]
    ctx = RaceContext(total_laps=w["total_laps"], pit_loss=w["pit_loss"], sc_rate=w["sc_rate"], vsc_rate=w["vsc_rate"])
    sim = RaceSim(model, ctx, cars, params, sims=RACE_SIMS)
    field = {d["driver"]: Plan.of(preds[d["driver"]]["predicted"]) for d in cars if preds.get(d["driver"], {}).get("predicted")}
    base = sim.field_arrays(field)
    chosen = dict(field)
    for d in cars:
        if d["driver"] not in field:
            continue
        opt = Optimizer(model, replace(ctx, team=d.get("team"), driver=d["driver"]))
        cands = candidate_plans(opt)
        likely = preds[d["driver"]].get("likely")
        if likely:
            p_stops = {int(k): v for k, v in likely["stop_probabilities"].items()}
            shortlist = [c for c in cands if p_stops.get(len(c.pit_laps), 0.0) * likely["start_probabilities"].get(c.sequence[0], 0.0) >= PLAUSIBLE_CLASS_P]
            cands = shortlist or cands
        ranked = sim.best_response(d["driver"], cands, field, base)
        preds[d["driver"]]["racesim"] = {"best": _sim_option(ranked[0]), "options": [_sim_option(o) for o in ranked[:6]]}
        chosen[d["driver"]] = ranked[0]["plan"]
    return {"sims": RACE_SIMS, "params": params.info | {"pace_sd": params.pace_sd}, "drivers": sim.forecast(chosen),
            "pace": params.pace}


def simulate_round(data: dict) -> None:
    """Last build stage for one round file: race simulation and its metrics."""
    params = sim_params(data["round"], data["drivers"], data["location"])
    if not params:
        return
    model = TyreModel.from_dict(data["weekend"]["model"])
    data["race_forecast"] = _clean(race_simulation(data["weekend"], data["drivers"], model, data["predictions"], params))
    for e in data["predictions"].values():
        if "racesim" in e:
            e["racesim"] = _clean(e["racesim"])
    if data.get("metrics") and data.get("overview"):
        data["metrics"].update(sim_metrics(data))


def sim_metrics(data: dict) -> dict:
    """Race-simulation picks vs what teams did, and the finishing-order forecast
    vs the grid, on the round's evaluated (classified, dry-plan) drivers."""
    rows = data["metrics"]["rows"]
    for r in rows:
        sim = data["predictions"].get(r["driver"], {}).get("racesim", {}).get("best")
        a = data["predictions"].get(r["driver"], {}).get("actual")
        if not sim or not a:
            continue
        r["sim_stops_match"] = sim["stops"] == a["stops"]
        r["sim_compound_set_match"] = set(sim["sequence"]) == set(a["sequence"])
        r["sim_start_compound_match"] = sim["sequence"][0] == a["sequence"][0]
        r["sim_uses_hard"] = "HARD" in sim["sequence"]
        if sim["pit_laps"] and a["pit_laps"]:
            r["sim_first_stop_error"] = a["pit_laps"][0] - sim["pit_laps"][0]
    sim_rows = [r for r in rows if "sim_stops_match" in r]
    out = {}
    if sim_rows:
        errs = [abs(r["sim_first_stop_error"]) for r in sim_rows if "sim_first_stop_error" in r]
        out = {"sim_stops_accuracy": float(np.mean([r["sim_stops_match"] for r in sim_rows])),
               "sim_compound_set_accuracy": float(np.mean([r["sim_compound_set_match"] for r in sim_rows])),
               "sim_start_compound_accuracy": float(np.mean([r["sim_start_compound_match"] for r in sim_rows])),
               "sim_first_stop_mae": float(np.mean(errs)) if errs else None}
    forecast = data["race_forecast"]["drivers"]
    pos = {d["driver"]: d["position"] for d in data["overview"]["drivers"]
           if d.get("position") and (d["status"] in ("Finished", "Lapped") or d["status"].startswith("+"))}
    grid = {d["driver"]: d["grid"] for d in data["drivers"] if d.get("grid")}
    common = [k for k in pos if k in grid and k in forecast]
    if len(common) >= 8:
        actual = [pos[k] for k in common]
        out["forecast_order_rho"] = float(spearmanr([forecast[k]["expected_position"] for k in common], actual)[0])
        out["grid_order_rho"] = float(spearmanr([grid[k] for k in common], actual)[0])
    return out


def build_round(rnd: int, behaviour: dict, q_first_stop: float | None) -> dict:
    w = build_weekend(rnd)
    raw_model = TyreModel.from_dict(w["model"])
    model = apply_behaviour(raw_model, behaviour)
    finished = race_is_finished(rnd)
    overview = race_overview(YEAR, rnd) if finished else None
    fit = race_fit(rnd) if finished else None
    # Hindsight uses the race's own fitted (physics-only) tyre model. Behaviour
    # corrections exist to mimic team *decisions*; for scoring what was actually
    # fastest we keep only the track-position cost per stop.
    hindsight_model = TyreModel.from_dict(fit["model"]) if fit and fit.get("model") else None
    hindsight_b = DEFAULT_BEHAVIOUR | {"soft_mult": 1.0, "fuel_wear": 0.25, "hard_start": 0.0, "stop_penalty": behaviour["stop_penalty"]}

    drivers = w["drivers"]
    if overview:
        meta = {d["driver"]: d for d in overview["drivers"]}
        drivers = [{**d, **{k: meta.get(d["driver"], {}).get(k) for k in ("position", "status", "points")}} for d in drivers]
        known = {d["driver"] for d in drivers}
        drivers += [d for d in overview["drivers"] if d["driver"] not in known]

    preds = {}
    for d in drivers:
        opt = Optimizer(model, _ctx(w, d.get("team"), behaviour, d["driver"]))
        res = opt.candidates(TyreState(), top=6)
        opt.monte_carlo(res, sims=400)
        # Risk-neutral choice: lowest expected time once SC/VSC and parameter
        # uncertainty are simulated (not just the deterministic optimum).
        res.sort(key=lambda s: s.total + s.risk.get("expected_delta", 0.0) - (res[0].total if res else 0))
        best = res[0] if res else None
        entry = {"alternatives": [r.to_dict() for r in res], "stop_probabilities": getattr(opt, "stop_probabilities", {}),
                 "wear_profile": model.driver_mult.get(d["driver"])}
        if best:
            entry["predicted"] = best.to_dict()
            if q_first_stop is not None and best.pit_laps:
                p = best.pit_laps[0]
                entry["first_stop_interval"] = [max(1, int(p - q_first_stop)), min(w["total_laps"] - 1, int(p + q_first_stop))]
        if overview and d["driver"] in overview["stints"]:
            act = actual_plan(overview["stints"][d["driver"]])
            entry["actual"] = act
            if hindsight_model and all(c in hindsight_model.compounds for c in act["sequence"]):
                h_opt = Optimizer(hindsight_model, _ctx(w, d.get("team"), hindsight_b))
                h_res = h_opt.optimise(top=1)
                if h_res:
                    entry["hindsight"] = h_res[0].to_dict()
                    act_cost = h_opt.evaluate(act["sequence"], act["pit_laps"], TyreState())
                    if act_cost < 1e8:
                        entry["actual_vs_hindsight"] = float(act_cost - h_res[0].total)
        preds[d["driver"]] = entry

    baseline = baseline_plan(rnd, w["total_laps"])
    metrics = evaluate(preds, overview, fit, baseline) if overview else None
    return {
        "round": rnd,
        "event": w["event"],
        "location": w["location"],
        "country": w["country"],
        "date": w["date"],
        "format": w["format"],
        "status": "finished" if finished else "upcoming",
        "race_start_utc": str(event_sessions(YEAR, rnd).get("Race")),
        "weekend": {k: w[k] for k in ("total_laps", "pit_loss", "sc_rate", "vsc_rate", "season_sc_rate", "season_vsc_rate", "venue_history", "prior_races", "transfer", "offset_transfer", "short_run_offsets", "prior", "practice")}
        | {"model": model.to_dict(), "raw_model": raw_model.to_dict(), "behaviour": behaviour,
           "stop_penalty": behaviour["stop_penalty"], "fuel_wear": behaviour["fuel_wear"], "first_stop_q": q_first_stop},
        "baseline": baseline,
        "practice_runs": w["practice_runs"],
        "drivers": drivers,
        "predictions": preds,
        "overview": overview,
        "hindsight_model": fit["model"] if fit else None,
        "race_forecast": None,  # filled by simulate_round after the behaviour model
        "metrics": metrics,
    }


def evaluate(preds: dict, overview: dict, fit: dict | None, baseline: dict | None = None) -> dict:
    wet = bool(overview.get("rain")) or (fit or {}).get("wet", False)
    finishers = {d["driver"] for d in overview["drivers"] if d["status"] in ("Finished", "Lapped") or d["status"].startswith("+")}
    rows = []
    for drv, e in preds.items():
        if drv not in finishers or "predicted" not in e or "actual" not in e:
            continue
        p, a = e["predicted"], e["actual"]
        dry = all(c in ("SOFT", "MEDIUM", "HARD") for c in a["sequence"])
        row = {
            "uses_hard_pred": "HARD" in p["sequence"],
            "uses_hard_actual": "HARD" in a["sequence"],
            "driver": drv,
            "stops_match": p["stops"] == a["stops"],
            "compound_set_match": set(p["sequence"]) == set(a["sequence"]),
            "start_compound_match": p["sequence"][0] == a["sequence"][0],
            "dry": dry,
        }
        if p["pit_laps"] and a["pit_laps"]:
            row["first_stop_error"] = a["pit_laps"][0] - p["pit_laps"][0]
            lo, hi = p["windows"][0]
            row["first_stop_in_window"] = lo <= a["pit_laps"][0] <= hi
            if "first_stop_interval" in e:
                lo, hi = e["first_stop_interval"]
                row["first_stop_in_conformal"] = lo <= a["pit_laps"][0] <= hi
        if "actual_vs_hindsight" in e:
            row["actual_vs_hindsight"] = e["actual_vs_hindsight"]
        if baseline:
            row["baseline_stops_match"] = baseline["stops"] == a["stops"]
            row["baseline_compound_set_match"] = set(baseline["sequence"]) == set(a["sequence"])
            row["baseline_start_compound_match"] = baseline["sequence"][0] == a["sequence"][0]
            if baseline["pit_laps"] and a["pit_laps"]:
                row["baseline_first_stop_error"] = a["pit_laps"][0] - baseline["pit_laps"][0]
        rows.append(row)

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else None

    errs = [r["first_stop_error"] for r in rows if "first_stop_error" in r]
    return {
        "wet": wet,
        "drivers_evaluated": len(rows),
        "modal_stops": modal_stops(overview),
        "stops_accuracy": rate("stops_match"),
        "compound_set_accuracy": rate("compound_set_match"),
        "start_compound_accuracy": rate("start_compound_match"),
        "first_stop_mae": float(np.mean(np.abs(errs))) if errs else None,
        "first_stop_bias": float(np.mean(errs)) if errs else None,
        "window_coverage": rate("first_stop_in_window"),
        "conformal_coverage": rate("first_stop_in_conformal"),
        "median_time_lost_vs_hindsight": float(np.median([r["actual_vs_hindsight"] for r in rows if "actual_vs_hindsight" in r])) if any("actual_vs_hindsight" in r for r in rows) else None,
        "first_stop_errors": errs,
        "baseline_stops_accuracy": rate("baseline_stops_match"),
        "baseline_compound_set_accuracy": rate("baseline_compound_set_match"),
        "baseline_first_stop_mae": float(np.mean([abs(r["baseline_first_stop_error"]) for r in rows if "baseline_first_stop_error" in r])) if any("baseline_first_stop_error" in r for r in rows) else None,
        "rows": rows,
    }


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(only: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds = completed_rounds()
    sched = get_schedule(YEAR)
    upcoming = [int(r) for r in sched["RoundNumber"] if int(r) not in rounds]
    nxt = upcoming[0] if upcoming else None
    targets = rounds + ([nxt] if nxt else [])
    if only:
        targets = [r for r in targets if r <= only]

    export_replays(rounds)

    history: dict[int, dict] = {}
    residuals: list[float] = []
    for rnd in targets:
        behaviour = calibrate_behaviour(rnd)
        q = conformal_q(residuals)
        if only and rnd != only:
            # still need history for walk-forward calibration
            prev = OUT_DIR / f"r{rnd:02d}.json"
            if prev.exists():
                data = json.loads(prev.read_text())
                if data.get("metrics"):
                    history[rnd] = data["metrics"]
                    if not data["metrics"]["wet"]:
                        residuals += data["metrics"]["first_stop_errors"]
                continue
        data = _clean(build_round(rnd, behaviour, q))
        (OUT_DIR / f"r{rnd:02d}.json").write_text(json.dumps(data))
        m = data.get("metrics")
        if m:
            history[rnd] = m
            if not m["wet"]:
                residuals += m["first_stop_errors"]
            print(f"R{rnd:02d} {data['location']:<14} b={behaviour} q={q} stops={m['stops_accuracy']} set={m['compound_set_accuracy']} "
                  f"mae={m['first_stop_mae']} conf={m['conformal_coverage']} | base stops={m['baseline_stops_accuracy']} set={m['baseline_compound_set_accuracy']} mae={m['baseline_first_stop_mae']}", flush=True)
        else:
            print(f"R{rnd:02d} {data['location']:<14} b={behaviour} q={q} (pre-race build)", flush=True)
    write_index()
    # Second stage: learn team behaviour on top of the physics predictions;
    # third: race simulation over each driver's plausible strategies.
    from src.pitwall import behaviour_model

    behaviour_model.main(after_round=simulate_round)


def write_index() -> None:
    sched = get_schedule(YEAR)
    rounds = []
    agg_rows = []
    order_rho: list[tuple[float, float]] = []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        path = OUT_DIR / f"r{rnd:02d}.json"
        entry = {"round": rnd, "event": str(ev["EventName"]), "location": str(ev["Location"]), "country": str(ev["Country"]),
                 "date": str(ev["EventDate"].date()), "format": str(ev["EventFormat"]), "available": path.exists(),
                 "race_start_utc": str(ev["Session5DateUtc"])}
        if path.exists():
            data = json.loads(path.read_text())
            entry["status"] = data["status"]
            m = data.get("metrics")
            if data.get("overview"):
                win = next((d for d in data["overview"]["drivers"] if d.get("position") == 1), None)
                entry["winner"] = win and {k: win[k] for k in ("driver", "name", "team", "color")}
                entry["total_laps"] = data["overview"]["total_laps"]
                entry["neutralised"] = data["overview"]["neutralised"]
            if m:
                entry["metrics"] = {k: v for k, v in m.items() if k not in ("rows", "first_stop_errors")}
                if not m["wet"] and data["round"] > 3:  # rounds 1-3 have almost no prior: report separately
                    agg_rows += m["rows"]
                    if m.get("forecast_order_rho") is not None:
                        order_rho.append((m["forecast_order_rho"], m["grid_order_rho"]))
        else:
            entry["status"] = "finished" if race_is_finished(rnd) else "upcoming"
        rounds.append(entry)
    (OUT_DIR / "season.json").write_text(json.dumps(_clean({"year": YEAR, "rounds": rounds})))

    def rate(key):
        vals = [r[key] for r in agg_rows if key in r]
        return float(np.mean(vals)) if vals else None

    errs = [r["first_stop_error"] for r in agg_rows if "first_stop_error" in r]
    backtest = {
        "scope": "dry 2026 races from round 4 onward, classified finishers, walk-forward (no future data)",
        "driver_races": len(agg_rows),
        "stops_accuracy": rate("stops_match"),
        "compound_set_accuracy": rate("compound_set_match"),
        "start_compound_accuracy": rate("start_compound_match"),
        "first_stop_mae": float(np.mean(np.abs(errs))) if errs else None,
        "window_coverage": rate("first_stop_in_window"),
        "conformal_coverage": rate("first_stop_in_conformal"),
        "conformal_target": CONFORMAL_LEVEL,
        "baseline": {
            "description": "most common strategy of earlier dry 2026 races, median first-stop fraction",
            "stops_accuracy": rate("baseline_stops_match"),
            "compound_set_accuracy": rate("baseline_compound_set_match"),
            "start_compound_accuracy": rate("baseline_start_compound_match"),
            "first_stop_mae": float(np.mean([abs(r["baseline_first_stop_error"]) for r in agg_rows if "baseline_first_stop_error" in r])) if agg_rows else None,
        },
    }
    sim_rows = [r for r in agg_rows if "sim_stops_match" in r]
    if sim_rows:
        sim_errs = [abs(r["sim_first_stop_error"]) for r in sim_rows if "sim_first_stop_error" in r]
        phys_errs = [abs(r["first_stop_error"]) for r in sim_rows if "first_stop_error" in r]
        backtest["racesim"] = {
            "description": "full-field race simulation (traffic, overtaking, SC/VSC); each driver's plan chosen by expected finishing position",
            "driver_races": len(sim_rows),
            "stops_accuracy": float(np.mean([r["sim_stops_match"] for r in sim_rows])),
            "compound_set_accuracy": float(np.mean([r["sim_compound_set_match"] for r in sim_rows])),
            "start_compound_accuracy": float(np.mean([r["sim_start_compound_match"] for r in sim_rows])),
            "first_stop_mae": float(np.mean(sim_errs)) if sim_errs else None,
            "uses_hard": float(np.mean([r["sim_uses_hard"] for r in sim_rows])),
            "physics_on_same_rows": {
                "stops_accuracy": float(np.mean([r["stops_match"] for r in sim_rows])),
                "compound_set_accuracy": float(np.mean([r["compound_set_match"] for r in sim_rows])),
                "start_compound_accuracy": float(np.mean([r["start_compound_match"] for r in sim_rows])),
                "first_stop_mae": float(np.mean(phys_errs)) if phys_errs else None,
                "uses_hard": float(np.mean([r["uses_hard_pred"] for r in sim_rows])),
            },
            "actual_uses_hard": float(np.mean([r["uses_hard_actual"] for r in sim_rows])),
        }
        if order_rho:
            backtest["racesim"]["finishing_order_rho"] = float(np.mean([a for a, _ in order_rho]))
            backtest["racesim"]["grid_order_rho"] = float(np.mean([b for _, b in order_rho]))
    (OUT_DIR / "backtest.json").write_text(json.dumps(_clean(backtest), indent=1))
    print(json.dumps(_clean(backtest), indent=1))
    try:  # experiment tracking (kept from the original project)
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{SEASON_DIR.parent.parent / 'mlflow.db'}")
        mlflow.set_experiment("pitwall_strategy_backtest")
        with mlflow.start_run(run_name="walk_forward_2026"):
            mlflow.log_metrics({k: v for k, v in backtest.items() if isinstance(v, float)})
            mlflow.log_metrics({f"baseline_{k}": v for k, v in backtest["baseline"].items() if isinstance(v, float)})
            mlflow.log_dict(backtest, "backtest.json")
    except Exception as exc:  # MLflow is optional at runtime
        print(f"(mlflow logging skipped: {exc})")


def export_replays(rounds: list[int]) -> None:
    """Save finished races for the Replay source, so the server never needs F1's archive."""
    from src.pitwall.live_sources import ReplaySource, replay_path

    for rnd in rounds:
        if replay_path(rnd).exists():
            continue
        try:
            ReplaySource.export(rnd)
            print(f"replay saved for round {rnd}")
        except Exception as exc:
            print(f"(replay for round {rnd} skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()
    main(args.round)
