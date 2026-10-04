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
from functools import lru_cache

import numpy as np

from src.pitwall.optimizer import SHORT, Optimizer, RaceContext, TyreState
from src.pitwall.race_data import race_overview
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
        "weekend": {k: w[k] for k in ("total_laps", "pit_loss", "sc_rate", "vsc_rate", "prior_races", "transfer", "offset_transfer", "short_run_offsets", "prior", "practice")}
        | {"model": model.to_dict(), "raw_model": raw_model.to_dict(), "behaviour": behaviour,
           "stop_penalty": behaviour["stop_penalty"], "fuel_wear": behaviour["fuel_wear"], "first_stop_q": q_first_stop},
        "baseline": baseline,
        "practice_runs": w["practice_runs"],
        "drivers": drivers,
        "predictions": preds,
        "overview": overview,
        "hindsight_model": fit["model"] if fit else None,
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
    # Second stage: learn team behaviour on top of the physics predictions.
    from src.pitwall import behaviour_model

    behaviour_model.main()


def write_index() -> None:
    sched = get_schedule(YEAR)
    rounds = []
    agg_rows = []
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()
    main(args.round)
