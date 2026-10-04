"""Hybrid physics + ML model of what teams will actually do.

The physics optimiser answers "what is fastest?". Teams also weigh track
position, risk and convention, so this layer learns *behaviour* from past
races, using physics outputs as features:

    features  = [optimiser's 1-stop vs 2-stop gap, optimal first-stop fraction,
                 weekend wear/pace per compound, pit loss, laps, SC rate,
                 grid slot, sprint weekend]
    targets   = stop count (1/2/3), starting compound, first-stop fraction

Small, regularised models (multinomial logistic regression, ridge) trained
walk-forward: round R is predicted from rounds < R only. The final "likely"
plan is the physics-optimal plan *within* the predicted stop count / starting
tyre, with the first stop moved to the predicted lap.

    python -m src.pitwall.behaviour_model      # after build_season
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pitwall.optimizer import INF, SHORT, Optimizer, RaceContext, Strategy, TyreState
from src.pitwall.season import SEASON_DIR
from src.pitwall.tyre_model import TyreModel

OUT_DIR = SEASON_DIR / "2026"
MIN_TRAIN_ROUNDS = 3
DRY = ("SOFT", "MEDIUM", "HARD")
FEATURES = ["gap_1v2", "gap_2v3", "opt_first_frac", "deg_S", "deg_M", "deg_H", "off_S", "off_H",
            "pit_loss", "laps", "sc_rate", "grid_frac", "sprint"]


def _ctx(w: dict, team: str | None, driver: str | None = None) -> RaceContext:
    return RaceContext(total_laps=w["total_laps"], pit_loss=w["pit_loss"], sc_rate=w["sc_rate"], vsc_rate=w["vsc_rate"],
                       team=team, driver=driver, stop_penalty=w.get("stop_penalty", 0.0), fuel_wear=w.get("fuel_wear", 1.0))


def best_by_class(opt: Optimizer) -> dict[tuple[int, str], Strategy]:
    """Best plan for every (stops, starting compound) class."""
    out: dict[tuple[int, str], Strategy] = {}
    for s in opt.optimise(TyreState(), max_stops=3, top=500):
        key = (len(s.pit_laps), s.sequence[0])
        if key not in out:
            out[key] = s
    return out


def driver_features(data: dict, driver: dict, classes: dict) -> dict:
    w = data["weekend"]
    m = w["model"]["compounds"]
    L = w["total_laps"]
    by_stops = {}
    for (k, _), s in classes.items():
        by_stops[k] = min(by_stops.get(k, INF), s.total)
    best = min(classes.values(), key=lambda s: s.total)
    grid = driver.get("grid") or 11
    return {
        "gap_1v2": float(np.clip(by_stops.get(2, INF) - by_stops.get(1, INF), -60, 60)),
        "gap_2v3": float(np.clip(by_stops.get(3, INF) - by_stops.get(2, INF), -60, 60)),
        "opt_first_frac": best.pit_laps[0] / L if best.pit_laps else 0.5,
        "deg_S": m.get("SOFT", {}).get("deg", 0.1), "deg_M": m.get("MEDIUM", {}).get("deg", 0.07), "deg_H": m.get("HARD", {}).get("deg", 0.05),
        "off_S": m.get("SOFT", {}).get("offset", -0.3), "off_H": m.get("HARD", {}).get("offset", 0.3),
        "pit_loss": w["pit_loss"], "laps": L, "sc_rate": w["sc_rate"], "grid_frac": grid / 22.0,
        "sprint": 1.0 if "sprint" in data.get("format", "") else 0.0,
    }


def _finished(d: dict) -> bool:
    return d["status"] in ("Finished", "Lapped") or str(d["status"]).startswith("+")


def build_rows(data: dict) -> list[dict]:
    """Feature rows (+ targets when the race has run) for every driver of a round."""
    w = data["weekend"]
    rows = []
    for d in data["drivers"]:
        pred = data["predictions"].get(d["driver"])
        if not pred or not pred.get("predicted"):
            continue
        opt = Optimizer(TyreModel.from_dict(w["model"]), _ctx(w, d.get("team"), d["driver"]))
        classes = best_by_class(opt)
        row = {"round": data["round"], "driver": d["driver"], "team": d.get("team"), "x": driver_features(data, d, classes), "classes": classes, "opt": opt}
        act = pred.get("actual")
        if act and data.get("metrics") and not data["metrics"]["wet"] and _finished(d) and 1 <= act["stops"] <= 3 and all(c in DRY for c in act["sequence"]):
            row["y_stops"] = act["stops"]
            row["y_start"] = act["sequence"][0]
            row["y_first_frac"] = act["pit_laps"][0] / w["total_laps"]
        rows.append(row)
    return rows


def _fit_predict(train: list[dict], test: list[dict]) -> list[dict]:
    X = np.array([[r["x"][f] for f in FEATURES] for r in train])
    Xt = np.array([[r["x"][f] for f in FEATURES] for r in test])
    out = [{} for _ in test]
    for target in ("y_stops", "y_start"):
        y = np.array([r[target] for r in train])
        classes = sorted(set(y.tolist()))
        if len(classes) == 1:
            for o in out:
                o[target] = {classes[0]: 1.0}
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(X, y)
        proba = clf.predict_proba(Xt)
        for o, p in zip(out, proba):
            o[target] = {c: float(pv) for c, pv in zip(clf.classes_, p)}
    reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    reg.fit(X, np.array([r["y_first_frac"] for r in train]))
    for o, v in zip(out, reg.predict(Xt)):
        o["first_frac"] = float(np.clip(v, 0.08, 0.85))
    return out


def likely_plan(row: dict, pred: dict, total_laps: int) -> Strategy | None:
    """Most probable (stops, start) class that the physics model allows, timed by the predicted first stop."""
    opt: Optimizer = row["opt"]
    joint = sorted(
        ((ps * pc, k, c) for k, ps in pred["y_stops"].items() for c, pc in pred["y_start"].items()),
        reverse=True,
    )
    for prob, k, c in joint:
        base = row["classes"].get((int(k), c))
        if base is None:
            continue
        target = int(round(pred["first_frac"] * total_laps))
        s = opt.plan_with_first_stop(base.sequence, TyreState(), target) if base.pit_laps else base
        s = s or base
        s.delta = s.total - min(x.total for x in row["classes"].values())
        s.risk = {"class_probability": float(prob)}
        return s
    return None


def main() -> None:
    files = sorted(OUT_DIR.glob("r[0-9][0-9].json"))
    rounds = {int(f.stem[1:]): json.loads(f.read_text()) for f in files}
    rows = {r: build_rows(d) for r, d in rounds.items()}
    for rnd in sorted(rounds):
        data = rounds[rnd]
        train = [x for r in sorted(rows) if r < rnd for x in rows[r] if "y_stops" in x]
        n_rounds = len({x["round"] for x in train})
        if n_rounds < MIN_TRAIN_ROUNDS:
            continue
        test = rows[rnd]
        preds = _fit_predict(train, test)
        L = data["weekend"]["total_laps"]
        for row, p in zip(test, preds):
            plan = likely_plan(row, p, L)
            if plan is None:
                continue
            entry = data["predictions"][row["driver"]]
            pdict = plan.to_dict()
            entry["likely"] = pdict | {
                "stop_probabilities": {str(k): v for k, v in p["y_stops"].items()},
                "start_probabilities": p["y_start"],
                "trained_on_rounds": n_rounds,
            }
        if data.get("metrics"):
            data["metrics"].update(_evaluate_likely(data))
        (OUT_DIR / f"r{rnd:02d}.json").write_text(json.dumps(data))
        m = data.get("metrics") or {}
        print(f"R{rnd:02d} trained on {n_rounds} rounds | likely stops={m.get('likely_stops_accuracy')} set={m.get('likely_compound_set_accuracy')} "
              f"start={m.get('likely_start_compound_accuracy')} mae={m.get('likely_first_stop_mae')}", flush=True)
    from src.pitwall.build_season import write_index

    write_index()  # refresh season.json with the new per-round metrics (rewrites backtest.json)
    _update_backtest(rounds)


def _evaluate_likely(data: dict) -> dict:
    rows = []
    for r in data["metrics"]["rows"]:
        e = data["predictions"].get(r["driver"], {})
        if "likely" not in e or "actual" not in e:
            continue
        p, a = e["likely"], e["actual"]
        row = {"stops": p["stops"] == a["stops"], "set": set(p["sequence"]) == set(a["sequence"]), "start": p["sequence"][0] == a["sequence"][0]}
        if p["pit_laps"] and a["pit_laps"]:
            row["err"] = a["pit_laps"][0] - p["pit_laps"][0]
        rows.append(row)
        r["likely_stops_match"], r["likely_set_match"], r["likely_start_match"] = row["stops"], row["set"], row["start"]
        if "err" in row:
            r["likely_first_stop_error"] = row["err"]
    if not rows:
        return {}
    errs = [x["err"] for x in rows if "err" in x]
    return {
        "likely_stops_accuracy": float(np.mean([x["stops"] for x in rows])),
        "likely_compound_set_accuracy": float(np.mean([x["set"] for x in rows])),
        "likely_start_compound_accuracy": float(np.mean([x["start"] for x in rows])),
        "likely_first_stop_mae": float(np.mean(np.abs(errs))) if errs else None,
    }


def _update_backtest(rounds: dict[int, dict]) -> None:
    path = OUT_DIR / "backtest.json"
    bt = json.loads(path.read_text())
    agg = [r for rnd, d in rounds.items() if rnd > 3 and d.get("metrics") and not d["metrics"]["wet"] for r in d["metrics"]["rows"] if "likely_stops_match" in r]
    if agg:
        errs = [abs(r["likely_first_stop_error"]) for r in agg if "likely_first_stop_error" in r]
        bt["hybrid"] = {
            "description": "physics optimiser + walk-forward behaviour classifier (logistic regression on physics features)",
            "driver_races": len(agg),
            "stops_accuracy": float(np.mean([r["likely_stops_match"] for r in agg])),
            "compound_set_accuracy": float(np.mean([r["likely_set_match"] for r in agg])),
            "start_compound_accuracy": float(np.mean([r["likely_start_match"] for r in agg])),
            "first_stop_mae": float(np.mean(errs)) if errs else None,
        }
        # Same driver-races for the other two methods, so the comparison is like-for-like.
        same = [r for r in agg]
        bt["hybrid"]["physics_on_same_rows"] = {
            "stops_accuracy": float(np.mean([r["stops_match"] for r in same])),
            "compound_set_accuracy": float(np.mean([r["compound_set_match"] for r in same])),
            "start_compound_accuracy": float(np.mean([r["start_compound_match"] for r in same])),
            "first_stop_mae": float(np.mean([abs(r["first_stop_error"]) for r in same if "first_stop_error" in r])),
        }
        bt["hybrid"]["baseline_on_same_rows"] = {
            "stops_accuracy": float(np.mean([r["baseline_stops_match"] for r in same if "baseline_stops_match" in r])),
            "compound_set_accuracy": float(np.mean([r["baseline_compound_set_match"] for r in same if "baseline_compound_set_match" in r])),
            "start_compound_accuracy": float(np.mean([r["baseline_start_compound_match"] for r in same if "baseline_start_compound_match" in r])),
            "first_stop_mae": float(np.mean([abs(r["baseline_first_stop_error"]) for r in same if "baseline_first_stop_error" in r])),
        }
    path.write_text(json.dumps(bt, indent=1))
    print(json.dumps(bt.get("hybrid"), indent=1))
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{SEASON_DIR.parent.parent / 'mlflow.db'}")
        mlflow.set_experiment("pitwall_strategy_backtest")
        with mlflow.start_run(run_name="hybrid_behaviour_model"):
            mlflow.log_metrics({f"hybrid_{k}": v for k, v in bt.get("hybrid", {}).items() if isinstance(v, float)})
    except Exception as exc:
        print(f"(mlflow logging skipped: {exc})")


if __name__ == "__main__":
    main()
