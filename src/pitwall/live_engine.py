"""Turn a RaceState into a pit-wall call for one driver.

Pipeline per request (~100-300 ms):
  1. Live Bayesian update of compound degradation from every car's green-flag
     race laps (field), then a second shrinkage step for the chosen car's own
     stint (driver/team-specific wear).
  2. Pit-loss update from stops already observed in this race.
  3. Exact DP over all remaining strategies from the car's current tyre state
     (SC/VSC make a stop this lap cheaper).
  4. Monte Carlo over future SC/VSC + parameter uncertainty -> win probability
     and tail risk (CVaR) per plan.
  5. Race-craft context: rejoin position if boxing now, undercut threat from
     the car behind / opportunity on the car ahead, rival stop predictions,
     model-drift alert when recent laps leave the predictive band.

Two guards keep the call honest:
  * Conditions: rain, wet tyres, race-control wet-track messages or a field
    that is drying fast make dry-tyre wear maths unreliable. Those laps are
    kept out of the wear model, a car that ran wet tyres is freed from the
    two-dry-compound rule, and slick <-> intermediate switches are called from
    the observed crossover (who is faster right now), never from dry wear.
  * Decisiveness: "BOX, BOX" only when waiting one more lap costs real time;
    otherwise the call is "pit window open".
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from src.pitwall.optimizer import SC_PIT_FACTOR, VSC_PIT_FACTOR, Optimizer, RaceContext, TyreState
from src.pitwall.race_data import DRY_COMPOUNDS
from src.pitwall.tyre_model import (
    LIVE_MIN_STINT_LAPS,
    LIVE_PRIOR_SD_MIN,
    LIVE_STINT_NOISE_SD,
    MIN_FIELD_RATE,
    MULT_BOUNDS,
    TyreModel,
    _combine,
    _posterior,
    _theil_sen,
    live_update,
)

RACE_START_LAPS = 3  # excluded from wear estimation
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
SLOW_LAP_FACTOR = 1.05  # field lap this much slower than the race's best field lap: not dry green running
DRYING_RATE = 0.012  # field pace improving > 1.2 %/lap for 3 laps: track drying
RAIN_SLOWDOWN = 0.03  # field pace 3 % slower than the lap before: rain arriving
UNSETTLED_LAPS = 3  # conditions count as changing for this many laps after the last sign
# "BOX, BOX" thresholds. The dry wear model times stops to a window, not a lap:
# on R4/R7/R12/R15 no green-flag gate got 2-lap precision above ~0.35, so under
# green the call is the window and BOX is kept for a window that is closing.
# Under SC/VSC, calls with P(best) >= 0.6 were right 8/10 times.
WAIT_COST_GREEN = 3.0  # s lost by waiting a lap before a green-flag BOX
WAIT_COST_NEUTRAL = 8.0  # s lost by passing up an SC/VSC stop ...
P_BEST_NEUTRAL = 0.6  # ... or Monte Carlo P(best) of the boxing plan
CROSSOVER_HORIZON = 12  # laps a slick/wet pace gap is assumed to last
CROSSOVER_MARGIN = 2.0  # s of net gain required before switching tyre type
RC_WET = re.compile(r"\b(WET|RAIN(?:ING)?|SLIPPERY|INTERMEDIATE)\b")
OUTLAP_WARMUP = 0.8  # s lost on the out-lap while new tyres come in (undercut maths)
BAND_Z = 1.2816  # 80% central interval


def _clean_stints(car: dict, fuel: float, exclude: frozenset[int] = frozenset()) -> list[dict]:
    """Split a car's completed laps into stints of representative green laps."""
    laps = [l for l in car.get("laps", []) if l.get("time") and l.get("age")]
    if not laps:
        return []
    med = float(np.median([l["time"] for l in laps]))
    out: list[dict] = []
    cur: dict | None = None
    for l in laps:
        if cur is None or l["compound"] != cur["compound"] or l["age"] < (cur["ages"][-1] if cur["ages"] else 0):
            cur = {"compound": l["compound"], "ages": [], "times": [], "laps": [], "team": car["team"]}
            out.append(cur)
        # Laps 1-3 (start, DRS trains, tyre warm-up) say little about wear.
        # Wet, drying or rain-hit laps (exclude) say nothing about dry wear either.
        bad = l["lap"] <= RACE_START_LAPS or l["pit_in"] or l["pit_out"] or l["neutral"] or l["time"] > med * 1.05 or l["lap"] in exclude
        if not bad and l["compound"] in DRY_COMPOUNDS:
            cur["ages"].append(float(l["age"]))
            cur["times"].append(float(l["time"]))
            cur["laps"].append(float(l["lap"]))
    return [s for s in out if len(s["ages"]) >= 1]


def personal_wear(car: dict, model: TyreModel, exclude: frozenset[int] = frozenset()) -> dict:
    """A car's own wear multiplier vs the live field rate, from all its race stints.

    Prior = the driver's weekend estimate (season history + practice). Each green
    stint of >= 4 laps adds an observation slope/field_rate with its sampling
    error; a conflict-aware Gaussian update combines them.
    """
    w = model.driver_mult.get(car["driver"]) or {}
    mu0, sd0 = float(w.get("mult", 1.0)), max(float(w.get("sd", LIVE_PRIOR_SD_MIN)), LIVE_PRIOR_SD_MIN)
    obs, stints_used, laps_used = [], 0, 0
    current = None
    for st in _clean_stints(car, model.fuel_per_lap, exclude):
        n = len(st["ages"])
        if n < LIVE_MIN_STINT_LAPS or st["compound"] not in model.compounds:
            continue
        times = np.asarray(st["times"]) + model.fuel_per_lap * np.asarray(st["laps"])
        slope = _theil_sen(np.asarray(st["ages"]), times)
        se_slope = float(np.sqrt(12 * model.noise_sd**2 / n**3)) + 0.01
        field = model.compounds[st["compound"]].deg
        current = {"compound": st["compound"], "observed": slope, "field": field, "laps": n}
        if field < MIN_FIELD_RATE:
            continue  # near-zero wear: a ratio is meaningless, keep the prior
        # Stint-level noise (traffic, management, modes) dominates pure lap noise.
        obs.append((float(np.clip(slope / field, *MULT_BOUNDS)), float(np.hypot(se_slope / field, LIVE_STINT_NOISE_SD))))
        stints_used += 1
        laps_used += n
    comb = _combine(obs)
    mu, sd = _posterior(mu0, sd0, *comb) if comb else (mu0, sd0)
    mu = float(np.clip(mu, *MULT_BOUNDS))
    out = {"mult": mu, "sd": float(sd), "prior_mult": mu0, "prior_sd": sd0,
           "observed_mult": comb[0] if comb else None, "stints": stints_used, "laps": laps_used}
    if current:
        out |= {"compound": current["compound"], "observed": current["observed"], "field": current["field"],
                "posterior": current["field"] * mu, "current_stint_laps": current["laps"]}
    return out


def _observed_pit_losses(state: dict) -> list[float]:
    losses = []
    for car in state["cars"]:
        laps = car.get("laps", [])
        times = [l["time"] for l in laps if l.get("time") and not l["pit_in"] and not l["pit_out"] and not l["neutral"]]
        if len(times) < 5:
            continue
        ref = float(np.median(times[-10:]))
        by_lap = {l["lap"]: l for l in laps}
        for l in laps:
            if l["pit_in"] and not l["neutral"] and (l["lap"] + 1) in by_lap:
                o = by_lap[l["lap"] + 1]
                if o.get("time") and l.get("time") and not o["neutral"]:
                    loss = l["time"] + o["time"] - 2 * ref
                    if 10 < loss < 45:
                        losses.append(loss)
    return losses


def _undercut_gain(model: TyreModel, team: str | None, old_c: str, old_age: int, new_c: str, laps: int = 2, driver: str | None = None) -> float:
    """Lap-time advantage of switching to new tyres now, over `laps` laps (driver's own wear)."""
    if old_c not in model.compounds or new_c not in model.compounds:
        return 0.0
    old = model.lap_delta(old_c, np.arange(old_age + 1, old_age + laps + 1), team, driver).sum()
    new = model.lap_delta(new_c, np.arange(1, laps + 1), team, driver).sum()
    return float(old - new - OUTLAP_WARMUP)


def _ran_wets(car: dict) -> bool:
    return car["compound"] in WET_COMPOUNDS or any(c in WET_COMPOUNDS for c in car.get("compounds_used", []))


def _best_fresh_dry(model: TyreModel, ctx: RaceContext, lap: int, used: tuple[str, ...]) -> str:
    """Dry compound a car leaving wet tyres now should take (rule waived: it ran wets)."""
    c2 = RaceContext(**{**ctx.__dict__, "two_compound_rule": False})
    best = Optimizer(model, c2).optimise(TyreState(lap=lap, compound=None, used=used), max_stops=1, top=1)
    return best[0].sequence[0] if best else "MEDIUM"


def field_pace(cars: list[dict]) -> dict[int, float]:
    """Median green-lap time of the field, per lap (laps with >= 5 clean times)."""
    by_lap: dict[int, list[float]] = defaultdict(list)
    for c in cars:
        for l in c.get("laps", []):
            if l.get("time") and l["lap"] > 1 and not (l["pit_in"] or l["pit_out"] or l["neutral"]):
                by_lap[l["lap"]].append(l["time"])
    return {k: float(np.median(v)) for k, v in sorted(by_lap.items()) if len(v) >= 5}


def assess_conditions(state: dict) -> dict:
    """Is the track wet or changing? Which laps must stay out of the dry wear model?"""
    cars = [c for c in state["cars"] if not c["retired"]]
    wx = state.get("weather") or {}
    lap_now = max((c["laps_completed"] for c in cars), default=0)
    pace = field_pace(cars)
    laps = list(pace)
    best = min(pace.values()) if pace else None
    rain_laps = {int(l) for l in wx.get("rain_laps") or []}
    wet_laps = {l["lap"] for c in cars for l in c.get("laps", []) if l["compound"] in WET_COMPOUNDS}
    on_wets = [c["driver"] for c in cars if c["compound"] in WET_COMPOUNDS]

    # (lap, kind, text). Rain alone is reported but does not unsettle the call:
    # a drizzle that does not move lap times leaves dry strategy intact.
    signs: list[tuple[int, str, str]] = [(l, "rain", "rain") for l in rain_laps] + [(l, "wet_tyres", "cars on wet tyres") for l in wet_laps]
    if wx.get("rainfall"):
        signs.append((lap_now, "rain", "rain falling"))
    if on_wets:
        signs.append((lap_now, "wet_tyres", f"{len(on_wets)} car{'s' if len(on_wets) > 1 else ''} on intermediates/wets"))
    for a, b in zip(laps, laps[3:]):
        if b - a == 3 and (pace[a] - pace[b]) / pace[b] / 3 > DRYING_RATE:
            signs.append((b, "drying", "track drying fast"))
    for a, b in zip(laps, laps[1:]):
        if b - a == 1 and (pace[b] - pace[a]) / pace[a] > RAIN_SLOWDOWN:
            signs.append((b, "jump", "lap times jumped (rain?)"))
    for m in state.get("race_control") or []:
        msg = str(m.get("message") or "").upper()
        if RC_WET.search(msg) and "RISK OF RAIN" not in msg:
            signs.append((int(m.get("lap") or lap_now), "rc", "race control: " + msg.title()[:60]))

    recent = [(l, k, r) for l, k, r in signs if lap_now - l <= UNSETTLED_LAPS]
    slow = {l for l in laps if best and pace[l] > best * SLOW_LAP_FACTOR}
    return {
        "changing": bool(recent),
        "unsettled": any(k != "rain" for _, k, _ in recent),
        "raining": bool(wx.get("rainfall")),
        "reasons": list(dict.fromkeys(r for _, _, r in sorted(recent, reverse=True)))[:4],
        "cars_on_wets": on_wets,
        "excluded_laps": sorted(slow | rain_laps | wet_laps),
    }


def _class_pace(cars: list[dict], wet: bool, upto: int, n: int = 2) -> tuple[float | None, int]:
    """Median clean lap time over the last n laps of cars on wet (or dry) tyres, and how many cars."""
    times, who = [], set()
    for c in cars:
        for l in c.get("laps", []):
            if upto - n < l["lap"] <= upto and l.get("time") and not (l["pit_in"] or l["pit_out"] or l["neutral"]) \
                    and (l["compound"] in WET_COMPOUNDS) == wet and l["compound"] in DRY_COMPOUNDS + WET_COMPOUNDS:
                times.append(l["time"])
                who.add(c["driver"])
    return (float(np.median(times)), len(who)) if len(who) >= 2 else (None, len(who))


def crossover(cars: list[dict], car: dict, stop_cost: float, laps_left: int) -> dict:
    """Should this car switch between slicks and wet tyres, judged by who is faster *now*."""
    on_wet = car["compound"] in WET_COMPOUNDS
    upto = max((c["laps_completed"] for c in cars), default=0)
    mine, n_mine = _class_pace(cars, on_wet, upto)
    other, n_other = _class_pace(cars, not on_wet, upto)
    horizon = max(1, min(laps_left, CROSSOVER_HORIZON))
    need = (stop_cost + CROSSOVER_MARGIN) / horizon  # s/lap the other tyre must be faster
    gain = None if mine is None or other is None else mine - other
    switch = gain is not None and gain > need
    return {"on_wet": on_wet, "my_class_pace": mine, "other_class_pace": other, "other_cars": n_other,
            "gain_per_lap": None if gain is None else round(gain, 2), "needed_per_lap": round(need, 2), "switch": switch}


def analyse(state: dict, weekend: dict, driver: str, sims: int = 300) -> dict:
    cars = state["cars"]
    car = next((c for c in cars if c["driver"] == driver), None)
    if car is None:
        return {"error": f"{driver} is not in this session"}
    total = int(state.get("total_laps") or weekend["total_laps"])
    k = int(car["laps_completed"])
    base = {"driver": driver, "lap": k + 1, "total_laps": total, "track_status": state["track_status"], "car": {kk: v for kk, v in car.items() if kk != "laps"}}
    if car["retired"]:
        return base | {"call": {"action": "RETIRED", "headline": f"{driver} is out of the race"}}
    if k >= total:
        return base | {"call": {"action": "FINISHED", "headline": "Chequered flag"}}

    prior = TyreModel.from_dict(weekend["model"])
    cond = assess_conditions(state)
    excl = frozenset(cond["excluded_laps"])

    # 1. field-wide live update of each compound's wear, then a *personal* wear
    #    multiplier for every car from all of its own race stints (partial pooling:
    #    starts at the driver's weekend estimate, moves to their own data as laps accrue).
    field_stints = [s for c in cars for s in _clean_stints(c, prior.fuel_per_lap, excl)]
    model, evidence = live_update(prior, field_stints)
    team = car.get("team")
    personal = {c["driver"]: personal_wear(c, model, excl) for c in cars}
    for drv, p in personal.items():
        model.driver_mult[drv] = {"mult": p["mult"], "sd": p["sd"]}
    car_deg = personal.get(driver)
    own = _clean_stints(car, prior.fuel_per_lap, excl)
    own_cur = own[-1] if own and own[-1]["compound"] == car["compound"] else None

    # 2. pit loss
    observed = _observed_pit_losses(state)
    pit_loss = float((weekend["pit_loss"] * 3 + sum(observed)) / (3 + len(observed)))

    # 3. optimise from the current tyre state
    ts = state["track_status"]
    pit_now = pit_loss * SC_PIT_FACTOR if ts == "SC" else pit_loss * VSC_PIT_FACTOR if ts in ("VSC", "VSC_ENDING") else None
    compound = car["compound"] if car["compound"] in model.compounds else None
    used = tuple(c for c in car.get("compounds_used", []) if c in DRY_COMPOUNDS)
    tstate = TyreState(lap=k, compound=compound, age=int(car["tyre_age"]), used=used, pit_cost_now=pit_now)
    ctx = RaceContext(total_laps=total, pit_loss=pit_loss, sc_rate=weekend.get("sc_rate", 0.01), vsc_rate=weekend.get("vsc_rate", 0.006),
                      team=team, driver=driver, stop_penalty=weekend.get("stop_penalty", 0.0), fuel_wear=weekend.get("fuel_wear", 1.0), min_stint=2,
                      two_compound_rule=not _ran_wets(car))
    opt = Optimizer(model, ctx)
    plans = opt.candidates(tstate, max_stops=3, top=6)
    if not plans and compound:
        # Tyres already older than the model believes possible: never go silent.
        opt.relaxed = True
        plans = opt.candidates(tstate, max_stops=3, top=6)
    if not plans:
        return base | {"call": {"action": "NO_PLAN", "headline": "No legal plan found (check tyre data)"}}
    opt.monte_carlo(plans, tstate, sims=sims)
    best_det = plans[0].total
    plans.sort(key=lambda p: p.total + p.risk.get("expected_delta", 0) - best_det)
    best = plans[0]

    # 4. the call
    cur_lap = k + 1
    nxt = best.pit_laps[0] if best.pit_laps else None
    nxt_c = best.sequence[1] if len(best.sequence) > 1 else None
    stop_cost_now = pit_now if pit_now is not None else pit_loss
    if nxt is None:
        action, headline = "STAY_OUT", f"Stay out to the flag on {car['compound'].title()}s"
    elif nxt <= cur_lap:
        action = "BOX_NOW"
        headline = f"BOX, BOX — {nxt_c.title()}s" + (" (cheap stop under " + ts.replace("_ENDING", "") + ")" if pit_now else "")
    elif nxt - cur_lap <= 3:
        action, headline = "BOX_SOON", f"Box in {nxt - cur_lap} lap{'s' if nxt - cur_lap > 1 else ''} (lap {nxt}) for {nxt_c.title()}s"
    else:
        lo, hi = best.windows[0]
        action, headline = "STAY_OUT", f"Stay out — pit window laps {lo}–{hi}, target lap {nxt} for {nxt_c.title()}s"

    # 4a. decisiveness: inside a flat window the optimum lap is "now" by a few
    #     tenths every lap until the car stops. Only shout BOX when waiting costs.
    wait_cost = None
    if action == "BOX_NOW":
        # every alternative that does not stop this lap: one lap later, or any
        # other candidate plan whose first stop is later (or that never stops)
        later = [opt.best_with_first_stop(tstate, cur_lap + 1)] + [p.total for p in plans if not p.pit_laps or p.pit_laps[0] > cur_lap]
        later = [t for t in later if t is not None and np.isfinite(t)]
        wait_cost = float(min(later) - best.total) if later else float("inf")
        p_best = float(best.risk.get("win_probability") or 0)
        if pit_now is not None:
            if wait_cost < WAIT_COST_NEUTRAL and p_best < P_BEST_NEUTRAL:
                action = "WINDOW_OPEN"
                headline = f"Cheap stop under {ts.replace('_ENDING', '')} — worth ~{max(wait_cost, 0):.0f} s if you box for {nxt_c.title()}s (optional)"
        elif wait_cost < WAIT_COST_GREEN:
            hi = max(best.windows[0][1], cur_lap + 1) if best.windows else cur_lap + 1
            action = "WINDOW_OPEN"
            headline = f"Pit window open — box by lap {hi} for {nxt_c.title()}s"

    # 4b. conditions: wet tyres or a changing track override the dry maths
    reason = None
    xo = None
    if car["compound"] in WET_COMPOUNDS or cond["cars_on_wets"]:
        xo = crossover(cars, car, stop_cost_now, total - k)
        if xo["on_wet"]:
            dry_c = _best_fresh_dry(model, ctx, k, used)
            if xo["switch"]:
                action, nxt, nxt_c = "BOX_NOW", cur_lap, dry_c
                headline = f"BOX, BOX — slicks ({dry_c.title()}s): crossover reached"
                reason = f"Cars on slicks are {xo['gain_per_lap']:.1f} s/lap faster; {xo['needed_per_lap']:.1f} s/lap pays for the stop."
            else:
                action, nxt, nxt_c = "STAY_OUT", None, None
                headline = f"Stay out on {car['compound'].title()}s — slicks not faster yet"
                reason = (f"Slick runners are {xo['gain_per_lap']:+.1f} s/lap vs wet runners; need +{xo['needed_per_lap']:.1f} s/lap to box."
                          if xo["gain_per_lap"] is not None else f"Not enough cars on slicks to judge the crossover (need +{xo['needed_per_lap']:.1f} s/lap).")
        elif xo["switch"]:
            action, nxt, nxt_c = "BOX_NOW", cur_lap, "INTERMEDIATE"
            headline = "BOX, BOX — Intermediates: wet runners are faster"
            reason = f"Cars on wet tyres are {xo['gain_per_lap']:.1f} s/lap faster; {xo['needed_per_lap']:.1f} s/lap pays for the stop."
        elif action in ("BOX_NOW", "BOX_SOON", "WINDOW_OPEN") and pit_now is None:
            action = "HOLD"
            headline = "Wet tyres on track — stay out on slicks, no dry-tyre stop"
            reason = "Box for intermediates only once cars on them are clearly faster."
        else:
            reason = "Cars on wet tyres: watching the crossover."
    elif cond["unsettled"] and action in ("BOX_NOW", "BOX_SOON", "WINDOW_OPEN") and pit_now is None:
        planned = f"plan: box ~lap {nxt} for {nxt_c.title()}s" if nxt and nxt_c else "plan on hold"
        action = "HOLD"
        headline = "Track changing — hold the dry-tyre stop"
        reason = f"{', '.join(cond['reasons']).capitalize()}; lap times are not a fair read of tyre wear yet ({planned})."
    second = plans[1] if len(plans) > 1 else None
    confidence = best.risk.get("win_probability")

    # Cost of boxing right now instead of following the plan
    box_now_cost = None
    if best.pit_laps and best.pit_laps[0] != cur_lap and cur_lap < total:
        forced = opt.best_with_first_stop(tstate, cur_lap)
        if forced is not None:
            box_now_cost = float(forced - best.total)

    # 5. race-craft context
    running = [c for c in cars if not c["retired"]]
    me_gap = car.get("gap_to_leader")
    rejoin = None
    stop_cost = pit_now if pit_now is not None else pit_loss
    if me_gap is not None:
        proj = me_gap + stop_cost
        ahead_after = [c for c in running if c["driver"] != driver and c.get("gap_to_leader") is not None and c["laps_completed"] >= k - 1 and c["gap_to_leader"] < proj]
        pos = len(ahead_after) + 1
        car_ahead = max(ahead_after, key=lambda c: c["gap_to_leader"]) if ahead_after else None
        behind_after = [c for c in running if c["driver"] != driver and c.get("gap_to_leader") is not None and c["gap_to_leader"] >= proj]
        car_behind = min(behind_after, key=lambda c: c["gap_to_leader"]) if behind_after else None
        rejoin = {
            "position": pos,
            "stop_cost": round(stop_cost, 1),
            "ahead": car_ahead and {"driver": car_ahead["driver"], "gap": round(proj - car_ahead["gap_to_leader"], 1), "compound": car_ahead["compound"], "tyre_age": car_ahead["tyre_age"]},
            "behind": car_behind and {"driver": car_behind["driver"], "gap": round(car_behind["gap_to_leader"] - proj, 1), "compound": car_behind["compound"], "tyre_age": car_behind["tyre_age"]},
            "traffic": bool(car_ahead and proj - car_ahead["gap_to_leader"] < 1.5),
        }

    idx = next((i for i, c in enumerate(running) if c["driver"] == driver), None)
    threats = []
    if idx is not None:
        new_c = nxt_c or "HARD"
        if idx + 1 < len(running):
            b = running[idx + 1]
            gain = _undercut_gain(model, b.get("team"), b["compound"], b["tyre_age"], new_c, driver=b["driver"]) if b["compound"] in model.compounds else 0.0
            gap_b = b.get("interval")
            threats.append({"type": "undercut_threat", "driver": b["driver"], "gap": gap_b, "gain": round(gain, 2),
                            "live": bool(gap_b is not None and gain > gap_b and b["stops"] <= car["stops"])})
        if idx > 0:
            a = running[idx - 1]
            gain = _undercut_gain(model, team, car["compound"], car["tyre_age"], new_c, driver=driver) if car["compound"] in model.compounds else 0.0
            gap_a = car.get("interval")
            threats.append({"type": "undercut_opportunity", "driver": a["driver"], "gap": gap_a, "gain": round(gain, 2),
                            "live": bool(gap_a is not None and gain > gap_a and a["stops"] <= car["stops"])})

    # rival predictions (cheap: deterministic, <=2 stops)
    rivals = []
    for c in running:
        if c["driver"] == driver or c["compound"] not in model.compounds:
            continue
        rc = RaceContext(total_laps=total, pit_loss=pit_loss, team=c.get("team"), driver=c["driver"], stop_penalty=ctx.stop_penalty, fuel_wear=ctx.fuel_wear, min_stint=2,
                         two_compound_rule=not _ran_wets(c))
        st = TyreState(lap=c["laps_completed"], compound=c["compound"], age=int(c["tyre_age"]),
                       used=tuple(x for x in c.get("compounds_used", []) if x in DRY_COMPOUNDS), pit_cost_now=pit_now)
        r = Optimizer(model, rc).optimise(st, max_stops=2, top=1)
        if r:
            rivals.append({"driver": c["driver"], "position": c["position"], "compound": c["compound"], "tyre_age": c["tyre_age"],
                           "next_stop": r[0].pit_laps[0] if r[0].pit_laps else None,
                           "next_compound": r[0].sequence[1] if len(r[0].sequence) > 1 else None, "plan": r[0].name})

    # tyre curve + drift monitor for the current stint
    curve = None
    drift = None
    if compound:
        ages = np.arange(0, max(car["tyre_age"] + (best.pit_laps[0] - k if best.pit_laps else total - k), car["tyre_age"]) + 2)
        pred = model.lap_delta(compound, ages, team, driver)
        band = BAND_Z * model.noise_sd
        obs_pts = []
        if own_cur and own_cur["ages"]:
            ages_o = np.asarray(own_cur["ages"])
            times = np.asarray(own_cur["times"]) + model.fuel_per_lap * (np.asarray(own_cur["laps"]) - own_cur["laps"][0])
            pred_o = model.lap_delta(compound, ages_o, team, driver)
            # Anchor observed laps to the model with a robust (median) offset.
            offset = float(np.median(times - pred_o))
            obs_pts = [{"age": int(a), "delta": round(float(t - offset), 3)} for a, t in zip(ages_o, times)]
            if len(obs_pts) >= 6:
                resid = (times - offset - pred_o)[-3:]
                z = float(resid.mean() / (model.noise_sd / np.sqrt(3)))
                drift = {"z": round(z, 2), "alert": abs(z) > 2.5,
                         "message": ("Recent laps slower than the model: tyre falling off" if z > 0 else "Recent laps faster than the model: tyre holding up") if abs(z) > 2.5 else "Lap times inside the model band"}
        curve = {"compound": compound, "ages": ages.tolist(), "predicted": np.round(pred, 3).tolist(), "band": round(band, 3), "observed": obs_pts}

    return base | {
        "call": {
            "action": action,
            "headline": headline,
            "target_lap": nxt,
            "next_compound": nxt_c,
            "window": best.windows[0] if best.windows else None,
            "confidence": confidence,
            "margin_to_next": round(second.total - best.total, 2) if second else None,
            "box_now_cost": round(box_now_cost, 2) if box_now_cost is not None else None,
            # P(fastest plan from here has k more stops), over SC/VSC + wear uncertainty
            "stop_probabilities": getattr(opt, "stop_probabilities", {}),
            "wait_cost": round(wait_cost, 2) if wait_cost is not None and np.isfinite(wait_cost) else None,
            "box_win_probability": round(p_best, 3) if wait_cost is not None else None,
            "reason": reason,
        },
        "conditions": cond | {"crossover": xo},
        "plans": [p.to_dict() for p in plans],
        "model": {
            "pit_loss": round(pit_loss, 2),
            "pit_losses_observed": len(observed),
            "compounds": {c: {"deg": round(p.deg, 4), "deg_sd": round(p.deg_sd, 4), "offset": round(p.offset, 3)} for c, p in model.compounds.items()},
            "live_evidence": evidence,
            "car_deg": car_deg,
            "stop_penalty": ctx.stop_penalty,
        },
        "rejoin": rejoin,
        "threats": threats,
        "rivals": sorted(rivals, key=lambda r: r["position"] or 99),
        "tyre_curve": curve,
        "drift": drift,
    }
