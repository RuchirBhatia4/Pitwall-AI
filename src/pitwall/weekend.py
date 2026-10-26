"""Build the pre-race model for one race weekend (walk-forward, no leakage).

Everything about round R is estimated from:
  * race fits of rounds < R (season prior, SC rates, pit-loss fallback),
  * the same circuit's 2024-25 races (Safety Car / VSC record, see track_history),
  * round R's practice / sprint sessions,
  * round R's qualifying (grid only).
The finished race of round R is used only for evaluation, never for the
prediction.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.pitwall.race_data import (
    _int_or_none,
    event_sessions,
    load_session,
    measure_pit_loss,
    neutralised_laps,
    normalize_laps,
    qualifying_grid,
    stints_from_laps,
)
from src.pitwall import track_history
from src.pitwall.season import SEASON_DIR, get_schedule
from src.pitwall.tyre_model import (
    LONG_RUN_TYPES,
    TyreModel,
    classify_practice_runs,
    fit_race_model,
    fit_transfer,
    race_stint_wear,
    season_prior,
    short_run_offsets,
    summarise_practice,
    personalise,
    practice_driver_multipliers,
    season_driver_priors,
    weekend_model,
    weekend_severity,
    DEFAULT_COMPOUND_RATIOS,
)

YEAR = 2026
FITS_DIR = SEASON_DIR / str(YEAR) / "fits"
# Bump when what race_fit stores changes; older cached fits are recomputed.
FIT_VERSION = 2
# Wear a pre-race model is built to predict: "stint" = race_stint_wear (the
# within-stint drop-off), "fit" = fit_race_model's wear.
WEAR_TARGET = "stint"
PRACTICE_SESSIONS = ("Practice 1", "Practice 2", "Practice 3", "Sprint")
# Weight (in laps) of the 2026 season-wide Safety Car / VSC rate against a
# circuit's own 2024-25 record: ~2 races, so two quiet years at a track pull
# its rate down but cannot zero it.
VENUE_SC_PRIOR_LAPS = 120.0
# A compound needs this many clean race laps for its within-stint wear to count.
RACE_WEAR_MIN_LAPS = 60
# Pirelli's sets per driver for a weekend (sprint weekends trade two softs for
# a medium). Qualifying uses most of the softs, so a race soft is often a
# scrubbed set: in 2026 the median driver started the race with 1 new soft and
# 1 new medium. The two hardest compounds always had a new set kept back.
TYRE_ALLOCATION = {
    "conventional": {"SOFT": 8, "MEDIUM": 3, "HARD": 2},
    "sprint_qualifying": {"SOFT": 6, "MEDIUM": 4, "HARD": 2},
}
KEPT_FOR_RACE = ("MEDIUM", "HARD")
# Compound -> (new sets, laps on the freshest used set) when there is no
# running to count yet: the median 2026 driver at the start of a race.
TYPICAL_SETS = {"SOFT": (1, 4), "MEDIUM": (1, 6), "HARD": (2, 6)}
INVENTORY_SESSIONS = PRACTICE_SESSIONS + ("Sprint Qualifying", "Sprint Shootout", "Qualifying")
# Official race distances for events not yet run (laps). Completed races use
# the classified lap count.
SCHEDULED_LAPS = {"Kuala Lumpur": 56, "Marina Bay": 62, "Austin": 56, "Mexico City": 71,
                  "São Paulo": 71, "Las Vegas": 50, "Lusail": 57, "Yas Marina": 58}


def event_row(rnd: int) -> pd.Series:
    sched = get_schedule(YEAR)
    return sched.loc[sched["RoundNumber"] == rnd].iloc[0]


def race_is_finished(rnd: int) -> bool:
    start = event_sessions(YEAR, rnd).get("Race")
    return start is not None and pd.Timestamp.now(tz="UTC").tz_localize(None) > start + pd.Timedelta(hours=3)


def completed_rounds() -> list[int]:
    sched = get_schedule(YEAR)
    return [int(r) for r in sched["RoundNumber"] if race_is_finished(int(r))]


def race_fit(rnd: int) -> dict | None:
    """Fit (and cache on disk) the tyre model of a finished race."""
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    path = FITS_DIR / f"r{rnd:02d}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("version") == FIT_VERSION:
            return cached
    try:
        session = load_session(YEAR, rnd, "Race")
    except Exception:
        return None
    laps = normalize_laps(session)
    if laps.empty:
        return None
    neutral = neutralised_laps(laps)
    wet = laps["compound"].isin(["INTERMEDIATE", "WET"]).mean() > 0.15
    model = None if wet else fit_race_model(laps, neutral)
    stints = stints_from_laps(laps)
    neutral_starts = {k: sum(1 for i, lap in enumerate(v) if i == 0 or v[i - 1] != lap - 1) for k, v in neutral.items()}
    out = {
        "version": FIT_VERSION,
        "round": rnd,
        "total_laps": int(laps["lap"].max()),
        "pit_loss": measure_pit_loss(laps, neutral),
        "wet": bool(wet),
        "sc_starts": neutral_starts["sc"],
        "vsc_starts": neutral_starts["vsc"],
        "model": model.to_dict() if model else None,
        "stint_wear": race_stint_wear(laps, neutral, model.fuel_per_lap) if model else {},
        "max_age": stints.groupby("compound")["laps"].max().to_dict() if not stints.empty else {},
    }
    # Same-weekend practice evidence, used only to learn practice->race transfer
    # for *later* rounds.
    runs, session_laps = practice_evidence(rnd)
    out["practice"] = summarise_practice(runs, 0.055)
    out["short_offsets"] = short_run_offsets(session_laps)
    path.write_text(json.dumps(out))
    return out


def _prior_inputs(rnd: int) -> tuple[TyreModel, dict]:
    fits = [f for r in range(1, rnd) if race_is_finished(r) and (f := race_fit(r))]
    with_model = [f for f in fits if f.get("model")]
    models = [TyreModel.from_dict(f["model"]) for f in with_model]
    wear = [_race_wear(f) for f in with_model]
    prior = season_prior(models, wear)
    laps_total = sum(f["total_laps"] for f in fits) or 1
    pit_losses = [f["pit_loss"] for f in fits if f.get("pit_loss")]
    pairs = []
    for f in fits:
        if not f.get("model"):
            continue
        race = _race_wear(f)
        for c, pc in f.get("practice", {}).get("compounds", {}).items():
            if c in race and pc["laps"] >= 15:
                pairs.append((pc["deg"], race[c], min(pc["laps"], f["model"]["compounds"].get(c, {}).get("evidence_laps", pc["laps"])) ** 0.5))
    off_pairs = []
    for f in fits:
        if not f.get("model"):
            continue
        for c, so in f.get("short_offsets", {}).items():
            rc = f["model"]["compounds"].get(c)
            if rc and rc["evidence_laps"] >= 30:
                off_pairs.append((so["offset"], rc["offset"], 1.0))
    off_transfer = fit_transfer(off_pairs, clip=(0.0, 1.0))
    # Typical wear of each compound relative to the Medium, from earlier races.
    ratio_obs: dict[str, list[float]] = {}
    for race in wear:
        if race.get("MEDIUM", 0.0) > 0.02:
            for c, v in race.items():
                ratio_obs.setdefault(c, []).append(v / race["MEDIUM"])
    ratios = {c: (float(np.median(v)) if len(v) >= 2 else DEFAULT_COMPOUND_RATIOS[c]) for c, v in ratio_obs.items()}
    ratios = DEFAULT_COMPOUND_RATIOS | ratios
    stats = {
        "race_models": models,
        "compound_ratios": ratios,
        "offset_transfer": off_transfer,
        "transfer": fit_transfer(pairs),
        "races": len(fits),
        "sc_rate": (sum(f["sc_starts"] for f in fits) + 1) / (laps_total + 60),  # +1/60 smoothing
        "vsc_rate": (sum(f["vsc_starts"] for f in fits) + 1) / (laps_total + 60),
        "pit_loss_median": float(np.median(pit_losses)) if pit_losses else 22.5,
    }
    return prior, stats


def _race_wear(fit: dict) -> dict[str, float]:
    """Compound -> wear of a finished race, as WEAR_TARGET defines it."""
    if WEAR_TARGET == "fit":
        return {c: v["deg"] for c, v in fit["model"]["compounds"].items()}
    return {c: max(v["deg"], 0.005) for c, v in fit.get("stint_wear", {}).items() if v["laps"] >= RACE_WEAR_MIN_LAPS}


def practice_evidence(rnd: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sessions = event_sessions(YEAR, rnd)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    runs = []
    all_laps: dict[str, pd.DataFrame] = {}
    for name in PRACTICE_SESSIONS:
        if name not in sessions or sessions[name] > now:
            continue
        try:
            laps = normalize_laps(load_session(YEAR, rnd, name))
        except Exception:
            continue
        # Sprint races are full-fuel-ish race runs: use the stints directly.
        if name != "Sprint":
            all_laps[name] = laps
        r = classify_practice_runs(laps, name)
        if not r.empty:
            runs.append(r)
    return (pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()), all_laps


def tyre_sets(rnd: int, fmt: str) -> dict[str, dict[str, list[int]]]:
    """Each driver's tyre sets left for the race, counted from every session
    before it: compound -> [new sets, laps on the freshest used set].

    A stint starting on a new set (FastF1 / OpenF1 fresh flag, tyre age <= 2)
    uses one up. OpenF1 sometimes flags a used set as new, which can only
    under-count what is left, never invent sets: counts are capped by the
    allocation and a new medium and hard are always kept.
    """
    sessions = event_sessions(YEAR, rnd)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    alloc = TYRE_ALLOCATION.get(fmt, TYRE_ALLOCATION["conventional"])
    new_used: dict[tuple[str, str], int] = {}
    used_age: dict[tuple[str, str], float] = {}
    for name in INVENTORY_SESSIONS:
        if name not in sessions or sessions[name] > now:
            continue
        try:
            laps = normalize_laps(load_session(YEAR, rnd, name))
        except Exception:
            continue
        if laps.empty:
            continue
        for (drv, _), g in laps.groupby(["driver", "stint"]):
            c = g["compound"].iloc[0]
            life = g["tyre_life"].dropna()
            if c not in alloc or life.empty:
                continue
            fresh = g["fresh"].iloc[0]
            if float(life.iloc[0]) <= 2 and (pd.isna(fresh) or bool(fresh)):
                new_used[(drv, c)] = new_used.get((drv, c), 0) + 1
            used_age[(drv, c)] = min(used_age.get((drv, c), 99.0), float(life.max()))
    out: dict[str, dict[str, list[int]]] = {}
    for drv in {d for d, _ in new_used} | {d for d, _ in used_age}:
        out[drv] = {
            c: [max(n - new_used.get((drv, c), 0), 1 if c in KEPT_FOR_RACE else 0), int(used_age.get((drv, c), TYPICAL_SETS[c][1]))]
            for c, n in alloc.items()
        }
    return out


def field_tyre_sets(sets: dict[str, dict[str, list[int]]]) -> dict[str, list[int]]:
    """The median driver's sets (used for field-level plans)."""
    if not sets:
        return {c: list(v) for c, v in TYPICAL_SETS.items()}
    return {c: [int(np.median([d[c][0] for d in sets.values()])), int(np.median([d[c][1] for d in sets.values()]))]
            for c in TYPICAL_SETS}


@lru_cache(maxsize=32)
def build_weekend(rnd: int) -> dict:
    """Pre-race model + context for round `rnd` (JSON-serialisable)."""
    ev = event_row(rnd)
    prior, stats = _prior_inputs(rnd)
    runs, session_laps = practice_evidence(rnd)
    practice = summarise_practice(runs, prior.fuel_per_lap)
    raw_offsets = short_run_offsets(session_laps)
    # Low-fuel compound gaps shrink on race fuel; scale by the learned ratio.
    ot = stats["offset_transfer"]
    offsets = {c: v | {"offset": ot["beta"] * v["offset"], "offset_se": float(np.hypot(ot["beta"] * v["offset_se"], ot["sd"]))} for c, v in raw_offsets.items()}
    model = weekend_model(prior, practice, transfer=stats["transfer"], short_offsets=offsets, compound_ratios=stats["compound_ratios"])
    season_drivers = season_driver_priors(stats["race_models"])

    finished = race_is_finished(rnd)
    own = race_fit(rnd) if finished else None
    location = str(ev["Location"])
    total_laps = own["total_laps"] if own else SCHEDULED_LAPS.get(location, 57)
    venue = track_history.venue_history(location)
    # Pit-lane loss is a fixed property of the circuit that teams know before
    # the weekend, so the measured value is allowed here. Before the race: this
    # circuit's 2024-25 loss weighted 2:1 against the 2026 season median (best
    # on 2026 rounds: 2.2 s mean error vs 2.7 s for the median alone).
    pit_loss = (own or {}).get("pit_loss")
    if not pit_loss:
        pit_loss = (2 * venue["pit_loss"] + stats["pit_loss_median"]) / 3 if venue.get("pit_loss") else stats["pit_loss_median"]
    # Safety Cars cluster at some circuits (street tracks, walls, lap-1 pile-ups):
    # blend this circuit's earlier seasons with the 2026 season rate.
    m = VENUE_SC_PRIOR_LAPS
    sc_rate = (venue["sc_starts"] + stats["sc_rate"] * m) / (venue["laps"] + m)
    vsc_rate = (venue["vsc_starts"] + stats["vsc_rate"] * m) / (venue["laps"] + m)

    grid = qualifying_grid(YEAR, rnd)
    drivers = []
    try:
        q = load_session(YEAR, rnd, "Qualifying")
        for _, r in q.results.iterrows():
            color = str(r.get("TeamColor") or "888888")
            drivers.append({
                "driver": str(r["Abbreviation"]), "number": str(r["DriverNumber"]),
                "name": str(r.get("FullName") or r["Abbreviation"]), "team": str(r.get("TeamName") or ""),
                "color": "#" + color.lstrip("#"), "grid": grid.get(str(r["Abbreviation"])), "quali": _int_or_none(r.get("Position")),
                "headshot": str(r.get("HeadshotUrl") or "") or None,
            })
    except Exception:
        pass

    sets = tyre_sets(rnd, str(ev["EventFormat"]))
    field_sets = field_tyre_sets(sets)
    for d in drivers:
        d["tyre_sets"] = sets.get(d["driver"]) or field_sets

    # Personal wear multipliers: season history per driver + this weekend's long runs.
    team_of = {d["driver"]: d["team"] for d in drivers}
    long_runs = runs[runs["type"].isin(LONG_RUN_TYPES)] if not runs.empty else runs
    run_dicts = long_runs.to_dict("records") if not long_runs.empty else []
    model = personalise(model, season_drivers, practice_driver_multipliers(run_dicts, model, stats["transfer"]), team_of)

    runs_out = []
    for _, r in long_runs.iterrows():
        runs_out.append({k: r[k] for k in ("session", "driver", "team", "compound", "type", "laps", "level", "age_start", "age_end", "median_time", "raw_slope")} | {"ages": r["ages"], "times": r["times"]})

    return {
        "round": rnd,
        "event": str(ev["EventName"]),
        "location": location,
        "country": str(ev["Country"]),
        "date": str(pd.Timestamp(ev["EventDate"]).date()),
        "format": str(ev["EventFormat"]),
        "total_laps": int(total_laps),
        "pit_loss": float(pit_loss),
        "sc_rate": sc_rate,
        "vsc_rate": vsc_rate,
        "season_sc_rate": stats["sc_rate"],
        "season_vsc_rate": stats["vsc_rate"],
        "venue_history": venue,
        "tyre_sets": {d["driver"]: d["tyre_sets"] for d in drivers},
        "field_tyre_sets": field_sets,
        "prior_races": stats["races"],
        "transfer": stats["transfer"],
        "offset_transfer": stats["offset_transfer"],
        "compound_ratios": stats["compound_ratios"],
        "severity": weekend_severity(practice, stats["compound_ratios"]),
        "short_run_offsets": raw_offsets,
        "prior": prior.to_dict(),
        "practice": practice,
        "practice_runs": runs_out,
        "model": model.to_dict(),
        "drivers": sorted(drivers, key=lambda d: d["grid"] or 99),
    }
