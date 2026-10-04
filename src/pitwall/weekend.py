"""Build the pre-race model for one race weekend (walk-forward, no leakage).

Everything about round R is estimated from:
  * race fits of rounds < R (season prior, SC rates, pit-loss fallback),
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
    event_sessions,
    load_session,
    measure_pit_loss,
    neutralised_laps,
    normalize_laps,
    qualifying_grid,
    stints_from_laps,
)
from src.pitwall.season import SEASON_DIR, get_schedule
from src.pitwall.tyre_model import (
    TyreModel,
    summarise_practice,
    fit_race_model,
    fit_transfer,
    practice_long_runs,
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
PRACTICE_SESSIONS = ("Practice 1", "Practice 2", "Practice 3", "Sprint")
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
        return json.loads(path.read_text())
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
        "round": rnd,
        "total_laps": int(laps["lap"].max()),
        "pit_loss": measure_pit_loss(laps, neutral),
        "wet": bool(wet),
        "sc_starts": neutral_starts["sc"],
        "vsc_starts": neutral_starts["vsc"],
        "model": model.to_dict() if model else None,
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
    models = [TyreModel.from_dict(f["model"]) for f in fits if f.get("model")]
    prior = season_prior(models)
    laps_total = sum(f["total_laps"] for f in fits) or 1
    pit_losses = [f["pit_loss"] for f in fits if f.get("pit_loss")]
    pairs = []
    for f in fits:
        if not f.get("model"):
            continue
        for c, pc in f.get("practice", {}).get("compounds", {}).items():
            rc = f["model"]["compounds"].get(c)
            if rc and pc["laps"] >= 15:
                pairs.append((pc["deg"], rc["deg"], min(pc["laps"], rc["evidence_laps"]) ** 0.5))
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
    for f in fits:
        mc = (f.get("model") or {}).get("compounds", {})
        if "MEDIUM" in mc and mc["MEDIUM"]["deg"] > 0.02:
            for c, v in mc.items():
                ratio_obs.setdefault(c, []).append(v["deg"] / mc["MEDIUM"]["deg"])
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
        r = practice_long_runs(laps, name, min_laps=5 if name != "Sprint" else 6)
        if not r.empty:
            runs.append(r)
    return (pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()), all_laps


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
    # Pit-lane loss is a fixed property of the circuit that teams know before
    # the weekend, so the measured value is allowed here; otherwise fall back
    # to the season median.
    pit_loss = (own or {}).get("pit_loss") or stats["pit_loss_median"]

    grid = qualifying_grid(YEAR, rnd)
    drivers = []
    try:
        q = load_session(YEAR, rnd, "Qualifying")
        for _, r in q.results.iterrows():
            color = str(r.get("TeamColor") or "888888")
            drivers.append({
                "driver": str(r["Abbreviation"]), "number": str(r["DriverNumber"]),
                "name": str(r.get("FullName") or r["Abbreviation"]), "team": str(r.get("TeamName") or ""),
                "color": "#" + color.lstrip("#"), "grid": grid.get(str(r["Abbreviation"])),
                "headshot": str(r.get("HeadshotUrl") or "") or None,
            })
    except Exception:
        pass

    # Personal wear multipliers: season history per driver + this weekend's long runs.
    team_of = {d["driver"]: d["team"] for d in drivers}
    run_dicts = runs.to_dict("records") if not runs.empty else []
    model = personalise(model, season_drivers, practice_driver_multipliers(run_dicts, model, stats["transfer"]), team_of)

    runs_out = []
    if not runs.empty:
        for _, r in runs.iterrows():
            runs_out.append({k: r[k] for k in ("session", "driver", "team", "compound", "laps", "age_start", "age_end", "median_time", "raw_slope")} | {"ages": r["ages"], "times": r["times"]})

    return {
        "round": rnd,
        "event": str(ev["EventName"]),
        "location": location,
        "country": str(ev["Country"]),
        "date": str(pd.Timestamp(ev["EventDate"]).date()),
        "format": str(ev["EventFormat"]),
        "total_laps": int(total_laps),
        "pit_loss": float(pit_loss),
        "sc_rate": stats["sc_rate"],
        "vsc_rate": stats["vsc_rate"],
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
