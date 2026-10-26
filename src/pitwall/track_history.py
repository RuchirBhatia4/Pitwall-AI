"""What each circuit was like in earlier seasons: overtaking and neutralisations.

Teams go into a weekend knowing how hard a track is to pass on and how often
it brings out a Safety Car. One season of the 2026 cars can't tell us that for
a circuit before its race, but the 2024 and 2025 races there can.

    python -m src.pitwall.track_history     # writes data/season/track_history.json

Per circuit: the overtaking intercept of every earlier race (see
race_sim.fit_pass_model, fitted jointly so all races share one pace slope),
the Safety Car / VSC starts per lap raced, and the green-flag pit-lane loss.
"""
from __future__ import annotations

import json

import numpy as np

from src.pitwall.race_data import load_session, measure_pit_loss, neutralised_laps, normalize_laps
from src.pitwall.race_sim import fit_pass_model, pass_pairs
from src.pitwall.season import SEASON_DIR, get_schedule
from src.pitwall.tyre_model import fit_race_model

HISTORY_YEARS = (2024, 2025)
OUT = SEASON_DIR / "track_history.json"
# Same circuit, different calendar spelling between seasons.
ALIASES = {"Miami": "Miami Gardens", "Monaco": "Monte Carlo", "Yas Island": "Yas Marina"}


def venue(location: str) -> str:
    return ALIASES.get(location, location)


def _starts(laps: list[int]) -> int:
    return sum(1 for i, lap in enumerate(laps) if i == 0 or laps[i - 1] != lap - 1)


def race_summary(year: int, rnd: int) -> tuple[dict, np.ndarray] | None:
    session = load_session(year, rnd, "Race")
    laps = normalize_laps(session)
    if laps.empty:
        return None
    neutral = neutralised_laps(laps)
    wet = laps["compound"].isin(["INTERMEDIATE", "WET"]).mean() > 0.15
    summary = {"year": year, "round": rnd, "laps": int(laps["lap"].max()), "wet": bool(wet),
               "sc_starts": _starts(neutral["sc"]), "vsc_starts": _starts(neutral["vsc"]), "pit_loss": measure_pit_loss(laps, neutral)}
    if wet:
        return summary, np.zeros((0, 2))
    model = fit_race_model(laps, neutral)
    if model is None:
        return summary, np.zeros((0, 2))
    blocked = set(neutral["sc"]) | set(neutral["vsc"]) | set(neutral["red"])
    return summary, pass_pairs(laps, blocked, model.to_dict())


def build() -> dict:
    wanted = {venue(str(loc)) for loc in get_schedule(2026)["Location"]}
    races, pairs = [], {}
    for year in HISTORY_YEARS:
        for ev in get_schedule(year).itertuples():
            loc = venue(str(ev.Location))
            if loc not in wanted:
                continue
            try:
                got = race_summary(year, int(ev.RoundNumber))
            except Exception as exc:  # missing session, OpenF1 outage
                print(f"{year} {loc}: skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
            if got is None:
                continue
            summary, p = got
            key = year * 100 + int(ev.RoundNumber)
            summary["venue"] = loc
            summary["pairs"] = len(p)
            races.append(summary)
            if len(p):
                pairs[key] = p
            print(f"{year} {loc}: {len(p)} close-following laps, SC {summary['sc_starts']}, VSC {summary['vsc_starts']}", flush=True)
    fit = fit_pass_model(pairs)
    if fit:
        mean_a, slope, intercepts = fit
        for r in races:
            key = r["year"] * 100 + r["round"]
            if key in intercepts:
                r["pass_intercept"] = intercepts[key]
    out = {"years": list(HISTORY_YEARS), "pass_slope": fit[1] if fit else None, "pass_intercept_mean": fit[0] if fit else None, "races": races}
    OUT.write_text(json.dumps(out, indent=1))
    return out


def load() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {"races": []}


def venue_history(location: str, history: dict | None = None) -> dict:
    """Earlier-season evidence for one circuit: pass difficulty relative to the
    average track of those seasons, and its Safety Car / VSC record."""
    history = history if history is not None else load()
    loc = venue(location)
    races = [r for r in history.get("races", []) if r["venue"] == loc]
    ints = [r["pass_intercept"] for r in races if "pass_intercept" in r]
    mean = history.get("pass_intercept_mean")
    return {
        "races": len(races),
        "pass_offset": float(np.mean(ints) - mean) if ints and mean is not None else None,
        "pass_races": len(ints),
        "pit_loss": float(np.median(losses)) if (losses := [r["pit_loss"] for r in races if r.get("pit_loss")]) else None,
        "sc_starts": sum(r["sc_starts"] for r in races),
        "vsc_starts": sum(r["vsc_starts"] for r in races),
        "laps": sum(r["laps"] for r in races),
    }


if __name__ == "__main__":
    build()
