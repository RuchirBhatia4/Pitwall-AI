"""Load and normalise FastF1 session data for the strategy engine.

Every function returns plain pandas/python structures with stable, lowercase
column names so the tyre model, optimizer and API never touch FastF1 objects.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.pitwall.season import enable_cache, get_schedule

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

# FastF1 TrackStatus codes (a lap can carry several, e.g. "264").
STATUS_SC = "4"
STATUS_RED = "5"
STATUS_VSC = ("6", "7")


@lru_cache(maxsize=96)
def load_session(year: int, rnd: int, identifier: str):
    """Load a FastF1 session (laps, results, weather, race control). Cached in-process."""
    enable_cache()
    import fastf1

    session = fastf1.get_session(year, rnd, identifier)
    session.load(telemetry=False, weather=True, messages=True)
    return session


def event_sessions(year: int, rnd: int) -> dict[str, pd.Timestamp]:
    """Map session name -> UTC start for one event (e.g. {'Practice 1': ..., 'Race': ...})."""
    schedule = get_schedule(year)
    row = schedule.loc[schedule["RoundNumber"] == rnd].iloc[0]
    out: dict[str, pd.Timestamp] = {}
    for i in range(1, 6):
        name = row[f"Session{i}"]
        if name:
            out[str(name)] = row[f"Session{i}DateUtc"]
    return out


def _seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def normalize_laps(session) -> pd.DataFrame:
    """Return one row per driver-lap with numeric, lowercase columns."""
    raw = session.laps
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    laps = pd.DataFrame(
        {
            "driver": raw["Driver"].astype(str),
            "number": raw["DriverNumber"].astype(str),
            "team": raw["Team"].astype(str),
            "lap": raw["LapNumber"].astype("Int64"),
            "lap_time": _seconds(raw["LapTime"]),
            "compound": raw["Compound"].fillna("UNKNOWN").astype(str).str.upper(),
            "tyre_life": raw["TyreLife"].astype(float),
            "fresh": raw["FreshTyre"].astype("boolean"),
            "stint": raw["Stint"].astype("Int64"),
            "pit_in": raw["PitInTime"].notna(),
            "pit_out": raw["PitOutTime"].notna(),
            "track_status": raw["TrackStatus"].fillna("").astype(str),
            "position": raw["Position"].astype(float),
            "accurate": raw["IsAccurate"].astype(bool),
            "deleted": raw["Deleted"].fillna(False).astype(bool),
            "lap_end": _seconds(raw["Time"]),
            "lap_start": _seconds(raw["LapStartTime"]),
        }
    )
    laps = laps.dropna(subset=["lap"]).copy()
    laps["lap"] = laps["lap"].astype(int)
    laps["stint"] = laps["stint"].fillna(0).astype(int)
    return laps.sort_values(["driver", "lap"]).reset_index(drop=True)


def neutralised_laps(laps: pd.DataFrame) -> dict[str, list[int]]:
    """Laps run (at least partly) under Safety Car, VSC or red flag."""
    by_lap = laps.groupby("lap")["track_status"].apply(lambda s: "".join(s))
    sc = sorted(int(lap) for lap, st in by_lap.items() if STATUS_SC in st)
    vsc = sorted(int(lap) for lap, st in by_lap.items() if any(code in st for code in STATUS_VSC))
    red = sorted(int(lap) for lap, st in by_lap.items() if STATUS_RED in st)
    return {"sc": sc, "vsc": vsc, "red": red}


def _periods(laps_list: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted lap numbers into (start, end) periods."""
    periods: list[tuple[int, int]] = []
    for lap in laps_list:
        if periods and lap == periods[-1][1] + 1:
            periods[-1] = (periods[-1][0], lap)
        else:
            periods.append((lap, lap))
    return periods


def stints_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver stint: compound, first/last lap, tyre age at the start."""
    if laps.empty:
        return pd.DataFrame(columns=["driver", "stint", "compound", "start_lap", "end_lap", "laps", "age_start", "fresh"])
    rows = []
    for (driver, stint), grp in laps.groupby(["driver", "stint"], sort=True):
        grp = grp.sort_values("lap")
        compound = grp["compound"].mode().iloc[0] if not grp["compound"].mode().empty else "UNKNOWN"
        first = grp.iloc[0]
        age_start = float(first["tyre_life"]) - 1 if pd.notna(first["tyre_life"]) else 0.0
        rows.append(
            {
                "driver": driver,
                "stint": int(stint),
                "compound": compound,
                "start_lap": int(grp["lap"].min()),
                "end_lap": int(grp["lap"].max()),
                "laps": int(grp["lap"].max() - grp["lap"].min() + 1),
                "age_start": max(0.0, age_start),
                "fresh": bool(first["fresh"]) if pd.notna(first["fresh"]) else True,
            }
        )
    return pd.DataFrame(rows)


def driver_table(session) -> list[dict]:
    """Driver metadata + classification from session.results."""
    res = session.results
    out = []
    for _, r in res.iterrows():
        color = str(r.get("TeamColor") or "888888")
        out.append(
            {
                "driver": str(r["Abbreviation"]),
                "number": str(r["DriverNumber"]),
                "name": str(r.get("FullName") or r["Abbreviation"]),
                "team": str(r.get("TeamName") or ""),
                "color": f"#{color}" if not color.startswith("#") else color,
                "headshot": str(r.get("HeadshotUrl") or "") or None,
                "grid": _int_or_none(r.get("GridPosition")),
                "position": _int_or_none(r.get("Position")),
                "status": str(r.get("Status") or ""),
                "points": float(r["Points"]) if pd.notna(r.get("Points")) else 0.0,
            }
        )
    return out


def _int_or_none(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        value = int(value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def qualifying_grid(year: int, rnd: int) -> dict[str, int]:
    """Driver -> qualifying position (used as the predicted grid before the race)."""
    try:
        q = load_session(year, rnd, "Qualifying")
    except Exception:
        return {}
    grid = {}
    for _, r in q.results.iterrows():
        pos = _int_or_none(r.get("Position"))
        if pos:
            grid[str(r["Abbreviation"])] = pos
    return grid


def measure_pit_loss(laps: pd.DataFrame, neutral: dict[str, list[int]]) -> float | None:
    """Median green-flag pit loss: (in-lap + out-lap) minus two representative laps."""
    blocked = set(neutral["sc"]) | set(neutral["vsc"]) | set(neutral["red"])
    losses = []
    for driver, grp in laps.groupby("driver"):
        grp = grp.set_index("lap").sort_index()
        clean = grp[(~grp["pit_in"]) & (~grp["pit_out"]) & grp["lap_time"].notna()]
        clean = clean[~clean.index.isin(blocked)]
        if len(clean) < 8:
            continue
        for lap in grp.index[grp["pit_in"]]:
            out_lap = lap + 1
            if out_lap not in grp.index or not grp.loc[out_lap, "pit_out"]:
                continue
            if lap in blocked or out_lap in blocked or lap <= 1:
                continue
            t_in, t_out = grp.loc[lap, "lap_time"], grp.loc[out_lap, "lap_time"]
            if pd.isna(t_in) or pd.isna(t_out):
                continue
            nearby = clean[(clean.index >= lap - 4) & (clean.index <= lap + 5)]["lap_time"]
            if len(nearby) < 3:
                continue
            losses.append(t_in + t_out - 2 * float(nearby.median()))
    if len(losses) < 3:
        return None
    arr = np.array(losses)
    arr = arr[(arr > 10) & (arr < 45)]  # drop slow stops / drive-throughs / timing glitches
    return float(np.median(arr)) if len(arr) else None


def race_overview(year: int, rnd: int) -> dict:
    """Everything the UI needs about a finished race (results, stints, neutralisations)."""
    session = load_session(year, rnd, "Race")
    laps = normalize_laps(session)
    neutral = neutralised_laps(laps)
    stints = stints_from_laps(laps)
    total_laps = int(laps["lap"].max()) if not laps.empty else 0
    drivers = driver_table(session)
    stint_map: dict[str, list[dict]] = {}
    for d, grp in stints.groupby("driver"):
        stint_map[d] = grp.sort_values("stint").drop(columns=["driver"]).to_dict("records")
    weather = session.weather_data
    rain = bool(weather["Rainfall"].any()) if weather is not None and len(weather) else False
    return {
        "total_laps": total_laps,
        "drivers": drivers,
        "stints": stint_map,
        "neutralised": {k: _periods(v) for k, v in neutral.items()},
        "pit_loss": measure_pit_loss(laps, neutral),
        "rain": rain,
        "track_temp": float(weather["TrackTemp"].median()) if weather is not None and len(weather) else None,
        "air_temp": float(weather["AirTemp"].median()) if weather is not None and len(weather) else None,
    }
