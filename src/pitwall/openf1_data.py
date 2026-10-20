"""Finished sessions from OpenF1, shaped like FastF1 sessions.

F1's timing archive (livetiming.formula1.com/static) started returning 401 in
October 2026, so FastF1 can no longer load sessions that aren't already in the
local cache. OpenF1 serves the same timing data for free once a session is
over. ``OpenF1Session`` exposes the attributes the pipeline reads from a
FastF1 session (laps, results, weather_data, race_control_messages,
track_status) with FastF1's column names, so ``load_session`` can fall back to
it without the rest of the code knowing where the data came from.

Responses are cached on disk (data/raw/openf1) because finished sessions never
change and OpenF1 rate-limits free use.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.pitwall.season import PROJECT_ROOT

BASE = "https://api.openf1.org/v1"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "openf1"

# FastF1 TrackStatus codes
GREEN, YELLOW, SC, RED, VSC, VSC_ENDING = "1", "2", "4", "5", "6", "7"


class OpenF1Unavailable(RuntimeError):
    pass


def _get(endpoint: str, **params) -> list[dict]:
    key = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(2 + 3 * attempt)
            continue
        if r.status_code == 404:
            data: list[dict] = []  # OpenF1 answers 404 "No results found." for empty tables
            break
        r.raise_for_status()
        data = r.json()
        break
    else:
        raise OpenF1Unavailable(f"OpenF1 kept rate-limiting {endpoint}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(0.4)  # stay well under the free rate limit
    return data


def _ts(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601")


@lru_cache(maxsize=8)
def _sessions(year: int) -> pd.DataFrame:
    df = pd.DataFrame(_get("sessions", year=year))
    if df.empty:
        raise OpenF1Unavailable(f"OpenF1 has no sessions for {year}")
    df["start"] = _ts(df["date_start"])
    return df


def find_session_key(year: int, name: str, start_utc: pd.Timestamp) -> int:
    """The OpenF1 session with this name that starts within 3 h of the scheduled start."""
    df = _sessions(year)
    df = df[df["session_name"] == name]
    start = pd.Timestamp(start_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    gap = (df["start"] - start).abs()
    if df.empty or gap.min() > pd.Timedelta(hours=3):
        raise OpenF1Unavailable(f"No OpenF1 {name} session near {start}")
    return int(df.loc[gap.idxmin(), "session_key"])


class OpenF1Session:
    """The parts of fastf1.core.Session the pipeline uses, built from OpenF1."""

    def __init__(self, session_key: int, name: str):
        self.session_key = session_key
        self.name = name
        info = _sessions_row(session_key)
        self.t0 = _ts(pd.Series([info["date_start"]])).iloc[0]

    def load(self, **_):
        k = self.session_key
        drivers = pd.DataFrame(_get("drivers", session_key=k))
        laps = pd.DataFrame(_get("laps", session_key=k))
        if drivers.empty or laps.empty:
            raise OpenF1Unavailable(f"OpenF1 has no lap data for session {k} yet")
        drivers = drivers.drop_duplicates("driver_number", keep="last").set_index("driver_number")
        stints = pd.DataFrame(_get("stints", session_key=k))
        pits = pd.DataFrame(_get("pit", session_key=k))
        rc = pd.DataFrame(_get("race_control", session_key=k))
        weather = pd.DataFrame(_get("weather", session_key=k))
        results = pd.DataFrame(_get("session_result", session_key=k))
        positions = pd.DataFrame(_get("position", session_key=k)) if self.name in ("Race", "Sprint") else pd.DataFrame()

        self.track_status = self._track_status(rc)
        self.race_control_messages = self._rcm(rc)
        self.weather_data = self._weather(weather)
        if self.name in ("Race", "Sprint") and not results.empty and results["number_of_laps"].notna().any():
            # drop the lap some cars start after the flag (no time, not part of the race)
            laps = laps[laps["lap_number"] <= results["number_of_laps"].max()]
        self.laps = self._laps(laps, drivers, stints, pits, rc, positions)
        self.results = self._results(results, drivers, positions)
        return self

    # -- helpers -----------------------------------------------------------
    def _rel(self, dates) -> pd.Series:
        return _ts(dates) - self.t0

    def _track_status(self, rc: pd.DataFrame) -> pd.DataFrame:
        """Status changes over time, from race-control messages (FastF1 codes)."""
        rows = [(pd.Timedelta(0), GREEN)]
        if not rc.empty:
            status = GREEN
            for _, m in rc.sort_values("date").iterrows():
                msg = str(m.get("message") or "").upper()
                flag = str(m.get("flag") or "").upper() if pd.notna(m.get("flag")) else ""
                scope = m.get("scope") if pd.notna(m.get("scope")) else None
                new = None
                # race control writes both "VSC DEPLOYED" and "VIRTUAL SAFETY CAR DEPLOYED"
                if "VSC DEPLOYED" in msg or "VIRTUAL SAFETY CAR DEPLOYED" in msg:
                    new = VSC
                elif "VSC ENDING" in msg or "VIRTUAL SAFETY CAR ENDING" in msg:
                    new = VSC_ENDING
                elif "SAFETY CAR DEPLOYED" in msg:
                    new = SC
                elif flag == "RED" and scope in (None, "Track"):
                    new = RED
                elif (flag in ("GREEN", "CLEAR") and scope == "Track") or "TRACK CLEAR" in msg:
                    new = GREEN if status in (SC, VSC, VSC_ENDING, RED) else None
                if new and new != status:
                    rows.append((self._rel(pd.Series([m["date"]])).iloc[0], new))
                    status = new
        return pd.DataFrame(rows, columns=["Time", "Status"])

    def _rcm(self, rc: pd.DataFrame) -> pd.DataFrame:
        if rc.empty:
            return pd.DataFrame(columns=["Time", "Lap", "Category", "Message", "Flag"])
        rc = rc.sort_values("date")
        return pd.DataFrame({
            "Time": self._rel(rc["date"]).values,
            "Lap": rc["lap_number"].astype(float).values,
            "Category": rc["category"].astype(str).values,
            "Message": rc["message"].astype(str).values,
            "Flag": rc["flag"].values,
        })

    def _weather(self, w: pd.DataFrame) -> pd.DataFrame | None:
        if w.empty:
            return None
        w = w.sort_values("date")
        return pd.DataFrame({
            "Time": self._rel(w["date"]).values,
            "Rainfall": w["rainfall"].fillna(0).astype(bool).values,
            "TrackTemp": w["track_temperature"].astype(float).values,
            "AirTemp": w["air_temperature"].astype(float).values,
        })

    def _laps(self, laps, drivers, stints, pits, rc, positions) -> pd.DataFrame:
        laps = laps.dropna(subset=["lap_number"]).copy()
        laps["start"] = _ts(laps["date_start"])
        laps = laps.sort_values(["driver_number", "lap_number"])
        # A lap ends when the next one starts; fall back to the measured duration.
        nxt = laps.groupby("driver_number")["start"].shift(-1)
        dur = pd.to_timedelta(laps["lap_duration"], unit="s")
        laps["end"] = (laps["start"] + dur).fillna(nxt)

        pit_in = set()
        if not pits.empty:
            pit_in = {(int(d), int(l)) for d, l in zip(pits["driver_number"], pits["lap_number"]) if pd.notna(l)}

        deleted = set()
        for msg in (rc["message"] if not rc.empty else []):
            m = re.search(r"CAR (\d+) .*?(?:TIME [\d:.]+|LAP) DELETED.*?\bLAP (\d+)\b", str(msg))
            if m:
                deleted.add((int(m.group(1)), int(m.group(2))))

        ts = self.track_status
        pos_by_driver = {d: g.sort_values("date") for d, g in positions.groupby("driver_number")} if not positions.empty else {}

        rows = []
        for r in laps.itertuples():
            d, lap = int(r.driver_number), int(r.lap_number)
            drv = drivers.loc[d] if d in drivers.index else None
            st = None
            if not stints.empty:
                mine = stints[stints["driver_number"] == d].sort_values("stint_number")
                hit = mine[(mine["lap_start"] <= lap) & (mine["lap_end"] >= lap)]
                # OpenF1 stints stop one lap short for cars that retire; that lap is still on the last set
                st = hit.iloc[0] if len(hit) else (mine.iloc[-1] if len(mine) and lap > mine["lap_end"].max() else None)
            start_rel = r.start - self.t0 if pd.notna(r.start) else pd.NaT
            end_rel = r.end - self.t0 if pd.notna(r.end) else pd.NaT
            # statuses in force at any point during the lap, like FastF1's TrackStatus
            if pd.notna(start_rel) and pd.notna(end_rel):
                before = ts[ts["Time"] <= start_rel]
                during = ts[(ts["Time"] > start_rel) & (ts["Time"] < end_rel)]
                codes = ([before.iloc[-1]["Status"]] if len(before) else []) + during["Status"].tolist()
                track = "".join(dict.fromkeys(codes))
            else:
                track = ""
            position = np.nan
            if self.name in ("Race", "Sprint") and d in pos_by_driver and pd.notna(r.end):
                p = pos_by_driver[d]
                p = p[_ts(p["date"]) <= r.end]
                if len(p):
                    position = float(p.iloc[-1]["position"])
            lap_time = float(r.lap_duration) if pd.notna(r.lap_duration) else np.nan
            out_lap = bool(r.is_pit_out_lap) if pd.notna(r.is_pit_out_lap) else False
            in_lap = (d, lap) in pit_in
            rows.append({
                "Driver": str(drv["name_acronym"]) if drv is not None else str(d),
                "DriverNumber": str(d),
                "Team": str(drv["team_name"]) if drv is not None else "",
                "LapNumber": lap,
                "LapTime": pd.to_timedelta(lap_time, unit="s") if not np.isnan(lap_time) else pd.NaT,
                "Compound": str(st["compound"]).upper() if st is not None and pd.notna(st["compound"]) else None,
                "TyreLife": float(st["tyre_age_at_start"] + (lap - st["lap_start"]) + 1) if st is not None and pd.notna(st["tyre_age_at_start"]) else np.nan,
                "FreshTyre": bool(st["tyre_age_at_start"] == 0) if st is not None and pd.notna(st["tyre_age_at_start"]) else pd.NA,
                "Stint": int(st["stint_number"]) if st is not None else pd.NA,
                "PitInTime": end_rel if in_lap else pd.NaT,
                "PitOutTime": start_rel if out_lap else pd.NaT,
                "TrackStatus": track,
                "Position": position,
                # FastF1 flags laps with timing problems, in/out laps and the first lap as inaccurate.
                "IsAccurate": bool(not np.isnan(lap_time) and not in_lap and not out_lap and lap > 1),
                "Deleted": (d, lap) in deleted,
                "Time": end_rel,
                "LapStartTime": start_rel,
            })
        return pd.DataFrame(rows)

    def _results(self, res: pd.DataFrame, drivers: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
        grid = {}
        if self.name in ("Race", "Sprint") and not positions.empty:
            # the first position reading of each car is its grid slot (after penalties)
            first = positions.sort_values("date").drop_duplicates("driver_number", keep="first")
            grid = dict(zip(first["driver_number"].astype(int), first["position"]))
        rows = []
        lead_laps = res["number_of_laps"].max() if not res.empty and "number_of_laps" in res else None
        source = res if not res.empty else pd.DataFrame({"driver_number": drivers.index})
        # OpenF1 leaves retired cars unclassified; FastF1 places them after the
        # finishers by laps completed, which the pipeline expects.
        if not res.empty and "position" in res:
            placed = int(res["position"].max()) if res["position"].notna().any() else 0
            unplaced = res[res["position"].isna()].sort_values("number_of_laps", ascending=False)
            fill = {int(d): placed + i + 1 for i, d in enumerate(unplaced["driver_number"])}
            source = res.assign(position=[p if pd.notna(p) else fill.get(int(d)) for p, d in zip(res["position"], res["driver_number"])])
        for _, r in source.iterrows():
            d = int(r["driver_number"])
            drv = drivers.loc[d] if d in drivers.index else None
            if r.get("dsq") is True:
                status = "Disqualified"
            elif r.get("dns") is True:
                status = "Did not start"
            elif r.get("dnf") is True:
                status = "Retired"
            elif lead_laps and pd.notna(r.get("number_of_laps")) and r["number_of_laps"] < lead_laps:
                status = "Lapped"  # FastF1's label for classified cars a lap or more down
            else:
                status = "Finished"
            rows.append({
                "Abbreviation": str(drv["name_acronym"]) if drv is not None else str(d),
                "DriverNumber": str(d),
                "FullName": str(drv["full_name"]) if drv is not None else "",
                "TeamName": str(drv["team_name"]) if drv is not None else "",
                "TeamColor": str(drv["team_colour"] or "") if drv is not None else "",
                "HeadshotUrl": str(drv.get("headshot_url") or "") if drv is not None else "",
                "GridPosition": float(grid.get(d, np.nan)),
                "Position": float(r["position"]) if pd.notna(r.get("position")) else np.nan,
                "Status": status,
                "Points": float(r["points"]) if pd.notna(r.get("points")) else 0.0,
            })
        return pd.DataFrame(rows)


def _sessions_row(session_key: int) -> dict:
    rows = _get("sessions", session_key=session_key)
    if not rows:
        raise OpenF1Unavailable(f"Unknown OpenF1 session {session_key}")
    return rows[0]


def load_openf1_session(year: int, name: str, start_utc: pd.Timestamp) -> OpenF1Session:
    return OpenF1Session(find_session_key(year, name, start_utc), name).load()
