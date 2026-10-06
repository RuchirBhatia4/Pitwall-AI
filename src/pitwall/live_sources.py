"""Live race-state sources, all normalised to one RaceState dict.

RaceState = {
  "source", "round", "event", "lap", "total_laps", "track_status",
  "updated_at", "cars": [CarState...], "race_control": [...], "weather": {...}
}
CarState = {
  "driver", "number", "team", "color", "position", "gap_to_leader", "interval",
  "laps_completed", "compound", "tyre_age", "stint", "stops", "compounds_used",
  "in_pit", "retired", "last_lap", "laps": [{"lap","time","compound","age","pit_in","pit_out","neutral"}]
}

Sources
  ReplaySource   - any finished 2026 race, frozen at the end of a chosen lap
                   (FastF1). Used for demos/backtests of the live engine.
  OpenF1Source   - api.openf1.org. Free for finished sessions; live sessions
                   need an OpenF1 sponsor account (OPENF1_USERNAME/PASSWORD).
  SignalRSource  - F1's own live timing feed via FastF1 (needs an F1TV
                   subscription token; see src/pitwall/f1tv_login.py).
  manual_state   - one car typed in by the user; always works.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import threading
import time
import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.pitwall.race_data import load_session, normalize_laps
from src.pitwall.weekend import YEAR, event_row

log = logging.getLogger("pitwall.live")
STATUS_MAP = {"1": "GREEN", "2": "YELLOW", "4": "SC", "5": "RED", "6": "VSC", "7": "VSC_ENDING"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rank_cars(cars: list[dict]) -> list[dict]:
    """Order by laps completed then gap; fill interval to the car ahead."""
    running = [c for c in cars if not c["retired"]]
    if running and all(c.get("position") for c in running):
        # The source gives official positions (F1 live timing): trust them.
        running.sort(key=lambda c: c["position"])
    else:
        running.sort(key=lambda c: (-c["laps_completed"], c["gap_to_leader"] if c["gap_to_leader"] is not None else 1e6))
    retired = sorted((c for c in cars if c["retired"]), key=lambda c: -c["laps_completed"])
    ordered = running + retired
    for i, c in enumerate(ordered):
        c["position"] = i + 1
        if i == 0 or c["retired"]:
            c["interval"] = None if i else 0.0
            continue
        ahead = ordered[i - 1]
        # Gaps are measured at the timing line, so cars one lap-count apart (the
        # car ahead has crossed, we have not yet) still give a valid interval.
        if c["gap_to_leader"] is not None and ahead["gap_to_leader"] is not None and abs(ahead["laps_completed"] - c["laps_completed"]) <= 1:
            c["interval"] = round(c["gap_to_leader"] - ahead["gap_to_leader"], 3)
        elif c.get("interval") is None:
            c["interval"] = None
    return ordered


# ---------------------------------------------------------------------------
# Replay (FastF1, finished races)
# ---------------------------------------------------------------------------
class ReplaySource:
    def __init__(self, rnd: int):
        self.round = rnd
        self.session = load_session(YEAR, rnd, "Race")
        self.laps = normalize_laps(self.session)
        self.total_laps = int(self.laps["lap"].max())
        res = self.session.results
        self.meta = {
            str(r["Abbreviation"]): {
                "team": str(r["TeamName"]),
                "color": "#" + str(r["TeamColor"] or "888888").lstrip("#"),
                "status": str(r["Status"]),
                "number": str(r["DriverNumber"]),
            }
            for _, r in res.iterrows()
        }
        by_lap = self.laps.groupby("lap")["track_status"].apply(lambda s: "".join(s))
        self.neutral_laps = {int(l) for l, st in by_lap.items() if any(c in st for c in "4567")}
        leader = self.laps.sort_values("lap_end").drop_duplicates("lap", keep="first")
        self.leader_end = dict(zip(leader["lap"], leader["lap_end"]))
        self.ts = self.session.track_status
        self.rcm = self.session.race_control_messages
        self.wx = self.session.weather_data if self.session.weather_data is not None and len(self.session.weather_data) else None
        self.rain_laps: set[int] = set()
        if self.wx is not None:
            ends = sorted(self.leader_end.items())
            for t in self.wx.loc[self.wx["Rainfall"].astype(bool), "Time"].dt.total_seconds():
                lap = next((l for l, e in ends if e >= t), None)
                if lap is not None:
                    self.rain_laps.add(int(lap))

    def state(self, lap: int) -> dict:
        lap = int(np.clip(lap, 1, self.total_laps))
        T = self.leader_end.get(lap)
        cars = []
        for drv, grp in self.laps.groupby("driver"):
            grp = grp.sort_values("lap")
            done = grp[grp["lap_end"] <= T + 1e-6]
            k = int(done["lap"].max()) if len(done) else 0
            nxt = grp[grp["lap"] == k + 1]
            row = nxt.iloc[0] if len(nxt) else (done.iloc[-1] if len(done) else grp.iloc[0])
            age = float(row["tyre_life"]) - (1 if len(nxt) else 0) if pd.notna(row["tyre_life"]) else 0.0
            meta = self.meta.get(drv, {})
            finished_status = meta.get("status", "")
            retired = (grp["lap"].max() <= k and k < self.total_laps and not (finished_status in ("Finished", "Lapped") or finished_status.startswith("+")))
            gap = None
            if k > 0 and len(done):
                lead_t = self.leader_end.get(k)
                if lead_t is not None:
                    gap = round(float(done.iloc[-1]["lap_end"] - lead_t), 3)
            stints_seen = done["stint"].tolist() + ([int(row["stint"])] if len(nxt) else [])
            comps = []
            for c in list(done["compound"]) + ([row["compound"]] if len(nxt) else []):
                if c not in comps and c != "UNKNOWN":
                    comps.append(c)
            cars.append(
                {
                    "driver": drv,
                    "number": meta.get("number", ""),
                    "team": meta.get("team", str(row["team"])),
                    "color": meta.get("color", "#888888"),
                    "position": None,
                    "gap_to_leader": gap,
                    "interval": None,
                    "laps_completed": k,
                    "compound": str(row["compound"]),
                    "tyre_age": max(0, int(round(age))),
                    "stint": int(row["stint"]),
                    "stops": max(0, len(set(stints_seen)) - 1),
                    "compounds_used": comps,
                    "in_pit": bool(len(done) and done.iloc[-1]["pit_in"]),
                    "retired": bool(retired),
                    "last_lap": float(done.iloc[-1]["lap_time"]) if len(done) and pd.notna(done.iloc[-1]["lap_time"]) else None,
                    "laps": [
                        {
                            "lap": int(r.lap),
                            "time": float(r.lap_time) if pd.notna(r.lap_time) else None,
                            "compound": r.compound,
                            "age": int(r.tyre_life) if pd.notna(r.tyre_life) else None,
                            "pit_in": bool(r.pit_in),
                            "pit_out": bool(r.pit_out),
                            "neutral": int(r.lap) in self.neutral_laps,
                        }
                        for r in done.itertuples()
                    ],
                }
            )
        status = "GREEN"
        if self.ts is not None and len(self.ts):
            prior = self.ts[self.ts["Time"].dt.total_seconds() <= T]
            if len(prior):
                status = STATUS_MAP.get(str(prior.iloc[-1]["Status"]), "GREEN")
        rc = []
        if self.rcm is not None and len(self.rcm):
            m = self.rcm[self.rcm["Lap"].fillna(0) <= lap]
            for _, r in m.tail(12).iterrows():
                rc.append({"lap": int(r["Lap"]) if pd.notna(r["Lap"]) else None, "category": str(r["Category"]), "message": str(r["Message"]), "flag": None if pd.isna(r.get("Flag")) else str(r.get("Flag"))})
        ev = event_row(self.round)
        return {
            "source": "replay",
            "round": self.round,
            "event": str(ev["EventName"]),
            "location": str(ev["Location"]),
            "lap": min(lap + 1, self.total_laps),
            "laps_completed_leader": lap,
            "total_laps": self.total_laps,
            "track_status": status,
            "updated_at": _now(),
            "cars": _rank_cars(cars),
            "race_control": rc[::-1],
            "weather": self._weather(T, lap),
        }

    def _weather(self, T: float, lap: int) -> dict:
        if self.wx is None:
            return {}
        prior = self.wx[self.wx["Time"].dt.total_seconds() <= T]
        if not len(prior):
            return {}
        w = prior.iloc[-1]
        return {"rainfall": bool(w["Rainfall"]), "track_temp": float(w["TrackTemp"]), "air_temp": float(w["AirTemp"]),
                "rain_laps": sorted(l for l in self.rain_laps if l <= lap)}


# ---------------------------------------------------------------------------
# OpenF1
# ---------------------------------------------------------------------------
class OpenF1Source:
    BASE = "https://api.openf1.org/v1"
    TOKEN_URL = "https://api.openf1.org/token"

    def __init__(self, rnd: int, session_key: int | None = None, total_laps: int | None = None):
        self.round = rnd
        self.total_laps = total_laps
        self._token: str | None = os.environ.get("OPENF1_TOKEN") or None
        self._token_exp = 0.0
        self.session_key = session_key or self._find_session_key()
        self._cache: dict[str, Any] = {}
        self._last_fetch: dict[str, float] = {}

    # auth -------------------------------------------------------------------
    def _auth_header(self) -> dict:
        user, pw = os.environ.get("OPENF1_USERNAME"), os.environ.get("OPENF1_PASSWORD")
        if user and pw and time.time() > self._token_exp - 60:
            try:
                r = requests.post(self.TOKEN_URL, data={"username": user, "password": pw}, timeout=10)
                r.raise_for_status()
                body = r.json()
                self._token = body["access_token"]
                self._token_exp = time.time() + float(body.get("expires_in", 3600))
            except Exception as exc:
                log.warning("OpenF1 token request failed: %s", exc)
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _get(self, endpoint: str, **params) -> list[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        r = requests.get(f"{self.BASE}/{endpoint}", params=params, headers=self._auth_header(), timeout=15)
        if r.status_code == 401 or r.status_code == 403:
            raise PermissionError(
                "OpenF1 refused the request (live sessions need an OpenF1 sponsor account: set OPENF1_USERNAME and OPENF1_PASSWORD)."
            )
        if r.status_code == 429:
            raise RuntimeError("OpenF1 rate limit reached; slowing down.")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def _find_session_key(self) -> int:
        ev = event_row(self.round)
        start = pd.Timestamp(ev["Session5DateUtc"])
        sessions = self._get("sessions", year=YEAR, session_name="Race")
        best = None
        for s in sessions:
            ds = pd.Timestamp(s["date_start"]).tz_convert(None)
            if abs((ds - start).total_seconds()) < 36 * 3600:
                best = s
        if not best:
            raise LookupError(f"No OpenF1 race session found for round {self.round}")
        return int(best["session_key"])

    def _fetch(self, endpoint: str, every: float) -> list[dict]:
        if endpoint in self._cache and time.time() - self._last_fetch.get(endpoint, 0) < every:
            return self._cache[endpoint]
        data = self._get(endpoint, session_key=self.session_key)
        self._cache[endpoint] = data
        self._last_fetch[endpoint] = time.time()
        return data

    # state --------------------------------------------------------------------
    def state(self) -> dict:
        drivers = self._fetch("drivers", 3600)
        laps = self._fetch("laps", 15)
        stints = self._fetch("stints", 15)
        pits = self._fetch("pit", 20)
        rc = self._fetch("race_control", 8)
        intervals = self._fetch("intervals", 8)
        weather = self._fetch("weather", 60)
        return build_openf1_state(self.round, drivers, laps, stints, pits, rc, intervals, weather, self.total_laps)


def _track_status_from_rc(rc: list[dict]) -> tuple[str, set[int]]:
    status = "GREEN"
    neutral_laps: set[int] = set()
    current_start = None
    for m in sorted(rc, key=lambda m: m.get("date") or ""):
        msg = (m.get("message") or "").upper()
        flag = (m.get("flag") or "").upper()
        lap = m.get("lap_number")
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg:
            status, current_start = "VSC", lap
        elif "VIRTUAL SAFETY CAR ENDING" in msg:
            status = "VSC_ENDING"
        elif "SAFETY CAR DEPLOYED" in msg:
            status, current_start = "SC", lap
        elif "SAFETY CAR IN THIS LAP" in msg:
            status = "SC"  # still neutralised until the line
        elif flag == "RED":
            status, current_start = "RED", lap
        elif flag == "GREEN" or "TRACK CLEAR" in msg or flag == "CLEAR" and m.get("scope") == "Track":
            if status in ("SC", "VSC", "VSC_ENDING", "RED") and current_start and lap:
                neutral_laps |= set(range(int(current_start), int(lap) + 1))
            status, current_start = "GREEN", None
        elif flag == "CHEQUERED":
            status = "FINISHED"
    if status in ("SC", "VSC", "VSC_ENDING", "RED") and current_start:
        neutral_laps |= set(range(int(current_start), int(current_start) + 50))
    return status, neutral_laps


def _parse_gap(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None  # "+1 LAP" etc.


def build_openf1_state(rnd, drivers, laps, stints, pits, rc, intervals, weather, total_laps=None) -> dict:
    ev = event_row(rnd)
    status, neutral = _track_status_from_rc(rc)
    latest_int: dict[int, dict] = {}
    for it in intervals:
        latest_int[it["driver_number"]] = it  # API returns ascending by date
    by_num_laps: dict[int, list[dict]] = {}
    for l in laps:
        by_num_laps.setdefault(l["driver_number"], []).append(l)
    by_num_stints: dict[int, list[dict]] = {}
    for s in stints:
        by_num_stints.setdefault(s["driver_number"], []).append(s)
    pit_laps: dict[int, set[int]] = {}
    for p in pits:
        if p.get("lap_number"):
            pit_laps.setdefault(p["driver_number"], set()).add(int(p["lap_number"]))
    leader_laps = 0
    cars = []
    for d in drivers:
        num = d["driver_number"]
        dl = sorted(by_num_laps.get(num, []), key=lambda l: l["lap_number"])
        completed = [l for l in dl if l.get("lap_duration") is not None]
        k = max((l["lap_number"] for l in completed), default=0)
        leader_laps = max(leader_laps, k)
        st = sorted(by_num_stints.get(num, []), key=lambda s: s["stint_number"])
        cur = st[-1] if st else {}
        comp = str(cur.get("compound") or "UNKNOWN").upper()
        age0 = int(cur.get("tyre_age_at_start") or 0)
        lap_start = int(cur.get("lap_start") or 1)
        tyre_age = age0 + max(0, k - lap_start + 1)

        def stint_for(lapno):
            for s in st:
                if (s.get("lap_start") or 0) <= lapno <= (s.get("lap_end") or 10_000):
                    return s
            return cur

        lap_rows = []
        for l in completed:
            s = stint_for(l["lap_number"])
            lap_rows.append({
                "lap": int(l["lap_number"]),
                "time": float(l["lap_duration"]),
                "compound": str(s.get("compound") or "UNKNOWN").upper(),
                "age": int((s.get("tyre_age_at_start") or 0) + l["lap_number"] - (s.get("lap_start") or 1) + 1),
                "pit_in": int(l["lap_number"]) in pit_laps.get(num, set()),
                "pit_out": bool(l.get("is_pit_out_lap")),
                "neutral": int(l["lap_number"]) in neutral,
            })
        it = latest_int.get(num, {})
        gap = _parse_gap(it.get("gap_to_leader"))
        comps = []
        for s in st:
            c = str(s.get("compound") or "").upper()
            if c and c not in comps and c != "UNKNOWN":
                comps.append(c)
        retired = False
        cars.append({
            "driver": d.get("name_acronym") or str(num),
            "number": str(num),
            "team": d.get("team_name") or "",
            "color": "#" + str(d.get("team_colour") or "888888").lstrip("#"),
            "position": None,
            "gap_to_leader": 0.0 if gap is None and it.get("gap_to_leader") in (0, "0", None) and k == leader_laps and it else gap,
            "interval": _parse_gap(it.get("interval")),
            "laps_completed": k,
            "compound": comp,
            "tyre_age": int(tyre_age),
            "stint": int(cur.get("stint_number") or 1),
            "stops": max(0, len(st) - 1),
            "compounds_used": comps,
            "in_pit": False,
            "retired": retired,
            "last_lap": lap_rows[-1]["time"] if lap_rows else None,
            "laps": lap_rows,
        })
    # Cars that stopped completing laps well behind the leader are retired.
    for c in cars:
        if leader_laps - c["laps_completed"] >= 3 and c["gap_to_leader"] is None:
            c["retired"] = True
    total = total_laps or int(ev.get("total_laps", 0) or 0) or None
    wx = weather[-1] if weather else {}
    return {
        "source": "openf1",
        "round": rnd,
        "event": str(ev["EventName"]),
        "location": str(ev["Location"]),
        "lap": min(leader_laps + 1, total or leader_laps + 1),
        "laps_completed_leader": leader_laps,
        "total_laps": total,
        "track_status": status,
        "updated_at": _now(),
        "cars": _rank_cars(cars),
        "race_control": [
            {"lap": m.get("lap_number"), "category": m.get("category"), "message": m.get("message"), "flag": m.get("flag")}
            for m in sorted(rc, key=lambda m: m.get("date") or "", reverse=True)[:12]
        ],
        "weather": {"track_temp": wx.get("track_temperature"), "air_temp": wx.get("air_temperature"), "rainfall": wx.get("rainfall")},
    }


# ---------------------------------------------------------------------------
# F1 SignalR live timing (FastF1 client + incremental reducer)
# ---------------------------------------------------------------------------
def _truthy(v: Any) -> bool:
    """F1 live timing sends flags as "0"/"1" strings."""
    return str(v).strip().lower() in ("1", "true")


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        elif isinstance(v, dict) and isinstance(dst.get(k), list):
            lst = dst[k]
            for ik, iv in v.items():
                i = int(ik)
                while len(lst) <= i:
                    lst.append({})
                if isinstance(iv, dict) and isinstance(lst[i], dict):
                    _deep_merge(lst[i], iv)
                else:
                    lst[i] = iv
        else:
            dst[k] = v
    return dst


def _lap_seconds(s: str | None) -> float | None:
    if not s:
        return None
    try:
        parts = s.split(":")
        return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


class TimingReducer:
    """Applies F1 live-timing messages (full snapshots + diffs) to an in-memory state."""

    TOPICS = ["DriverList", "TimingData", "TimingAppData", "LapCount", "TrackStatus", "RaceControlMessages", "WeatherData", "SessionInfo", "SessionStatus"]

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.lap_history: dict[str, list[dict]] = {}
        self.neutral_laps: set[int] = set()
        self.rain_laps: set[int] = set()
        self.lock = threading.Lock()

    def apply(self, topic: str, payload: Any) -> None:
        if topic.endswith(".z") and isinstance(payload, str):
            payload = json.loads(zlib.decompress(__import__("base64").b64decode(payload), -zlib.MAX_WBITS))
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return
        with self.lock:
            if topic == "TimingData" and isinstance(payload, dict):
                self._record_laps(payload)
            if topic == "RaceControlMessages" and isinstance(payload, dict):
                msgs = payload.get("Messages")
                cur = self.data.setdefault("RaceControlMessages", {"Messages": []})
                if isinstance(msgs, list):
                    cur["Messages"].extend(msgs)
                elif isinstance(msgs, dict):
                    cur["Messages"].extend(msgs.values())
                return
            if topic == "TrackStatus" and isinstance(payload, dict):
                st = STATUS_MAP.get(str(payload.get("Status")), None)
                lap = int(self.data.get("LapCount", {}).get("CurrentLap") or 0)
                if st in ("SC", "VSC", "RED", "VSC_ENDING") and lap:
                    self.neutral_laps.add(lap)
            if topic == "WeatherData" and isinstance(payload, dict) and _truthy(payload.get("Rainfall")):
                lap = int(self.data.get("LapCount", {}).get("CurrentLap") or 0)
                if lap:
                    self.rain_laps.add(lap)
            if isinstance(payload, dict) and isinstance(self.data.get(topic), dict):
                _deep_merge(self.data[topic], payload)
            else:
                self.data[topic] = payload

    def apply_line(self, line: str) -> None:
        """Parse one line written by fastf1.livetiming (python-repr lists)."""
        line = line.strip()
        if not line:
            return
        try:
            msg = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                return
        if isinstance(msg, list) and len(msg) >= 2 and isinstance(msg[0], str):
            self.apply(msg[0], msg[1])

    def _record_laps(self, payload: dict) -> None:
        lines = payload.get("Lines") or {}
        cur_lines = self.data.get("TimingData", {}).get("Lines", {})
        app = self.data.get("TimingAppData", {}).get("Lines", {})
        lap_now = int(self.data.get("LapCount", {}).get("CurrentLap") or 0)
        # A lap is neutralised if a SC/VSC/red status was seen during it or is still active.
        status_now = str((self.data.get("TrackStatus") or {}).get("Status") or "1")
        neutral_now = lap_now in self.neutral_laps or status_now in ("4", "5", "6", "7")
        for num, upd in lines.items():
            if not isinstance(upd, dict) or "NumberOfLaps" not in upd:
                continue
            prev = cur_lines.get(num, {})
            n = int(upd["NumberOfLaps"])
            if n == int(prev.get("NumberOfLaps") or 0):
                continue
            last = upd.get("LastLapTime", {}).get("Value") if isinstance(upd.get("LastLapTime"), dict) else None
            if last is None:
                last = (prev.get("LastLapTime") or {}).get("Value")
            stints = (app.get(num) or {}).get("Stints") or []
            st = stints[-1] if isinstance(stints, list) and stints else {}
            self.lap_history.setdefault(num, []).append({
                "lap": n,
                "time": _lap_seconds(last),
                "compound": str(st.get("Compound") or "UNKNOWN").upper(),
                "age": int(st.get("TotalLaps") or 0),
                "pit_in": bool(prev.get("InPit")),
                "pit_out": bool(upd.get("PitOut") or prev.get("PitOut")),
                "neutral": neutral_now,
            })

    def _weather(self) -> dict:
        w = self.data.get("WeatherData") or {}
        if not w and not self.rain_laps:
            return {}
        return {"rainfall": _truthy(w.get("Rainfall")), "track_temp": _num(w.get("TrackTemp")), "air_temp": _num(w.get("AirTemp")),
                "rain_laps": sorted(self.rain_laps)}

    def state(self, rnd: int, total_laps: int | None) -> dict:
        with self.lock:
            dl = self.data.get("DriverList", {}) or {}
            td = (self.data.get("TimingData", {}) or {}).get("Lines", {}) or {}
            app = (self.data.get("TimingAppData", {}) or {}).get("Lines", {}) or {}
            lc = self.data.get("LapCount", {}) or {}
            ts = self.data.get("TrackStatus", {}) or {}
            cars = []
            for num, info in dl.items():
                if not isinstance(info, dict) or "Tla" not in info:
                    continue
                t = td.get(num, {}) or {}
                stints = (app.get(num) or {}).get("Stints") or []
                if isinstance(stints, dict):
                    stints = [stints[k] for k in sorted(stints, key=int)]
                cur = stints[-1] if stints else {}
                comps = []
                for s in stints:
                    c = str(s.get("Compound") or "").upper()
                    if c and c not in comps and c != "UNKNOWN":
                        comps.append(c)
                gap = t.get("GapToLeader")
                interval = (t.get("IntervalToPositionAhead") or {}).get("Value") if isinstance(t.get("IntervalToPositionAhead"), dict) else None
                cars.append({
                    "driver": info["Tla"],
                    "number": str(num),
                    "team": info.get("TeamName", ""),
                    "color": "#" + str(info.get("TeamColour") or "888888").lstrip("#"),
                    "position": int(t["Position"]) if str(t.get("Position", "")).isdigit() else None,
                    "gap_to_leader": 0.0 if str(t.get("Position")) == "1" else _parse_gap(gap),
                    "interval": _parse_gap(interval),
                    "laps_completed": int(t.get("NumberOfLaps") or 0),
                    "compound": str(cur.get("Compound") or "UNKNOWN").upper(),
                    "tyre_age": int(cur.get("TotalLaps") or 0),
                    "stint": len(stints),
                    "stops": int(t.get("NumberOfPitStops") or max(0, len(stints) - 1)),
                    "compounds_used": comps,
                    "in_pit": bool(t.get("InPit")),
                    "retired": bool(t.get("Retired") or t.get("Stopped")),
                    "last_lap": _lap_seconds((t.get("LastLapTime") or {}).get("Value")),
                    "laps": list(self.lap_history.get(num, [])),
                })
            msgs = (self.data.get("RaceControlMessages", {}) or {}).get("Messages", [])
        ev = event_row(rnd)
        lap = int(lc.get("CurrentLap") or 0)
        total = int(lc.get("TotalLaps") or 0) or total_laps
        cars = _rank_cars(cars)
        return {
            "source": "signalr",
            "round": rnd,
            "event": str(ev["EventName"]),
            "location": str(ev["Location"]),
            "lap": lap,
            "laps_completed_leader": max(lap - 1, 0),
            "total_laps": total,
            "track_status": STATUS_MAP.get(str(ts.get("Status")), "GREEN"),
            "updated_at": _now(),
            "cars": cars,
            "race_control": [{"lap": m.get("Lap"), "category": m.get("Category"), "message": m.get("Message"), "flag": m.get("Flag")} for m in msgs[-12:][::-1]],
            "weather": self._weather(),
        }


class SignalRSource:
    """Connects to F1 live timing in a background thread (requires F1TV token)."""

    def __init__(self, rnd: int, total_laps: int | None, record_path: str | None = None, no_auth: bool = False):
        self.round = rnd
        self.total_laps = total_laps
        self.reducer = TimingReducer()
        self.record_path = record_path
        self.no_auth = no_auth
        self.error: str | None = None
        self.connected = False
        self.reconnects = 0
        self._stop = threading.Event()
        self._client = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._client is not None and self._client._connection is not None:
                self._client._connection.stop()
        except Exception:
            pass

    def _run(self) -> None:
        try:
            from fastf1.internals import f1auth
            from fastf1.livetiming.client import SignalRClient

            if not self.no_auth and not f1auth.AUTH_DATA_FILE.read_text().strip():
                # No F1TV token: the public feed still carries timing, tyres and
                # race control (no car telemetry), so connect without auth.
                self.no_auth = True
            reducer = self.reducer
            source = self
            rec = open(self.record_path, "a") if self.record_path else None

            class _Client(SignalRClient):
                def _on_message(self, msg):
                    self._t_last_message = time.time()
                    try:
                        from signalrcore.messages.completion_message import CompletionMessage

                        if isinstance(msg, CompletionMessage):
                            for key, val in (msg.result or {}).items():
                                reducer.apply(key, val)
                                if rec:
                                    rec.write(str([key, json.dumps(val), ""]) + "\n")
                        elif isinstance(msg, list) and len(msg) >= 2:
                            reducer.apply(msg[0], msg[1])
                            if rec:
                                rec.write(str(msg) + "\n")
                        if rec:
                            rec.flush()
                        source.connected = True
                    except Exception as exc:  # never kill the feed on one bad message
                        log.warning("SignalR message error: %s", exc)

                def _run(self):
                    # Same as FastF1's SignalRClient._run, except that no-auth mode
                    # passes a token factory returning "" (FastF1 3.8 passes None,
                    # which signalrcore rejects with "access_token_factory is not function").
                    import requests as _rq
                    from signalrcore.hub_connection_builder import HubConnectionBuilder

                    self._output_file = open(self.filename, self.filemode)
                    r = _rq.options(self._negotiate_url, headers=self.headers, timeout=15)
                    if "AWSALBCORS" in r.cookies:
                        self.headers.update({"Cookie": f"AWSALBCORS={r.cookies['AWSALBCORS']}"})
                    options = {
                        "verify_ssl": True,
                        "access_token_factory": (lambda: "") if self._no_auth else f1auth.get_auth_token,
                        "headers": self.headers,
                    }
                    self._connection = HubConnectionBuilder().with_url(self._connection_url, options=options).build()
                    self._connection.on_open(self._on_connect)
                    self._connection.on_close(self._on_close)
                    self._connection.on("feed", self._on_message)
                    self._connection.start()
                    t0 = time.time()
                    while not self._is_connected:
                        if time.time() - t0 > 20:
                            raise ConnectionError("Timed out connecting to F1 live timing")
                        time.sleep(0.1)
                    self._connection.send("Subscribe", [self.topics], on_invocation=self._on_message)

            # Supervisor: (re)connect until stopped. Live-timing connections do
            # drop during a race; the reducer keeps state across reconnects and
            # the snapshot sent on resubscribe re-syncs it.
            backoff = 2.0
            while not self._stop.is_set():
                client = _Client(filename=os.devnull, timeout=0, no_auth=self.no_auth)
                client.topics = TimingReducer.TOPICS
                self._client = client
                try:
                    client._run()
                    self.error = None
                    backoff = 2.0
                    while client._is_connected and not self._stop.is_set():
                        time.sleep(1.0)
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc} (retrying)"
                    log.warning("live timing connection failed: %s", exc)
                finally:
                    try:
                        if client._connection is not None:
                            client._connection.stop()
                    except Exception:
                        pass
                if self._stop.is_set():
                    break
                self.reconnects += 1
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.connected = False

    def state(self) -> dict:
        if self.error and not self.reducer.data:
            raise ConnectionError(self.error)
        st = self.reducer.state(self.round, self.total_laps)
        st["auth"] = "f1tv" if not self.no_auth else "public"
        return st


# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------
def manual_state(rnd: int, total_laps: int, car: dict, track_status: str = "GREEN", rivals: list[dict] | None = None) -> dict:
    ev = event_row(rnd)
    k = int(car["laps_completed"])
    base = {
        "driver": car.get("driver", "YOU"), "number": "", "team": car.get("team", ""), "color": car.get("color", "#e10600"),
        "position": car.get("position", 1), "gap_to_leader": car.get("gap_to_leader", 0.0), "interval": car.get("interval"),
        "laps_completed": k, "compound": car["compound"].upper(), "tyre_age": int(car["tyre_age"]), "stint": len(car.get("compounds_used", [])) or 1,
        "stops": max(0, len(car.get("compounds_used", [car["compound"]])) - 1), "compounds_used": [c.upper() for c in car.get("compounds_used", [car["compound"]])],
        "in_pit": False, "retired": False, "last_lap": None, "laps": [],
    }
    cars = [base] + [r | {"laps": [], "retired": False, "in_pit": False} for r in (rivals or [])]
    return {
        "source": "manual", "round": rnd, "event": str(ev["EventName"]), "location": str(ev["Location"]),
        "lap": min(k + 1, total_laps), "laps_completed_leader": k, "total_laps": total_laps,
        "track_status": track_status.upper(), "updated_at": _now(), "cars": cars, "race_control": [], "weather": {},
    }
