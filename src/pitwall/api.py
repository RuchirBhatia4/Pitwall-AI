"""HTTP API for the Next.js frontend (mounted by src/api/main.py)."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.pitwall.live_engine import analyse
from src.pitwall.optimizer import Optimizer, RaceContext, TyreState
from src.pitwall.season import SEASON_DIR
from src.pitwall.tyre_model import TyreModel

log = logging.getLogger("pitwall.api")
router = APIRouter(prefix="/api")
DATA = SEASON_DIR / "2026"


# ---------------------------------------------------------------------------
# season artefacts
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _round_file(rnd: int, mtime: float) -> dict:
    return json.loads((DATA / f"r{rnd:02d}.json").read_text())


def load_round(rnd: int) -> dict:
    path = DATA / f"r{rnd:02d}.json"
    if not path.exists():
        raise HTTPException(404, f"Round {rnd} has not been built yet (python -m src.pitwall.build_season)")
    return _round_file(rnd, path.stat().st_mtime)


def weekend_for(rnd: int) -> dict:
    return load_round(rnd)["weekend"]


def _season() -> dict:
    path = DATA / "season.json"
    if not path.exists():
        raise HTTPException(503, "Season not built yet: run python -m src.pitwall.build_season")
    season = json.loads(path.read_text())
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    live_round, next_round = None, None
    for r in season["rounds"]:
        start = pd.Timestamp(r["race_start_utc"]) if r.get("race_start_utc") not in (None, "NaT") else None
        if start is None:
            continue
        r["seconds_to_start"] = (start - now).total_seconds()
        if start - pd.Timedelta(minutes=45) <= now <= start + pd.Timedelta(hours=3):
            live_round = r["round"]
        if next_round is None and start > now - pd.Timedelta(hours=3):
            next_round = r["round"]
    season["live_round"] = live_round
    season["next_round"] = next_round
    bt = DATA / "backtest.json"
    season["backtest"] = json.loads(bt.read_text()) if bt.exists() else None
    return season


@router.get("/season")
def get_season() -> dict:
    return _season()


@router.get("/races/{rnd}")
def get_race(rnd: int) -> dict:
    return load_round(rnd)


class EvaluateRequest(BaseModel):
    round: int
    driver: str | None = None
    sequence: list[str]
    pit_laps: list[int]


@router.post("/strategy/evaluate")
def evaluate_plan(req: EvaluateRequest) -> dict:
    """What-if: model time of a user-built plan vs the optimum for that car."""
    data = load_round(req.round)
    w = data["weekend"]
    team = next((d["team"] for d in data["drivers"] if d["driver"] == req.driver), None)
    model = TyreModel.from_dict(w["model"])
    seq = [s.upper() for s in req.sequence]
    if any(c not in model.compounds for c in seq):
        raise HTTPException(400, f"Unknown compound in {seq}")
    if len(set(seq)) < 2:
        raise HTTPException(400, "A dry race must use at least two different compounds")
    if len(req.pit_laps) != len(seq) - 1:
        raise HTTPException(400, "Need exactly one pit lap per compound change")
    ctx = RaceContext(total_laps=w["total_laps"], pit_loss=w["pit_loss"], team=team,
                      stop_penalty=w.get("stop_penalty", 0.0), fuel_wear=w.get("fuel_wear", 1.0))
    opt = Optimizer(model, ctx)
    best = opt.optimise(top=1)[0]
    total = opt.evaluate(seq, sorted(req.pit_laps), TyreState())
    if total >= 1e8:
        raise HTTPException(400, "Plan exceeds the usable life of a tyre set")
    return {"delta_to_optimal": total - best.total, "optimal": best.to_dict(), "plan_total": total}


# ---------------------------------------------------------------------------
# replay (stateless)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=6)
def _replay(rnd: int):
    from src.pitwall.live_sources import ReplaySource

    return ReplaySource(rnd)


def _slim(state: dict) -> dict:
    return state | {"cars": [{k: v for k, v in c.items() if k != "laps"} for c in state["cars"]]}


@router.get("/replay/{rnd}")
def replay(rnd: int, lap: int, driver: str) -> dict:
    src = _replay(rnd)
    state = src.state(lap)
    return {"state": _slim(state), "analysis": analyse(state, weekend_for(rnd), driver.upper())}


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
# Live tracking is per visitor: every browser sends a random session id
# (X-Pitwall-Session). A session only *chooses* what to watch; real live feeds
# (F1 live timing, OpenF1) are shared, read-only and keyed by race, so one
# visitor can never switch what another visitor sees.
SESSION_HEADER = "X-Pitwall-Session"
MAX_SESSIONS = 2000
SESSION_TTL_S = 6 * 3600
MAX_FEEDS = 3
FEED_IDLE_STOP_S = 20 * 60  # stop a shared feed nobody has read for 20 minutes
STATE_CACHE_S = 3.0
_SID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def session_id(raw: str | None) -> str:
    return raw if raw and _SID_RE.match(raw) else "default"


class Feed:
    """One shared live data source (F1 live timing or OpenF1) for one race."""

    def __init__(self, kind: str, rnd: int, source) -> None:
        self.kind, self.round, self.source = kind, rnd, source
        self.lock = threading.Lock()
        self.last_state: dict | None = None
        self.last_ok: float | None = None
        self.error: str | None = None
        self.last_read = time.time()
        self._cache_t = 0.0

    def state(self) -> dict:
        with self.lock:
            self.last_read = time.time()
            if self.last_state and time.time() - self._cache_t < STATE_CACHE_S:
                return self.last_state
            try:
                st = self.source.state()
                self.last_state, self.last_ok, self.error, self._cache_t = st, time.time(), None, time.time()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("live feed %s/R%s error: %s", self.kind, self.round, self.error)
                if self.last_state is None:
                    raise HTTPException(502, self.error) from exc
            return self.last_state

    def stop(self) -> None:
        if hasattr(self.source, "stop"):
            self.source.stop()


@dataclass
class LiveSession:
    kind: str | None = None
    round: int | None = None
    replay_t0: float = 0.0
    replay_start_lap: int = 1
    seconds_per_lap: float = 8.0
    last_seen: float = field(default_factory=time.time)


class LiveManager:
    def __init__(self) -> None:
        self.sessions: dict[str, LiveSession] = {}
        self.feeds: dict[tuple[str, int], Feed] = {}
        self.lock = threading.Lock()

    # -- bookkeeping ------------------------------------------------------
    def _gc(self) -> None:
        now = time.time()
        for sid, s in list(self.sessions.items()):
            if now - s.last_seen > SESSION_TTL_S:
                del self.sessions[sid]
        for key, f in list(self.feeds.items()):
            if now - f.last_read > FEED_IDLE_STOP_S:
                f.stop()
                del self.feeds[key]

    def session(self, sid: str) -> LiveSession:
        with self.lock:
            self._gc()
            s = self.sessions.get(sid)
            if s is None:
                if len(self.sessions) >= MAX_SESSIONS:
                    oldest = min(self.sessions, key=lambda k: self.sessions[k].last_seen)
                    del self.sessions[oldest]
                s = self.sessions[sid] = LiveSession()
            s.last_seen = time.time()
            return s

    def _feed(self, kind: str, rnd: int, no_auth: bool = False) -> Feed:
        from src.pitwall.live_sources import OpenF1Source, SignalRSource

        key = (kind, rnd)
        with self.lock:
            if key in self.feeds:
                return self.feeds[key]
            if len(self.feeds) >= MAX_FEEDS:
                idle = min(self.feeds.values(), key=lambda f: f.last_read)
                if time.time() - idle.last_read < 60:
                    raise HTTPException(429, "Too many live feeds are open right now; try again in a minute.")
                idle.stop()
                del self.feeds[(idle.kind, idle.round)]
            w = weekend_for(rnd)
            if kind == "openf1":
                source = OpenF1Source(rnd, total_laps=w["total_laps"])
            else:
                rec = str(SEASON_DIR / f"live_r{rnd:02d}_{int(time.time())}.txt")
                source = SignalRSource(rnd, w["total_laps"], record_path=rec, no_auth=no_auth)
            feed = self.feeds[key] = Feed(kind, rnd, source)
            return feed

    # -- per-visitor API ---------------------------------------------------
    def connect(self, sid: str, kind: str, rnd: int, seconds_per_lap: float = 8.0, start_lap: int = 1, no_auth: bool = False) -> None:
        weekend_for(rnd)  # 404 for rounds that have not been built
        if kind == "signalr":
            # F1 live timing streams whatever session is on air, so it can only
            # be paired with the current race weekend's model.
            season = _season()
            current = season.get("live_round") or season.get("next_round")
            if rnd != current:
                raise HTTPException(400, f"F1 live timing only carries the current session (round {current}). Use Replay or OpenF1 for other races.")
        s = self.session(sid)
        if kind in ("openf1", "signalr"):
            self._feed(kind, rnd, no_auth)
        elif kind == "replay":
            _replay(rnd)
            s.replay_t0, s.replay_start_lap, s.seconds_per_lap = time.time(), start_lap, seconds_per_lap
        else:
            raise ValueError(kind)
        s.kind, s.round = kind, rnd

    def state(self, sid: str) -> dict:
        s = self.session(sid)
        if s.kind is None or s.round is None:
            raise HTTPException(409, "Live tracker not connected. POST /api/live/connect first.")
        if s.kind == "replay":
            src = _replay(s.round)
            lap = s.replay_start_lap + int((time.time() - s.replay_t0) / s.seconds_per_lap)
            st = src.state(min(lap, src.total_laps))
            st["source"] = "replay-live"
            return st
        feed = self.feeds.get((s.kind, s.round)) or self._feed(s.kind, s.round)
        return feed.state()

    def status(self, sid: str) -> dict:
        from fastf1.internals import f1auth

        s = self.session(sid)
        feed = self.feeds.get((s.kind, s.round)) if s.kind in ("openf1", "signalr") else None
        try:
            f1tv = bool(f1auth.AUTH_DATA_FILE.read_text().strip())
        except Exception:
            f1tv = False
        viewers = sum(1 for x in self.sessions.values() if x.kind == s.kind and x.round == s.round) if s.kind else 0
        return {
            "connected": s.kind is not None,
            "source": s.kind,
            "round": s.round,
            "error": (feed.error or getattr(feed.source, "error", None)) if feed else None,
            "last_update_age": round(time.time() - feed.last_ok, 1) if feed and feed.last_ok else None,
            "shared_feed": feed is not None,
            "viewers": viewers,
            "openf1_credentials": bool(os.environ.get("OPENF1_USERNAME") or os.environ.get("OPENF1_TOKEN")),
            "f1tv_token": f1tv,
        }


LIVE = LiveManager()


class ConnectRequest(BaseModel):
    source: Literal["openf1", "signalr", "replay"]
    round: int
    seconds_per_lap: float = Field(8.0, ge=1, le=120)
    start_lap: int = Field(1, ge=1)
    no_auth: bool = False


@router.post("/live/connect")
def live_connect(req: ConnectRequest, x_pitwall_session: str | None = Header(default=None)) -> dict:
    sid = session_id(x_pitwall_session)
    try:
        LIVE.connect(sid, req.source, req.round, req.seconds_per_lap, req.start_lap, req.no_auth)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    return LIVE.status(sid)


@router.get("/live/status")
def live_status(x_pitwall_session: str | None = Header(default=None)) -> dict:
    return LIVE.status(session_id(x_pitwall_session))


@router.get("/live/state")
def live_state(driver: str | None = None, x_pitwall_session: str | None = Header(default=None)) -> dict:
    sid = session_id(x_pitwall_session)
    state = LIVE.state(sid)
    out = {"state": _slim(state), "status": LIVE.status(sid)}
    if driver:
        out["analysis"] = analyse(state, weekend_for(state["round"]), driver.upper())
    return out


class ManualRequest(BaseModel):
    round: int
    driver: str = "YOU"
    team: str | None = None
    laps_completed: int = Field(..., ge=0)
    compound: str
    tyre_age: int = Field(..., ge=0)
    compounds_used: list[str] = []
    track_status: Literal["GREEN", "YELLOW", "SC", "VSC", "RED"] = "GREEN"
    position: int | None = None
    gap_to_leader: float | None = None
    gap_ahead: float | None = None
    gap_behind: float | None = None


@router.post("/live/manual")
def live_manual(req: ManualRequest) -> dict:
    from src.pitwall.live_sources import manual_state

    w = weekend_for(req.round)
    used = list(dict.fromkeys([c.upper() for c in req.compounds_used] + [req.compound.upper()]))
    pos = req.position or 1
    gap = req.gap_to_leader if req.gap_to_leader is not None else 0.0
    rivals = []
    if req.gap_ahead is not None and pos > 1:
        rivals.append({"driver": "AHEAD", "number": "", "team": "", "color": "#888888", "position": pos - 1, "gap_to_leader": gap - req.gap_ahead,
                       "interval": None, "laps_completed": req.laps_completed, "compound": req.compound.upper(), "tyre_age": req.tyre_age, "stint": 1, "stops": 0, "compounds_used": used})
    if req.gap_behind is not None:
        rivals.append({"driver": "BEHIND", "number": "", "team": "", "color": "#888888", "position": pos + 1, "gap_to_leader": gap + req.gap_behind,
                       "interval": req.gap_behind, "laps_completed": req.laps_completed, "compound": req.compound.upper(), "tyre_age": req.tyre_age, "stint": 1, "stops": 0, "compounds_used": used})
    car = {"driver": req.driver.upper(), "team": req.team or "", "laps_completed": req.laps_completed, "compound": req.compound, "tyre_age": req.tyre_age,
           "compounds_used": used, "position": pos, "gap_to_leader": gap, "interval": req.gap_ahead}
    state = manual_state(req.round, w["total_laps"], car, req.track_status, rivals)
    return {"state": _slim(state), "analysis": analyse(state, w, req.driver.upper())}


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=40)
    context: dict = {}


@router.post("/chat")
def chat_endpoint(req: ChatRequest, x_pitwall_session: str | None = Header(default=None)) -> dict:
    from src.pitwall.chat import chat

    return chat([m.model_dump() for m in req.messages], req.context | {"session": session_id(x_pitwall_session)})
