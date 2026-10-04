"""HTTP API for the Next.js frontend (mounted by src/api/main.py)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
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
class LiveManager:
    def __init__(self) -> None:
        self.kind: str | None = None
        self.round: int | None = None
        self.source = None
        self.error: str | None = None
        self.last_state: dict | None = None
        self.last_ok: float | None = None
        self.replay_t0: float | None = None
        self.replay_start_lap = 1
        self.seconds_per_lap = 8.0
        self.lock = threading.Lock()
        self._cache_t = 0.0

    def connect(self, kind: str, rnd: int, seconds_per_lap: float = 8.0, start_lap: int = 1, no_auth: bool = False) -> None:
        from src.pitwall.live_sources import OpenF1Source, SignalRSource

        w = weekend_for(rnd)
        with self.lock:
            if self.source is not None and hasattr(self.source, "stop"):
                self.source.stop()  # never leave two live-timing connections open
            self.kind, self.round, self.error, self.last_state = kind, rnd, None, None
            if kind == "openf1":
                self.source = OpenF1Source(rnd, total_laps=w["total_laps"])
            elif kind == "signalr":
                rec = str(SEASON_DIR / f"live_r{rnd:02d}_{int(time.time())}.txt")
                self.source = SignalRSource(rnd, w["total_laps"], record_path=rec, no_auth=no_auth)
            elif kind == "replay":
                self.source = _replay(rnd)
                self.replay_t0, self.replay_start_lap, self.seconds_per_lap = time.time(), start_lap, seconds_per_lap
            else:
                raise ValueError(kind)

    def state(self) -> dict:
        if self.source is None:
            raise HTTPException(409, "Live tracker not connected. POST /api/live/connect first.")
        with self.lock:
            if self.last_state and time.time() - self._cache_t < 3:
                return self.last_state
            try:
                if self.kind == "replay":
                    lap = self.replay_start_lap + int((time.time() - self.replay_t0) / self.seconds_per_lap)
                    st = self.source.state(min(lap, self.source.total_laps))
                    st["source"] = "replay-live"
                else:
                    st = self.source.state()
                self.last_state, self.last_ok, self.error, self._cache_t = st, time.time(), None, time.time()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("live source error: %s", self.error)
                if self.last_state is None:
                    raise HTTPException(502, self.error) from exc
            return self.last_state

    def status(self) -> dict:
        from fastf1.internals import f1auth

        try:
            f1tv = bool(f1auth.AUTH_DATA_FILE.read_text().strip())
        except Exception:
            f1tv = False
        return {
            "connected": self.source is not None,
            "source": self.kind,
            "round": self.round,
            "error": self.error or getattr(self.source, "error", None),
            "last_update_age": round(time.time() - self.last_ok, 1) if self.last_ok else None,
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
def live_connect(req: ConnectRequest) -> dict:
    try:
        LIVE.connect(req.source, req.round, req.seconds_per_lap, req.start_lap, req.no_auth)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    return LIVE.status()


@router.get("/live/status")
def live_status() -> dict:
    return LIVE.status()


@router.get("/live/state")
def live_state(driver: str | None = None) -> dict:
    state = LIVE.state()
    out = {"state": _slim(state), "status": LIVE.status()}
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
def chat_endpoint(req: ChatRequest) -> dict:
    from src.pitwall.chat import chat

    return chat([m.model_dump() for m in req.messages], req.context)
