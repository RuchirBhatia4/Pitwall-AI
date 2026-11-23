"""'Ask the pit wall': a chat assistant grounded in the strategy engine.

Two modes:
  * LLM mode (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN set, or
    PITWALL_USE_LLM=1 with an `ant auth login` profile): an LLM answers using
    tool calls into the engine, so every number it quotes comes from the
    strategy model, not from the LLM's memory. Model: PITWALL_CHAT_MODEL.
  * Engine mode (no credentials / API unreachable): a deterministic intent
    parser calls the same tools and formats the answer itself.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
import unicodedata
from datetime import datetime
from typing import Any

from src.pitwall.optimizer import SHORT, Optimizer, RaceContext, TyreState
from src.pitwall.tyre_model import TyreModel

log = logging.getLogger("pitwall.chat")
_round = round  # tool parameters are named `round`, which shadows the builtin
# Visitor session for the live tracker, set per chat request.
_SESSION: contextvars.ContextVar[str] = contextvars.ContextVar("pitwall_session", default="default")
MODEL = os.environ.get("PITWALL_CHAT_MODEL", "claude-opus-5-5")
MAX_TOOL_ROUNDS = 8

SYSTEM = """You are the race strategy engineer on the PitWall AI pit wall, talking to a user about the 2026 Formula 1 season.

Ground every number you state in a tool result from this conversation: pit laps, time deltas, probabilities, tyre wear rates and accuracy figures all come from the strategy engine tools. If a tool cannot answer, say what is missing rather than estimating.

How the engine works, so you can explain it: a lap-time model (compound offset + wear per lap of tyre age, fuel-load interaction, fuel burn) is fitted per weekend from earlier 2026 races plus that weekend's practice long runs, combined with Bayesian updating; strategies are optimised exactly by dynamic programming; risk comes from Monte Carlo over Safety Cars / VSCs and parameter uncertainty. "P(best)" is the share of simulated races in which a plan was fastest; "expected regret" is the average time lost against the best plan in the same simulated race.

Style: answer like a race engineer on the radio when the question is about a decision (lead with the call), and like an analyst when it is about the model. Keep answers short (under ~150 words unless asked for detail), use driver three-letter codes, compound names (Soft/Medium/Hard) and lap numbers. Be candid about uncertainty and about the backtest's limits.
During a race, use live_race for the running order, gaps, tyres, flags and race control, and live_call for one driver's pit call."""


# ---------------------------------------------------------------------------
# Tools (thin wrappers over the engine + season artefacts)
# ---------------------------------------------------------------------------
def _load(rnd: int) -> dict:
    from src.pitwall.api import load_round

    return load_round(rnd)


def _season() -> dict:
    from src.pitwall.api import _season as season

    return season()


def _plan_summary(p: dict | None) -> dict | None:
    if not p:
        return None
    out = {"plan": p.get("name"), "compounds": p.get("sequence"), "pit_laps": p.get("pit_laps")}
    if p.get("windows"):
        out["pit_windows"] = p["windows"]
    if p.get("delta") is not None:
        out["seconds_slower_than_fastest"] = _round(p["delta"], 1)
    r = p.get("risk") or {}
    if "win_probability" in r:
        out["p_best"] = _round(r["win_probability"], 2)
        out["expected_regret_s"] = _round(r.get("expected_delta", 0), 1)
        out["cvar90_s"] = _round(r.get("cvar90", 0), 1)
    if "class_probability" in r:
        out["probability_of_this_stop_count_and_start_tyre"] = _round(r["class_probability"], 2)
    return out


def tool_list_races() -> dict:
    s = _season()
    return {
        "next_round": s.get("next_round"),
        "live_round": s.get("live_round"),
        "rounds": [
            {"round": r["round"], "event": r["event"], "location": r["location"], "date": r["date"], "status": r.get("status"),
             "winner": (r.get("winner") or {}).get("driver")}
            for r in s["rounds"]
        ],
    }


def tool_race_summary(round: int) -> dict:
    d = _load(round)
    out = {"round": round, "event": d["event"], "location": d["location"], "status": d["status"], "laps": d["weekend"]["total_laps"],
           "pit_loss_s": _round(d["weekend"]["pit_loss"], 1),
           "tyre_model": {c: {"wear_s_per_lap": _round(p["deg"], 3), "pace_vs_medium_s": _round(p["offset"], 2)} for c, p in d["weekend"]["model"]["compounds"].items()}}
    preds = d["predictions"]
    counts: dict[str, int] = {}
    for e in preds.values():
        if e.get("predicted"):
            counts[e["predicted"]["name"]] = counts.get(e["predicted"]["name"], 0) + 1
    out["predicted_strategies_count"] = counts
    if d.get("overview"):
        ov = d["overview"]
        res = sorted([x for x in ov["drivers"] if x.get("position")], key=lambda x: x["position"])[:5]
        out["top5"] = [f"{x['position']}. {x['driver']} ({x['team']})" for x in res]
        out["safety_car_laps"] = ov["neutralised"]["sc"]
        out["vsc_laps"] = ov["neutralised"]["vsc"]
        act: dict[str, int] = {}
        for drv, st in ov["stints"].items():
            name = "-".join(SHORT.get(s["compound"], "?") for s in st)
            act[name] = act.get(name, 0) + 1
        out["actual_strategies_count"] = dict(sorted(act.items(), key=lambda kv: -kv[1])[:6])
    if d.get("metrics"):
        m = d["metrics"]
        out["prediction_accuracy_here"] = {k: m.get(k) for k in ("stops_accuracy", "compound_set_accuracy", "first_stop_mae", "baseline_stops_accuracy", "wet")}
    return out


def _driver_code(d: dict, driver: str) -> str | None:
    q = driver.strip().upper()
    for x in d["drivers"]:
        if x["driver"] == q or x["name"].upper().split()[-1] == q or x["name"].upper() == q:
            return x["driver"]
    return None


def tool_driver_strategy(round: int, driver: str) -> dict:
    d = _load(round)
    code = _driver_code(d, driver)
    if not code or code not in d["predictions"]:
        return {"error": f"No driver '{driver}' in round {round}. Drivers: {', '.join(x['driver'] for x in d['drivers'])}"}
    e = d["predictions"][code]
    meta = next(x for x in d["drivers"] if x["driver"] == code)
    out = {
        "round": round, "event": d["event"], "driver": code, "name": meta["name"], "team": meta["team"], "grid": meta.get("grid"),
        "fastest_plan_physics": _plan_summary(e.get("predicted")),
        "likely_plan_team_behaviour_model": (_plan_summary(e["likely"]) | {"stop_count_probabilities": e["likely"].get("stop_probabilities"),
                                                                         "start_tyre_probabilities": e["likely"].get("start_probabilities")}) if e.get("likely") else None,
        "first_stop_80pct_interval": e.get("first_stop_interval"),
        "stop_count_odds_physics": e.get("stop_probabilities"),
        "personal_wear_vs_field": ({k: e["wear_profile"][k] for k in ("mult", "sd", "season_races")} if e.get("wear_profile") else None),
        "alternatives": [_plan_summary(p) for p in e.get("alternatives", [])[:5]],
    }
    if e.get("actual"):
        out["actual"] = {"plan": e["actual"]["name"], "pit_laps": e["actual"]["pit_laps"], "finish": meta.get("position"), "status": meta.get("status")}
    if e.get("hindsight"):
        out["hindsight_optimum"] = _plan_summary(e["hindsight"])
        out["actual_vs_hindsight_s"] = _round(e["actual_vs_hindsight"], 1) if e.get("actual_vs_hindsight") is not None else None
    return out


def tool_evaluate_plan(round: int, driver: str, compounds: list[str], pit_laps: list[int]) -> dict:
    from src.pitwall.api import EvaluateRequest, evaluate_plan

    d = _load(round)
    code = _driver_code(d, driver) or driver.upper()
    try:
        res = evaluate_plan(EvaluateRequest(round=round, driver=code, sequence=compounds, pit_laps=pit_laps))
    except Exception as exc:  # HTTPException carries a readable detail
        return {"error": getattr(exc, "detail", str(exc))}
    return {"driver": code, "plan": "-".join(SHORT.get(c.upper(), "?") for c in compounds), "pit_laps": pit_laps,
            "seconds_slower_than_optimal": _round(res["delta_to_optimal"], 1), "optimal": _plan_summary(res["optimal"])}


def tool_safety_car_whatif(round: int, driver: str, sc_lap: int) -> dict:
    """If a Safety Car comes out on `sc_lap`, should the driver box (following the predicted plan until then)?"""
    d = _load(round)
    code = _driver_code(d, driver)
    if not code:
        return {"error": f"Unknown driver {driver}"}
    w = d["weekend"]
    e = d["predictions"][code]
    plan = e.get("predicted")
    if not plan:
        return {"error": "no plan"}
    L = w["total_laps"]
    lap = max(1, min(int(sc_lap), L - 1))
    stint = next(s for s in plan["stints"] if s["start_lap"] <= lap <= s["end_lap"])
    used = tuple(dict.fromkeys(s["compound"] for s in plan["stints"] if s["start_lap"] <= lap))
    age = lap - stint["start_lap"] + (stint.get("age_start") or 0)
    team = next((x["team"] for x in d["drivers"] if x["driver"] == code), None)
    model = TyreModel.from_dict(w["model"])
    ctx = RaceContext(total_laps=L, pit_loss=w["pit_loss"], team=team, stop_penalty=w.get("stop_penalty", 0), fuel_wear=w.get("fuel_wear", 1.0), min_stint=2)
    opt = Optimizer(model, ctx)
    from src.pitwall.optimizer import SC_PIT_FACTOR

    under_sc = TyreState(lap=lap - 1, compound=stint["compound"], age=age, used=used, pit_cost_now=w["pit_loss"] * SC_PIT_FACTOR)
    best = opt.optimise(under_sc, top=3)
    stay = [p for p in best if not p.pit_laps or p.pit_laps[0] != lap]
    box = [p for p in best if p.pit_laps and p.pit_laps[0] == lap]
    gain = None
    if box:
        alt = min((p.total for p in stay), default=None)
        if alt is None:
            no_box = opt.optimise(TyreState(lap=lap - 1, compound=stint["compound"], age=age, used=used), top=1)
            alt = no_box[0].total if no_box else None
        gain = None if alt is None else _round(alt - box[0].total, 1)
    return {
        "driver": code, "sc_lap": lap, "tyre_at_sc": {"compound": stint["compound"], "age_laps": age},
        "pit_cost_under_sc_s": _round(w["pit_loss"] * SC_PIT_FACTOR, 1), "green_pit_cost_s": _round(w["pit_loss"], 1),
        "call": "BOX under the Safety Car" if box and best[0] in box else "STAY OUT",
        "best_plan_from_there": _plan_summary(best[0].to_dict()),
        "time_saved_by_boxing_s": gain,
    }


def _live_state() -> tuple[dict | None, str | None]:
    """The live state this visitor sees: their own tracker if they connected one,
    else the shared F1 live-timing feed while a race is on air."""
    from src.pitwall.api import LIVE

    sid = _SESSION.get()
    if LIVE.session(sid).kind is not None:
        return LIVE.state(sid), None
    live = _season().get("live_round")
    if live is None:
        return None, "No race is on air right now. Connect a source (or a replay) on the Live pit wall page to ask about one."
    st = LIVE._feed("signalr", live).state()
    if (st.get("session") or {}).get("waiting"):
        return None, f"F1 live timing is still carrying {st['session'].get('name') or 'the previous session'}; the race feed has not started yet."
    return st, None


def tool_live_race() -> dict:
    from src.pitwall.forecast import rain_forecast

    st, err = _live_state()
    if err:
        return {"error": err}
    cars = [{"position": c.get("position"), "driver": c["driver"], "team": c.get("team"), "gap_to_leader_s": c.get("gap_to_leader"), "interval_s": c.get("interval"),
             "tyre": c.get("compound"), "tyre_age_laps": c.get("tyre_age"), "stops": c.get("stops"), "in_pit": c.get("in_pit"),
             "retired": c.get("retired"), "penalty_s": c.get("penalty_s") or None} for c in st["cars"]]
    out = {"source": st.get("source"), "round": st["round"], "event": st.get("event"), "lap": st.get("lap"), "total_laps": st.get("total_laps"),
           "track_status": st.get("track_status"), "cars": cars, "race_control": (st.get("race_control") or [])[:8], "weather": st.get("weather")}
    if st.get("source") in ("signalr", "openf1"):
        out["rain_next_2h"] = rain_forecast(st.get("location"))
    return out


def tool_live_session() -> dict:
    """Timing of the race weekend's session on air (or the last one run)."""
    from src.pitwall.api import DATA, LIVE
    from src.pitwall.live_sources import session_timing

    s = _season()
    rnd = s.get("live_round") or s.get("next_round")
    if rnd is None:
        return {"error": "The 2026 season is over."}
    if not (DATA / f"r{rnd:02d}.json").exists():  # the feed needs the weekend's model, built after first practice
        ev = next((r for r in s["rounds"] if r["round"] == rnd), {})
        fp1 = (ev.get("sessions") or {}).get("Practice 1")
        when = f" ({datetime.fromisoformat(fp1):%a %d %b %H:%M} UTC)" if fp1 else ""
        return {"error": f"No session timing yet: R{rnd} {ev.get('event', '')} timing is available here once its first practice{when} has been processed, about two hours after it starts."}
    feed = LIVE._feed("signalr", rnd)
    reducer = getattr(feed.source, "reducer", None)
    for _ in range(16):  # a feed opened by this question takes a few seconds to send its snapshot
        if reducer is None or reducer.data.get("SessionInfo"):
            break
        time.sleep(0.5)
    if reducer is None or not reducer.data.get("SessionInfo"):
        return {"error": "F1 live timing has not sent anything yet; try again in a few seconds."}
    out = session_timing(reducer)
    for c in out["cars"]:
        if c["best_lap"] is not None:
            c["best_lap"] = round(c["best_lap"], 3)
    return {"round": rnd} | out


def tool_circuit_history(round: int) -> dict:
    from src.pitwall import track_history

    d = _load(round)
    w = d["weekend"]
    v = track_history.venue_history(d["location"])
    return {"round": round, "event": d["event"], "circuit": d["location"], "laps": w["total_laps"],
            "pit_loss_s": _round(w["pit_loss"], 1), "safety_car_per_lap": _round(w["sc_rate"], 4), "vsc_per_lap": _round(w["vsc_rate"], 4),
            "earlier_races_here": v.get("races"), "venue": {k: v[k] for k in v if k != "races"}}


def tool_race_stints(round: int, driver: str | None = None) -> dict:
    d = _load(round)
    ov = d.get("overview")
    if not ov:
        return {"error": f"R{round} {d['event']} has not been raced yet."}
    code = _driver_code(d, driver) if driver else None
    stints = {k: [{"compound": s["compound"], "laps": f"{s['start_lap']}-{s['end_lap']}", "new": s.get("fresh"), "red_flag_change": s.get("red_flag") or None}
                  for s in v] for k, v in ov["stints"].items() if not code or k == code}
    return {"round": round, "event": d["event"], "neutralised_laps": ov.get("neutralised"), "stints": stints}


def tool_live_call(driver: str) -> dict:
    from src.pitwall.api import shared_analysis

    st, err = _live_state()
    if err:
        return {"error": err}
    code = driver.strip().upper()
    by_name = {c["driver"]: c["driver"] for c in st["cars"]}
    if code not in by_name:
        return {"error": f"{driver} not found in the live session; drivers: {', '.join(by_name)}"}
    a = shared_analysis(st, code)
    if "call" not in a or "car" not in a:
        return {"error": a.get("error") or "no call for this car"}
    return {
        "source": st["source"], "lap": a.get("lap"), "total_laps": a.get("total_laps"), "track_status": a.get("track_status"),
        "position": a["car"]["position"], "tyre": {"compound": a["car"]["compound"], "age": a["car"]["tyre_age"]},
        "call": a["call"], "rejoin_if_box_now": a.get("rejoin"), "threats": a.get("threats"),
        "top_plans": [_plan_summary(p) for p in (a.get("plans") or [])[:3]], "drift": a.get("drift"),
        "race_simulation": {"sims": a["race_sim"]["sims"], "options": a["race_sim"]["options"][:4]} if (a.get("race_sim") or {}).get("options") else None,
    }


def tool_backtest() -> dict:
    return _season().get("backtest") or {"error": "backtest not built"}


TOOLS: list[dict] = [
    {"name": "list_races", "description": "List all 2026 rounds with status (finished/upcoming), winner, and which round is next or live.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []}},
    {"name": "race_summary", "description": "Overview of one round: laps, pit loss, tyre model, predicted strategy mix, and for finished races the top 5, Safety Car laps, actual strategy mix and prediction accuracy.",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}}, "additionalProperties": False, "required": ["round"]}},
    {"name": "driver_strategy", "description": "Pre-race strategy for a driver at a round: the physics-fastest plan (pit laps, windows, P(best), regret, alternatives, 80% interval) and the most likely team call from the behaviour model (with stop-count and start-tyre probabilities), plus actual and hindsight-optimal strategy if the race is finished. Driver can be a 3-letter code or surname.",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}, "driver": {"type": "string"}}, "additionalProperties": False, "required": ["round", "driver"]}},
    {"name": "evaluate_plan", "description": "What-if: model time of an explicit plan (compounds in order, pit laps = last lap of each stint except the final one) vs the optimal plan for that driver.",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}, "driver": {"type": "string"},
                                                       "compounds": {"type": "array", "items": {"type": "string", "enum": ["SOFT", "MEDIUM", "HARD"]}},
                                                       "pit_laps": {"type": "array", "items": {"type": "integer"}}},
                      "additionalProperties": False, "required": ["round", "driver", "compounds", "pit_laps"]}},
    {"name": "safety_car_whatif", "description": "If a Safety Car is deployed on a given lap, should the driver box? Assumes they followed their predicted plan until then.",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}, "driver": {"type": "string"}, "sc_lap": {"type": "integer"}},
                      "additionalProperties": False, "required": ["round", "driver", "sc_lap"]}},
    {"name": "live_race", "description": "The live race right now: lap, track status (GREEN/SC/VSC/RED), every car's position, gap, interval, tyre and age, stops and penalties, recent race-control messages, weather and the rain chance for the next 2 hours. Uses the visitor's live tracker, or F1 live timing while a race is on air.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []}},
    {"name": "live_session", "description": "Timing of the race weekend's session on F1 live timing right now (practice, qualifying, sprint or race; or the last one run): each driver's position, best and last lap, gap to fastest, laps run, tyre and age, knocked out in qualifying, plus race control and weather.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []}},
    {"name": "circuit_history", "description": "A circuit's race context for a round: laps, pit-lane loss, Safety Car / VSC rate per lap used by the model, and its 2024-25 races (Safety Cars, VSCs, pit loss, overtaking difficulty).",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}}, "additionalProperties": False, "required": ["round"]}},
    {"name": "race_stints", "description": "Tyre stints every driver (or one driver) actually ran in a finished race: compound and laps of each stint, whether the set was new, red-flag changes, and the Safety Car / VSC / red-flag laps.",
     "input_schema": {"type": "object", "properties": {"round": {"type": "integer"}, "driver": {"type": ["string", "null"]}}, "additionalProperties": False, "required": ["round", "driver"]}},
    {"name": "live_call", "description": "Current live pit-wall call for a driver: box/stay out, target lap, next tyre, rejoin position, undercut threats, the top plans and the race simulation. Uses the visitor's live tracker, or F1 live timing while a race is on air.",
     "input_schema": {"type": "object", "properties": {"driver": {"type": "string"}}, "additionalProperties": False, "required": ["driver"]}},
    {"name": "backtest", "description": "Season walk-forward validation metrics of the prediction model vs a naive baseline.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []}},
]
for _t in TOOLS:
    _t["strict"] = True

HANDLERS = {
    "list_races": lambda a: tool_list_races(),
    "race_summary": lambda a: tool_race_summary(int(a["round"])),
    "driver_strategy": lambda a: tool_driver_strategy(int(a["round"]), a["driver"]),
    "evaluate_plan": lambda a: tool_evaluate_plan(int(a["round"]), a["driver"], a["compounds"], [int(x) for x in a["pit_laps"]]),
    "safety_car_whatif": lambda a: tool_safety_car_whatif(int(a["round"]), a["driver"], int(a["sc_lap"])),
    "live_race": lambda a: tool_live_race(),
    "live_session": lambda a: tool_live_session(),
    "circuit_history": lambda a: tool_circuit_history(int(a["round"])),
    "race_stints": lambda a: tool_race_stints(int(a["round"]), a.get("driver")),
    "live_call": lambda a: tool_live_call(a["driver"]),
    "backtest": lambda a: tool_backtest(),
}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    try:
        return json.dumps(HANDLERS[name](args), default=str), False
    except Exception as exc:
        log.exception("tool %s failed", name)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------
_llm_down_until = 0.0
# A public site with a paid key: cap AI-mode questions per visitor and in total
# per hour (PITWALL_CHAT_LLM_PER_VISITOR / PITWALL_CHAT_LLM_PER_HOUR); past the
# cap the engine answers.
LLM_PER_VISITOR = int(os.environ.get("PITWALL_CHAT_LLM_PER_VISITOR", "30"))
LLM_PER_HOUR = int(os.environ.get("PITWALL_CHAT_LLM_PER_HOUR", "400"))
_llm_calls: dict[str, list[float]] = {}
_llm_calls_lock = threading.Lock()


def _llm_allowed(visitor: str) -> bool:
    now = time.time()
    with _llm_calls_lock:
        for k in list(_llm_calls):
            _llm_calls[k] = [t for t in _llm_calls[k] if now - t < 3600]
            if not _llm_calls[k]:
                del _llm_calls[k]
        mine = _llm_calls.setdefault(visitor, [])
        if len(mine) >= LLM_PER_VISITOR or sum(len(v) for v in _llm_calls.values()) >= LLM_PER_HOUR:
            return False
        mine.append(now)
        return True


def llm_enabled() -> bool:
    if time.time() < _llm_down_until:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("PITWALL_USE_LLM") == "1")


def _season_note() -> str:
    from src.pitwall.api import default_round

    s = _season()
    rnd = s.get("live_round") or s.get("next_round")
    ev = next((r for r in s["rounds"] if r["round"] == rnd), None)
    if not ev:
        return " The 2026 season is over."
    when = "Its race is on air now." if s.get("live_round") else f"Race start (UTC): {ev.get('race_start_utc')}."
    built = default_round(s)
    data = "" if built == rnd else f" Its data is built after its practice sessions; the latest built round is R{built}."
    return f" Current race weekend: R{rnd} {ev['event']} ({ev['location']}). {when}{data}"


def chat_llm(messages: list[dict], context: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    convo: list[Any] = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
    page = context.get("page") or "/"
    note = f"\n\nThe user is on page {page}" + (f" (round {context['round']})." if context.get("round") else ".") + _season_note()
    used: list[str] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM + note,
            tools=TOOLS,
            messages=convo,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return {"reply": "I can't help with that one — ask me about race strategy, tyres or the model.", "tools_used": used, "mode": "LLM"}
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            return {"reply": text or "(no answer)", "tools_used": used, "mode": "LLM + engine tools"}
        convo.append({"role": "assistant", "content": resp.content})
        results = []
        for b in tool_uses:
            used.append(b.name)
            content, is_err = run_tool(b.name, dict(b.input))
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_err})
        convo.append({"role": "user", "content": results})
    return {"reply": "That needed more engine calls than I allow per question — try narrowing it down.", "tools_used": used, "mode": "LLM"}


# ---------------------------------------------------------------------------
# Engine mode (no LLM): intent parsing + templated answers
# ---------------------------------------------------------------------------
# What people call a circuit when it isn't the schedule's location or country.
ROUND_ALIASES = {"vegas": "las vegas", "interlagos": "sao paulo", "cota": "austin", "spa": "spa-francorchamps", "albert park": "melbourne"}


def _plain(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _round_named(text: str) -> int | None:
    """The round a question names (race, place, country, nickname or 'round N'), if any."""
    t = _plain(text)
    for alias, loc in ROUND_ALIASES.items():
        t = re.sub(rf"\b{alias}\b", loc, t)
    for r in _season()["rounds"]:
        keys = {_plain(r["location"]), _plain(r["event"]).replace(" grand prix", ""), _plain(r["country"])}
        if any(k and re.search(rf"\b{re.escape(k)}\b", t) for k in keys):
            return r["round"]
    m = re.search(r"\b(?:round|r)\s?(\d{1,2})\b", t)
    return int(m.group(1)) if m else None


def _find_round(text: str, context: dict) -> tuple[int, str]:
    """The round a question is about, and a note when it had to fall back: a
    round is answerable only once its rNN.json is built (after its practice)."""
    from src.pitwall.api import DATA, default_round

    s = _season()
    current = default_round(s) or 16
    rnd = None
    if not any(k in _plain(text) for k in ("tonight", "today", "this race", "live")):
        rnd = _round_named(text)
        if rnd is None:
            rnd = context.get("round")
    if rnd is None or rnd == current or (DATA / f"r{rnd:02d}.json").exists():
        return rnd or current, ""
    ev = next((r["event"] for r in s["rounds"] if r["round"] == rnd), f"Round {rnd}")
    return current, f"R{rnd} {ev} is built once its practice sessions have run, so this is R{current}. "


def _with_grid(rnd: int) -> int:
    """rnd, or the latest earlier round with a grid: until qualifying, the next
    race has no drivers or predictions yet."""
    for r in range(rnd, 0, -1):
        try:
            if _load(r)["drivers"]:
                return r
        except Exception:  # round not built
            continue
    return rnd


def _find_driver(text: str, rnd: int) -> str | None:
    d = _load(rnd)
    t = text.upper()
    for x in d["drivers"]:
        if re.search(rf"\b{x['driver']}\b", t) or x["name"].upper().split()[-1] in t:
            return x["driver"]
    return None


def _fmt_plan(p: dict | None) -> str:
    if not p:
        return "no plan"
    comps = " → ".join(c.title() for c in p["compounds"])
    laps = ", ".join(f"L{l}" for l in p["pit_laps"]) or "no stop"
    extra = f" (P(best) {p['p_best']:.0%})" if "p_best" in p else ""
    return f"{comps}, pit {laps}{extra}"


SESSION_WORDS = ("practice", "fp1", "fp2", "fp3", "qualifying", "quali", "fastest lap", "pole")


def _season_current() -> int:
    from src.pitwall.api import default_round

    return default_round(_season()) or 16


def _lap(sec: float | None) -> str:
    return f"{int(sec // 60)}:{sec % 60:06.3f}" if sec else "-"


LIVE_RACE_WORDS = ("leading", "leader", "running order", "positions", "race control", "track status", "right now", "happening", "who is winning", "who's winning")

# General questions the engine answers without an LLM, when no driver or race
# is named (the AI mode explains anything). Specific topics first.
CONCEPTS = (
    (("tyre model", "tire model", "degradation", "tyre wear", "tire wear", "wear model"),
     "Every lap time is modelled as driver pace + compound offset + wear × tyre age (wear grows with fuel load) − fuel burn per lap. "
     "Wear is measured as the drop-off inside each stint of every earlier 2026 race, updated with the weekend's practice race sims "
     "(scaled by a practice-to-race ratio learned on earlier weekends) and, during a race, with every car's green-flag laps. "
     "Each driver also has a personal wear multiplier learned from their own stints."),
    (("undercut", "overcut"),
     "An undercut is pitting before a rival so the fresh tyres gain more than the stop costs, and you come out ahead when they stop. "
     "An overcut is staying out longer, when old tyres are still quick or the rival would rejoin in traffic. "
     "On the Live pit wall, the race-craft card shows where the selected car would rejoin if it boxed now, and which rivals threaten an "
     "undercut, including what it still gains if they cover on the next lap."),
    (("safety car", "vsc", "virtual safety"),
     "A stop under a Safety Car or VSC costs much less than a green-flag stop because the field is slowed, so it is often a cheap stop. "
     "The engine uses per-lap Safety Car and VSC chances from 2026 races blended with each circuit's 2024-25 record, includes them in every "
     "simulated race, and the live call flags a cheap stop when one comes out. Ask e.g. 'Safety Car on lap 20 for Norris?' for a what-if."),
    (("monte carlo", "simulated races", "race simulation", "simulation"),
     "Two kinds of simulation. The strategy Monte Carlo replays the rest of the race hundreds of times with random Safety Cars, VSCs and "
     "tyre-model uncertainty, and reports how often each plan is fastest (close calls get up to four times as many runs). The race simulation "
     "runs the whole field lap by lap, with traffic, overtaking learned per circuit, pit stops and Safety Cars, and ranks each driver's plans "
     "by expected finishing position."),
    (("optimiser", "optimizer", "optimise", "optimize", "dynamic programming"),
     "The optimiser is exact dynamic programming over every 0-3 stop compound sequence and every pit lap, with the two-compound rule, "
     "stint-life limits and each driver's remaining tyre sets. It solves a race in about 10 ms, so it re-plans every lap during a race."),
    (("conformal", "prediction interval", "uncertainty", "confidence interval"),
     "First-stop intervals are split-conformal: the errors on earlier races set an 80% interval (with a finite-sample correction), and "
     "coverage is measured rather than assumed: {coverage} of actual first stops fell inside on the 2026 backtest."),
    (("walk-forward", "walk forward", "leakage", "how was it tested", "how is it tested", "how do you test"),
     "Every race is predicted with models trained only on earlier races (walk-forward), never on random splits, so the scorecard shows how "
     "the engine would have done in real time. Ask 'how accurate are you?' for the numbers."),
    (("behaviour model", "behavior model", "likely call", "team call", "what teams will do"),
     "The 'likely call' comes from a walk-forward logistic regression that learns how teams actually choose the stop count and starting tyre, "
     "from physics features (1- vs 2-stop gap, optimal stop lap, wear, grid slot, practice tyre use). The optimiser then times the plan inside "
     "that choice. 'Fastest' and 'likely' are reported separately."),
    (("how does it work", "how does this work", "how do you work", "how does pitwall", "what is pitwall", "what can you do",
      "what do you do", "methodology", "how are the predictions made", "how do you predict"),
     "PitWall AI models every car's lap times (pace, tyre compound, wear, fuel), learns them from 2026 races and each weekend's practice, "
     "finds the fastest plan exactly, stress-tests it in simulated races with Safety Cars, and simulates the whole field to rank plans by "
     "finishing position. During a race it re-plans every lap from F1 live timing. Ask me about a driver's strategy, a race result, a Safety "
     "Car what-if, who's leading right now, or how accurate the model is."),
)
QUESTION_WORDS = ("how", "what", "what's", "whats", "why", "explain", "mean", "means", "tell me about")


def _concept(t: str) -> str | None:
    if not any(re.search(rf"(?<![\w']){re.escape(w)}(?![\w'])", t) for w in QUESTION_WORDS):
        return None
    text = next((text for keys, text in CONCEPTS if any(k in t for k in keys)), None)
    if text and "{coverage}" in text:
        cov = (tool_backtest() or {}).get("conformal_coverage")
        text = text.replace("{coverage}", f"{cov:.0%}" if cov is not None else "the reported share")
    return text


def chat_engine(messages: list[dict], context: dict) -> dict:
    q = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    t = q.lower()
    used: list[str] = []
    try:
        if any(k in t for k in ("accurate", "accuracy", "backtest", "validation", "how good")):
            used.append("backtest")
            b = tool_backtest()
            return {"reply": (f"Walk-forward backtest ({b['driver_races']} driver-races, dry races from R4): stops correct {b['stops_accuracy']:.0%} "
                              f"(baseline {b['baseline']['stops_accuracy']:.0%}), compound set {b['compound_set_accuracy']:.0%} (baseline {b['baseline']['compound_set_accuracy']:.0%}), "
                              f"first-stop error {b['first_stop_mae']:.1f} laps (baseline {b['baseline']['first_stop_mae']:.1f}). "
                              f"80% conformal intervals covered {b['conformal_coverage']:.0%} of actual first stops."),
                    "tools_used": used, "mode": "engine (no LLM key configured)"}
        asked, note = _find_round(q, context)
        rnd = _with_grid(asked)
        drv = _find_driver(q, rnd)
        if rnd != asked:
            if drv:
                note += f"R{asked} {_load(asked)['event']} has no grid until qualifying, so this is from R{rnd} {_load(rnd)['event']}. "
            else:
                rnd = asked
        if not drv and _round_named(q) is None and (text := _concept(t)):
            return {"reply": text, "tools_used": used, "mode": "engine (no LLM key configured)"}
        if any(k in t for k in SESSION_WORDS) and not note and asked == _season_current():
            used.append("live_session")
            r = tool_live_session()
            if "error" not in r:
                ses = r["session"]
                timed = [c for c in r["cars"] if c.get("best_lap")]
                order = " · ".join(f"{c['position']}. {c['driver']} {_lap(c['best_lap'])} ({c['tyre'].title()})" for c in timed[:10])
                state = {"Started": "live", "Ends": "finished", "Finished": "finished", "Finalised": "finished", "Inactive": "not started yet"}.get(ses.get("status"), "on air")
                return {"reply": f"{ses.get('meeting') or ''} {ses.get('name') or 'session'} ({state}): "
                                 f"{order or 'no timed laps yet'}.", "tools_used": used, "mode": "engine (no LLM key configured)"}
            if any(k in t for k in ("qualifying", "quali", "pole")):  # the latest qualifying instead
                d = _load(rnd)
                order = sorted((x for x in d["drivers"] if x.get("quali")), key=lambda x: x["quali"])
                if order:
                    return {"reply": f"{r['error']} Latest qualifying, {d['event']}: " + " · ".join(f"{x['quali']}. {x['driver']}" for x in order[:10]) + ".",
                            "tools_used": used, "mode": "engine (no LLM key configured)"}
            return {"reply": r["error"], "tools_used": used, "mode": "engine (no LLM key configured)"}
        if any(k in t for k in ("history", "last year", "previous years", "before at", "usually")):
            used.append("circuit_history")
            r = tool_circuit_history(rnd if rnd == asked else asked)
            v = r["venue"]
            return {"reply": (f"{note}{r['circuit']} ({r['event']}): {r['laps']} laps, pit loss {r['pit_loss_s']}s. In {r['earlier_races_here']} race(s) here in "
                              f"2024-25: {v.get('sc_starts')} Safety Car(s), {v.get('vsc_starts')} VSC(s). The model uses {r['safety_car_per_lap'] * 100:.1f}% SC "
                              f"and {r['vsc_per_lap'] * 100:.1f}% VSC per lap."), "tools_used": used, "mode": "engine (no LLM key configured)"}
        if any(k in t for k in ("stint", "tyres did", "tyres were", "what tyres", "which tyres")) and _load(asked).get("overview"):
            used.append("race_stints")
            r = tool_race_stints(asked, drv)
            lines = [f"{k}: " + " → ".join(f"{x['compound'].title()} L{x['laps']}" for x in v) for k, v in list(r["stints"].items())[:22]]
            return {"reply": f"{r['event']} stints:\n" + "\n".join(lines), "tools_used": used, "mode": "engine (no LLM key configured)"}
        if not drv and any(k in t for k in LIVE_RACE_WORDS):
            used.append("live_race")
            r = tool_live_race()
            if "error" in r:
                return {"reply": r["error"], "tools_used": used, "mode": "engine (no LLM key configured)"}
            order = " · ".join(f"{c['position']}. {c['driver']} {'+' + format(c['gap_to_leader_s'], '.1f') + ' ' if c.get('gap_to_leader_s') else ''}"
                               f"{SHORT.get(c['tyre'] or '', '?')}{c['tyre_age_laps']}" for c in r["cars"][:10] if c.get("position") and not c.get("retired"))
            rc = f" Latest from race control: {r['race_control'][0]['message']}." if r.get("race_control") else ""
            return {"reply": f"{r['event']}, lap {r['lap']}/{r['total_laps']}, track {r['track_status']}. {order}.{rc}",
                    "tools_used": used, "mode": "engine (no LLM key configured)"}
        if drv and ("safety car" in t or re.search(r"\bsc\b", t)):
            m = re.search(r"lap\s?(\d{1,2})", t)
            lap = int(m.group(1)) if m else 20
            used.append("safety_car_whatif")
            r = tool_safety_car_whatif(rnd, drv, lap)
            gain = f" — saves about {r['time_saved_by_boxing_s']}s vs staying out" if r.get("time_saved_by_boxing_s") else ""
            return {"reply": (f"{note}SC on lap {lap}: {drv} would be on {r['tyre_at_sc']['compound'].title()}s, {r['tyre_at_sc']['age_laps']} laps old. "
                              f"A stop costs ~{r['pit_cost_under_sc_s']}s under SC vs {r['green_pit_cost_s']}s green. Call: {r['call']}{gain}. "
                              f"Plan from there: {_fmt_plan(r['best_plan_from_there'])}."), "tools_used": used, "mode": "engine (no LLM key configured)"}
        if drv and any(k in t for k in ("now", "live", "box", "this lap")):
            used.append("live_call")
            r = tool_live_call(drv)
            if "error" not in r:
                return {"reply": f"Lap {r['lap']}/{r['total_laps']}, P{r['position']} on {r['tyre']['compound'].title()}s ({r['tyre']['age']} laps): {r['call']['headline']}.",
                        "tools_used": used, "mode": "engine (no LLM key configured)"}
            note += f"{r['error']} The pre-race plan instead: "
        if drv:
            used.append("driver_strategy")
            r = tool_driver_strategy(rnd, drv)
            if "error" in r:
                return {"reply": r["error"], "tools_used": used, "mode": "engine (no LLM key configured)"}
            lines = [f"{note}{r['name']} ({r['team']}), {r['event']}: fastest plan {_fmt_plan(r['fastest_plan_physics'])}."]
            if r.get("likely_plan_team_behaviour_model"):
                lk = r["likely_plan_team_behaviour_model"]
                probs = ", ".join(f"{k}-stop {v:.0%}" for k, v in sorted((lk.get("stop_count_probabilities") or {}).items()))
                lines.append(f"Most likely team call: {_fmt_plan(lk)} ({probs}).")
            if r.get("stop_count_odds_physics"):
                odds = ", ".join(f"{k}-stop {v:.0%}" for k, v in sorted(r["stop_count_odds_physics"].items()) if v > 0)
                lines.append(f"Stop-count odds (simulated): {odds}.")
            if r.get("first_stop_80pct_interval"):
                lo, hi = r["first_stop_80pct_interval"]
                lines.append(f"80% interval for the first stop: laps {lo}–{hi}.")
            if any(k in t for k in ("compare", "one-stop", "two-stop", "1-stop", "2-stop", "options", "alternative")):
                for a in r["alternatives"][:4]:
                    lines.append(f"• {a['plan']} pit {a['pit_laps']}: +{a['seconds_slower_than_fastest']}s, P(best) {a['p_best']:.0%}, expected regret {a['expected_regret_s']}s")
            if r.get("actual"):
                lines.append(f"Actual: {r['actual']['plan']} pitting {r['actual']['pit_laps']} (finished P{r['actual']['finish']}).")
                if r.get("actual_vs_hindsight_s") is not None:
                    lines.append(f"Hindsight optimum was {r['hindsight_optimum']['plan']}; actual cost ~{r['actual_vs_hindsight_s']}s in model time.")
            return {"reply": "\n".join(lines), "tools_used": used, "mode": "engine (no LLM key configured)"}
        used.append("race_summary")
        r = tool_race_summary(rnd)
        mix = ", ".join(f"{k} ×{v}" for k, v in r["predicted_strategies_count"].items())
        mix = f"Predicted strategy mix: {mix}." if mix else "Strategy predictions appear once qualifying sets the grid."
        lines = [f"{note}R{rnd} {r['event']} ({r['location']}, {r['laps']} laps, pit loss {r['pit_loss_s']}s). {mix}"]
        if r.get("top5"):
            lines.append("Result: " + "; ".join(r["top5"]) + ".")
            lines.append("Actual strategies: " + ", ".join(f"{k} ×{v}" for k, v in r["actual_strategies_count"].items()) + ".")
        lines.append("Name a driver (e.g. 'VER' or 'Leclerc') for their plan, or ask about a Safety Car on a given lap.")
        return {"reply": "\n".join(lines), "tools_used": used, "mode": "engine (no LLM key configured)"}
    except Exception as exc:
        log.exception("engine chat failed")
        detail = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
        return {"reply": f"The engine couldn't answer that one ({detail}). Try naming a driver or a race, e.g. 'Verstappen at Las Vegas'.",
                "tools_used": used, "mode": "engine"}


def chat(messages: list[dict], context: dict) -> dict:
    global _llm_down_until
    _SESSION.set(context.get("session") or "default")
    if llm_enabled() and _llm_allowed(_SESSION.get()):
        import anthropic

        try:
            return chat_llm(messages, context)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            log.warning("LLM credentials rejected; engine mode for 1h: %s", exc)
            _llm_down_until = time.time() + 3600
        except anthropic.RateLimitError as exc:
            log.warning("LLM rate limited; answering from the engine: %s", exc)
        except anthropic.APIStatusError as exc:
            log.warning("LLM API error %s; engine mode for 5 min", exc.status_code)
            _llm_down_until = time.time() + 300
        except anthropic.APIConnectionError as exc:
            log.warning("LLM unreachable; engine mode for 5 min: %s", exc)
            _llm_down_until = time.time() + 300
        except Exception as exc:  # e.g. no resolvable credentials
            log.warning("LLM unavailable (%s); engine mode for 5 min", exc)
            _llm_down_until = time.time() + 300
    return chat_engine(messages, context)
