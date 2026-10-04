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
import time
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
Round 16, the Bahrain Grand Prix held at Sepang (Kuala Lumpur), is the current race weekend."""


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


def tool_live_call(driver: str) -> dict:
    from src.pitwall.api import LIVE, weekend_for
    from src.pitwall.live_engine import analyse

    sid = _SESSION.get()
    if LIVE.session(sid).kind is None:
        return {"error": "The live tracker is not connected. Connect a source on the Live pit wall page first."}
    st = LIVE.state(sid)
    code = driver.strip().upper()
    if not any(c["driver"] == code for c in st["cars"]):
        surname = {c["driver"]: c["driver"] for c in st["cars"]}
        return {"error": f"{driver} not found in live session; drivers: {', '.join(surname)}"}
    a = analyse(st, weekend_for(st["round"]), code, sims=200)
    return {
        "source": st["source"], "lap": a.get("lap"), "total_laps": a.get("total_laps"), "track_status": a.get("track_status"),
        "position": a["car"]["position"], "tyre": {"compound": a["car"]["compound"], "age": a["car"]["tyre_age"]},
        "call": a["call"], "rejoin_if_box_now": a.get("rejoin"), "threats": a.get("threats"),
        "top_plans": [_plan_summary(p) for p in (a.get("plans") or [])[:3]], "drift": a.get("drift"),
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
    {"name": "live_call", "description": "Current live pit-wall call for a driver (requires the live tracker to be connected): box/stay out, target lap, next tyre, rejoin position, undercut threats.",
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


def llm_enabled() -> bool:
    if time.time() < _llm_down_until:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("PITWALL_USE_LLM") == "1")


def chat_llm(messages: list[dict], context: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    convo: list[Any] = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
    page = context.get("page") or "/"
    note = f"\n\nThe user is on page {page}" + (f" (round {context['round']})." if context.get("round") else ".")
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
def _find_round(text: str, context: dict) -> int:
    s = _season()
    t = text.lower()
    if any(k in t for k in ("tonight", "today", "this race", "live", "sepang", "malaysia", "kuala lumpur", "bahrain")):
        return s.get("live_round") or s.get("next_round") or 16
    for r in s["rounds"]:
        keys = {r["location"].lower(), r["event"].lower().replace(" grand prix", ""), r["country"].lower()}
        if any(k and k in t for k in keys):
            return r["round"]
    m = re.search(r"\b(?:round|r)\s?(\d{1,2})\b", t)
    if m:
        return int(m.group(1))
    return context.get("round") or s.get("live_round") or s.get("next_round") or 16


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
        rnd = _find_round(q, context)
        drv = _find_driver(q, rnd)
        if drv and ("safety car" in t or re.search(r"\bsc\b", t)):
            m = re.search(r"lap\s?(\d{1,2})", t)
            lap = int(m.group(1)) if m else 20
            used.append("safety_car_whatif")
            r = tool_safety_car_whatif(rnd, drv, lap)
            gain = f" — saves about {r['time_saved_by_boxing_s']}s vs staying out" if r.get("time_saved_by_boxing_s") else ""
            return {"reply": (f"SC on lap {lap}: {drv} would be on {r['tyre_at_sc']['compound'].title()}s, {r['tyre_at_sc']['age_laps']} laps old. "
                              f"A stop costs ~{r['pit_cost_under_sc_s']}s under SC vs {r['green_pit_cost_s']}s green. Call: {r['call']}{gain}. "
                              f"Plan from there: {_fmt_plan(r['best_plan_from_there'])}."), "tools_used": used, "mode": "engine (no LLM key configured)"}
        if drv and any(k in t for k in ("now", "live", "box", "this lap")):
            used.append("live_call")
            r = tool_live_call(drv)
            if "error" not in r:
                return {"reply": f"Lap {r['lap']}/{r['total_laps']}, P{r['position']} on {r['tyre']['compound'].title()}s ({r['tyre']['age']} laps): {r['call']['headline']}.",
                        "tools_used": used, "mode": "engine (no LLM key configured)"}
        if drv:
            used.append("driver_strategy")
            r = tool_driver_strategy(rnd, drv)
            if "error" in r:
                return {"reply": r["error"], "tools_used": used, "mode": "engine (no LLM key configured)"}
            lines = [f"{r['name']} ({r['team']}), {r['event']}: fastest plan {_fmt_plan(r['fastest_plan_physics'])}."]
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
        lines = [f"R{rnd} {r['event']} ({r['location']}, {r['laps']} laps, pit loss {r['pit_loss_s']}s). Predicted strategy mix: {mix}."]
        if r.get("top5"):
            lines.append("Result: " + "; ".join(r["top5"]) + ".")
            lines.append("Actual strategies: " + ", ".join(f"{k} ×{v}" for k, v in r["actual_strategies_count"].items()) + ".")
        lines.append("Name a driver (e.g. 'VER' or 'Leclerc') for their plan, or ask about a Safety Car on a given lap.")
        return {"reply": "\n".join(lines), "tools_used": used, "mode": "engine (no LLM key configured)"}
    except Exception as exc:
        log.exception("engine chat failed")
        return {"reply": f"The engine couldn't answer that ({type(exc).__name__}: {exc}).", "tools_used": used, "mode": "engine"}


def chat(messages: list[dict], context: dict) -> dict:
    global _llm_down_until
    _SESSION.set(context.get("session") or "default")
    if llm_enabled():
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
