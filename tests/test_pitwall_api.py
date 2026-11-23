"""API contract tests for the PitWall endpoints (use the built season artefacts)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.pitwall import api

SEASON = Path(__file__).resolve().parents[1] / "data" / "season" / "2026" / "season.json"
pytestmark = pytest.mark.skipif(not SEASON.exists(), reason="run python -m src.pitwall.build_season first")
client = TestClient(app)


@pytest.fixture
def after_mexico(monkeypatch):
    """The day after R19: the next round (R20) has no rNN.json until its practice."""
    monkeypatch.setattr(api, "_now", lambda: pd.Timestamp("2026-11-02 12:00"))


def test_season_lists_rounds_and_backtest():
    j = client.get("/api/season").json()
    assert len(j["rounds"]) >= 16
    assert j["backtest"]["driver_races"] > 0
    assert 0 <= j["backtest"]["conformal_coverage"] <= 1


def test_race_payload_has_predictions_for_grid():
    j = client.get("/api/races/16").json()
    assert j["weekend"]["total_laps"] > 40
    preds = [p for p in j["predictions"].values() if p.get("predicted")]
    assert len(preds) >= 18
    for p in preds:
        plan = p["predicted"]
        assert len(set(plan["sequence"])) >= 2
        assert len(plan["pit_laps"]) == len(plan["sequence"]) - 1


def test_unknown_round_is_404():
    assert client.get("/api/races/99").status_code == 404


def test_evaluate_rejects_single_compound_and_accepts_valid_plan():
    bad = client.post("/api/strategy/evaluate", json={"round": 16, "driver": "RUS", "sequence": ["HARD", "HARD"], "pit_laps": [20]})
    assert bad.status_code == 400
    ok = client.post("/api/strategy/evaluate", json={"round": 16, "driver": "RUS", "sequence": ["MEDIUM", "HARD"], "pit_laps": [22]})
    assert ok.status_code == 200 and ok.json()["delta_to_optimal"] >= -1e-6


def test_manual_live_call_under_safety_car():
    j = client.post("/api/live/manual", json={"round": 16, "driver": "RUS", "laps_completed": 20, "compound": "MEDIUM",
                                               "tyre_age": 20, "track_status": "SC", "position": 1, "gap_behind": 1.5}).json()
    assert j["analysis"]["call"]["action"] in {"BOX_NOW", "BOX_SOON", "STAY_OUT"}
    assert j["analysis"]["plans"]


def test_default_round_falls_back_to_latest_built_round(tmp_path, monkeypatch):
    for rnd in (1, 2, 3):
        (tmp_path / f"r{rnd:02d}.json").write_text("{}")
    monkeypatch.setattr(api, "DATA", tmp_path)
    assert api.default_round({"live_round": None, "next_round": 3}) == 3
    assert api.default_round({"live_round": 2, "next_round": 2}) == 2
    assert api.default_round({"live_round": None, "next_round": 4}) == 3  # R4 not built until its practice
    assert api.default_round({"live_round": None, "next_round": None}) == 3  # season over


def test_chat_engine_mode_answers_from_tools(monkeypatch, after_mexico):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    j = client.post("/api/chat", json={"messages": [{"role": "user", "content": "When should Verstappen pit tonight?"}]}).json()
    assert "driver_strategy" in j["tools_used"]
    assert "Verstappen" in j["reply"]


def test_chat_never_fails_on_a_round_that_is_not_built(monkeypatch, after_mexico):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    for q, ctx in (("What strategy will they run in Qatar?", {}), ("Best plan for Verstappen?", {"round": 23}), ("Who wins in Abu Dhabi?", {"round": 22})):
        j = client.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "context": ctx}).json()
        assert "couldn't answer" not in j["reply"], j["reply"]
        assert "is built once its practice sessions have run" in j["reply"]


def test_chat_explains_the_engine_without_an_llm(monkeypatch, after_mexico):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    ask = lambda q: client.post("/api/chat", json={"messages": [{"role": "user", "content": q}]}).json()
    for q, word in (("How does the tyre model work?", "stint"), ("What is an undercut?", "rejoin"), ("What can you do?", "Ask me"),
                    ("How do you handle uncertainty?", "conformal")):
        j = ask(q)
        assert word in j["reply"] and not j["tools_used"], (q, j["reply"])
    # a named race or driver still gets that race's or driver's answer
    assert ask("What happened with the safety car at Monza?")["tools_used"] == ["race_summary"]
    assert "driver_strategy" in ask("What is Verstappen's best strategy at Monza?")["tools_used"]


def test_chat_says_when_no_race_is_on(monkeypatch, after_mexico):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    j = client.post("/api/chat", headers={"X-Pitwall-Session": "visitor-eeee-5555"},
                    json={"messages": [{"role": "user", "content": "Should Verstappen box now?"}]}).json()
    assert j["reply"].startswith("No race is on air right now") and "pre-race plan" in j["reply"]


def test_session_question_between_weekends_gives_the_latest_qualifying(monkeypatch, tmp_path, after_mexico):
    # Between weekends the next round has no rNN.json, so no live feed, until its first practice.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    from src.pitwall import chat

    season = {**api._season(), "live_round": None, "next_round": 20}
    for f in api.DATA.glob("r[01][0-9].json"):  # rounds 1-19
        (tmp_path / f.name).symlink_to(f)
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(chat, "_season", lambda: season)
    ask = lambda q: client.post("/api/chat", json={"messages": [{"role": "user", "content": q}]}).json()["reply"]
    reply = ask("Who was fastest in qualifying?")
    assert reply.startswith("No session timing yet: R20") and "Latest qualifying, Mexico City Grand Prix: 1. " in reply, reply
    assert "first practice" in ask("Who topped FP1?")


def test_chat_live_answers_use_the_race_feed_without_connecting(monkeypatch):
    # During a race a visitor can ask the chat without opening the Live pit wall page.
    from src.pitwall import chat
    from src.pitwall.live_sources import ReplaySource

    st = ReplaySource(19).state(30) | {"source": "signalr", "session": {"name": "Race", "type": "Race", "waiting": False}}
    feed = type("Feed", (), {"state": lambda self: st})()
    monkeypatch.setattr(chat, "_season", lambda: {"live_round": 19, "next_round": 19, "rounds": []})
    monkeypatch.setattr(api.LIVE, "_feed", lambda kind, rnd, no_auth=False: feed)
    monkeypatch.setattr("src.pitwall.forecast.rain_forecast", lambda loc: None)
    chat._SESSION.set("visitor-not-connected-0001")
    race = chat.tool_live_race()
    assert race["lap"] == st["lap"] and len(race["cars"]) == len(st["cars"])
    call = chat.tool_live_call(st["cars"][0]["driver"])
    assert call["call"]["action"] and call["position"] == st["cars"][0]["position"]


def test_ai_mode_is_capped_per_visitor_and_per_hour(monkeypatch):
    from src.pitwall import chat
    monkeypatch.setattr(chat, "_llm_calls", {})
    monkeypatch.setattr(chat, "LLM_PER_VISITOR", 2)
    monkeypatch.setattr(chat, "LLM_PER_HOUR", 3)
    assert chat._llm_allowed("a") and chat._llm_allowed("a") and not chat._llm_allowed("a")
    assert chat._llm_allowed("b") and not chat._llm_allowed("c")


def test_chat_finds_races_by_nickname_and_without_accents():
    from src.pitwall.chat import _find_round
    assert _find_round("How did Interlagos go?", {})[0] == 20
    assert _find_round("sao paulo strategies", {})[0] == 20
    assert _find_round("Bahrain one-stop?", {})[0] == 16


# ---------------------------------------------------------------------------
# live tracker: per-visitor sessions
# ---------------------------------------------------------------------------
A = {"X-Pitwall-Session": "visitor-aaaa-1111"}
B = {"X-Pitwall-Session": "visitor-bbbb-2222"}


def test_visitors_do_not_switch_each_others_live_source():
    ra = client.post("/api/live/connect", headers=A, json={"source": "replay", "round": 15, "seconds_per_lap": 60, "start_lap": 10})
    rb = client.post("/api/live/connect", headers=B, json={"source": "replay", "round": 14, "seconds_per_lap": 60, "start_lap": 30})
    assert ra.status_code == 200 and rb.status_code == 200
    # B connecting afterwards must not change A
    assert client.get("/api/live/status", headers=A).json()["round"] == 15
    sa = client.get("/api/live/state", headers=A).json()["state"]
    sb = client.get("/api/live/state", headers=B).json()["state"]
    assert sa["round"] == 15 and sb["round"] == 14
    assert sa["laps_completed_leader"] == 10 and sb["laps_completed_leader"] == 30


def test_missing_or_invalid_session_header_uses_default_session():
    fresh = {"X-Pitwall-Session": "visitor-cccc-3333"}
    assert client.get("/api/live/status", headers=fresh).json()["connected"] is False
    bad = {"X-Pitwall-Session": "x"}  # too short: treated as the default session
    client.post("/api/live/connect", json={"source": "replay", "round": 13, "start_lap": 5})
    assert client.get("/api/live/status", headers=bad).json()["round"] == 13


def test_live_timing_only_for_current_round():
    r = client.post("/api/live/connect", headers=A, json={"source": "signalr", "round": 3})
    assert r.status_code == 400
    # A's existing replay selection is untouched by the rejected request
    assert client.get("/api/live/status", headers=A).json()["round"] == 15


def test_chat_live_answer_follows_its_own_session(monkeypatch, after_mexico):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client.post("/api/live/connect", headers=A, json={"source": "replay", "round": 15, "seconds_per_lap": 60, "start_lap": 10})
    q = {"messages": [{"role": "user", "content": "Should Russell box now?"}]}
    a = client.post("/api/chat", headers=A, json=q).json()
    other = client.post("/api/chat", headers={"X-Pitwall-Session": "visitor-dddd-4444"}, json=q).json()
    assert a["reply"].startswith("Lap ")  # live call from A's own replay
    assert not other["reply"].startswith("Lap ")  # an unconnected visitor gets the pre-race plan instead
