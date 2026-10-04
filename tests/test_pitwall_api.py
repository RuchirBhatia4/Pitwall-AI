"""API contract tests for the PitWall endpoints (use the built season artefacts)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

SEASON = Path(__file__).resolve().parents[1] / "data" / "season" / "2026" / "season.json"
pytestmark = pytest.mark.skipif(not SEASON.exists(), reason="run python -m src.pitwall.build_season first")
client = TestClient(app)


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


def test_chat_engine_mode_answers_from_tools(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PITWALL_USE_LLM", raising=False)
    j = client.post("/api/chat", json={"messages": [{"role": "user", "content": "When should Verstappen pit tonight?"}]}).json()
    assert "driver_strategy" in j["tools_used"]
    assert "Verstappen" in j["reply"]


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


def test_chat_live_answer_follows_its_own_session(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client.post("/api/live/connect", headers=A, json={"source": "replay", "round": 15, "seconds_per_lap": 60, "start_lap": 10})
    q = {"messages": [{"role": "user", "content": "Should Russell box now?"}]}
    a = client.post("/api/chat", headers=A, json=q).json()
    other = client.post("/api/chat", headers={"X-Pitwall-Session": "visitor-dddd-4444"}, json=q).json()
    assert a["reply"].startswith("Lap ")  # live call from A's own replay
    assert not other["reply"].startswith("Lap ")  # an unconnected visitor gets the pre-race plan instead
