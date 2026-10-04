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
