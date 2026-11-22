"""The scheduled data update: which steps are due, and when."""
import json
from datetime import datetime

from src.pitwall import auto_update


def _season(tmp_path, monkeypatch, done=None):
    season = {"rounds": [
        {"round": 20, "sessions": {"Practice 1": "2026-11-06 15:30:00", "Qualifying": "2026-11-07 18:00:00", "Race": "2026-11-08 17:00:00"}},
        {"round": 21, "sessions": {"Practice 1": "2026-11-20 00:30:00", "Practice 2": "2026-11-20 04:00:00", "Practice 3": "2026-11-21 00:30:00",
                                   "Qualifying": "2026-11-21 04:00:00", "Race": "2026-11-22 04:00:00"}},
        {"round": 22, "sessions": {"Practice 1": "2026-11-27 13:30:00", "Race": "2026-11-29 16:00:00"}},
    ]}
    (tmp_path / "season.json").write_text(json.dumps(season))
    if done is not None:
        (tmp_path / "updates.json").write_text(json.dumps(done))
    monkeypatch.setattr(auto_update, "OUT", tmp_path)
    monkeypatch.setattr(auto_update, "STATE", tmp_path / "updates.json")


def test_practice_is_due_once_openf1_has_had_time(tmp_path, monkeypatch):
    _season(tmp_path, monkeypatch, {"21": ["Practice 1", "Practice 2"]})
    assert auto_update.due(datetime(2026, 11, 21, 1, 0)) == []  # FP3 still running
    assert auto_update.due(datetime(2026, 11, 21, 2, 30)) == [(21, "Practice 3")]
    assert auto_update.due(datetime(2026, 11, 21, 6, 0)) == [(21, "Practice 3"), (21, "Qualifying")]


def test_grid_refresh_before_the_race_but_never_close_to_the_start(tmp_path, monkeypatch):
    _season(tmp_path, monkeypatch, {"21": ["Practice 1", "Practice 2", "Practice 3", "Qualifying"]})
    assert auto_update.due(datetime(2026, 11, 22, 1, 30)) == [(21, "Grid")]
    assert auto_update.due(datetime(2026, 11, 22, 3, 40)) == []  # 20 min to the start: no redeploy
    assert auto_update.due(datetime(2026, 11, 22, 7, 30)) == [(21, "Race")]


def test_old_weekends_and_finished_steps_are_left_alone(tmp_path, monkeypatch):
    _season(tmp_path, monkeypatch, {"21": ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Grid", "Race"]})
    assert auto_update.due(datetime(2026, 11, 24, 12, 0)) == []  # R20 is weeks old, R21 is done, R22 has not run
    assert auto_update.due(datetime(2026, 11, 27, 15, 30)) == [(22, "Practice 1")]


def test_nothing_is_published_while_a_session_is_on(tmp_path, monkeypatch):
    _season(tmp_path, monkeypatch, {"21": ["Practice 1", "Practice 2", "Practice 3"]})
    # qualifying ended long ago in this test's eyes, but the race is running: a deploy would drop the live feed
    for t in (datetime(2026, 11, 22, 3, 35), datetime(2026, 11, 22, 4, 30), datetime(2026, 11, 22, 6, 55)):
        assert auto_update.due(t) == [], t
    assert auto_update.due(datetime(2026, 11, 22, 7, 20)) == [(21, "Qualifying"), (21, "Race")]
    _season(tmp_path, monkeypatch, {"21": ["Practice 1"]})
    assert auto_update.due(datetime(2026, 11, 20, 4, 30)) == []  # FP2 is on
    assert auto_update.due(datetime(2026, 11, 20, 5, 45)) == [(21, "Practice 2")]
