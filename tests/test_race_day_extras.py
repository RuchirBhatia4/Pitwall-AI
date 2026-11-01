"""Red flags, time penalties, double-stacking and free stops in the live call."""
import pytest

from src.pitwall.api import weekend_for
from src.pitwall.live_engine import analyse
from src.pitwall.live_sources import ReplaySource, outstanding_penalties


def test_outstanding_penalties():
    msgs = [
        "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 63 (RUS) - LEAVING THE TRACK AND GAINING AN ADVANTAGE",
        "FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 87 (BEA) - CAUSING A COLLISION",
        "FIA STEWARDS: PENALTY SERVED - 5 SECOND TIME PENALTY FOR CAR 63 (RUS) - LEAVING THE TRACK",
        "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 87 (BEA) - SPEEDING IN THE PIT LANE",
        "CAR 44 (HAM) TIME 1:22.123 DELETED - TRACK LIMITS AT TURN 4",
    ]
    assert outstanding_penalties(msgs) == {"87": 15}


@pytest.fixture(scope="module")
def austin():
    src = ReplaySource(18)
    return src, weekend_for(18)


def _state(src, lap):
    st = src.state(lap)
    return st, next(c for c in st["cars"] if c["position"] == 5)


def test_red_flag_is_a_free_tyre_change(austin):
    src, w = austin
    st, car = _state(src, 30)
    st["track_status"] = "RED"
    call = analyse(st, w, car["driver"], sims=40)["call"]
    assert call["headline"].startswith("Red flag:")
    assert call["action"] in ("BOX_NOW", "STAY_OUT")


def test_penalty_is_served_at_the_stop(austin):
    src, w = austin
    st, car = _state(src, 30)
    base = analyse(st, w, car["driver"], sims=40)["rejoin"]
    car["penalty_s"] = 5
    rj = analyse(st, w, car["driver"], sims=40)["rejoin"]
    assert rj["penalty_s"] == 5
    assert rj["stop_cost"] == pytest.approx(base["stop_cost"] + 5, abs=0.11)


def test_queue_behind_teammate_in_the_pit_box(austin):
    src, w = austin
    st, car = _state(src, 30)
    mate = next(c for c in st["cars"] if c is not car and c["team"] == car["team"])
    mate["in_pit"] = True
    mate["gap_to_leader"] = car["gap_to_leader"] - 1.5
    rj = analyse(st, w, car["driver"], sims=40)["rejoin"]
    assert rj["stack_wait"] == pytest.approx(3.0, abs=0.05)


def test_free_stop_flag(austin):
    src, w = austin
    st, car = _state(src, 30)
    rj = analyse(st, w, car["driver"], sims=40)["rejoin"]
    assert rj["free"] == (rj["position"] <= car["position"])


def test_hards_warm_up_slower_than_mediums(austin):
    from src.pitwall.live_engine import OUTLAP_WARMUP
    assert OUTLAP_WARMUP["HARD"] > OUTLAP_WARMUP["MEDIUM"] == OUTLAP_WARMUP["SOFT"]


def test_threats_say_what_a_cover_leaves(austin):
    src, w = austin
    st, car = _state(src, 12)
    for t in analyse(st, w, car["driver"], sims=40)["threats"]:
        assert "covered_gain" in t and isinstance(t["live_if_covered"], bool)
        # a cover can only take gain away, never make an undercut work that would not
        if t["live_if_covered"]:
            assert t["gap"] is not None and t["covered_gain"] > t["gap"]


def test_held_up_names_the_car_ahead(austin):
    src, w = austin
    st, car = _state(src, 30)
    car["interval"] = 0.6
    car["in_pit"] = False
    a = analyse(st, w, car["driver"], sims=40)
    ahead = next(c for c in st["cars"] if c["position"] == 4)
    assert a["held_up"] == {"driver": ahead["driver"], "gap": 0.6}


def test_graining_shape():
    import numpy as np
    from src.pitwall.live_engine import _graining_cleared
    ages = np.arange(1, 11)
    assert _graining_cleared(np.array([0, 0.1, 0.9, 1.0, 1.1, 0.8, 0.2, 0.1, 0.0, 0.1]), ages, 0.6)
    assert not _graining_cleared(np.array([0, 0.1, 0.9, 0.1, 0.0, 0.8, 0.2, 0.1, 0.0, 0.1]), ages, 0.6)  # one-off slow laps
    assert not _graining_cleared(np.array([0, 0.1, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]), ages, 0.6)  # wear, not graining


def test_track_temperature_note():
    from src.pitwall.live_engine import track_temperature
    assert track_temperature({"track_temp": 50.0, "track_temp_start": 44.0})["note"].startswith("Track +6°C")
    assert track_temperature({"track_temp": 46.0, "track_temp_start": 44.0})["note"] is None
    assert track_temperature({}) is None


def test_rain_forecast(monkeypatch):
    from src.pitwall import forecast
    forecast._cache.clear()
    monkeypatch.setattr(forecast, "_fetch", lambda lat, lon: {"next_hour": 10, "hour_after": 60, "mm_2h": 1.2, "from": "x", "source": "Open-Meteo"})
    assert forecast.rain_forecast("Mexico City")["hour_after"] == 60
    assert forecast.rain_forecast("Nowhere") is None

    def boom(lat, lon):
        raise OSError("offline")
    forecast._cache.clear()
    monkeypatch.setattr(forecast, "_fetch", boom)
    assert forecast.rain_forecast("Mexico City") is None  # never takes the live page down
