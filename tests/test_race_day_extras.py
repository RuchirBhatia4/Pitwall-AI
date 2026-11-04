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


def test_text_only_red_flag_lasts_until_the_restart():
    import pandas as pd
    from src.pitwall.openf1_data import OpenF1Session
    t = pd.Timestamp("2026-11-01T21:00:00Z")
    rc = pd.DataFrame([
        {"date": t, "message": "SAFETY CAR DEPLOYED", "flag": None, "scope": None, "category": "SafetyCar"},
        {"date": t + pd.Timedelta(minutes=2), "message": "RED FLAG - RACE SUSPENDED", "flag": None, "scope": None, "category": "Other"},
        {"date": t + pd.Timedelta(minutes=4), "message": "TRACK CLEAR", "flag": "CLEAR", "scope": "Track", "category": "Flag"},
        {"date": t + pd.Timedelta(minutes=41), "message": "SESSION STARTED", "flag": None, "scope": None, "category": "SessionStatus"},
    ])
    s = OpenF1Session.__new__(OpenF1Session)
    s._rel = lambda d: pd.to_datetime(d) - t
    ts = s._track_status(rc)
    assert list(ts["Status"]) == ["1", "4", "5", "1"]
    assert ts["Time"].iloc[-1] == pd.Timedelta(minutes=41)  # not at TRACK CLEAR, mid-suspension


def test_safety_car_stop_just_before_a_red_flag_is_a_stop():
    import pandas as pd
    from src.pitwall.race_data import stints_from_laps
    rows = []
    for drv, in_status in (("SCSTOP", "14"), ("REDCHG", "45")):
        for lap in range(1, 48):
            stint, comp, life = (1, "MEDIUM", lap) if lap <= 44 else (2, "SOFT", lap - 44)
            ts = in_status if lap == 44 else "45" if lap == 45 else "1"
            rows.append({"driver": drv, "lap": lap, "stint": stint, "compound": comp, "tyre_life": float(life), "fresh": True, "track_status": ts})
    st = stints_from_laps(pd.DataFrame(rows)).set_index("driver")
    assert not st.loc["SCSTOP"].iloc[1]["red_flag"]  # pitted under the Safety Car, before the red flag
    assert st.loc["REDCHG"].iloc[1]["red_flag"]  # came in after the red flag: a free change, not a stop


def _synthetic_cars(fuel, deg, stop_laps, total=50, seed=3):
    """Cars whose lap time = pace + deg * tyre age - fuel * lap + noise; one stop each."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cars = []
    for i, stop in enumerate(stop_laps):
        laps, pace = [], 80 + 0.3 * i
        for lap in range(1, total + 1):
            first = lap <= stop
            age = lap if first else lap - stop
            t = pace + deg * age - fuel * lap + rng.normal(0, 0.15)
            laps.append({"lap": lap, "time": t, "compound": "MEDIUM" if first else "HARD", "age": age,
                         "pit_in": lap == stop, "pit_out": lap == stop + 1, "neutral": False})
        cars.append({"driver": f"D{i}", "team": f"T{i % 5}", "laps": laps})
    return cars


def test_live_fuel_recovers_the_race_rate():
    from src.pitwall.api import weekend_for
    from src.pitwall.live_engine import live_fuel
    from src.pitwall.tyre_model import TyreModel
    prior = TyreModel.from_dict(weekend_for(18)["model"])
    prior.fuel_per_lap, prior.quad = 0.058, 0.0
    cars = _synthetic_cars(fuel=0.030, deg=0.04, stop_laps=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36])
    f = live_fuel(cars, prior)
    assert f["cars"] == 10
    assert abs(f["observed"] - 0.030) < 0.006
    assert abs(f["posterior"] - 0.030) < abs(prior.fuel_per_lap - 0.030) / 2  # pulled most of the way


def test_live_fuel_keeps_the_pre_race_rate_until_cars_have_stopped():
    from src.pitwall.api import weekend_for
    from src.pitwall.live_engine import FUEL_MIN_CARS, live_fuel
    from src.pitwall.tyre_model import TyreModel
    prior = TyreModel.from_dict(weekend_for(18)["model"])
    cars = _synthetic_cars(fuel=0.030, deg=0.04, stop_laps=[20] * (FUEL_MIN_CARS - 1) + [60] * 6)
    f = live_fuel(cars, prior)
    assert f["observed"] is None and f["posterior"] == prior.fuel_per_lap
