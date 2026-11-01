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
