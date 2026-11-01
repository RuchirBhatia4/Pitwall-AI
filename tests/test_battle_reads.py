"""Cover / overcut reads against a rival (advisory cards next to the pit call)."""
import json
from pathlib import Path

import pytest

from src.pitwall.live_engine import battle_reads
from src.pitwall.tyre_model import TyreModel

R18 = Path(__file__).resolve().parents[1] / "data" / "season" / "2026" / "r18.json"


@pytest.fixture(scope="module")
def model():
    return TyreModel.from_dict(json.loads(R18.read_text())["weekend"]["model"])


def _car(drv, gap, stops=0, age=20, compound="MEDIUM", interval=None, in_pit=False):
    return {"driver": drv, "team": "", "gap_to_leader": gap, "stops": stops, "tyre_age": age, "compound": compound,
            "interval": interval, "in_pit": in_pit, "retired": False}


def test_cover_when_the_car_behind_has_just_pitted(model):
    me = _car("AAA", 10.0)
    rival = _car("BBB", 10.8 + 22.0 + 2.0, stops=1, age=1, compound="HARD")  # was 0.8 s behind, lost 22 s + a bit
    reads = battle_reads(me, [_car("LDR", 0.0), me, rival], model, stop_cost=22.0)
    assert [r["type"] for r in reads] == ["cover"] and reads[0]["on"]
    assert "box now" in reads[0]["text"]


def test_overcut_when_the_rival_rejoins_in_traffic(model):
    me = _car("AAA", 10.0)
    rival = _car("BBB", 31.6, stops=1, age=1, compound="HARD")  # rejoined 0.6 s behind CCC
    reads = battle_reads(me, [_car("LDR", 0.0), me, _car("CCC", 31.0), rival], model, stop_cost=22.0, sc_hazard=0.02)
    assert reads and reads[0]["type"] == "overcut" and reads[0]["on"]
    assert "Safety Car" in reads[0]["text"]


def test_no_reads_far_from_the_stop(model):
    ahead = _car("BBB", 9.0)
    me = _car("AAA", 10.0, interval=1.0)
    traffic = _car("CCC", 30.5)  # BBB would rejoin 0.5 s behind it
    running = [_car("LDR", 0.0), ahead, me, traffic]
    assert battle_reads(me, running, model, stop_cost=22.0, laps_to_stop=15) == []
    near = battle_reads(me, running, model, stop_cost=22.0, laps_to_stop=3)
    assert near and near[0]["type"] == "overcut" and near[0]["driver"] == "BBB"


def test_no_cover_that_needs_an_extra_stop(model):
    me = _car("AAA", 10.0)
    rival = _car("BBB", 10.8 + 22.0 + 2.0, stops=1, age=1, compound="HARD")
    reads = battle_reads(me, [_car("LDR", 0.0), me, rival], model, stop_cost=22.0, stop_now_cost=None)
    assert reads[0]["type"] == "cover" and not reads[0]["on"]
    assert "extra stop" in reads[0]["text"]
