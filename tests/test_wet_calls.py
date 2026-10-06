"""Live calls in wet and changing conditions, and the BOX/window gate.

No 2026 race so far has used intermediates, so these build synthetic race
states: what matters is that the call follows who is faster *now* on each
tyre type, and never comes from dry-tyre wear maths on a wet or drying track.
"""
from __future__ import annotations

from src.pitwall.live_engine import WAIT_COST_GREEN, analyse, assess_conditions
from src.pitwall.live_sources import TimingReducer
from src.pitwall.optimizer import Optimizer, RaceContext, TyreState
from src.pitwall.tyre_model import CompoundParams, TyreModel

TOTAL = 50


def model() -> TyreModel:
    return TyreModel(
        compounds={
            "SOFT": CompoundParams(-0.6, 0.12, 0.2, 0.02, 30),
            "MEDIUM": CompoundParams(0.0, 0.07, 0.01, 0.02, 40),
            "HARD": CompoundParams(0.4, 0.04, 0.2, 0.02, 50),
        },
        fuel_per_lap=0.055,
        quad=0.0,
        noise_sd=0.4,
    )


WEEKEND = {"model": model().to_dict(), "total_laps": TOTAL, "pit_loss": 21.0, "sc_rate": 0.01, "vsc_rate": 0.006}


def car(drv: str, pos: int, done: int, stints: list[tuple[str, int]], pace) -> dict:
    """stints: [(compound, first_lap), ...]; pace(lap, compound) -> lap time."""
    laps = []
    for l in range(1, done + 1):
        comp, start = [s for s in stints if s[1] <= l][-1]
        laps.append({"lap": l, "time": pace(l, comp), "compound": comp, "age": l - start + 1,
                     "pit_in": any(s[1] == l + 1 for s in stints), "pit_out": l == start and start > 1, "neutral": False})
    comp, start = stints[-1]
    used = list(dict.fromkeys(c for c, _ in stints))
    return {"driver": drv, "team": "T" + drv, "position": pos, "gap_to_leader": 2.0 * (pos - 1), "interval": 2.0 if pos > 1 else 0.0,
            "laps_completed": done, "compound": comp, "tyre_age": done - start + 1, "stint": len(stints), "stops": len(stints) - 1,
            "compounds_used": used, "retired": False, "laps": laps}


def state(cars: list[dict], weather: dict | None = None, rc: list | None = None, ts: str = "GREEN") -> dict:
    return {"cars": cars, "total_laps": TOTAL, "track_status": ts, "weather": weather or {}, "race_control": rc or []}


def wet_dry_pace(slick: float, wet: float):
    return lambda l, c: wet if c in ("INTERMEDIATE", "WET") else slick


def field(n_slick: int, n_wet: int, pace, done: int = 20, slick_from: int = 12) -> list[dict]:
    cars = []
    for i in range(n_slick):
        cars.append(car(f"S{i}", len(cars) + 1, done, [("INTERMEDIATE", 1), ("MEDIUM", slick_from)], pace))
    for i in range(n_wet):
        cars.append(car(f"W{i}", len(cars) + 1, done, [("INTERMEDIATE", 1)], pace))
    return cars


def test_on_inters_boxes_for_slicks_once_slick_runners_are_clearly_faster():
    a = analyse(state(field(4, 6, wet_dry_pace(slick=90.0, wet=93.0))), WEEKEND, "W0", sims=40)
    assert a["call"]["action"] == "BOX_NOW"
    assert "slicks" in a["call"]["headline"]
    assert a["call"]["next_compound"] in ("SOFT", "MEDIUM", "HARD")


def test_on_inters_stays_out_while_slicks_are_not_faster_enough():
    a = analyse(state(field(4, 6, wet_dry_pace(slick=92.6, wet=93.0))), WEEKEND, "W0", sims=40)
    assert a["call"]["action"] == "STAY_OUT"
    assert "Intermediate" in a["call"]["headline"]


def test_on_inters_with_nobody_on_slicks_stays_out():
    a = analyse(state(field(0, 10, wet_dry_pace(slick=90.0, wet=93.0))), WEEKEND, "W0", sims=40)
    assert a["call"]["action"] == "STAY_OUT"
    assert a["conditions"]["crossover"]["other_class_pace"] is None


def test_on_slicks_boxes_for_inters_when_wet_runners_are_faster():
    # rain: the cars already on intermediates are 4 s/lap quicker
    a = analyse(state(field(6, 4, wet_dry_pace(slick=97.0, wet=93.0)), weather={"rainfall": True}), WEEKEND, "S0", sims=40)
    assert a["call"]["action"] == "BOX_NOW"
    assert a["call"]["next_compound"] == "INTERMEDIATE"


def test_car_that_ran_wets_is_free_of_the_two_compound_rule():
    m = model()
    st = TyreState(lap=40, compound="MEDIUM", age=10, used=("MEDIUM",))
    with_rule = Optimizer(m, RaceContext(total_laps=TOTAL, pit_loss=21.0)).optimise(st, top=50)
    waived = Optimizer(m, RaceContext(total_laps=TOTAL, pit_loss=21.0, two_compound_rule=False)).optimise(st, top=50)
    assert all(p.pit_laps for p in with_rule)  # dry race: must still stop
    assert waived[0].pit_laps == []  # ran inters: one dry set to the flag is legal and best


def test_drizzle_that_does_not_slow_the_field_does_not_hold_dry_strategy():
    cars = [car(f"D{i}", i + 1, 20, [("MEDIUM", 1)], lambda l, c: 90.0) for i in range(10)]
    c = assess_conditions(state(cars, weather={"rainfall": True, "rain_laps": [19, 20]}))
    assert c["changing"] and not c["unsettled"]


def test_drying_track_holds_a_dry_stop():
    # field pace improving ~2.5 %/lap for the last laps, and a soft past its life that the dry maths wants to box
    drying = lambda l, c: 100.0 * (0.975 ** max(0, l - 14)) if l > 14 else 100.0
    cars = [car("AAA", 1, 20, [("SOFT", 1)], drying)] + [car(f"D{i}", i + 2, 20, [("MEDIUM", 1)], drying) for i in range(9)]
    a = analyse(state(cars), WEEKEND, "AAA", sims=40)
    assert a["conditions"]["unsettled"]
    assert a["call"]["action"] in ("HOLD", "STAY_OUT")
    # and the wet/drying laps are kept out of the wear model
    assert set(range(15, 21)) & set(a["conditions"]["excluded_laps"])


def test_green_flag_box_only_when_waiting_costs_real_time():
    for done in (16, 22, 28):  # soft getting older through its window
        cars = [car("AAA", 1, done, [("SOFT", 1)], lambda l, c: 90.0)] + \
               [car(f"D{i}", i + 2, done, [("MEDIUM", 1)], lambda l, c: 90.2) for i in range(9)]
        call = analyse(state(cars), WEEKEND, "AAA", sims=40)["call"]
        if call["action"] == "BOX_NOW":
            assert call["wait_cost"] >= WAIT_COST_GREEN


def test_tyres_past_model_life_still_get_a_call():
    cars = [car("AAA", 1, 46, [("SOFT", 1)], lambda l, c: 90.0)] + \
           [car(f"D{i}", i + 2, 46, [("HARD", 1)], lambda l, c: 90.0) for i in range(9)]
    cars[0]["compounds_used"] = ["MEDIUM", "SOFT"]  # rule already met
    assert analyse(state(cars), WEEKEND, "AAA", sims=40)["call"]["action"] != "NO_PLAN"


def test_live_feed_rainfall_reaches_the_state():
    r = TimingReducer()
    r.apply("LapCount", {"CurrentLap": 12, "TotalLaps": TOTAL})
    r.apply("WeatherData", {"Rainfall": "1", "TrackTemp": "24.1", "AirTemp": "18.0"})
    w = r._weather()
    assert w["rainfall"] is True and w["rain_laps"] == [12] and w["track_temp"] == 24.1
