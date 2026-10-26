"""Full-field race simulation (synthetic field, no data files)."""
from __future__ import annotations

import numpy as np

from src.pitwall.optimizer import RaceContext
from src.pitwall.race_sim import Plan, RaceSim, SimParams, fit_pass_model
from src.pitwall.tyre_model import default_model

CTX = RaceContext(total_laps=40, pit_loss=21.0, sc_rate=0.0, vsc_rate=0.0)
ONE_STOP = Plan(["MEDIUM", "HARD"], [18])


def _field(n=6):
    return [{"driver": f"D{i}", "team": f"T{i}", "grid": i + 1} for i in range(n)]


def _sim(pace, pass_a, **kw):
    params = SimParams(pace=pace, pace_sd=0.0, pass_a=pass_a, pass_b=1.4)
    return RaceSim(default_model(), CTX, _field(), params, sims=60, **kw)


def test_hard_to_pass_tracks_keep_a_faster_car_behind():
    # D5 starts last but is 0.8 s/lap faster than everyone.
    pace = {f"D{i}": 0.0 for i in range(6)} | {"D5": -0.8}
    plans = {f"D{i}": ONE_STOP for i in range(6)}
    monaco = _sim(pace, pass_a=-9.0).forecast(plans)["D5"]["expected_position"]
    monza = _sim(pace, pass_a=-1.0).forecast(plans)["D5"]["expected_position"]
    # Gains at a Monaco-like track come only from the start and the stops
    # (lap 1 shuffles ~2 places on average in 2026); at Monza it drives through.
    assert monza < 2.5
    assert monaco > monza + 1.5


def test_best_response_ranks_a_needless_extra_stop_last():
    pace = {f"D{i}": 0.0 for i in range(6)}
    plans = {f"D{i}": ONE_STOP for i in range(6)}
    sim = _sim(pace, pass_a=-2.5)
    ranked = sim.best_response("D0", [Plan(["MEDIUM", "HARD", "HARD"], [10, 25]), ONE_STOP], plans)
    assert ranked[0]["plan"] == ONE_STOP
    assert ranked[0]["expected_position"] < ranked[1]["expected_position"]


def test_live_start_handles_rival_stops_on_laps_already_done():
    pace = {f"D{i}": 0.0 for i in range(6)}
    gaps = {f"D{i}": 2.0 * i for i in range(6)}
    tyres = {f"D{i}": ("MEDIUM", 20) for i in range(6)}
    sim = _sim(pace, pass_a=-2.5, start_lap=20, gaps=gaps, tyres=tyres)
    # D3 is a lap behind in its own count: its plan still has a stop at lap 19.
    plans = {f"D{i}": Plan(["MEDIUM", "HARD"], [24]) for i in range(6)} | {"D3": Plan(["MEDIUM", "HARD"], [19])}
    out = sim.forecast(plans)
    assert set(out) == set(plans)
    assert out["D0"]["expected_position"] < out["D5"]["expected_position"]


def test_pass_model_learns_track_difficulty():
    rng = np.random.default_rng(0)

    def race(a):
        adv = rng.normal(0, 0.8, 400)
        p = 1 / (1 + np.exp(-(a + 1.5 * adv)))
        return np.c_[adv, (rng.random(400) < p).astype(float)]

    mean_a, slope, per_race = fit_pass_model({1: race(-1.5), 2: race(-2.5), 3: race(-5.0)})
    assert per_race[1] > per_race[2] > per_race[3]
    assert 1.0 < slope < 2.0
