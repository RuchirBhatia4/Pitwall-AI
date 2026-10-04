"""Unit tests for the PitWall strategy engine (no network, no FastF1 cache needed)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.pitwall.build_season import conformal_q, plan_loss
from src.pitwall.live_sources import TimingReducer, _parse_gap, _rank_cars, _track_status_from_rc
from src.pitwall.optimizer import INF, Optimizer, RaceContext, TyreState
from src.pitwall.tyre_model import (
    CompoundParams,
    TyreModel,
    _pav,
    apply_behaviour,
    enforce_compound_order,
    fit_transfer,
    live_update,
    weekend_model,
)


def model(soft=0.12, med=0.07, hard=0.04, max_stint=(30, 40, 50)) -> TyreModel:
    return TyreModel(
        compounds={
            "SOFT": CompoundParams(-0.6, soft, 0.2, 0.02, max_stint[0]),
            "MEDIUM": CompoundParams(0.0, med, 0.01, 0.02, max_stint[1]),
            "HARD": CompoundParams(0.4, hard, 0.2, 0.02, max_stint[2]),
        },
        fuel_per_lap=0.055,
        quad=0.0,
        noise_sd=0.4,
    )


def brute_force(opt: Optimizer, state: TyreState, max_stops: int) -> float:
    """Exhaustive search over every sequence and pit-lap combination."""
    best = INF
    start = state.lap + 1
    for seq in opt.sequences(state, max_stops):
        k = len(seq) - 1
        if k == 0:
            best = min(best, opt.evaluate(seq, [], state))
            continue
        lo = start + (opt.ctx.min_stint - 1 if state.compound is None else 0)
        for pits in itertools.combinations(range(lo, opt.L), k):
            gaps = np.diff([state.lap] + list(pits) + [opt.L])
            if (gaps[1:] < opt.ctx.min_stint).any():
                continue
            best = min(best, opt.evaluate(seq, list(pits), state))
    return best


@pytest.mark.parametrize("stop_penalty", [0.0, 6.0])
def test_dp_matches_brute_force_prerace(stop_penalty):
    opt = Optimizer(model(), RaceContext(total_laps=24, pit_loss=20.0, stop_penalty=stop_penalty, min_stint=3))
    res = opt.optimise(TyreState(), max_stops=2, top=50)
    assert res[0].total == pytest.approx(brute_force(opt, TyreState(), 2), abs=1e-6)


def test_dp_matches_brute_force_live_and_respects_used_compounds():
    opt = Optimizer(model(), RaceContext(total_laps=26, pit_loss=21.0, min_stint=2))
    st = TyreState(lap=9, compound="MEDIUM", age=9, used=("MEDIUM",))
    res = opt.optimise(st, max_stops=2, top=50)
    assert res[0].total == pytest.approx(brute_force(opt, st, 2), abs=1e-6)
    # the plan must still add a second dry compound
    assert any(c != "MEDIUM" for c in res[0].sequence)


def test_no_stop_allowed_once_two_compounds_used():
    opt = Optimizer(model(), RaceContext(total_laps=30, pit_loss=60.0))
    st = TyreState(lap=25, compound="HARD", age=10, used=("MEDIUM", "HARD"))
    res = opt.optimise(st, top=3)
    assert res[0].pit_laps == []  # huge pit loss, 5 laps left: stay out


def test_two_compound_rule_prerace():
    opt = Optimizer(model(), RaceContext(total_laps=40, pit_loss=20.0))
    for s in opt.optimise(top=30):
        assert len(set(s.sequence)) >= 2


def test_safety_car_makes_boxing_now_attractive():
    m = model()
    ctx = RaceContext(total_laps=50, pit_loss=22.0, min_stint=2)
    opt = Optimizer(m, ctx)
    green = opt.optimise(TyreState(lap=17, compound="MEDIUM", age=17, used=("MEDIUM",)), top=1)[0]
    sc = opt.optimise(TyreState(lap=17, compound="MEDIUM", age=17, used=("MEDIUM",), pit_cost_now=11.0), top=1)[0]
    assert sc.pit_laps[0] == 18
    assert sc.total < green.total


def test_windows_contain_optimum_and_monte_carlo_probabilities_sum_to_one():
    opt = Optimizer(model(), RaceContext(total_laps=40, pit_loss=20.0))
    res = opt.optimise(top=5)
    for s in res:
        for p, (lo, hi) in zip(s.pit_laps, s.windows):
            assert lo <= p <= hi
    opt.monte_carlo(res, sims=200)
    assert sum(s.risk["win_probability"] for s in res) == pytest.approx(1.0)
    assert all(s.risk["expected_delta"] >= 0 for s in res)


def test_max_stint_is_enforced():
    opt = Optimizer(model(max_stint=(8, 12, 15)), RaceContext(total_laps=40, pit_loss=20.0))
    for s in opt.optimise(top=10):
        for st in s.stints:
            assert st["laps"] <= {"SOFT": 8, "MEDIUM": 12, "HARD": 15}[st["compound"]]


def test_start_penalty_changes_opening_tyre():
    m = model(soft=0.07, med=0.07, hard=0.07)
    free = Optimizer(m, RaceContext(total_laps=40, pit_loss=20.0)).optimise(top=1)[0]
    if free.sequence[0] == "HARD":
        pen = Optimizer(m, RaceContext(total_laps=40, pit_loss=20.0, start_penalty={"HARD": 50.0})).optimise(top=1)[0]
        assert pen.sequence[0] != "HARD"


def test_pav_and_compound_order():
    assert _pav([3.0, 1.0, 2.0], [1, 1, 1], increasing=True) == pytest.approx([2.0, 2.0, 2.0])
    m = model(soft=0.03, med=0.08, hard=0.05)  # inverted: soft wears least
    m.compounds["SOFT"].offset = 0.5  # inverted: soft slower than medium
    enforce_compound_order(m)
    assert m.compounds["SOFT"].deg >= m.compounds["MEDIUM"].deg >= m.compounds["HARD"].deg
    assert m.compounds["SOFT"].offset <= m.compounds["MEDIUM"].offset == 0.0 <= m.compounds["HARD"].offset


def test_weekend_model_moves_toward_practice_with_learned_transfer():
    prior = model()
    practice = {"compounds": {"MEDIUM": {"deg": 0.30, "deg_se": 0.01, "runs": 10, "laps": 100}}, "teams": {}}
    w = weekend_model(prior, practice, transfer={"beta": 0.5, "sd": 0.01})
    # 0.5 * 0.30 = 0.15 observed; posterior must sit between prior 0.07 and 0.15
    assert 0.07 < w.compounds["MEDIUM"].deg < 0.15
    assert w.compounds["MEDIUM"].deg_sd < prior.compounds["MEDIUM"].deg_sd


def test_fit_transfer_recovers_ratio():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.4, 40)
    pairs = [(xi, 0.45 * xi + rng.normal(0, 0.005), 1.0) for xi in x]
    t = fit_transfer(pairs)
    assert t["beta"] == pytest.approx(0.45, abs=0.03)
    assert t["source"] == "learned"


def test_live_update_pulls_toward_observed_wear():
    m = model(med=0.05)
    laps = np.arange(5, 25)
    ages = laps - 2
    times = 90 + 0.20 * ages - 0.055 * laps  # true wear 0.20 s/lap, fuel burn hidden in raw times
    obs = [{"compound": "MEDIUM", "ages": ages.tolist(), "times": times.tolist(), "laps": laps.tolist(), "team": "X"}] * 3
    upd, ev = live_update(m, obs)
    assert ev["MEDIUM"]["observed"] == pytest.approx(0.20, abs=0.01)
    assert upd.compounds["MEDIUM"].deg > 0.15


def test_apply_behaviour_scales_soft_only():
    m = model()
    b = apply_behaviour(m, {"soft_mult": 2.0})
    assert b.compounds["SOFT"].deg == pytest.approx(2 * m.compounds["SOFT"].deg)
    assert b.compounds["MEDIUM"].deg == m.compounds["MEDIUM"].deg
    assert m.compounds["SOFT"].deg == pytest.approx(0.12)  # original untouched


def test_conformal_quantile_finite_sample():
    assert conformal_q(list(range(5))) is None
    res = list(range(1, 21))  # |r| = 1..20, n=20, k=ceil(21*0.8)=17
    assert conformal_q(res) == 17


def test_plan_loss_perfect_match_is_zero():
    pred = {"sequence": ["MEDIUM", "HARD"], "pit_laps": [20]}
    assert plan_loss(pred, ((("MEDIUM", "HARD"), (20,)),), 50) == 0.0
    assert plan_loss(pred, ((("SOFT", "MEDIUM", "HARD"), (10, 30)),), 50) > 2.0


# ---------------------------------------------------------------------------
# live data plumbing
# ---------------------------------------------------------------------------
def test_parse_gap():
    assert _parse_gap("+1.234") == pytest.approx(1.234)
    assert _parse_gap("+1 LAP") is None
    assert _parse_gap(3) == 3.0


def test_rank_cars_orders_and_fills_intervals():
    cars = [
        {"driver": "B", "laps_completed": 10, "gap_to_leader": 2.0, "retired": False},
        {"driver": "A", "laps_completed": 10, "gap_to_leader": 0.0, "retired": False},
        {"driver": "C", "laps_completed": 9, "gap_to_leader": 5.5, "retired": False},
        {"driver": "D", "laps_completed": 4, "gap_to_leader": None, "retired": True},
    ]
    out = _rank_cars(cars)
    assert [c["driver"] for c in out] == ["A", "B", "C", "D"]
    assert out[1]["interval"] == pytest.approx(2.0)
    assert out[2]["interval"] == pytest.approx(3.5)  # one lap-count apart at the line is still comparable


def test_track_status_from_race_control():
    rc = [
        {"date": "1", "message": "SAFETY CAR DEPLOYED", "lap_number": 12, "category": "SafetyCar"},
        {"date": "2", "message": "SAFETY CAR IN THIS LAP", "lap_number": 15, "category": "SafetyCar"},
    ]
    status, neutral = _track_status_from_rc(rc)
    assert status == "SC"
    rc.append({"date": "3", "message": "TRACK CLEAR", "flag": "GREEN", "lap_number": 16, "category": "Flag"})
    status, neutral = _track_status_from_rc(rc)
    assert status == "GREEN" and {12, 13, 14, 15, 16} <= neutral


def test_timing_reducer_snapshot_then_diffs():
    r = TimingReducer()
    r.apply("DriverList", {"1": {"Tla": "NOR", "TeamName": "McLaren", "TeamColour": "F47600"}, "63": {"Tla": "RUS", "TeamName": "Mercedes", "TeamColour": "00D7B6"}})
    r.apply("LapCount", {"CurrentLap": 5, "TotalLaps": 56})
    r.apply("TimingAppData", {"Lines": {"1": {"Stints": [{"Compound": "MEDIUM", "New": "true", "TotalLaps": 4}]},
                                         "63": {"Stints": [{"Compound": "HARD", "New": "true", "TotalLaps": 4}]}}})
    r.apply("TimingData", {"Lines": {"1": {"Position": "2", "GapToLeader": "+1.2", "IntervalToPositionAhead": {"Value": "+1.2"}, "NumberOfLaps": 4},
                                      "63": {"Position": "1", "GapToLeader": "", "NumberOfLaps": 4}}})
    # a diff: NOR completes lap 5 with a lap time; stint list arrives as an index-keyed dict
    r.apply("TimingData", {"Lines": {"1": {"NumberOfLaps": 5, "LastLapTime": {"Value": "1:36.500"}, "GapToLeader": "+0.9"}}})
    r.apply("TimingAppData", {"Lines": {"1": {"Stints": {"0": {"TotalLaps": 5}}}}})
    r.apply_line("['TrackStatus', {'Status': '4', 'Message': 'SCDeployed'}, '2026-10-04T07:20:00Z']")
    st = r.state(16, None)
    nor = next(c for c in st["cars"] if c["driver"] == "NOR")
    assert st["total_laps"] == 56 and st["track_status"] == "SC"
    assert nor["tyre_age"] == 5 and nor["compound"] == "MEDIUM"
    assert nor["gap_to_leader"] == pytest.approx(0.9)
    assert nor["laps"][-1]["time"] == pytest.approx(96.5)
    assert [c["driver"] for c in st["cars"]][:2] == ["RUS", "NOR"]


# ---------------------------------------------------------------------------
# personal wear + severity fill
# ---------------------------------------------------------------------------
def test_severity_fill_infers_missing_compound_from_weekend():
    from src.pitwall.tyre_model import weekend_model

    prior = model(soft=0.04, med=0.03, hard=0.03)
    practice = {"compounds": {"MEDIUM": {"deg": 0.30, "deg_se": 0.02, "runs": 12, "laps": 120},
                              "SOFT": {"deg": 0.35, "deg_se": 0.02, "runs": 20, "laps": 200}}, "teams": {}}
    w = weekend_model(prior, practice, transfer={"beta": 0.5, "sd": 0.03}, compound_ratios={"SOFT": 1.2, "MEDIUM": 1.0, "HARD": 0.8})
    # Hard had no long runs: it must move off the 0.03 prior toward the weekend's
    # severity (0.5 * ~0.29 * 0.8 ~= 0.12), weighted by the two uncertainties.
    assert 0.045 < w.compounds["HARD"].deg < 0.12
    without = weekend_model(prior, {"compounds": {}, "teams": {}}, transfer={"beta": 0.5, "sd": 0.03})
    assert without.compounds["HARD"].deg == pytest.approx(0.03)


def test_personal_multiplier_changes_only_that_driver():
    m = model()
    m.driver_mult["AAA"] = {"mult": 1.3, "sd": 0.1}
    assert m.deg_for("MEDIUM", driver="AAA") == pytest.approx(1.3 * m.deg_for("MEDIUM"))
    assert m.deg_for("MEDIUM", driver="BBB") == pytest.approx(m.deg_for("MEDIUM"))
    hi = Optimizer(m, RaceContext(total_laps=50, pit_loss=21.0, driver="AAA")).optimise(top=1)[0]
    lo = Optimizer(m, RaceContext(total_laps=50, pit_loss=21.0, driver="BBB")).optimise(top=1)[0]
    assert hi.total > lo.total  # higher personal wear costs more time


def test_personal_wear_shrinks_one_stint_toward_prior():
    from src.pitwall.live_engine import personal_wear

    m = model(med=0.10)
    laps = list(range(5, 20))
    car = {"driver": "AAA", "team": "T", "laps": [
        {"lap": l, "time": 90 + 0.20 * (l - 4) - 0.055 * l, "compound": "MEDIUM", "age": l - 4, "pit_in": False, "pit_out": False, "neutral": False}
        for l in laps]}
    p = personal_wear(car, m)
    assert p["observed_mult"] == pytest.approx(2.0, abs=0.2) or p["observed_mult"] == pytest.approx(1.8, abs=0.05)  # clipped at 1.8
    assert 1.0 < p["mult"] < 1.5  # one stint moves it, but does not take it all the way


def test_stop_probabilities_cover_all_stop_counts():
    opt = Optimizer(model(), RaceContext(total_laps=50, pit_loss=21.0))
    plans = opt.candidates(top=3)
    assert {len(p.pit_laps) for p in plans} >= {1, 2, 3}
    opt.monte_carlo(plans, sims=100)
    assert sum(opt.stop_probabilities.values()) == pytest.approx(1.0)
