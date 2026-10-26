"""Full-field race simulation: every car, lap by lap, with track position.

The DP optimiser prices a plan in isolation (tyre time + pit-lane time). A pit
wall ranks plans by where the car *finishes*, and that depends on the other
cars: a stop that rejoins behind slower traffic, or a faster car that cannot
get past, costs places the isolated model never sees. Simulating the whole
field is how team strategy tools handle this (see Heilmeier et al., 2020,
"Application of Monte Carlo Methods to Consider Probabilistic Effects in a
Race Simulation for Circuit Motorsport").

Each lap, every car's free-air time is

    race pace (driver) + tyre delta (compound, age, personal wear) + noise

then cars interact in running order. A car within ATTACK_GAP of the one ahead
passes with probability sigmoid(pass_a + pass_b * pace advantage), learned
from earlier 2026 races; otherwise it is held MIN_GAP behind. Stops add the
pit loss plus stationary-time noise. Safety Cars cheapen stops and bunch the
field at the restart; VSCs cheapen stops and freeze the gaps. Cars with a stop
due soon box under a neutralisation, like a real pit wall.

All randomness (SC/VSC timeline, lap noise, pace errors, stop times, the
start) is drawn once per simulation and shared by every plan being compared
(common random numbers), so a difference between two plans is the plan, not
noise. Everything is vectorised over simulations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pitwall.optimizer import INF, SC_PIT_FACTOR, VSC_PIT_FACTOR, Optimizer, RaceContext, Strategy, TyreState
from src.pitwall.tyre_model import TyreModel

MIN_GAP = 0.5  # s: closest a held-up car runs behind the car ahead at the line
ATTACK_GAP = 1.0  # s: within this at the start of a lap a car can try to pass (DRS range)
LAP_NOISE_SD = 0.35  # s: lap-to-lap free-air variation
START_SPACING = 0.25  # s between grid slots crossing the line on lap 1
START_NOISE_SD = 0.6  # s: launch and lap-1 shuffle
STOP_SD = 0.8  # s: stationary-time variation per stop
SLOW_STOP_P = 0.04  # share of stops with a problem (wheel gun, unsafe-release hold)
SLOW_STOP_S = (2.0, 8.0)
SC_RESTART_GAP = 0.6  # s between cars when the Safety Car pulls in
REACT_LAPS = 10  # a neutralisation this many laps before a planned stop brings it forward
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 30, dtype=float)


@dataclass
class SimParams:
    pace: dict[str, float]  # driver -> race pace vs the field median, s/lap (lower is faster)
    pace_sd: float = 0.3  # error of that pace prediction, s/lap
    pass_a: float = -2.0  # logit of P(pass in one lap | within ATTACK_GAP, equal pace)
    pass_b: float = 1.0  # change in that logit per s/lap of pace advantage
    info: dict = field(default_factory=dict)


@dataclass
class Plan:
    sequence: list[str]
    pit_laps: list[int]

    @classmethod
    def of(cls, s: Strategy | dict) -> "Plan":
        if isinstance(s, dict):
            return cls(list(s["sequence"]), list(s["pit_laps"]))
        return cls(list(s.sequence), list(s.pit_laps))


class RaceSim:
    """Monte Carlo race of the whole field. ``cars`` are dicts with driver, team, grid.

    Pre-race the cars line up in grid order. Live, pass ``start_lap`` (laps
    completed), ``gaps`` (s behind the leader) and ``tyres`` (driver ->
    (compound, age)): the race is simulated from there, plans' pit laps are
    absolute race laps after ``start_lap``, and ``neutral_now`` (1 VSC, 2 SC)
    keeps a running neutralisation going for its first laps."""

    def __init__(self, model: TyreModel, ctx: RaceContext, cars: list[dict], params: SimParams, sims: int = 160, seed: int = 11,
                 start_lap: int = 0, gaps: dict[str, float] | None = None, tyres: dict[str, tuple[str, int]] | None = None, neutral_now: int = 0):
        self.model, self.ctx, self.params = model, ctx, params
        key = (lambda c: gaps.get(c["driver"], 1e6)) if gaps else (lambda c: c.get("grid") or 99)
        self.cars = sorted(cars, key=key)
        self.index = {c["driver"]: i for i, c in enumerate(self.cars)}
        self.k0, self.tyres = start_lap, tyres or {}
        self.N, self.L, self.S = len(self.cars), ctx.total_laps, sims
        self.n = self.L - start_lap  # laps to simulate
        self.opt = Optimizer(model, ctx)  # wear-vs-fuel weights and the SC reaction rule
        rng = np.random.default_rng(seed)
        S, N, L, k0 = self.S, self.N, self.L, start_lap
        # Neutralisations: 0 green, 1 VSC, 2 SC, per simulation and race lap (1-based).
        self.neutral = np.zeros((S, L + 2), dtype=np.int8)
        for s in range(S):
            lap = k0 + 1
            if neutral_now:
                dur = int(rng.integers(1, 4)) if neutral_now == 2 else int(rng.integers(1, 3))
                self.neutral[s, lap : lap + dur] = neutral_now
                lap += dur
            while lap <= L:
                u = rng.random()
                if u < ctx.sc_rate:
                    dur = int(rng.integers(3, 6))
                    self.neutral[s, lap : lap + dur] = 2
                    lap += dur
                elif u < ctx.sc_rate + ctx.vsc_rate:
                    dur = int(rng.integers(1, 4))
                    self.neutral[s, lap : lap + dur] = np.maximum(self.neutral[s, lap : lap + dur], 1)
                    lap += dur
                else:
                    lap += 1
        self.neutral[:, L + 1 :] = 0
        # New neutralisations a pit wall reacts to (a running one is already in the plans).
        first = k0 + 2 if neutral_now else k0 + 1
        self.neutral_starts = [[x for x in range(first, L) if self.neutral[s, x] and not self.neutral[s, x - 1]] for s in range(S)]
        self.lap_noise = rng.normal(0, LAP_NOISE_SD, (S, N, self.n)).astype(np.float32)
        self.pace_err = rng.normal(0, params.pace_sd, (S, N)).astype(np.float32)
        if gaps:
            self.start = np.tile(np.array([gaps.get(c["driver"], 0.0) for c in self.cars], float), (S, 1))
        else:
            self.start = (START_SPACING * np.arange(N))[None, :] + rng.normal(0, START_NOISE_SD, (S, N))
        self.pass_u = rng.random((S, N, self.n)).astype(np.float32)
        slow = rng.random((S, N, 4)) < SLOW_STOP_P
        self.stop_noise = rng.normal(0, STOP_SD, (S, N, 4)) + slow * rng.uniform(*SLOW_STOP_S, (S, N, 4))
        # Tyre-parameter uncertainty: one posterior draw per simulation, shared by the field.
        self.deg_draw = {c: np.maximum(0.003, p.deg + rng.normal(0, p.deg_sd, S)) for c, p in model.compounds.items()}
        self.pace = np.array([params.pace.get(c["driver"], 0.0) for c in self.cars], dtype=np.float32)
        self.wear_w = self.opt.wear_w
        self._mult = {}
        for i, c in enumerate(self.cars):
            for comp, p in model.compounds.items():
                self._mult[i, comp] = model.deg_for(comp, c.get("team"), c["driver"]) / max(p.deg, 1e-6)

    # -- per-car lap costs ------------------------------------------------------
    def car_laps(self, i: int, plan: Plan) -> tuple[np.ndarray, np.ndarray]:
        """For car i on `plan`: tyre delta and pit cost per (sim, simulated lap),
        with the plan's stops brought forward under SC/VSC where a pit wall would."""
        S, L, k0 = self.S, self.L, self.k0
        tyre = np.zeros((S, self.n), dtype=np.float32)
        pit = np.zeros((S, self.n), dtype=np.float32)
        laps = np.arange(1, L + 1)
        age0 = int(self.tyres.get(self.cars[i]["driver"], (None, 0))[1]) if self.tyres else 0
        here = TyreState(lap=k0)
        # A rival a lap behind can have a stop on a lap the leaders have done: box now.
        planned = []
        for p_lap in plan.pit_laps:
            p_lap = max(int(p_lap), k0 + 1, planned[-1] + 1 if planned else 0)
            if p_lap < L:
                planned.append(p_lap)
        seq = plan.sequence[: len(planned) + 1]
        for s in range(S):
            pits = planned
            if pits and self.neutral_starts[s]:
                pits = self.opt._react(pits, self.neutral_starts[s], here)
            bounds = [k0 + 1] + [p + 1 for p in pits] + [L + 1]
            for k, comp in enumerate(seq):
                a, b = bounds[k], bounds[k + 1]
                if b <= a or comp not in self.model.compounds:
                    continue
                p = self.model.compounds[comp]
                age = np.arange(1, b - a + 1, dtype=float) + (age0 if k == 0 else 0)
                d = self.deg_draw[comp][s] * self._mult[i, comp]
                cost = p.offset + d * age * self.wear_w[laps[a - 1 : b - 1]] + self.model.quad * age**2
                over = age - p.max_stint * 1.3
                cost += np.where(over > 0, 0.5 * over, 0.0)  # pushed past its life after an early SC stop
                tyre[s, a - 1 - k0 : b - 1 - k0] = cost
            for k, p_lap in enumerate(pits[:4]):
                if k0 < p_lap < L:
                    status = self.neutral[s, p_lap]
                    factor = SC_PIT_FACTOR if status == 2 else VSC_PIT_FACTOR if status == 1 else 1.0
                    pit[s, p_lap - 1 - k0] = self.ctx.pit_loss * factor + self.stop_noise[s, i, k]
        return tyre, pit

    # -- the race ---------------------------------------------------------------
    def run(self, tyre: np.ndarray, pit: np.ndarray, sim_idx: np.ndarray) -> np.ndarray:
        """tyre, pit: (B, N, laps) per batch row; sim_idx: (B,) which simulation
        each row replays. Returns race time per (B, N) (only the order matters)."""
        B, N, L = tyre.shape
        rows = np.arange(B)
        neutral = self.neutral[sim_idx]
        pace = self.pace[None, :] + self.pace_err[sim_idx]
        t = self.start[sim_idx].astype(np.float64)
        a, b = self.params.pass_a, self.params.pass_b
        for n in range(L):
            lap = self.k0 + n + 1
            status = neutral[:, lap]
            green = status == 0
            expected = pace + tyre[:, :, n]
            free = np.where(green[:, None], expected + self.lap_noise[sim_idx, :, n], 0.0)
            pitting = pit[:, :, n] > 0
            tent = t + free + pit[:, :, n]
            new = tent.copy()
            if green.any():
                order = np.argsort(t, axis=1)
                ahead = order[:, 0].copy()
                has_ahead = ~pitting[rows, ahead]
                for k in range(1, N):
                    i = order[:, k]
                    i_pit = pitting[rows, i]
                    gap = t[rows, i] - t[rows, ahead]
                    adv = expected[rows, ahead] - expected[rows, i]
                    p = 1.0 / (1.0 + np.exp(-(a + b * adv)))
                    fight = green & has_ahead & ~i_pit & (gap <= ATTACK_GAP)
                    passed = fight & (self.pass_u[sim_idx, i, n] < p)
                    ahead_new = new[rows, ahead]
                    cand = tent[rows, i]
                    held = green & has_ahead & ~i_pit & ~passed
                    res = np.where(passed, np.minimum(cand, ahead_new - 0.05), np.where(held, np.maximum(cand, ahead_new + MIN_GAP), cand))
                    new[rows, i] = res
                    on_track = ~i_pit
                    ahead = np.where(on_track, i, ahead)
                    has_ahead = has_ahead | on_track
            t = new
            # Safety Car comes in: the field restarts nose to tail, order kept.
            restart = (status == 2) & (neutral[:, lap + 1] != 2) if lap < self.L else np.zeros(B, bool)
            if restart.any():
                ranks = np.argsort(np.argsort(t[restart], axis=1), axis=1)
                lead = t[restart].min(axis=1, keepdims=True)
                t[restart] = lead + SC_RESTART_GAP * ranks
        return t

    def field_arrays(self, plans: dict[str, Plan]) -> tuple[np.ndarray, np.ndarray]:
        S, N, n = self.S, self.N, self.n
        tyre = np.zeros((S, N, n), dtype=np.float32)
        pit = np.zeros((S, N, n), dtype=np.float32)
        for d, i in self.index.items():
            if d in plans:
                tyre[:, i], pit[:, i] = self.car_laps(i, plans[d])
        return tyre, pit

    def forecast(self, plans: dict[str, Plan]) -> dict[str, dict]:
        """Finishing-position distribution of every car with the field on `plans`."""
        tyre, pit = self.field_arrays(plans)
        t = self.run(tyre, pit, np.arange(self.S))
        pos = np.argsort(np.argsort(t, axis=1), axis=1) + 1
        out = {}
        for d, i in self.index.items():
            p = pos[:, i]
            out[d] = {"expected_position": float(p.mean()), "p_win": float((p == 1).mean()), "p_podium": float((p <= 3).mean()),
                      "p_points": float((p <= 10).mean()), "expected_points": float(POINTS[p - 1].mean())}
        return out

    def best_response(self, driver: str, candidates: list[Plan], plans: dict[str, Plan], base: tuple | None = None) -> list[dict]:
        """Score each candidate plan for `driver` with the rest of the field on
        `plans`. Returns one dict per candidate, best (lowest expected
        finishing position) first."""
        i = self.index[driver]
        tyre_f, pit_f = base if base is not None else self.field_arrays(plans)
        C, S = len(candidates), self.S
        tyre = np.repeat(tyre_f, C, axis=0)
        pit = np.repeat(pit_f, C, axis=0)
        for c, plan in enumerate(candidates):
            tc, pc = self.car_laps(i, plan)
            tyre[c::C, i], pit[c::C, i] = tc, pc
        sim_idx = np.repeat(np.arange(S), C)
        t = self.run(tyre, pit, sim_idx)
        pos = (np.argsort(np.argsort(t, axis=1), axis=1) + 1)[:, i].reshape(S, C)
        own = t[:, i].reshape(S, C)
        out = []
        for c, plan in enumerate(candidates):
            p = pos[:, c]
            out.append({"plan": plan, "expected_position": float(p.mean()), "position_sd": float(p.std()),
                        "p_points": float((p <= 10).mean()), "expected_points": float(POINTS[p - 1].mean()),
                        "expected_time": float(own[:, c].mean())})
        best_pos = pos.min(axis=1, keepdims=True)
        for c, o in enumerate(out):
            o["p_best"] = float((pos[:, c] == best_pos[:, 0]).mean())
        out.sort(key=lambda o: (round(o["expected_position"], 3), o["expected_time"]))
        return out


def params_from_forecast(forecast: dict | None) -> SimParams | None:
    """Rebuild the pre-race simulation parameters stored in a round file."""
    if not forecast or not forecast.get("params"):
        return None
    p = forecast["params"]
    return SimParams(pace=dict(forecast.get("pace") or {}), pace_sd=float(p.get("pace_sd", 0.3)),
                     pass_a=float(p["pass_a_2026"]) + float(p.get("venue_offset", 0.0)), pass_b=float(p.get("pass_b", 1.4)), info=p)


def candidate_plans(opt: Optimizer, top: int = 8, shifts: tuple[int, ...] = (-6, -3, 3, 6)) -> list[Plan]:
    """DP plans to put through the race simulation: the fastest few, the best
    for every stop count and starting compound, and each of the leading plans
    with its first stop moved earlier/later (later stops re-optimised), since
    traffic changes *when* to stop as much as *what* to fit."""
    allp = opt.optimise(TyreState(), max_stops=3, top=400)
    chosen: list[Strategy] = list(allp[:top])
    seen_cls = {(len(s.pit_laps), s.sequence[0]) for s in chosen}
    for s in allp:
        key = (len(s.pit_laps), s.sequence[0])
        if key not in seen_cls:
            chosen.append(s)
            seen_cls.add(key)
    plans = [Plan.of(s) for s in chosen]
    seen = {(tuple(p.sequence), tuple(p.pit_laps)) for p in plans}
    for s in chosen[:4]:
        if not s.pit_laps:
            continue
        for d in shifts:
            v = opt.plan_with_first_stop(s.sequence, TyreState(), s.pit_laps[0] + d)
            if v is None or v.total >= INF:
                continue
            key = (tuple(v.sequence), tuple(v.pit_laps))
            if key not in seen:
                seen.add(key)
                plans.append(Plan.of(v))
    return plans


# -- parameters learned from earlier races ---------------------------------------
FORM_HALF_LIFE = 4.0  # rounds: car development makes recent races more telling
FORM_PRIOR_WEIGHT = 1.0  # pseudo-races pulling a short history toward the field median


def race_pace(race_model: dict) -> dict[str, float]:
    """Per-driver race pace from a race fit, vs the field median (s/lap), with
    fits broken by a handful of laps (retirements) clipped."""
    dp = race_model.get("driver_pace") or {}
    if not dp:
        return {}
    med = float(np.median(list(dp.values())))
    return {d: float(np.clip(v - med, -3.0, 3.0)) for d, v in dp.items()}


def driver_form(history: list[tuple[int, dict[str, float]]], rnd: int) -> dict[str, float]:
    """Recency-weighted race pace of each driver over rounds before `rnd`."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for r, pace in history:
        if r >= rnd:
            continue
        w = 0.5 ** ((rnd - 1 - r) / FORM_HALF_LIFE)
        for d, v in pace.items():
            num[d] = num.get(d, 0.0) + w * v
            den[d] = den.get(d, 0.0) + w
    return {d: num[d] / (den[d] + FORM_PRIOR_WEIGHT) for d in num}


def fit_pace_model(history: list[tuple[int, dict[str, float], dict[str, int]]], rnd: int):
    """Learn race pace from qualifying position and recent form, on rounds < rnd.

    history: (round, race pace by driver, qualifying position by driver).
    Returns predict(quali) -> ({driver: pace}, prediction sd), or None with too
    little data. Qualifying, not the grid: a penalty moves where a car starts,
    not how fast it is."""
    from sklearn.linear_model import Ridge

    past = [(r, p, g) for r, p, g in history if r < rnd and p and g]
    X, y = [], []
    for r, pace, grid in past:
        form = driver_form([(r2, p2) for r2, p2, _ in past], r)
        for d, v in pace.items():
            if grid.get(d):
                X.append([grid[d], form.get(d, 0.0)])
                y.append(v)
    if len({r for r, _, _ in past}) < 3 or len(y) < 40:
        return None
    X, y = np.array(X, float), np.array(y, float)
    reg = Ridge(alpha=1.0).fit(X, y)
    sd = float(np.std(y - reg.predict(X)) * 1.1)
    form_now = driver_form([(r, p) for r, p, _ in past], rnd)

    def predict(grid: dict[str, int]) -> tuple[dict[str, float], float]:
        rows = [(d, [g or 22, form_now.get(d, 0.0)]) for d, g in grid.items()]
        if not rows:
            return {}, sd
        pred = reg.predict(np.array([x for _, x in rows], float))
        return {d: float(v) for (d, _), v in zip(rows, pred)}, sd

    predict.coef = reg.coef_.tolist()  # type: ignore[attr-defined]
    return predict


def pass_pairs(laps: pd.DataFrame, neutral_laps: set[int], race_model: dict) -> np.ndarray:
    """Close-following situations in a finished race: (pace advantage, passed).

    A pair is a car within ATTACK_GAP of the car ahead at the start of a green
    lap, neither pitting; 'passed' is whether it was ahead at the end of the lap.
    Pace advantage is the gap in expected lap time (race pace + tyre state) from
    that race's own fit."""
    pace = race_pace(race_model)
    comps, quad = race_model.get("compounds", {}), race_model.get("quad", 0.0)
    laps = laps.dropna(subset=["lap_end"])
    exp = np.full(len(laps), np.nan)
    for j, r in enumerate(laps.itertuples()):
        c = comps.get(r.compound)
        if c is not None and r.driver in pace and pd.notna(r.tyre_life):
            exp[j] = pace[r.driver] + c["offset"] + c["deg"] * r.tyre_life + quad * r.tyre_life**2
    laps = laps.assign(exp=exp)
    by_lap = {lap: g.sort_values("lap_end") for lap, g in laps.groupby("lap")}
    out = []
    for lap, cur in by_lap.items():
        if lap < 3 or lap - 1 not in by_lap or lap in neutral_laps or lap - 1 in neutral_laps:
            continue
        prev = by_lap[lap - 1]
        order, t_prev = list(prev["driver"]), dict(zip(prev["driver"], prev["lap_end"]))
        now = cur.set_index("driver")
        for k in range(1, len(order)):
            lead, foll = order[k - 1], order[k]
            if t_prev[foll] - t_prev[lead] > ATTACK_GAP or lead not in now.index or foll not in now.index:
                continue
            a, f = now.loc[lead], now.loc[foll]
            if a.pit_in or a.pit_out or f.pit_in or f.pit_out or np.isnan(a.exp) or np.isnan(f.exp):
                continue
            out.append((a.exp - f.exp, float(f.lap_end < a.lap_end)))
    return np.array(out, float).reshape(-1, 2)


def fit_pass_model(pairs: dict[int, np.ndarray]) -> tuple[float, float, dict[int, float]] | None:
    """Logistic P(pass) = sigmoid(a_race + b * advantage): a shared slope and one
    intercept per race (how hard that track is to pass on). Returns the mean
    intercept, the slope and the per-race intercepts."""
    from sklearn.linear_model import LogisticRegression

    races = [r for r, p in pairs.items() if len(p) >= 30]
    if len(races) < 3:
        return None
    X, y = [], []
    for j, r in enumerate(races):
        onehot = np.zeros((len(pairs[r]), len(races)))
        onehot[:, j] = 1.0
        X.append(np.c_[pairs[r][:, 0], onehot])
        y.append(pairs[r][:, 1])
    X, y = np.vstack(X), np.concatenate(y)
    m = LogisticRegression(C=10.0, fit_intercept=False, max_iter=2000).fit(X, y)
    b = float(m.coef_[0][0])
    intercepts = {r: float(m.coef_[0][1 + j]) for j, r in enumerate(races)}
    weights = np.array([len(pairs[r]) for r in races], float)
    a_mean = float(np.average(list(intercepts.values()), weights=weights))
    return a_mean, b, intercepts
