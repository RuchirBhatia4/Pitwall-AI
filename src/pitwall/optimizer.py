"""Exact strategy optimisation by dynamic programming + Monte Carlo risk.

For every legal compound sequence (0-3 stops) a backward DP finds the pit laps
that minimise total tyre time + pit-lane time. Because the DP is exact for the
model, there is no need for heuristic search or RL; the uncertainty lives in
the model parameters, which we handle with Monte Carlo instead.

Lap indices are 1-based race laps throughout. "Pit at lap p" means the car
enters the pit lane at the end of lap p and starts lap p+1 on new tyres.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from src.pitwall.race_data import DRY_COMPOUNDS
from src.pitwall.tyre_model import TyreModel

INF = 1e9
# Heavier cars put more energy through the tyres: wear is scaled by
# 1 + FUEL_WEAR_FACTOR * remaining_fuel_fraction. Assumption (not fitted);
# it is what makes "softer tyre first" orders differ from the reverse.
FUEL_WEAR_FACTOR = 0.25
# Pit loss multipliers under neutralisation (field slows, pit lane does not).
SC_PIT_FACTOR = 0.50
VSC_PIT_FACTOR = 0.65
SHORT = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}


@dataclass
class RaceContext:
    total_laps: int
    pit_loss: float
    sc_rate: float = 0.010  # Safety Car starts per lap
    vsc_rate: float = 0.006  # VSC starts per lap
    team: str | None = None
    driver: str | None = None  # personal wear multiplier lookup
    available: tuple[str, ...] = DRY_COMPOUNDS
    min_stint: int = 3
    # Track-position cost of each stop on top of pit-lane time (traffic, lost
    # places that must be re-passed). Learned from teams' revealed choices in
    # earlier races (see calibration.py); 0 = pure lap-time optimum.
    stop_penalty: float = 0.0
    # Wear sensitivity to fuel load (see FUEL_WEAR_FACTOR); learned per round.
    fuel_wear: float = FUEL_WEAR_FACTOR
    # Pre-race only: extra cost of starting on a compound (launch grip, lap-1
    # position). Learned from teams' revealed start-tyre choices.
    start_penalty: dict = field(default_factory=dict)


@dataclass
class TyreState:
    """Where the car is now. Pre-race: lap=0, compound=None."""
    lap: int = 0  # laps completed
    compound: str | None = None
    age: int = 0  # laps already on the current set
    used: tuple[str, ...] = ()  # dry compounds already raced (incl. current)
    pit_cost_now: float | None = None  # override for pitting at the end of this lap (SC/VSC)


@dataclass
class Strategy:
    sequence: list[str]
    pit_laps: list[int]
    total: float  # model time for the remaining race (tyre + pit), seconds
    delta: float = 0.0
    windows: list[tuple[int, int]] = field(default_factory=list)
    stints: list[dict] = field(default_factory=list)
    risk: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "-".join(SHORT.get(c, c[0]) for c in self.sequence)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        d["stops"] = len(self.pit_laps)
        d["windows"] = [list(w) for w in self.windows]
        return d


class Optimizer:
    def __init__(self, model: TyreModel, ctx: RaceContext):
        self.model = model
        self.ctx = ctx
        L = ctx.total_laps
        self.L = L
        laps = np.arange(L + 2, dtype=float)
        self.wear_w = 1.0 + ctx.fuel_wear * np.clip((L - laps) / L, 0, 1)
        self._fresh: dict[str, np.ndarray] = {}

    # -- stint costs --------------------------------------------------------
    def _lap_cost(self, compound: str, ages: np.ndarray, laps: np.ndarray) -> np.ndarray:
        p = self.model.compounds[compound]
        k = self.model.deg_for(compound, self.ctx.team, self.ctx.driver)
        return p.offset + k * ages * self.wear_w[laps] + self.model.quad * ages**2

    def fresh_matrix(self, compound: str) -> np.ndarray:
        """M[s, e] = tyre time of a new-tyre stint covering laps s..e (inclusive)."""
        if compound in self._fresh:
            return self._fresh[compound]
        L = self.L
        M = np.full((L + 2, L + 2), INF)
        max_len = self.model.compounds[compound].max_stint
        n = np.arange(1, L + 1)
        for s in range(1, L + 1):
            laps = n[n >= s]
            ages = (laps - s + 1).astype(float)
            cum = np.cumsum(self._lap_cost(compound, ages, laps))
            lengths = laps - s + 1
            cum[lengths > max_len] = INF
            M[s, laps] = cum
        self._fresh[compound] = M
        return M

    def current_cost(self, compound: str, age: int, start: int) -> np.ndarray:
        """c[e] = tyre time from lap `start` to lap e on the current (used) set."""
        L = self.L
        c = np.full(L + 2, INF)
        laps = np.arange(start, L + 1)
        ages = (age + laps - start + 1).astype(float)
        cum = np.cumsum(self._lap_cost(compound, ages, laps))
        # A worn set can always be pushed a bit, but not beyond ~1.3x its life.
        too_long = ages > self.model.compounds[compound].max_stint * 1.3
        cum[too_long] = INF
        c[laps] = cum
        return c

    # -- DP -----------------------------------------------------------------
    def _pit_cost(self, state: TyreState) -> np.ndarray:
        P = np.full(self.L + 2, self.ctx.pit_loss + self.ctx.stop_penalty)
        if state.pit_cost_now is not None:
            P[state.lap + 1] = state.pit_cost_now + self.ctx.stop_penalty  # box at end of the lap being driven
        return P

    def _suffix(self, seq: list[str], P: np.ndarray, min_stint: int):
        """H[j][s]: best time for laps s..L using seq[j:], new tyres at s."""
        L = self.L
        H = [np.full(L + 2, INF) for _ in seq]
        arg = [np.zeros(L + 2, dtype=int) for _ in seq]
        last = self.fresh_matrix(seq[-1])
        H[-1][1 : L + 1] = last[1 : L + 1, L]
        lengths = L - np.arange(1, L + 1) + 1
        H[-1][1 : L + 1][lengths < min_stint] = INF
        e_idx = np.arange(L + 2)
        for j in range(len(seq) - 2, -1, -1):
            M = self.fresh_matrix(seq[j])
            nxt = np.full(L + 2, INF)
            nxt[: L + 1] = H[j + 1][1 : L + 2]  # nxt[e] = H[j+1][e+1]
            T = M + P[None, :] + nxt[None, :]
            s_idx = np.arange(L + 2)[:, None]
            T[(e_idx[None, :] - s_idx + 1) < min_stint] = INF
            T[:, L:] = INF  # cannot pit on the last lap
            arg[j] = np.argmin(T, axis=1)
            H[j] = T[np.arange(L + 2), arg[j]]
        return H, arg

    def _first_costs(self, seq: list[str], state: TyreState) -> tuple[np.ndarray, int]:
        start = state.lap + 1
        if state.compound is None:
            return self.fresh_matrix(seq[0])[start].copy(), start
        return self.current_cost(state.compound, state.age, start), start

    def evaluate(self, seq: list[str], pit_laps: list[int], state: TyreState, P: np.ndarray | None = None) -> float:
        """Model time for an explicit plan (used for windows, what-ifs and Monte Carlo)."""
        P = self._pit_cost(state) if P is None else P
        start = state.lap + 1
        bounds = [start] + [p + 1 for p in pit_laps] + [self.L + 1]
        if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
            return INF
        total = 0.0
        for i, c in enumerate(seq):
            s, e = bounds[i], bounds[i + 1] - 1
            if i == 0 and state.compound is not None:
                total += self.current_cost(state.compound, state.age, start)[e]
            else:
                total += self.fresh_matrix(c)[s, e]
            if i < len(pit_laps):
                total += P[pit_laps[i]]
        return float(total)

    def sequences(self, state: TyreState, max_stops: int) -> list[list[str]]:
        avail = [c for c in self.ctx.available if c in self.model.compounds]
        out = []
        first_opts = [state.compound] if state.compound else avail
        for stops in range(0, max_stops + 1):
            for first in first_opts:
                for rest in itertools.product(avail, repeat=stops):
                    seq = [first, *rest]
                    used = set(state.used) | set(seq)
                    if len(used & set(DRY_COMPOUNDS)) < 2:
                        continue  # FIA: two different dry compounds in a dry race
                    out.append(seq)
        return out

    def optimise(self, state: TyreState | None = None, max_stops: int = 3, top: int = 10, window_tol: float = 2.0) -> list[Strategy]:
        state = state or TyreState()
        P = self._pit_cost(state)
        min_stint = self.ctx.min_stint
        results: list[Strategy] = []
        for seq in self.sequences(state, max_stops):
            first, start = self._first_costs(seq, state)
            if len(seq) == 1:
                total = first[self.L] + (self.ctx.start_penalty.get(seq[0], 0.0) if state.compound is None else 0.0)
                if total >= INF:
                    continue
                results.append(Strategy(seq, [], float(total)))
                continue
            H, arg = self._suffix(seq[1:], P, min_stint)
            nxt = np.full(self.L + 2, INF)
            nxt[: self.L + 1] = H[0][1 : self.L + 2]
            totals = first + P + nxt
            # pre-race the opening stint must be >= min_stint; live we may box this lap
            lo = start + (min_stint - 1 if state.compound is None else 0)
            totals[:lo] = INF
            totals[self.L :] = INF
            e0 = int(np.argmin(totals))
            best = float(totals[e0])
            if best >= INF:
                continue
            pits = [e0]
            s = e0 + 1
            for j in range(len(seq) - 2):
                e = int(arg[j][s])
                pits.append(e)
                s = e + 1
            ok = np.where(totals <= best + window_tol)[0]
            win0 = (int(ok.min()), int(ok.max())) if len(ok) else (e0, e0)
            strat = Strategy(seq, pits, best, windows=[win0])
            strat.windows += [self._stop_window(strat, i, state, P, window_tol) for i in range(1, len(pits))]
            results.append(strat)
        results.sort(key=lambda s: s.total)
        results = results[:top]
        if results:
            best = results[0].total
            for r in results:
                r.delta = r.total - best
                r.stints = self._stints(r, state)
        return results

    def candidates(self, state: TyreState | None = None, max_stops: int = 3, top: int = 6) -> list[Strategy]:
        """Top plans plus the best plan for every stop count, so Monte Carlo can
        say how likely each stop count is to be fastest (not just the leader)."""
        state = state or TyreState()
        allp = self.optimise(state, max_stops=max_stops, top=400)
        chosen = allp[:top]
        seen = {len(p.pit_laps) for p in chosen}
        for p in allp:
            k = len(p.pit_laps)
            if k not in seen:
                chosen.append(p)
                seen.add(k)
        return chosen

    def plan_with_first_stop(self, seq: list[str], state: TyreState, target: int, window_tol: float = 2.0) -> Strategy | None:
        """Best plan for a fixed compound sequence whose first stop is the
        feasible lap closest to `target` (later stops re-optimised)."""
        if len(seq) < 2:
            return None
        P = self._pit_cost(state)
        first, start = self._first_costs(seq, state)
        H, arg = self._suffix(seq[1:], P, self.ctx.min_stint)
        nxt = np.full(self.L + 2, INF)
        nxt[: self.L + 1] = H[0][1 : self.L + 2]
        totals = first + P + nxt
        lo = start + (self.ctx.min_stint - 1 if state.compound is None else 0)
        totals[:lo] = INF
        totals[self.L :] = INF
        feasible = np.where(totals < INF)[0]
        if not len(feasible):
            return None
        e0 = int(feasible[np.argmin(np.abs(feasible - target))])
        pits, s = [e0], e0 + 1
        for j in range(len(seq) - 2):
            e = int(arg[j][s])
            pits.append(e)
            s = e + 1
        best = float(totals.min())
        ok = np.where(totals <= best + window_tol)[0]
        strat = Strategy(list(seq), pits, float(totals[e0]), windows=[(int(ok.min()), int(ok.max()))])
        strat.windows += [self._stop_window(strat, i, state, P, window_tol) for i in range(1, len(pits))]
        strat.stints = self._stints(strat, state)
        return strat

    def best_with_first_stop(self, state: TyreState, lap: int, max_stops: int = 3) -> float | None:
        """Best model time over all plans whose first stop is at `lap`."""
        P = self._pit_cost(state)
        best = None
        for seq in self.sequences(state, max_stops):
            if len(seq) < 2:
                continue
            first, _ = self._first_costs(seq, state)
            H, _ = self._suffix(seq[1:], P, self.ctx.min_stint)
            if lap + 1 > self.L:
                continue
            total = first[lap] + P[lap] + H[0][lap + 1]
            if total < INF and (best is None or total < best):
                best = float(total)
        return best

    def _stop_window(self, strat: Strategy, i: int, state: TyreState, P: np.ndarray, tol: float) -> tuple[int, int]:
        lo = hi = strat.pit_laps[i]
        base = strat.total
        for direction in (-1, 1):
            p = strat.pit_laps[i]
            while True:
                p += direction
                laps = list(strat.pit_laps)
                laps[i] = p
                if p <= state.lap or p >= self.L:
                    break
                if self.evaluate(strat.sequence, laps, state, P) > base + tol:
                    break
                lo, hi = min(lo, p), max(hi, p)
        return (lo, hi)

    def _stints(self, strat: Strategy, state: TyreState) -> list[dict]:
        bounds = [state.lap + 1] + [p + 1 for p in strat.pit_laps] + [self.L + 1]
        out = []
        for i, c in enumerate(strat.sequence):
            s, e = bounds[i], bounds[i + 1] - 1
            age0 = state.age if (i == 0 and state.compound) else 0
            out.append({"compound": c, "start_lap": s, "end_lap": e, "laps": e - s + 1, "age_start": age0, "new": not (i == 0 and state.compound)})
        return out

    # -- Monte Carlo risk -----------------------------------------------------
    def monte_carlo(self, strategies: list[Strategy], state: TyreState | None = None, sims: int = 600, seed: int = 7) -> None:
        """Safety-car robustness with common random numbers across strategies.

        Each simulation draws SC / VSC periods from a per-lap hazard. A strategy
        reacts like a real pit wall: if a neutralisation starts inside the
        window of its next planned stop (up to 10 laps early), it stops then.
        Degradation parameters are also resampled from their posterior sd, so
        the spread reflects both race events and model uncertainty.
        """
        if not strategies:
            return
        state = state or TyreState()
        rng = np.random.default_rng(seed)
        L, start = self.L, state.lap + 1
        base_P = self._pit_cost(state)
        results = np.zeros((sims, len(strategies)))
        saved = {c: (p.deg, p.offset) for c, p in self.model.compounds.items()}
        try:
            for k in range(sims):
                # Parameter uncertainty: one posterior draw per simulation.
                for c, p in self.model.compounds.items():
                    d0, o0 = saved[c]
                    p.deg = max(0.003, d0 + rng.normal(0, p.deg_sd))
                    p.offset = o0 + (rng.normal(0, p.offset_sd) if c != "MEDIUM" else 0.0)
                self._fresh.clear()
                P = base_P.copy()
                neutral = np.zeros(L + 2, dtype=int)  # 0 green, 1 VSC, 2 SC
                lap = start
                while lap <= L:
                    u = rng.random()
                    if u < self.ctx.sc_rate:
                        dur = int(rng.integers(3, 6))
                        neutral[lap : lap + dur] = 2
                        lap += dur
                    elif u < self.ctx.sc_rate + self.ctx.vsc_rate:
                        dur = int(rng.integers(1, 4))
                        neutral[lap : lap + dur] = np.maximum(neutral[lap : lap + dur], 1)
                        lap += dur
                    else:
                        lap += 1
                P[neutral == 2] = self.ctx.pit_loss * SC_PIT_FACTOR + self.ctx.stop_penalty
                P[neutral == 1] = self.ctx.pit_loss * VSC_PIT_FACTOR + self.ctx.stop_penalty
                starts = [x for x in range(start, L) if neutral[x] and not neutral[x - 1]]
                for j, s in enumerate(strategies):
                    planned = self.evaluate(s.sequence, s.pit_laps, state, P)
                    reacted = self.evaluate(s.sequence, self._react(s.pit_laps, starts, state), state, P)
                    # A pit wall only reacts if the reaction is feasible and helps.
                    results[k, j] = min(planned, reacted)
        finally:
            for c, p in self.model.compounds.items():
                p.deg, p.offset = saved[c]
            self._fresh.clear()
        results = np.where(results >= INF / 2, np.nan, results)
        results = np.where(np.isnan(results), np.nanmax(results, axis=0, initial=0.0), results)
        # Regret: time lost vs the best of these plans *in the same simulated race*.
        regret = results - results.min(axis=1, keepdims=True)
        winners = np.argmin(results, axis=1)
        stops = np.array([len(s.pit_laps) for s in strategies])
        self.stop_probabilities = {int(k): float((stops[winners] == k).mean()) for k in sorted(set(stops.tolist()))}
        for j, s in enumerate(strategies):
            col = regret[:, j]
            worst = np.sort(col)[-max(1, sims // 10) :]
            s.risk = {
                "expected_delta": float(col.mean()),
                "p50": float(np.percentile(col, 50)),
                "p90": float(np.percentile(col, 90)),
                "cvar90": float(worst.mean()),
                "win_probability": float((winners == j).mean()),
            }

    def _react(self, pits: list[int], neutral_starts: list[int], state: TyreState) -> list[int]:
        pits = list(pits)
        for ns in neutral_starts:
            for i, p in enumerate(pits):
                prev = pits[i - 1] if i else state.lap
                if p > ns and p - ns <= 10 and ns - prev >= self.ctx.min_stint:
                    shift = ns - p
                    pits[i] = ns
                    # later stops keep their spacing, clipped to the race
                    for j in range(i + 1, len(pits)):
                        pits[j] = min(max(pits[j] + shift, pits[j - 1] + self.ctx.min_stint), self.L - 1)
                    break
                if p >= ns:
                    break
        return pits
