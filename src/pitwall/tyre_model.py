"""Hierarchical Bayesian tyre model.

Lap time for driver d on compound c, tyre age a (laps), race lap n:

    t = base_d + offset_c + deg_c * a + quad * a^2 - fuel * n + noise

Three sources of evidence are combined per race weekend:

1. Season prior   - the same model fitted to every *earlier* 2026 race
                    (walk-forward: never uses the race being predicted).
2. Practice runs  - FP long runs fitted with a robust Theil-Sen slope.
3. Live race laps - conjugate Gaussian updates while the race is running.

Every estimate carries a standard deviation, so the strategy layer can
propagate uncertainty instead of pretending point estimates are exact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.pitwall.race_data import DRY_COMPOUNDS

# ---------------------------------------------------------------------------
# Modelling assumptions (documented so a reviewer can challenge each one)
# ---------------------------------------------------------------------------
# Used only before any 2026 race exists (round 1). Values are typical
# Pirelli-era magnitudes; they are deliberately vague (large sd).
DEFAULT_PRIOR = {
    "SOFT": {"offset": -0.55, "deg": 0.10, "offset_sd": 0.4, "deg_sd": 0.06},
    "MEDIUM": {"offset": 0.0, "deg": 0.065, "offset_sd": 0.01, "deg_sd": 0.05},
    "HARD": {"offset": 0.45, "deg": 0.045, "offset_sd": 0.4, "deg_sd": 0.04},
}
DEFAULT_FUEL_PER_LAP = 0.055  # s/lap gained per lap of fuel burnt (overridden by fitted value)
# Extra variance when transferring practice degradation to the race: fuel load,
# track evolution and engine modes differ between FP long runs and Sunday.
PRACTICE_TRANSFER_SD_DEG = 0.025
PRACTICE_TRANSFER_SD_OFFSET = 0.25
# Minimum clean laps in a live stint before its slope is used (shorter stints
# give slopes that overshoot; Barcelona replay: 4 laps -> 0.26 vs 0.215 true).
LIVE_MIN_STINT_LAPS = 6
# Pseudo-count (laps) for shrinking team-specific degradation toward the field.
TEAM_SHRINK_LAPS = 25.0
# Upper bound used when nothing better is known.
DEFAULT_MAX_STINT = {"SOFT": 30, "MEDIUM": 42, "HARD": 55}


@dataclass
class CompoundParams:
    offset: float  # s/lap vs MEDIUM on fresh tyres
    deg: float  # s/lap added per lap of tyre age
    offset_sd: float
    deg_sd: float
    max_stint: int
    evidence_laps: int = 0


@dataclass
class TyreModel:
    compounds: dict[str, CompoundParams]
    fuel_per_lap: float = DEFAULT_FUEL_PER_LAP
    quad: float = 0.0  # shared wear-acceleration term (s/lap^2)
    team_deg: dict[str, dict[str, float]] = field(default_factory=dict)
    driver_pace: dict[str, float] = field(default_factory=dict)
    noise_sd: float = 0.6
    source: str = "default"
    # Personal wear: driver -> {"mult": wear relative to the field, "sd": ..., evidence...}
    driver_mult: dict[str, dict] = field(default_factory=dict)
    # Race fits only: per-driver raw wear observations used to build season priors.
    driver_obs: dict[str, dict] = field(default_factory=dict)

    def deg_for(self, compound: str, team: str | None = None, driver: str | None = None) -> float:
        base = self.compounds[compound].deg
        if driver and driver in self.driver_mult:
            # The personal multiplier already includes the team effect (hierarchical prior).
            return max(base * self.driver_mult[driver]["mult"], 0.005)
        if team and team in self.team_deg:
            base += self.team_deg[team].get(compound, 0.0)
        return max(base, 0.005)

    def lap_delta(self, compound: str, age: np.ndarray | float, team: str | None = None, driver: str | None = None) -> np.ndarray:
        """Tyre-only lap-time delta (no fuel, no base pace) at the given tyre ages."""
        p = self.compounds[compound]
        age = np.asarray(age, dtype=float)
        return p.offset + self.deg_for(compound, team, driver) * age + self.quad * age**2

    def to_dict(self) -> dict:
        return {
            "compounds": {c: asdict(p) for c, p in self.compounds.items()},
            "fuel_per_lap": self.fuel_per_lap,
            "quad": self.quad,
            "team_deg": self.team_deg,
            "driver_pace": self.driver_pace,
            "noise_sd": self.noise_sd,
            "source": self.source,
            "driver_mult": self.driver_mult,
            "driver_obs": self.driver_obs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TyreModel":
        return cls(
            compounds={c: CompoundParams(**p) for c, p in d["compounds"].items()},
            fuel_per_lap=d.get("fuel_per_lap", DEFAULT_FUEL_PER_LAP),
            quad=d.get("quad", 0.0),
            team_deg=d.get("team_deg", {}),
            driver_pace=d.get("driver_pace", {}),
            noise_sd=d.get("noise_sd", 0.6),
            source=d.get("source", "unknown"),
            driver_mult={k: dict(v) for k, v in d.get("driver_mult", {}).items()},
            driver_obs=d.get("driver_obs", {}),
        )


def _pav(values: list[float], weights: list[float], increasing: bool) -> list[float]:
    """Weighted pool-adjacent-violators (isotonic regression) for a short list."""
    vals = list(values) if increasing else [-v for v in values]
    blocks = [[v, w, 1] for v, w in zip(vals, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return out if increasing else [-v for v in out]


def enforce_compound_order(model: TyreModel) -> TyreModel:
    """Physics constraint: softer compounds are faster when new and wear faster.

    Projects the estimates onto offset(S) <= offset(M) <= offset(H) and
    deg(S) >= deg(M) >= deg(H) with inverse-variance weights. Noisy race fits
    (selection effects, traffic) otherwise produce inverted compound orders.
    """
    order = [c for c in DRY_COMPOUNDS if c in model.compounds]
    if len(order) < 2:
        return model
    ps = [model.compounds[c] for c in order]
    offs = _pav([p.offset for p in ps], [1 / max(p.offset_sd, 0.02) ** 2 for p in ps], increasing=True)
    degs = _pav([p.deg for p in ps], [1 / max(p.deg_sd, 0.005) ** 2 for p in ps], increasing=False)
    ref = offs[order.index("MEDIUM")] if "MEDIUM" in order else 0.0
    for p, o, d in zip(ps, offs, degs):
        p.offset, p.deg = o - ref, max(d, 0.003)
    return model


def default_model() -> TyreModel:
    return TyreModel(
        compounds={
            c: CompoundParams(v["offset"], v["deg"], v["offset_sd"], v["deg_sd"], DEFAULT_MAX_STINT[c])
            for c, v in DEFAULT_PRIOR.items()
        },
        source="default-prior",
    )


# ---------------------------------------------------------------------------
# Race fit (used for the season prior and for post-race "hindsight" analysis)
# ---------------------------------------------------------------------------
def clean_race_laps(laps: pd.DataFrame, neutral: dict[str, list[int]]) -> pd.DataFrame:
    """Green-flag, representative race laps on dry tyres."""
    blocked = set(neutral.get("sc", [])) | set(neutral.get("vsc", [])) | set(neutral.get("red", []))
    # The lap after a neutralisation is a restart lap: not representative.
    blocked |= {lap + 1 for lap in blocked}
    df = laps[
        (laps["lap"] > 1)
        & (~laps["pit_in"])
        & (~laps["pit_out"])
        & laps["lap_time"].notna()
        & laps["tyre_life"].notna()
        & laps["compound"].isin(DRY_COMPOUNDS)
        & (~laps["lap"].isin(blocked))
    ].copy()
    if df.empty:
        return df
    med = df.groupby("driver")["lap_time"].transform("median")
    df = df[df["lap_time"] < med * 1.05]
    return df


def _robust_lstsq(X: np.ndarray, y: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with iterative MAD trimming (cheap, robust to traffic laps)."""
    keep = np.ones(len(y), dtype=bool)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        resid = y - X @ beta
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826 + 1e-6
        new_keep = np.abs(resid) < 3.0 * mad
        if new_keep.sum() < X.shape[1] * 3 or (new_keep == keep).all():
            break
        keep = new_keep
    return beta, keep


def fit_race_model(laps: pd.DataFrame, neutral: dict[str, list[int]], fuel_per_lap: float | None = None) -> TyreModel | None:
    """Fit the lap-time model to a finished race.

    If fuel_per_lap is None it is estimated jointly (identifiable because tyre
    age resets at each stop while race lap does not).
    """
    df = clean_race_laps(laps, neutral)
    compounds = [c for c in DRY_COMPOUNDS if (df["compound"] == c).sum() >= 30] if not df.empty else []
    if len(df) < 200 or not compounds:
        return None
    drivers = sorted(df["driver"].unique())
    d_idx = {d: i for i, d in enumerate(drivers)}
    ref = "MEDIUM" if "MEDIUM" in compounds else compounds[0]
    other = [c for c in compounds if c != ref]

    n = len(df)
    cols = len(drivers) + len(other) + len(compounds) + 1 + (1 if fuel_per_lap is None else 0)
    X = np.zeros((n, cols))
    X[np.arange(n), df["driver"].map(d_idx).to_numpy()] = 1.0
    off0 = len(drivers)
    comp = df["compound"].to_numpy()
    age = df["tyre_life"].to_numpy(dtype=float)
    lap = df["lap"].to_numpy(dtype=float)
    for j, c in enumerate(other):
        X[:, off0 + j] = comp == c
    deg0 = off0 + len(other)
    for j, c in enumerate(compounds):
        X[:, deg0 + j] = np.where(comp == c, age, 0.0)
    q_col = deg0 + len(compounds)
    X[:, q_col] = age**2
    y = df["lap_time"].to_numpy(dtype=float)
    if fuel_per_lap is None:
        X[:, q_col + 1] = -lap
    else:
        y = y + fuel_per_lap * lap

    beta, keep = _robust_lstsq(X, y)
    resid = (y - X @ beta)[keep]
    noise_sd = float(np.std(resid))
    fuel = float(beta[q_col + 1]) if fuel_per_lap is None else fuel_per_lap
    quad = float(np.clip(beta[q_col], 0.0, 0.004))

    # Standard errors from (X'X)^-1 * sigma^2 on the kept rows.
    try:
        cov = np.linalg.inv(X[keep].T @ X[keep]) * noise_sd**2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(cols, 0.05)

    max_age = df.groupby("compound")["tyre_life"].max().to_dict()
    params: dict[str, CompoundParams] = {}
    for j, c in enumerate(compounds):
        offset = 0.0 if c == ref else float(beta[off0 + other.index(c)])
        offset_sd = 0.01 if c == ref else float(se[off0 + other.index(c)])
        params[c] = CompoundParams(
            offset=offset,
            deg=float(max(beta[deg0 + j], 0.0)),
            offset_sd=offset_sd,
            deg_sd=float(se[deg0 + j]),
            max_stint=int(max(max_age.get(c, DEFAULT_MAX_STINT[c]), 10)),
            evidence_laps=int((comp[keep] == c).sum()),
        )

    base = beta[: len(drivers)]
    pace = {d: float(base[i] - np.median(base)) for d, i in d_idx.items()}

    # Team-specific degradation: shrink the per-team slope of the residuals
    # toward zero (empirical-Bayes style) so sparse teams don't overfit.
    team_deg: dict[str, dict[str, float]] = {}
    df_k = df.iloc[np.where(keep)[0]].assign(resid=resid)
    for (team, c), grp in df_k.groupby(["team", "compound"]):
        if c not in params or len(grp) < 8:
            continue
        x = grp["tyre_life"].to_numpy(dtype=float)
        x = x - x.mean()
        sxx = float((x**2).sum())
        slope = float((x * grp["resid"].to_numpy()).sum() / (sxx + TEAM_SHRINK_LAPS * 40.0))
        team_deg.setdefault(team, {})[c] = slope

    # Personal wear: each driver's extra wear vs the fitted field rate, from the
    # slope of their residuals against tyre age *within* each stint.
    driver_obs: dict[str, dict] = {}
    deg_by_c = {c: params[c].deg for c in params}
    for drv, grp in df_k.groupby("driver"):
        x = grp["tyre_life"].to_numpy(dtype=float) - grp.groupby("stint")["tyre_life"].transform("mean").to_numpy(dtype=float)
        sxx = float((x**2).sum())
        if len(grp) < 12 or sxx < 50:
            continue
        delta = float((x * grp["resid"].to_numpy()).sum() / sxx)
        field_rate = float(np.mean([deg_by_c.get(c, 0.0) for c in grp["compound"]]))
        driver_obs[drv] = {"team": str(grp["team"].iloc[0]), "delta": delta, "se": float(noise_sd / np.sqrt(sxx)),
                           "field": field_rate, "laps": int(len(grp))}

    return enforce_compound_order(TyreModel(
        compounds=params,
        fuel_per_lap=fuel,
        quad=quad,
        team_deg=team_deg,
        driver_pace=pace,
        noise_sd=noise_sd,
        source="race-fit",
        driver_obs=driver_obs,
    ))


# ---------------------------------------------------------------------------
# Practice long runs
# ---------------------------------------------------------------------------
def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return 0.0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def practice_long_runs(laps: pd.DataFrame, session_name: str, min_laps: int = 5) -> pd.DataFrame:
    """Detect race-simulation runs in a practice session.

    A long run is a stint where >= min_laps laps sit within 3% of the stint's
    fastest representative pace (filters out cool-down and push laps).
    """
    rows = []
    if laps.empty:
        return pd.DataFrame()
    df = laps[
        laps["lap_time"].notna()
        & (~laps["pit_in"])
        & (~laps["pit_out"])
        & laps["compound"].isin(DRY_COMPOUNDS)
        & laps["tyre_life"].notna()
    ]
    for (driver, stint), grp in df.groupby(["driver", "stint"]):
        if len(grp) < min_laps:
            continue
        t = grp["lap_time"].to_numpy(dtype=float)
        ref = np.percentile(t, 20)
        mask = (t < ref * 1.03) & (t > ref * 0.985)
        if mask.sum() < min_laps:
            continue
        g = grp[mask]
        age = g["tyre_life"].to_numpy(dtype=float)
        lt = g["lap_time"].to_numpy(dtype=float)
        rows.append(
            {
                "session": session_name,
                "driver": driver,
                "team": g["team"].iloc[0],
                "compound": g["compound"].iloc[0],
                "laps": int(mask.sum()),
                "age_start": float(age.min()),
                "age_end": float(age.max()),
                "median_time": float(np.median(lt)),
                "raw_slope": _theil_sen(age, lt),
                "ages": age.tolist(),
                "times": lt.tolist(),
            }
        )
    return pd.DataFrame(rows)


def short_run_offsets(session_laps: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Compound pace deltas from each driver's best push lap per compound.

    Only pairs where both laps are within 2.5% of that driver's session best
    are used, so a high-fuel medium run is never compared with a low-fuel
    soft run.
    """
    diffs: dict[str, list[float]] = {"SOFT": [], "HARD": []}
    for _, laps in session_laps.items():
        df = laps[laps["lap_time"].notna() & laps["accurate"] & ~laps["deleted"] & ~laps["pit_in"] & ~laps["pit_out"]]
        df = df[df["compound"].isin(DRY_COMPOUNDS)]
        for _, grp in df.groupby("driver"):
            best = grp.groupby("compound")["lap_time"].min()
            overall = best.min()
            best = best[best < overall * 1.025]
            if "MEDIUM" not in best:
                continue
            for c in ("SOFT", "HARD"):
                if c in best:
                    diffs[c].append(float(best[c] - best["MEDIUM"]))
    out = {}
    for c, vals in diffs.items():
        arr = np.array([v for v in vals if abs(v) < 2.0])
        if len(arr) >= 3:
            med = float(np.median(arr))
            out[c] = {"offset": med, "offset_se": float(1.25 * np.median(np.abs(arr - med)) / np.sqrt(len(arr)) + 0.05), "pairs": int(len(arr))}
    return out


def summarise_practice(runs: pd.DataFrame, fuel_per_lap: float) -> dict:
    """Per-compound practice estimates of degradation and pace offset (vs MEDIUM)."""
    out: dict[str, dict] = {"compounds": {}, "teams": {}}
    if runs.empty:
        return out
    runs = runs.copy()
    # Fuel burn makes a car faster each lap, hiding true wear: add it back.
    runs["deg"] = runs["raw_slope"] + fuel_per_lap
    # Clip absurd slopes (traffic, mode changes) before aggregating.
    runs = runs[(runs["deg"] > -0.15) & (runs["deg"] < 0.6)]
    for c, grp in runs.groupby("compound"):
        w = grp["laps"].to_numpy(dtype=float)
        d = grp["deg"].to_numpy(dtype=float)
        order = np.argsort(d)
        cum = np.cumsum(w[order]) / w.sum()
        wmed = float(d[order][np.searchsorted(cum, 0.5)])
        mad = float(np.median(np.abs(d - wmed))) * 1.4826 + 0.01
        out["compounds"][c] = {
            "deg": wmed,
            "deg_se": mad / np.sqrt(max(len(d), 1)),
            "runs": int(len(d)),
            "laps": int(w.sum()),
        }
    # Pace offsets from drivers who ran two compounds in the same session.
    pairs: dict[str, list[float]] = {}
    for (_, driver), grp in runs.groupby(["session", "driver"]):
        by_c = {}
        for _, r in grp.iterrows():
            # Normalise to a fresh tyre using the run's own wear.
            mid_age = (r["age_start"] + r["age_end"]) / 2
            by_c.setdefault(r["compound"], []).append(r["median_time"] - r["deg"] * mid_age)
        if "MEDIUM" in by_c:
            m = float(np.median(by_c["MEDIUM"]))
            for c in ("SOFT", "HARD"):
                if c in by_c:
                    pairs.setdefault(c, []).append(float(np.median(by_c[c])) - m)
    for c, diffs in pairs.items():
        if c in out["compounds"] and len(diffs) >= 2:
            arr = np.array(diffs)
            arr = arr[np.abs(arr) < 2.0]
            if len(arr):
                out["compounds"][c]["offset"] = float(np.median(arr))
                out["compounds"][c]["offset_se"] = float(np.std(arr) / np.sqrt(len(arr)) + 0.1)
                out["compounds"][c]["offset_pairs"] = int(len(arr))
    # Team-level degradation (for hierarchical shrinkage later).
    for (team, c), grp in runs.groupby(["team", "compound"]):
        out["teams"].setdefault(team, {})[c] = {
            "deg": float(np.average(grp["deg"], weights=grp["laps"])),
            "laps": int(grp["laps"].sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Combining evidence
# ---------------------------------------------------------------------------
def season_prior(race_models: list[TyreModel]) -> TyreModel:
    """Aggregate earlier race fits into a prior (mean and between-race spread)."""
    if not race_models:
        return default_model()
    base = default_model()
    comps: dict[str, CompoundParams] = {}
    for c in DRY_COMPOUNDS:
        degs = [m.compounds[c].deg for m in race_models if c in m.compounds]
        offs = [m.compounds[c].offset for m in race_models if c in m.compounds]
        stints = [m.compounds[c].max_stint for m in race_models if c in m.compounds]
        d0 = base.compounds[c]
        if len(degs) >= 2:
            deg = float(np.median(degs))
            # Between-race spread is the honest prior uncertainty for a new track.
            deg_sd = float(max(np.std(degs), 0.02))
        else:
            deg, deg_sd = d0.deg, d0.deg_sd
        if len(offs) >= 2 and c != "MEDIUM":
            off, off_sd = float(np.median(offs)), float(max(np.std(offs), 0.15))
        else:
            off, off_sd = d0.offset, d0.offset_sd
        comps[c] = CompoundParams(
            offset=off,
            deg=deg,
            offset_sd=off_sd,
            deg_sd=deg_sd,
            max_stint=int(np.percentile(stints, 75)) if stints else d0.max_stint,
            evidence_laps=0,
        )
    fuels = [m.fuel_per_lap for m in race_models if 0.0 < m.fuel_per_lap < 0.15]
    quads = [m.quad for m in race_models]
    return TyreModel(
        compounds=comps,
        fuel_per_lap=float(np.median(fuels)) if fuels else DEFAULT_FUEL_PER_LAP,
        quad=float(np.median(quads)) if quads else 0.0,
        noise_sd=float(np.median([m.noise_sd for m in race_models])),
        source=f"season-prior({len(race_models)} races)",
    )


def _posterior(mu0: float, sd0: float, obs: float, sd_obs: float) -> tuple[float, float]:
    """Conjugate Gaussian update."""
    p0, p1 = 1.0 / sd0**2, 1.0 / sd_obs**2
    var = 1.0 / (p0 + p1)
    return (mu0 * p0 + obs * p1) * var, float(np.sqrt(var))


def fit_transfer(pairs: list[tuple[float, float, float]], clip: tuple[float, float] = (0.2, 1.2)) -> dict:
    """Learn race_deg ~ beta * practice_deg from earlier weekends.

    pairs: (practice_deg, race_deg, weight). Weighted least squares through the
    origin; the weighted residual sd becomes the transfer uncertainty, so a
    poorly predictive practice signal automatically gets less weight.
    """
    if len(pairs) < 4:
        return {"beta": 0.6, "sd": 0.05, "n": len(pairs), "source": "default"}
    x, y, w = (np.array(v, dtype=float) for v in zip(*pairs))
    beta = float(np.sum(w * x * y) / np.sum(w * x * x))
    beta = float(np.clip(beta, *clip))
    resid = y - beta * x
    sd = float(np.sqrt(np.sum(w * resid**2) / np.sum(w)))
    return {"beta": beta, "sd": max(sd, PRACTICE_TRANSFER_SD_DEG), "n": len(pairs), "source": "learned"}


def _robust_posterior(mu0: float, sd0: float, obs: float, sd_obs: float) -> tuple[float, float]:
    """Gaussian update with a prior-data conflict guard.

    When the observation sits far outside the prior (> 2 combined sd), the
    prior is widened to |obs - mu0| / 2 before updating - a cheap stand-in for
    a heavy-tailed (Student-t) prior. Lets live data overrule a wrong pre-race
    estimate quickly instead of being anchored by it.
    """
    gap = abs(obs - mu0)
    if gap > 2.0 * float(np.hypot(sd0, sd_obs)):
        sd0 = max(sd0, gap / 2.0)
    return _posterior(mu0, sd0, obs, sd_obs)


DEFAULT_COMPOUND_RATIOS = {"SOFT": 1.25, "MEDIUM": 1.0, "HARD": 0.8}
# Extra uncertainty when a compound's wear is inferred from the other compounds.
SEVERITY_FILL_SD = 0.02
MIN_PRACTICE_LAPS = 8


def weekend_severity(practice: dict, ratios: dict[str, float]) -> float | None:
    """This weekend's practice wear in 'Medium-equivalent' units.

    Each compound with enough long-run laps gives deg_c / ratio_c, where ratio_c
    is that compound's typical wear relative to the Medium (learned from earlier
    races). Weighted by laps.
    """
    pc = practice.get("compounds", {})
    vals = [(pc[c]["deg"] / ratios[c], pc[c]["laps"]) for c in pc if pc[c]["laps"] >= MIN_PRACTICE_LAPS and ratios.get(c, 0) > 0]
    if not vals:
        return None
    return float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))


def weekend_model(
    prior: TyreModel,
    practice: dict,
    max_stint_hint: dict[str, int] | None = None,
    transfer: dict | None = None,
    short_offsets: dict | None = None,
    compound_ratios: dict[str, float] | None = None,
) -> TyreModel:
    """Blend the season prior with this weekend's practice running.

    Compounds with too little practice running (often the Hard) are not left on
    the season prior: their practice wear is inferred from the weekend's overall
    severity times the compound's typical wear ratio. Without this, a weekend
    with no Hard long runs silently gets a low-wear Hard from easier tracks.
    """
    comps: dict[str, CompoundParams] = {}
    pc = practice.get("compounds", {})
    transfer = transfer or {"beta": 1.0, "sd": PRACTICE_TRANSFER_SD_DEG}
    short_offsets = short_offsets or {}
    ratios = compound_ratios or DEFAULT_COMPOUND_RATIOS
    severity = weekend_severity(practice, ratios)
    for c, p in prior.compounds.items():
        deg, deg_sd, off, off_sd = p.deg, p.deg_sd, p.offset, p.offset_sd
        laps = 0
        if c in pc and pc[c]["laps"] >= MIN_PRACTICE_LAPS:
            obs = transfer["beta"] * pc[c]["deg"]
            obs_sd = float(np.hypot(transfer["beta"] * pc[c]["deg_se"], transfer["sd"]))
            deg, deg_sd = _posterior(deg, deg_sd, obs, obs_sd)
            laps = pc[c]["laps"]
        elif severity is not None and c in ratios:
            obs = transfer["beta"] * severity * ratios[c]
            obs_sd = float(np.hypot(transfer["sd"], SEVERITY_FILL_SD))
            deg, deg_sd = _posterior(deg, deg_sd, obs, obs_sd)
        if c in short_offsets and c != "MEDIUM":
            obs_sd = float(short_offsets[c]["offset_se"])
            off, off_sd = _posterior(off, off_sd, short_offsets[c]["offset"], obs_sd)
        elif c in pc and "offset" in pc[c] and c != "MEDIUM":
            obs_sd = float(np.hypot(pc[c]["offset_se"], PRACTICE_TRANSFER_SD_OFFSET))
            off, off_sd = _posterior(off, off_sd, pc[c]["offset"], obs_sd)
        max_stint = (max_stint_hint or {}).get(c, p.max_stint)
        comps[c] = CompoundParams(off, max(deg, 0.005), off_sd, deg_sd, int(max_stint), laps)

    # Hierarchical team effects: team deviation from the practice field mean,
    # shrunk by evidence (laps / (laps + TEAM_SHRINK_LAPS)).
    team_deg: dict[str, dict[str, float]] = {}
    for team, by_c in practice.get("teams", {}).items():
        for c, v in by_c.items():
            if c not in pc:
                continue
            w = v["laps"] / (v["laps"] + TEAM_SHRINK_LAPS)
            team_deg.setdefault(team, {})[c] = float(w * (v["deg"] - pc[c]["deg"]))

    return enforce_compound_order(TyreModel(
        compounds=comps,
        fuel_per_lap=prior.fuel_per_lap,
        quad=prior.quad,
        team_deg=team_deg,
        driver_pace={},
        noise_sd=prior.noise_sd,
        source=f"{prior.source} + practice",
    ))


def live_update(model: TyreModel, stint_obs: list[dict], driver_team: str | None = None) -> tuple[TyreModel, dict]:
    """Update compound degradation from live race stints (conjugate, per compound).

    stint_obs: [{"compound", "ages": [...], "times": [...], "laps": [...], "team"}]
    Lap times are fuel-corrected before the slope is estimated.
    """
    updated = TyreModel.from_dict(model.to_dict())
    evidence: dict[str, dict] = {}
    slopes: dict[str, list[tuple[float, float]]] = {}
    for s in stint_obs:
        ages = np.asarray(s["ages"], dtype=float)
        if len(ages) < LIVE_MIN_STINT_LAPS or s["compound"] not in updated.compounds:
            continue
        times = np.asarray(s["times"], dtype=float) + model.fuel_per_lap * np.asarray(s["laps"], dtype=float)
        slope = _theil_sen(ages, times)
        if not -0.2 < slope < 0.8:
            continue
        slopes.setdefault(s["compound"], []).append((slope, float(len(ages))))
    for c, vals in slopes.items():
        arr = np.array(vals)
        w = arr[:, 1]
        obs = float(np.average(arr[:, 0], weights=w))
        # Per-stint slope noise ~ sigma * sqrt(12 / n^3); pooled over stints.
        se = float(np.sqrt(1.0 / np.sum(w**3 / (12 * model.noise_sd**2))) + 0.01)
        p = updated.compounds[c]
        mu, sd = _robust_posterior(p.deg, p.deg_sd, obs, se)
        evidence[c] = {"prior": p.deg, "observed": obs, "posterior": mu, "stints": len(vals)}
        p.deg, p.deg_sd = max(mu, 0.005), sd
        p.evidence_laps += int(w.sum())
    updated.source = model.source + " + live"
    return updated, evidence


def apply_behaviour(model: TyreModel, behaviour: dict | None) -> TyreModel:
    """Apply revealed-preference corrections learned from teams' past choices.

    soft_mult scales SOFT wear: a linear wear curve misses the soft's thermal
    degradation / cliff on race fuel, which teams clearly price in (they rarely
    race it for long). Returns a copy.
    """
    out = TyreModel.from_dict(model.to_dict())
    if behaviour and "SOFT" in out.compounds:
        out.compounds["SOFT"].deg *= float(behaviour.get("soft_mult", 1.0))
    return out


# ---------------------------------------------------------------------------
# Personal (per-driver) wear
# ---------------------------------------------------------------------------
MULT_BOUNDS = (0.5, 1.8)
MIN_FIELD_RATE = 0.04  # below this, wear ratios are numerically meaningless (near-zero wear races)
DRIVER_WITHIN_TEAM_SD = 0.08  # assumed spread of teammates' wear around their team (same car)
# Variance components measured on 2026 race data (consecutive green stints, >= 8 laps):
#   persistent driver effect sd ~0.25-0.34, per-stint measurement noise sd ~0.44-0.55,
#   race-to-race variation of a driver's multiplier ~0.30. A weight of ~0.15 per
#   stint, learned on rounds 1-8, beat the field average on rounds 9-15 (MAE 0.531
#   vs 0.557); trusting raw per-driver numbers did worse (0.564).
RACE_TO_RACE_SD = 0.30
PRACTICE_RUN_SD = 0.50
LIVE_STINT_NOISE_SD = 0.55
LIVE_PRIOR_SD_MIN = 0.25


def _combine(obs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Precision-weighted mean and its sd of (value, sd) observations."""
    obs = [(v, s) for v, s in obs if np.isfinite(v) and s > 0]
    if not obs:
        return None
    w = np.array([1 / s**2 for _, s in obs])
    v = np.array([v for v, _ in obs])
    return float((w * v).sum() / w.sum()), float(np.sqrt(1 / w.sum()))


def season_driver_priors(race_models: list[TyreModel]) -> dict[str, dict]:
    """Two-level empirical-Bayes estimate of each driver's wear multiplier
    (driver within team, team within field) from earlier race fits."""
    per_driver: dict[str, list[tuple[float, float]]] = {}
    team_of: dict[str, str] = {}
    for m in race_models:
        for drv, o in m.driver_obs.items():
            if o["field"] < MIN_FIELD_RATE:
                continue
            mult = 1.0 + o["delta"] / o["field"]
            # one race says little about the next: add race-to-race variation
            se = float(np.hypot(o["se"] / o["field"], RACE_TO_RACE_SD))
            per_driver.setdefault(drv, []).append((float(np.clip(mult, *MULT_BOUNDS)), se))
            team_of[drv] = o["team"]
    drv_est = {d: c for d, obs in per_driver.items() if (c := _combine(obs))}
    if not drv_est:
        return {}
    # Team level: pool teammates, then shrink toward the field (1.0).
    team_obs: dict[str, list[tuple[float, float]]] = {}
    for d, (m, s) in drv_est.items():
        team_obs.setdefault(team_of[d], []).append((m, float(np.hypot(s, DRIVER_WITHIN_TEAM_SD))))
    team_raw = {t: _combine(o) for t, o in team_obs.items()}
    spread = np.array([m for m, _ in team_raw.values()])
    noise = np.array([s for _, s in team_raw.values()])
    # Between-team variance, method of moments (floored so it never collapses to 0).
    tau_t = float(np.sqrt(max(np.var(spread) - np.mean(noise**2), 0.05**2))) if len(spread) > 2 else 0.12
    team_post = {t: _posterior(1.0, tau_t, m, s) for t, (m, s) in team_raw.items()}
    out = {}
    for d, (m, s) in drv_est.items():
        tm, ts = team_post[team_of[d]]
        mu, sd = _posterior(tm, float(np.hypot(ts, DRIVER_WITHIN_TEAM_SD)), m, s)
        # Predictive sd for the *next* race includes race-to-race variation.
        out[d] = {"mult": float(np.clip(mu, *MULT_BOUNDS)), "sd": float(np.hypot(sd, RACE_TO_RACE_SD)), "team": team_of[d], "team_mult": tm,
                  "races": len(per_driver[d]), "raw_mult": m}
    return out


def practice_driver_multipliers(runs: list[dict], model: TyreModel, transfer: dict) -> dict[str, list[tuple[float, float]]]:
    """Per-driver wear observations (multiplier vs the weekend field rate) from FP long runs."""
    out: dict[str, list[tuple[float, float]]] = {}
    beta = transfer.get("beta", 1.0)
    rel_transfer_sd = PRACTICE_RUN_SD  # practice->race transfer is noisy per run (fuel, modes, traffic)
    for r in runs:
        c = r["compound"]
        if c not in model.compounds or r["laps"] < 5:
            continue
        field_rate = model.compounds[c].deg
        if field_rate < MIN_FIELD_RATE:
            continue
        slope = (r["raw_slope"] + model.fuel_per_lap) * beta
        n = r["laps"]
        se_slope = np.sqrt(12 * model.noise_sd**2 / n**3) * beta
        mult = slope / field_rate
        se = float(np.hypot(se_slope / field_rate, rel_transfer_sd))
        out.setdefault(r["driver"], []).append((float(np.clip(mult, *MULT_BOUNDS)), se))
    return out


def personalise(model: TyreModel, season: dict[str, dict], practice_obs: dict[str, list[tuple[float, float]]], team_of: dict[str, str]) -> TyreModel:
    """Weekend personal multipliers: season prior (or team prior) updated by practice runs."""
    team_means: dict[str, list[float]] = {}
    for d, v in season.items():
        team_means.setdefault(v["team"], []).append(v["team_mult"])
    for drv, team in team_of.items():
        if drv in season:
            mu0, sd0 = season[drv]["mult"], season[drv]["sd"]
            races = season[drv]["races"]
        elif team in team_means:  # e.g. a rookie: start from the team car
            mu0, sd0, races = float(np.mean(team_means[team])), RACE_TO_RACE_SD, 0
        else:
            mu0, sd0, races = 1.0, RACE_TO_RACE_SD, 0
        prac = _combine(practice_obs.get(drv, []))
        # Practice runs are recorded per driver but NOT used to move the pre-race
        # estimate: walk-forward, adding them made per-driver race-wear predictions
        # worse (MAE 0.168 vs 0.146 for the field average). Personalisation comes
        # from the driver's own race stints (live), where it was shown to help.
        mu, sd = mu0, sd0
        model.driver_mult[drv] = {
            "mult": float(np.clip(mu, *MULT_BOUNDS)), "sd": float(sd), "prior_mult": mu0, "prior_sd": sd0,
            "season_races": races, "practice_runs": len(practice_obs.get(drv, [])),
            "practice_mult": prac[0] if prac else None, "practice_used": False, "team": team,
        }
    return model
