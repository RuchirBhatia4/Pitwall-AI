from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.practice_loader import (
    filter_clean_practice_laps,
    load_practice_csvs,
)


SESSION_RELEVANCE_WEIGHTS = {
    "FP1": 0.80,
    "FP2": 1.00,
    "FP3": 1.15,
}

# Modeling assumption:
# Around 8 clean long-run laps are enough for a high-confidence session slope.
TARGET_CLEAN_LAPS_FOR_FULL_CONFIDENCE = 8


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        return 0.0

    return max(0.0, 1.0 - (ss_res / ss_tot))


def _fit_slope(df: pd.DataFrame) -> dict[str, float]:
    x = df["stint_lap"].to_numpy(dtype=float)
    y = df["lap_time_seconds"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = (slope * x) + intercept
    r2 = _safe_r2(y, y_pred)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _session_confidence(clean_lap_count: int, r2: float, slope: float, session: str) -> float:
    lap_conf = _clip(clean_lap_count / TARGET_CLEAN_LAPS_FOR_FULL_CONFIDENCE)
    fit_conf = _clip(r2)

    # Modeling assumption:
    # Strongly negative practice slopes are likely noisy/contaminated rather than
    # true tire improvement, so we reduce confidence instead of discarding them.
    slope_sign_penalty = 0.60 if slope < 0 else 1.00

    base_conf = (0.60 * lap_conf) + (0.40 * fit_conf)
    session_weight = SESSION_RELEVANCE_WEIGHTS.get(session, 1.00)

    return _clip(base_conf * slope_sign_penalty * session_weight)


def estimate_session_degradation(
    practice_laps: pd.DataFrame,
    min_clean_laps: int = 3,
) -> pd.DataFrame:
    """
    Estimates degradation slope per race/session/driver/team/compound.

    Input should already be clean-filtered practice laps.
    """
    if practice_laps.empty:
        return pd.DataFrame(
            columns=[
                "race",
                "session",
                "driver",
                "team",
                "compound",
                "clean_lap_count",
                "degradation_slope_per_lap",
                "degradation_intercept",
                "fit_r2",
                "confidence",
            ]
        )

    required_cols = [
        "race",
        "session",
        "driver",
        "team",
        "compound",
        "stint_lap",
        "lap_time_seconds",
    ]
    missing_cols = [col for col in required_cols if col not in practice_laps.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for calibration: {missing_cols}")

    rows: list[dict[str, Any]] = []

    group_cols = ["race", "session", "driver", "team", "compound"]

    for keys, group in practice_laps.groupby(group_cols):
        race, session, driver, team, compound = keys

        # Modeling assumption:
        # If multiple long runs exist in one session, we average repeated stint_lap
        # values so no single run dominates the slope fit.
        fit_df = (
            group.groupby("stint_lap", as_index=False)["lap_time_seconds"]
            .mean()
            .sort_values("stint_lap")
        )

        clean_lap_count = int(len(fit_df))
        if clean_lap_count < min_clean_laps:
            continue

        fit = _fit_slope(fit_df)

        rows.append(
            {
                "race": race,
                "session": session,
                "driver": driver,
                "team": team,
                "compound": compound,
                "clean_lap_count": clean_lap_count,
                "degradation_slope_per_lap": fit["slope"],
                "degradation_intercept": fit["intercept"],
                "fit_r2": fit["r2"],
                "confidence": _session_confidence(
                    clean_lap_count=clean_lap_count,
                    r2=fit["r2"],
                    slope=fit["slope"],
                    session=session,
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "race",
                "session",
                "driver",
                "team",
                "compound",
                "clean_lap_count",
                "degradation_slope_per_lap",
                "degradation_intercept",
                "fit_r2",
                "confidence",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["race", "driver", "compound", "session"]
    ).reset_index(drop=True)


def aggregate_practice_degradation(
    session_estimates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates FP1/FP2/FP3 session slopes into one race-weekend practice signal.
    """
    if session_estimates.empty:
        return pd.DataFrame(
            columns=[
                "race",
                "driver",
                "team",
                "compound",
                "practice_degradation_slope_per_lap",
                "practice_confidence",
                "total_clean_laps",
                "sessions_used",
            ]
        )

    rows = []

    for keys, group in session_estimates.groupby(["race", "driver", "team", "compound"]):
        race, driver, team, compound = keys

        # Confidence-weighted session blend to prioritize cleaner/higher-signal sessions.
        weights = group["confidence"].to_numpy(dtype=float)
        slopes = group["degradation_slope_per_lap"].to_numpy(dtype=float)

        if np.allclose(weights.sum(), 0.0):
            blended_slope = float(np.mean(slopes))
            weighted_confidence = 0.0
        else:
            blended_slope = float(np.average(slopes, weights=weights))
            weighted_confidence = float(np.average(weights, weights=weights))

        total_clean_laps = int(group["clean_lap_count"].sum())

        # Modeling assumption:
        # Across sessions, ~12 total clean laps is enough to treat practice signal as mature.
        lap_coverage = _clip(total_clean_laps / 12.0)
        practice_confidence = _clip((0.70 * weighted_confidence) + (0.30 * lap_coverage))

        rows.append(
            {
                "race": race,
                "driver": driver,
                "team": team,
                "compound": compound,
                "practice_degradation_slope_per_lap": blended_slope,
                "practice_confidence": practice_confidence,
                "total_clean_laps": total_clean_laps,
                "sessions_used": int(group["session"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["race", "driver", "compound"]
    ).reset_index(drop=True)


def _canonicalize_key_columns(df: pd.DataFrame, specs: dict[str, str]) -> pd.DataFrame:
    clean = df.copy()

    for source_col, target_col in specs.items():
        if source_col not in clean.columns:
            raise ValueError(f"Missing required column '{source_col}' in input dataframe.")
        clean[target_col] = clean[source_col]

    clean["race_key"] = clean["race"].astype(str).str.strip().str.lower()
    clean["driver_key"] = clean["driver"].astype(str).str.strip().str.upper()
    clean["team_key"] = clean["team"].astype(str).str.strip().str.lower()
    clean["compound_key"] = clean["compound"].astype(str).str.strip().str.upper()

    return clean


def _estimate_historical_slopes(
    predictions_df: pd.DataFrame,
    prediction_col: str,
    tyre_life_col: str,
) -> pd.DataFrame:
    rows = []

    group_cols = ["race_key", "driver_key", "team_key", "compound_key"]
    for keys, group in predictions_df.groupby(group_cols):
        fit_df = (
            group[[tyre_life_col, prediction_col]]
            .dropna()
            .sort_values(tyre_life_col)
        )

        if len(fit_df) < 2:
            slope = 0.0
        else:
            slope = _fit_slope(
                fit_df.rename(
                    columns={
                        tyre_life_col: "stint_lap",
                        prediction_col: "lap_time_seconds",
                    }
                )
            )["slope"]

        race_key, driver_key, team_key, compound_key = keys
        rows.append(
            {
                "race_key": race_key,
                "driver_key": driver_key,
                "team_key": team_key,
                "compound_key": compound_key,
                "historical_degradation_slope_per_lap": slope,
            }
        )

    return pd.DataFrame(rows)


def blend_practice_with_historical_predictions(
    historical_predictions: pd.DataFrame,
    practice_aggregate: pd.DataFrame,
    prediction_col: str = "predicted_degradation_seconds",
    tyre_life_col: str = "TyreLife",
    output_col: str = "practice_calibrated_prediction",
    max_practice_blend_weight: float = 0.75,
) -> pd.DataFrame:
    """
    Blends historical prediction curves with practice-derived degradation slopes.

    The blend is slope-based to preserve the model's baseline while adjusting
    tire-age growth rate toward current-weekend practice evidence.
    """
    history = _canonicalize_key_columns(
        historical_predictions,
        {
            "race": "race",
            "Driver": "driver",
            "Team": "team",
            "Compound": "compound",
        },
    )

    if prediction_col not in history.columns:
        raise ValueError(
            f"Prediction column '{prediction_col}' not found in historical predictions."
        )

    if tyre_life_col not in history.columns:
        raise ValueError(f"Tyre-life column '{tyre_life_col}' not found in historical predictions.")

    practice = _canonicalize_key_columns(
        practice_aggregate,
        {
            "race": "race",
            "driver": "driver",
            "team": "team",
            "compound": "compound",
        },
    )

    if "practice_degradation_slope_per_lap" not in practice.columns:
        raise ValueError("practice_aggregate must include 'practice_degradation_slope_per_lap'.")

    if "practice_confidence" not in practice.columns:
        raise ValueError("practice_aggregate must include 'practice_confidence'.")

    history = history.copy()
    history[tyre_life_col] = pd.to_numeric(history[tyre_life_col], errors="coerce")
    history[prediction_col] = pd.to_numeric(history[prediction_col], errors="coerce")
    history = history.dropna(subset=[tyre_life_col, prediction_col])

    historical_slopes = _estimate_historical_slopes(
        history,
        prediction_col=prediction_col,
        tyre_life_col=tyre_life_col,
    )

    merged = history.merge(
        historical_slopes,
        on=["race_key", "driver_key", "team_key", "compound_key"],
        how="left",
    )

    merged = merged.merge(
        practice[
            [
                "race_key",
                "driver_key",
                "team_key",
                "compound_key",
                "practice_degradation_slope_per_lap",
                "practice_confidence",
                "total_clean_laps",
                "sessions_used",
            ]
        ],
        on=["race_key", "driver_key", "team_key", "compound_key"],
        how="left",
    )

    merged["practice_confidence"] = merged["practice_confidence"].fillna(0.0)
    merged["practice_weight"] = (
        merged["practice_confidence"].clip(0.0, 1.0) * max_practice_blend_weight
    )

    merged["practice_degradation_slope_per_lap"] = merged[
        "practice_degradation_slope_per_lap"
    ].fillna(merged["historical_degradation_slope_per_lap"])

    merged["blended_degradation_slope_per_lap"] = (
        (1.0 - merged["practice_weight"]) * merged["historical_degradation_slope_per_lap"]
        + merged["practice_weight"] * merged["practice_degradation_slope_per_lap"]
    )

    merged["group_tyre_life_anchor"] = merged.groupby(
        ["race_key", "driver_key", "team_key", "compound_key"]
    )[tyre_life_col].transform("min")

    slope_delta = (
        merged["blended_degradation_slope_per_lap"]
        - merged["historical_degradation_slope_per_lap"]
    )

    merged[output_col] = (
        merged[prediction_col]
        + slope_delta * (merged[tyre_life_col] - merged["group_tyre_life_anchor"])
    )

    return merged


def calibrate_predictions_from_practice_csvs(
    practice_csv_paths: list[str | Path],
    historical_predictions: pd.DataFrame,
    prediction_col: str = "predicted_degradation_seconds",
    tyre_life_col: str = "TyreLife",
    output_col: str = "practice_calibrated_prediction",
    min_clean_laps_per_session: int = 3,
    allow_drs_laps: bool = False,
    max_practice_blend_weight: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end helper:
    1) load practice CSVs
    2) filter clean long-run laps
    3) estimate session slopes/confidence
    4) aggregate practice signal
    5) blend with historical predictions
    """
    practice_raw = load_practice_csvs(practice_csv_paths)
    practice_clean = filter_clean_practice_laps(
        practice_raw,
        allow_drs_laps=allow_drs_laps,
    )

    session_estimates = estimate_session_degradation(
        practice_clean,
        min_clean_laps=min_clean_laps_per_session,
    )
    practice_aggregate = aggregate_practice_degradation(session_estimates)

    calibrated_predictions = blend_practice_with_historical_predictions(
        historical_predictions=historical_predictions,
        practice_aggregate=practice_aggregate,
        prediction_col=prediction_col,
        tyre_life_col=tyre_life_col,
        output_col=output_col,
        max_practice_blend_weight=max_practice_blend_weight,
    )

    return calibrated_predictions, session_estimates, practice_aggregate


def run_sample_practice_calibration() -> None:
    practice_sample_path = Path("data/processed/practice/practice_long_run_sample.csv")
    historical_curve_path = Path("data/predictions/degradation_curves/combined_degradation_curves.csv")

    if not practice_sample_path.exists() or not historical_curve_path.exists():
        print("Sample run skipped. Required files:")
        print(f"- {practice_sample_path}")
        print(f"- {historical_curve_path}")
        return

    historical_df = pd.read_csv(historical_curve_path)

    calibrated_df, session_df, aggregate_df = calibrate_predictions_from_practice_csvs(
        practice_csv_paths=[practice_sample_path],
        historical_predictions=historical_df,
    )

    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrated_path = output_dir / "practice_calibrated_curves.csv"
    session_path = output_dir / "practice_session_estimates.csv"
    aggregate_path = output_dir / "practice_aggregate_estimates.csv"

    calibrated_df.to_csv(calibrated_path, index=False)
    session_df.to_csv(session_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("Practice calibration complete")
    print(f"- Calibrated curves: {calibrated_path}")
    print(f"- Session estimates: {session_path}")
    print(f"- Aggregate estimates: {aggregate_path}")


if __name__ == "__main__":
    run_sample_practice_calibration()
