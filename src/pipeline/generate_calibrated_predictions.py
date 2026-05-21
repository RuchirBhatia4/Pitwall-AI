from argparse import ArgumentParser
from pathlib import Path
import warnings

import joblib
import pandas as pd

from src.data.practice_loader import (
    filter_clean_practice_laps,
    load_practice_csv,
)
from src.models.practice_calibration import (
    aggregate_practice_degradation,
    blend_practice_with_historical_predictions,
    estimate_session_degradation,
)
from src.models.train_degradation_best import FEATURE_COLUMNS


DEFAULT_PRACTICE_CSV_PATH = Path("data/raw/practice_uploads/practice_long_run_sample.csv")
DEFAULT_HISTORICAL_CURVES_PATH = Path(
    "data/predictions/degradation_curves/combined_degradation_curves.csv"
)
DEFAULT_BEST_MODEL_PATH = Path("data/models/best_degradation_model.pkl")
DEFAULT_OUTPUT_PATH = Path("data/predictions/calibrated_degradation_predictions.csv")

DEFAULT_PREDICTION_COL = "predicted_degradation_seconds"
DEFAULT_TYRE_LIFE_COL = "TyreLife"
DEFAULT_CALIBRATION_METHOD = "practice_confidence_weighted_slope_blend_v1"
DEFAULT_FALLBACK_METHOD = "historical_only_no_practice_data"
DEFAULT_ASSUMED_SEASON = 2026
DEFAULT_ASSUMED_RACE_LAPS = 57
DEFAULT_MAX_TYRE_LIFE = 25


def _load_historical_predictions(path: str | Path) -> pd.DataFrame:
    history_path = Path(path)
    if not history_path.exists():
        raise FileNotFoundError(
            f"Historical prediction file not found: {history_path}. "
            "Generate degradation curves first with "
            "`python -m src.models.predict_degradation_curves`."
        )

    df = pd.read_csv(history_path)

    required_cols = [
        "race",
        "Driver",
        "Team",
        "Compound",
        DEFAULT_TYRE_LIFE_COL,
        DEFAULT_PREDICTION_COL,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Historical prediction file is missing required columns: {missing}"
        )

    return df


def _build_model_feature_grid(
    practice_df: pd.DataFrame,
    assumed_season: int = DEFAULT_ASSUMED_SEASON,
    race_laps: int = DEFAULT_ASSUMED_RACE_LAPS,
    max_tyre_life: int = DEFAULT_MAX_TYRE_LIFE,
) -> pd.DataFrame:
    unique_keys = (
        practice_df[["race", "driver", "team", "compound"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows = []
    for _, item in unique_keys.iterrows():
        race = str(item["race"])
        driver = str(item["driver"]).upper()
        team = str(item["team"])
        compound = str(item["compound"]).upper()

        for tyre_life in range(1, max_tyre_life + 1):
            lap_number = tyre_life

            # Modeling assumption:
            # Practice upload rows do not contain season/race-distance metadata
            # required by the historical model, so we default to 2026 and a
            # representative 57-lap race distance for feature compatibility.
            race_phase = min(1.0, lap_number / race_laps)
            stint_progress = tyre_life / max_tyre_life

            rows.append(
                {
                    "season": assumed_season,
                    "race": race,
                    "Driver": driver,
                    "Team": team,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "Stint": 1,
                    "LapNumber": lap_number,
                    "race_phase": race_phase,
                    "stint_progress": stint_progress,
                    "tyre_life_squared": tyre_life ** 2,
                    "lap_number_squared": lap_number ** 2,
                    "is_first_stint": 1,
                    "is_second_stint": 0,
                    "is_late_race": int(race_phase >= 0.66),
                    "is_early_race": int(race_phase <= 0.33),
                    "soft_tyre_life": tyre_life if compound == "SOFT" else 0,
                    "medium_tyre_life": tyre_life if compound == "MEDIUM" else 0,
                    "hard_tyre_life": tyre_life if compound == "HARD" else 0,
                }
            )

    return pd.DataFrame(rows)


def _load_or_build_historical_predictions(
    practice_raw: pd.DataFrame,
    best_model_path: str | Path,
    historical_predictions_path: str | Path,
) -> pd.DataFrame:
    best_model_file = Path(best_model_path)
    if not best_model_file.exists():
        warnings.warn(
            f"Best degradation model not found at {best_model_file}. "
            "Falling back to historical curve CSV estimates.",
            UserWarning,
        )
        return _load_historical_predictions(historical_predictions_path)

    try:
        model = joblib.load(best_model_file)

        feature_grid = _build_model_feature_grid(practice_raw)
        if feature_grid.empty:
            raise ValueError("No valid race/driver/team/compound rows found in practice CSV.")

        for col in FEATURE_COLUMNS:
            if col not in feature_grid.columns:
                raise ValueError(
                    f"Generated feature grid is missing required model feature column '{col}'."
                )

        preds = model.predict(feature_grid[FEATURE_COLUMNS])

        historical_predictions = feature_grid[
            ["race", "Driver", "Team", "Compound", "TyreLife"]
        ].copy()
        historical_predictions[DEFAULT_PREDICTION_COL] = preds

        return historical_predictions
    except Exception as exc:  # pragma: no cover - defensive fallback path.
        warnings.warn(
            "Failed to use best degradation model for historical estimates; "
            f"falling back to historical curve CSV. Error: {exc}",
            UserWarning,
        )
        return _load_historical_predictions(historical_predictions_path)


def _build_report_table(
    blended_rows: pd.DataFrame,
    practice_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = ["race", "driver", "team", "compound"]

    # Modeling assumption:
    # Group-level degradation estimate is represented by the fitted slope-per-lap
    # signal, which is what the calibration blend modifies.
    report = (
        blended_rows.groupby(group_cols, as_index=False)
        .agg(
            historical_deg_estimate=("historical_degradation_slope_per_lap", "first"),
            calibrated_deg_estimate=("blended_degradation_slope_per_lap", "first"),
        )
    )

    practice_info = practice_aggregate.rename(
        columns={
            "practice_degradation_slope_per_lap": "practice_deg_estimate",
            "total_clean_laps": "clean_laps_used",
        }
    )[
        [
            "race",
            "driver",
            "team",
            "compound",
            "practice_deg_estimate",
            "practice_confidence",
            "clean_laps_used",
            "sessions_used",
        ]
    ]

    report = report.merge(
        practice_info,
        on=group_cols,
        how="left",
    )

    report["practice_confidence"] = report["practice_confidence"].fillna(0.0)
    report["clean_laps_used"] = report["clean_laps_used"].fillna(0).astype(int)
    report["sessions_used"] = report["sessions_used"].fillna(0).astype(int)
    report["calibration_method"] = report["clean_laps_used"].apply(
        lambda laps: DEFAULT_CALIBRATION_METHOD if laps > 0 else DEFAULT_FALLBACK_METHOD
    )

    output_cols = [
        "race",
        "driver",
        "team",
        "compound",
        "historical_deg_estimate",
        "practice_deg_estimate",
        "practice_confidence",
        "calibrated_deg_estimate",
        "clean_laps_used",
        "sessions_used",
        "calibration_method",
    ]

    return report[output_cols].sort_values(
        ["race", "driver", "compound"]
    ).reset_index(drop=True)


def generate_calibrated_prediction_report(
    practice_csv_path: str | Path = DEFAULT_PRACTICE_CSV_PATH,
    historical_predictions_path: str | Path = DEFAULT_HISTORICAL_CURVES_PATH,
    best_model_path: str | Path = DEFAULT_BEST_MODEL_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Generates a practice-calibrated degradation report and saves it to CSV.
    """
    practice_path = Path(practice_csv_path)
    if not practice_path.exists():
        raise FileNotFoundError(
            f"Practice CSV not found: {practice_path}. "
            "Upload a practice file to data/raw/practice_uploads/."
        )

    practice_raw = load_practice_csv(practice_path)
    practice_clean = filter_clean_practice_laps(practice_raw)
    historical_predictions = _load_or_build_historical_predictions(
        practice_raw=practice_raw,
        best_model_path=best_model_path,
        historical_predictions_path=historical_predictions_path,
    )

    session_estimates = estimate_session_degradation(practice_clean)
    practice_aggregate = aggregate_practice_degradation(session_estimates)

    blended_rows = blend_practice_with_historical_predictions(
        historical_predictions=historical_predictions,
        practice_aggregate=practice_aggregate,
        prediction_col=DEFAULT_PREDICTION_COL,
        tyre_life_col=DEFAULT_TYRE_LIFE_COL,
    )

    report = _build_report_table(
        blended_rows=blended_rows,
        practice_aggregate=practice_aggregate,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)

    return report


def summarize_calibration_report(
    report: pd.DataFrame,
    practice_df: pd.DataFrame | None = None,
) -> dict[str, str | float]:
    """
    Builds high-level calibration summary metrics for CLI reporting.
    """
    if report.empty:
        return {
            "drivers_covered": "N/A",
            "compounds_covered": "N/A",
            "sessions_covered": "N/A",
            "average_practice_confidence": 0.0,
            "highest_confidence_row": "N/A",
            "lowest_confidence_row": "N/A",
        }

    drivers = sorted(report["driver"].dropna().astype(str).unique().tolist())
    compounds = sorted(report["compound"].dropna().astype(str).unique().tolist())

    if practice_df is not None and "session" in practice_df.columns:
        sessions = sorted(practice_df["session"].dropna().astype(str).unique().tolist())
    else:
        sessions = []

    confidence_view = report[
        ["race", "driver", "team", "compound", "practice_confidence"]
    ].copy()

    highest = confidence_view.sort_values(
        ["practice_confidence", "race", "driver", "compound"],
        ascending=[False, True, True, True],
    ).iloc[0]
    lowest = confidence_view.sort_values(
        ["practice_confidence", "race", "driver", "compound"],
        ascending=[True, True, True, True],
    ).iloc[0]

    return {
        "drivers_covered": ", ".join(drivers) if drivers else "N/A",
        "compounds_covered": ", ".join(compounds) if compounds else "N/A",
        "sessions_covered": ", ".join(sessions) if sessions else "N/A",
        "average_practice_confidence": float(report["practice_confidence"].mean()),
        "highest_confidence_row": (
            f"{highest['race']} | {highest['driver']} | {highest['team']} | "
            f"{highest['compound']} | conf={float(highest['practice_confidence']):.3f}"
        ),
        "lowest_confidence_row": (
            f"{lowest['race']} | {lowest['driver']} | {lowest['team']} | "
            f"{lowest['compound']} | conf={float(lowest['practice_confidence']):.3f}"
        ),
    }


def main() -> None:
    parser = ArgumentParser(description="Generate calibrated degradation predictions from practice data.")
    parser.add_argument(
        "--practice-csv",
        default=str(DEFAULT_PRACTICE_CSV_PATH),
        help="Path to uploaded practice race-pace CSV.",
    )
    parser.add_argument(
        "--historical-predictions",
        default=str(DEFAULT_HISTORICAL_CURVES_PATH),
        help="Path to historical degradation prediction curves CSV.",
    )
    parser.add_argument(
        "--best-model",
        default=str(DEFAULT_BEST_MODEL_PATH),
        help=(
            "Path to best degradation model (.pkl). "
            "If missing, pipeline falls back to historical prediction curves CSV."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for calibrated degradation prediction report CSV.",
    )
    args = parser.parse_args()

    report = generate_calibrated_prediction_report(
        practice_csv_path=args.practice_csv,
        historical_predictions_path=args.historical_predictions,
        best_model_path=args.best_model,
        output_path=args.output,
    )

    print("Calibrated degradation prediction report saved.")
    print(f"- Rows: {len(report)}")
    print(f"- Output: {Path(args.output)}")
    practice_summary_df = load_practice_csv(args.practice_csv)
    summary = summarize_calibration_report(report, practice_df=practice_summary_df)
    print("Calibration summary:")
    print(f"- Drivers covered: {summary['drivers_covered']}")
    print(f"- Compounds covered: {summary['compounds_covered']}")
    print(f"- Sessions covered: {summary['sessions_covered']}")
    print(
        "- Average practice confidence: "
        f"{float(summary['average_practice_confidence']):.3f}"
    )
    print(
        "- Highest-confidence row: "
        f"{summary['highest_confidence_row']}"
    )
    print(
        "- Lowest-confidence row: "
        f"{summary['lowest_confidence_row']}"
    )
    print(report.head(10))


if __name__ == "__main__":
    main()
