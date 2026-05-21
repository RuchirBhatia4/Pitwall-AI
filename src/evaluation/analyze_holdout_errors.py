from pathlib import Path
import pandas as pd


PREDICTIONS_PATH = Path("data/predictions/holdout_2026_japan_predictions.csv")
OUTPUT_DIR = Path("reports/post_race_evaluations")

TARGET_COL = "actual_fuel_adjusted_degradation_delta"


def summarize_errors(df: pd.DataFrame, error_col: str, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols)
        .agg(
            rows=(error_col, "count"),
            mae=(error_col, "mean"),
            median_error=(error_col, "median"),
            max_error=(error_col, "max"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    return summary


def get_best_model_error_column(df: pd.DataFrame) -> tuple[str, str]:
    candidate_error_cols = [
        "ridge_error",
        "xgboost_error",
        "lightgbm_error",
    ]

    available_error_cols = [col for col in candidate_error_cols if col in df.columns]

    if not available_error_cols:
        raise ValueError("No model error columns found in prediction file.")

    mean_errors = {
        col: df[col].mean()
        for col in available_error_cols
    }

    best_error_col = min(mean_errors, key=mean_errors.get)
    pred_col = best_error_col.replace("_error", "_prediction")

    return best_error_col, pred_col


def analyze_single_model(df: pd.DataFrame, error_col: str, pred_col: str) -> None:
    print("\n" + "=" * 80)
    print(f"Error analysis for: {error_col}")
    print("=" * 80)

    print("\nHoldout prediction rows:")
    print(len(df))

    print(f"\nOverall {error_col} summary:")
    print(df[error_col].describe())

    by_driver = summarize_errors(df, error_col, ["Driver"])
    by_team = summarize_errors(df, error_col, ["Team"])
    by_compound = summarize_errors(df, error_col, ["Compound"])
    by_stint = summarize_errors(df, error_col, ["Stint"])
    by_driver_compound = summarize_errors(df, error_col, ["Driver", "Compound"])

    by_driver.to_csv(OUTPUT_DIR / f"holdout_japan_{error_col}_by_driver.csv", index=False)
    by_team.to_csv(OUTPUT_DIR / f"holdout_japan_{error_col}_by_team.csv", index=False)
    by_compound.to_csv(OUTPUT_DIR / f"holdout_japan_{error_col}_by_compound.csv", index=False)
    by_stint.to_csv(OUTPUT_DIR / f"holdout_japan_{error_col}_by_stint.csv", index=False)
    by_driver_compound.to_csv(
        OUTPUT_DIR / f"holdout_japan_{error_col}_by_driver_compound.csv",
        index=False,
    )

    print("\nWorst drivers by MAE:")
    print(by_driver.head(10))

    print("\nWorst teams by MAE:")
    print(by_team.head(10))

    print("\nError by compound:")
    print(by_compound)

    print("\nError by stint:")
    print(by_stint)

    print("\nWorst driver-compound pairs:")
    print(by_driver_compound.head(15))

    biggest_misses = df.sort_values(error_col, ascending=False).head(25)
    biggest_misses.to_csv(
        OUTPUT_DIR / f"holdout_japan_{error_col}_biggest_misses.csv",
        index=False,
    )

    print("\nBiggest individual misses:")
    print(
        biggest_misses[
            [
                "Driver",
                "Team",
                "Compound",
                "Stint",
                "TyreLife",
                "LapNumber",
                "race_phase",
                "stint_progress",
                TARGET_COL,
                pred_col,
                error_col,
            ]
        ]
    )


def analyze_holdout_errors():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_PATH)

    base_required_cols = [
        "Driver",
        "Team",
        "Compound",
        "Stint",
        "TyreLife",
        "LapNumber",
        "race_phase",
        "stint_progress",
        TARGET_COL,
    ]

    missing_base = [col for col in base_required_cols if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required columns in holdout prediction file: {missing_base}")

    model_pairs = [
        ("ridge_prediction", "ridge_error"),
        ("xgboost_prediction", "xgboost_error"),
        ("lightgbm_prediction", "lightgbm_error"),
    ]

    available_pairs = [
        (pred_col, error_col)
        for pred_col, error_col in model_pairs
        if pred_col in df.columns and error_col in df.columns
    ]

    if not available_pairs:
        raise ValueError("No model prediction/error columns found in holdout prediction file.")

    print("\nAvailable model error columns:")
    for pred_col, error_col in available_pairs:
        print(f"- {error_col}: mean MAE = {df[error_col].mean():.4f}")

    best_error_col, best_pred_col = get_best_model_error_column(df)

    print("\nBest model according to prediction file:")
    print(f"- Error column: {best_error_col}")
    print(f"- Prediction column: {best_pred_col}")
    print(f"- MAE: {df[best_error_col].mean():.4f}")

    analyze_single_model(df, best_error_col, best_pred_col)

    print("\nSaved reports to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    analyze_holdout_errors()