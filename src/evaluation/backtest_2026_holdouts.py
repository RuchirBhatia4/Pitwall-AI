from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_degradation_best import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET,
    assign_regulation_weight,
    build_model_pipeline,
    get_default_lgbm_search_space,
    get_default_xgb_search_space,
    load_training_data,
    run_hyperparameter_search,
    split_train_validation,
)


HOLDOUT_SEASON = 2026
BACKTEST_OUTPUT_CSV = Path("reports/model_cards/2026_holdout_backtest.csv")
BACKTEST_OUTPUT_MD = Path("reports/model_cards/2026_holdout_backtest.md")

BACKTEST_REQUIRED_COLUMNS = [
    "heldout_race",
    "best_model_type",
    "rows",
    "mae",
    "rmse",
    "r2",
]


def evaluate_one_holdout_race(
    df: pd.DataFrame,
    heldout_race: str,
    xgb_search_space: list[dict[str, Any]],
    lgbm_search_space: list[dict[str, Any]],
    random_state: int = 42,
) -> dict[str, Any]:
    holdout_mask = (
        (df["season"] == HOLDOUT_SEASON)
        & (df["race"].str.lower() == heldout_race.lower())
    )
    train_df = df[~holdout_mask].copy()
    test_df = df[holdout_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid train/test split for holdout race {heldout_race}. "
            f"Train rows: {len(train_df)}, test rows: {len(test_df)}"
        )

    hp_train_df, hp_val_df, split_info = split_train_validation(
        df=train_df,
        validation_season=HOLDOUT_SEASON,
        excluded_race=heldout_race,
        random_state=random_state,
    )

    _, best_config = run_hyperparameter_search(
        train_df=hp_train_df,
        val_df=hp_val_df,
        xgb_search_space=xgb_search_space,
        lgbm_search_space=lgbm_search_space,
        log_to_mlflow=False,
        random_state=random_state,
        split_info=split_info,
    )

    best_pipeline = build_model_pipeline(
        model_type=best_config["model_type"],
        params=best_config["params"],
        random_state=random_state,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    weights_train = train_df["season"].apply(assign_regulation_weight)

    best_pipeline.fit(
        X_train,
        y_train,
        regressor__sample_weight=weights_train,
    )

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]
    preds = best_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    return {
        "heldout_race": heldout_race,
        "best_model_type": best_config["model_type"],
        "rows": int(len(test_df)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_test, preds)),
    }


def build_backtest_markdown(results_df: pd.DataFrame) -> str:
    lines = [
        "# 2026 Future-Race Holdout Backtest",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "This report shows **historical-only pre-practice performance**.",
        "No FP1/FP2/FP3 race-pace calibration is included in these backtest scores.",
        "",
        "## Per-Race Results",
        "",
        "| heldout_race | best_model_type | rows | mae | rmse | r2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['heldout_race']} | {row['best_model_type']} | {int(row['rows'])} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- races_evaluated: {len(results_df)}",
            f"- mean_mae: {results_df['mae'].mean():.4f}",
            f"- mean_rmse: {results_df['rmse'].mean():.4f}",
            f"- mean_r2: {results_df['r2'].mean():.4f}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_2026_holdout_backtest(
    data_path: str | Path = DATA_PATH,
    output_csv_path: str | Path = BACKTEST_OUTPUT_CSV,
    output_md_path: str | Path = BACKTEST_OUTPUT_MD,
    xgb_search_space: list[dict[str, Any]] | None = None,
    lgbm_search_space: list[dict[str, Any]] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = load_training_data(path=data_path, dry_only=True)

    races_2026 = sorted(
        df[df["season"] == HOLDOUT_SEASON]["race"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not races_2026:
        raise ValueError(f"No {HOLDOUT_SEASON} races found in dataset for backtest.")

    xgb_grid = xgb_search_space or get_default_xgb_search_space()
    lgbm_grid = lgbm_search_space or get_default_lgbm_search_space()

    rows = []
    for race in races_2026:
        result = evaluate_one_holdout_race(
            df=df,
            heldout_race=race,
            xgb_search_space=xgb_grid,
            lgbm_search_space=lgbm_grid,
            random_state=random_state,
        )
        rows.append(result)

    results_df = pd.DataFrame(rows)[BACKTEST_REQUIRED_COLUMNS].sort_values(
        "heldout_race"
    ).reset_index(drop=True)

    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    md_path = Path(output_md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_backtest_markdown(results_df), encoding="utf-8")

    print("\n2026 holdout backtest complete")
    print(f"Races evaluated: {len(results_df)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")

    return results_df


if __name__ == "__main__":
    run_2026_holdout_backtest()
