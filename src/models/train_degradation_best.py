from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


DATA_PATH = Path("data/processed/historical_laps_clean.csv")
MODEL_OUTPUT_PATH = Path("data/models/best_degradation_model.pkl")
METRICS_OUTPUT_PATH = Path("data/predictions/best_degradation_model_metrics.csv")

EXPERIMENT_NAME = "pitwall_best_degradation_model"

TARGET = "fuel_adjusted_degradation_delta"
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

FEATURE_COLUMNS = [
    "season",
    "race",
    "Driver",
    "Team",
    "Compound",
    "TyreLife",
    "Stint",
    "LapNumber",
    "race_phase",
    "stint_progress",
    "tyre_life_squared",
    "lap_number_squared",
    "is_first_stint",
    "is_second_stint",
    "is_late_race",
    "is_early_race",
    "soft_tyre_life",
    "medium_tyre_life",
    "hard_tyre_life",
]

NUMERIC_COLUMNS = [
    "season",
    "TyreLife",
    "Stint",
    "LapNumber",
    "race_phase",
    "stint_progress",
    "tyre_life_squared",
    "lap_number_squared",
    "is_first_stint",
    "is_second_stint",
    "is_late_race",
    "is_early_race",
    "soft_tyre_life",
    "medium_tyre_life",
    "hard_tyre_life",
]

CATEGORICAL_COLUMNS = ["race", "Driver", "Team", "Compound"]

BEST_METRIC_COLUMNS = [
    "model_type",
    "params_json",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "train_rows",
    "validation_rows",
    "split_type",
    "validation_season",
    "validation_race",
    "dry_compounds_only",
    "target",
    "is_best_model",
]


def assign_regulation_weight(season: int) -> float:
    if season == 2026:
        return 5.0
    if season == 2025:
        return 1.5
    return 1.0


def get_default_xgb_search_space() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    ]


def get_default_lgbm_search_space() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "num_leaves": 31,
        },
        {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "num_leaves": 31,
        },
        {
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "num_leaves": 63,
        },
    ]


def filter_dry_compounds(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["Compound"] = clean["Compound"].astype(str).str.upper()
    return clean[clean["Compound"].isin(DRY_COMPOUNDS)].copy()


def load_training_data(
    path: str | Path = DATA_PATH,
    dry_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = FEATURE_COLUMNS + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    df = df.dropna(subset=required_cols).copy()

    for col in NUMERIC_COLUMNS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_COLUMNS + [TARGET]).copy()

    if dry_only:
        df = filter_dry_compounds(df)

    if df.empty:
        raise ValueError("No rows available after filtering and numeric conversion.")

    df["season"] = df["season"].astype(int)
    df["race"] = df["race"].astype(str)
    df["Driver"] = df["Driver"].astype(str)
    df["Team"] = df["Team"].astype(str)
    df["Compound"] = df["Compound"].astype(str).str.upper()

    return df.reset_index(drop=True)


def choose_validation_race(
    df: pd.DataFrame,
    validation_season: int = 2026,
    excluded_race: str | None = None,
) -> str | None:
    races = sorted(
        df[df["season"] == validation_season]["race"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if excluded_race is not None:
        races = [race for race in races if race.lower() != excluded_race.lower()]

    if not races:
        return None

    # Modeling assumption:
    # Use the latest lexicographic available 2026 race as the default validation
    # race for tuning to prioritize near-future generalization over random split.
    return races[-1]


def split_train_validation(
    df: pd.DataFrame,
    validation_season: int = 2026,
    validation_race: str | None = None,
    excluded_race: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    chosen_race = validation_race or choose_validation_race(
        df=df,
        validation_season=validation_season,
        excluded_race=excluded_race,
    )

    if chosen_race is not None:
        val_mask = (
            (df["season"] == validation_season)
            & (df["race"].str.lower() == chosen_race.lower())
        )

        if val_mask.any() and (~val_mask).any():
            train_df = df[~val_mask].copy()
            val_df = df[val_mask].copy()
            return train_df, val_df, {
                "split_type": "future_race_validation",
                "validation_season": validation_season,
                "validation_race": chosen_race,
            }

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
    )
    return train_df.copy(), val_df.copy(), {
        "split_type": "random_validation_fallback",
        "validation_season": None,
        "validation_race": None,
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERIC_COLUMNS,
            ),
        ]
    )


def build_model_pipeline(
    model_type: str,
    params: dict[str, Any],
    random_state: int = 42,
) -> Pipeline:
    if model_type == "XGBoost":
        model_params: dict[str, Any] = {
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
        }
        model_params.update(params)
        regressor = XGBRegressor(**model_params)
    elif model_type == "LightGBM":
        model_params = {
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        model_params.update(params)
        regressor = LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def evaluate_predictions(y_true: pd.Series, preds: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, preds)
    return {
        "mae": float(mean_absolute_error(y_true, preds)),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_true, preds)),
    }


def run_hyperparameter_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    xgb_search_space: list[dict[str, Any]] | None = None,
    lgbm_search_space: list[dict[str, Any]] | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    run_name_prefix: str = "best_degradation",
    log_to_mlflow: bool = True,
    random_state: int = 42,
    split_info: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    xgb_grid = xgb_search_space or get_default_xgb_search_space()
    lgbm_grid = lgbm_search_space or get_default_lgbm_search_space()

    split_info = split_info or {}

    if log_to_mlflow:
        mlflow.set_experiment(experiment_name)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET]
    weights_train = train_df["season"].apply(assign_regulation_weight)

    candidates: list[tuple[str, dict[str, Any]]] = []
    candidates.extend([("XGBoost", params) for params in xgb_grid])
    candidates.extend([("LightGBM", params) for params in lgbm_grid])

    results = []
    for idx, (model_type, params) in enumerate(candidates, start=1):
        pipeline = build_model_pipeline(
            model_type=model_type,
            params=params,
            random_state=random_state,
        )
        pipeline.fit(
            X_train,
            y_train,
            regressor__sample_weight=weights_train,
        )

        preds = pipeline.predict(X_val)
        metrics = evaluate_predictions(y_val, preds)

        result = {
            "model_type": model_type,
            "params_json": json.dumps(params, sort_keys=True),
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_r2": metrics["r2"],
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "split_type": split_info.get("split_type"),
            "validation_season": split_info.get("validation_season"),
            "validation_race": split_info.get("validation_race"),
            "dry_compounds_only": True,
            "target": TARGET,
            "is_best_model": False,
        }
        results.append(result)

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{run_name_prefix}_{model_type}_trial_{idx}"):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("params_json", result["params_json"])
                mlflow.log_param("target", TARGET)
                mlflow.log_param("dry_compounds_only", True)
                mlflow.log_param("weight_2026", 5.0)
                mlflow.log_param("weight_2025", 1.5)
                mlflow.log_param("weight_2024_or_older", 1.0)
                mlflow.log_param("split_type", result["split_type"])
                mlflow.log_param("validation_season", result["validation_season"])
                mlflow.log_param("validation_race", result["validation_race"])
                mlflow.log_metric("validation_mae_seconds", metrics["mae"])
                mlflow.log_metric("validation_rmse_seconds", metrics["rmse"])
                mlflow.log_metric("validation_r2", metrics["r2"])

    results_df = pd.DataFrame(results).sort_values("validation_mae").reset_index(drop=True)
    if results_df.empty:
        raise RuntimeError("Hyperparameter search produced no results.")

    results_df.loc[0, "is_best_model"] = True
    best_row = results_df.iloc[0].to_dict()
    best_params = json.loads(best_row["params_json"])

    best_config = {
        "model_type": best_row["model_type"],
        "params": best_params,
        "validation_mae": float(best_row["validation_mae"]),
        "validation_rmse": float(best_row["validation_rmse"]),
        "validation_r2": float(best_row["validation_r2"]),
    }
    return results_df, best_config


def train_best_model(
    data_path: str | Path = DATA_PATH,
    model_output_path: str | Path = MODEL_OUTPUT_PATH,
    metrics_output_path: str | Path = METRICS_OUTPUT_PATH,
    experiment_name: str = EXPERIMENT_NAME,
    xgb_search_space: list[dict[str, Any]] | None = None,
    lgbm_search_space: list[dict[str, Any]] | None = None,
    log_to_mlflow: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_training_data(path=data_path, dry_only=True)

    train_df, val_df, split_info = split_train_validation(
        df=df,
        validation_season=2026,
        random_state=random_state,
    )

    search_results_df, best_config = run_hyperparameter_search(
        train_df=train_df,
        val_df=val_df,
        xgb_search_space=xgb_search_space,
        lgbm_search_space=lgbm_search_space,
        experiment_name=experiment_name,
        run_name_prefix="best_degradation",
        log_to_mlflow=log_to_mlflow,
        random_state=random_state,
        split_info=split_info,
    )

    best_pipeline = build_model_pipeline(
        model_type=best_config["model_type"],
        params=best_config["params"],
        random_state=random_state,
    )

    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET]
    weights_full = df["season"].apply(assign_regulation_weight)
    best_pipeline.fit(
        X_full,
        y_full,
        regressor__sample_weight=weights_full,
    )

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    search_results_df[BEST_METRIC_COLUMNS].to_csv(metrics_path, index=False)

    if log_to_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="best_degradation_final_refit"):
            mlflow.log_param("model_type", best_config["model_type"])
            mlflow.log_param("params_json", json.dumps(best_config["params"], sort_keys=True))
            mlflow.log_param("target", TARGET)
            mlflow.log_param("dry_compounds_only", True)
            mlflow.log_param("train_rows_full_refit", int(len(df)))
            mlflow.log_param("weight_2026", 5.0)
            mlflow.log_param("weight_2025", 1.5)
            mlflow.log_param("weight_2024_or_older", 1.0)
            mlflow.log_metric("selection_validation_mae_seconds", best_config["validation_mae"])
            mlflow.log_metric("selection_validation_rmse_seconds", best_config["validation_rmse"])
            mlflow.log_metric("selection_validation_r2", best_config["validation_r2"])
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

    print("\nBest model training complete")
    print(f"Model type: {best_config['model_type']}")
    print(f"Validation MAE: {best_config['validation_mae']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    return {
        "best_model_type": best_config["model_type"],
        "best_params": best_config["params"],
        "validation_mae": best_config["validation_mae"],
        "validation_rmse": best_config["validation_rmse"],
        "validation_r2": best_config["validation_r2"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    train_best_model()
