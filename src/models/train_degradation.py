from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.features.degradation_priors import (
    add_degradation_priors,
    get_prior_feature_columns,
)


DATA_PATH = "data/processed/historical_laps_clean.csv"
MODEL_DIR = Path("data/models")
PREDICTION_DIR = Path("data/predictions")

EXPERIMENT_NAME = "pitwall_tire_degradation"

TARGET = "fuel_adjusted_degradation_delta"


def assign_regulation_weight(season: int) -> float:
    """
    Assigns higher training weight to 2026 data because 2026 is a new regulation era.

    2026: primary truth
    2025: recent historical prior
    2024 or older: weaker historical prior
    """
    if season == 2026:
        return 5.0
    elif season == 2025:
        return 1.5
    else:
        return 1.0


def get_base_feature_columns() -> list[str]:
    return [
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


def get_feature_columns() -> list[str]:
    return get_base_feature_columns() + get_prior_feature_columns()


def get_numeric_columns() -> list[str]:
    return [
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
    ] + get_prior_feature_columns()


def get_categorical_columns() -> list[str]:
    return ["race", "Driver", "Team", "Compound"]


def load_training_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = get_base_feature_columns() + [TARGET]

    df = df.dropna(subset=required_cols).copy()

    numeric_cols = [
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
        TARGET,
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["season"] = df["season"].astype(int)
    df["race"] = df["race"].astype(str)
    df["Driver"] = df["Driver"].astype(str)
    df["Team"] = df["Team"].astype(str)
    df["Compound"] = df["Compound"].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_features = get_categorical_columns()
    numerical_features = get_numeric_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    return preprocessor


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"\n{name} Results")
    print("-" * 40)
    print(f"MAE:  {mae:.4f} seconds")
    print(f"RMSE: {rmse:.4f} seconds")
    print(f"R²:   {r2:.4f}")

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": preds,
    }


def log_model_to_mlflow(
    model_name: str,
    model: Pipeline,
    results: dict,
    features: list,
    target: str,
    train_rows: int,
    test_rows: int,
    data_path: str,
    weighted_training: bool = True,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param("data_path", data_path)

        mlflow.log_param("weighted_training", weighted_training)
        mlflow.log_param("weight_2026", 5.0)
        mlflow.log_param("weight_2025", 1.5)
        mlflow.log_param("weight_2024_or_older", 1.0)
        mlflow.log_param("fuel_adjusted_target", True)
        mlflow.log_param("race_phase_features", True)
        mlflow.log_param("nonlinear_tire_features", True)
        mlflow.log_param("degradation_prior_features", True)

        mlflow.log_metric("mae_seconds", results["mae"])
        mlflow.log_metric("rmse_seconds", results["rmse"])
        mlflow.log_metric("r2", results["r2"])

        mlflow.sklearn.log_model(model, artifact_path="model")


def train_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data()

    features = get_feature_columns()
    target = TARGET

    sample_weights = df["season"].apply(assign_regulation_weight)

    train_df, test_df, weights_train, weights_test = train_test_split(
        df,
        sample_weights,
        test_size=0.2,
        random_state=42,
    )

    train_df = add_degradation_priors(
        train_df=train_df,
        target_df=train_df,
        target_col=target,
    )

    test_df = add_degradation_priors(
        train_df=train_df,
        target_df=test_df,
        target_col=target,
    )

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("\nDataset summary:")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("\nRows by season:")
    print(df["season"].value_counts().sort_index())

    print("\nSample weight distribution in training set:")
    print(weights_train.value_counts().sort_index())

    print("\nPrior feature columns:")
    print(get_prior_feature_columns())

    ridge_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            ),
        ]
    )

    lgbm_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                LGBMRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )

    print("\nTraining Ridge baseline with regulation-aware weights...")
    ridge_model.fit(
        X_train,
        y_train,
        regressor__sample_weight=weights_train,
    )

    print("Training XGBoost model with regulation-aware weights...")
    xgb_model.fit(
        X_train,
        y_train,
        regressor__sample_weight=weights_train,
    )

    print("Training LightGBM model with regulation-aware weights...")
    lgbm_model.fit(
        X_train,
        y_train,
        regressor__sample_weight=weights_train,
    )

    ridge_results = evaluate_model(
        "Ridge Regression Weighted Fuel Adjusted Engineered Priors",
        ridge_model,
        X_test,
        y_test,
    )

    xgb_results = evaluate_model(
        "XGBoost Regressor Weighted Fuel Adjusted Engineered Priors",
        xgb_model,
        X_test,
        y_test,
    )

    lgbm_results = evaluate_model(
        "LightGBM Regressor Weighted Fuel Adjusted Engineered Priors",
        lgbm_model,
        X_test,
        y_test,
    )

    log_model_to_mlflow(
        model_name="Ridge Regression Weighted Fuel Adjusted Engineered Priors",
        model=ridge_model,
        results=ridge_results,
        features=features,
        target=target,
        train_rows=len(X_train),
        test_rows=len(X_test),
        data_path=DATA_PATH,
        weighted_training=True,
    )

    log_model_to_mlflow(
        model_name="XGBoost Regressor Weighted Fuel Adjusted Engineered Priors",
        model=xgb_model,
        results=xgb_results,
        features=features,
        target=target,
        train_rows=len(X_train),
        test_rows=len(X_test),
        data_path=DATA_PATH,
        weighted_training=True,
    )

    log_model_to_mlflow(
        model_name="LightGBM Regressor Weighted Fuel Adjusted Engineered Priors",
        model=lgbm_model,
        results=lgbm_results,
        features=features,
        target=target,
        train_rows=len(X_train),
        test_rows=len(X_test),
        data_path=DATA_PATH,
        weighted_training=True,
    )

    results_df = pd.DataFrame(
        [
            {
                "model": ridge_results["model"],
                "mae": ridge_results["mae"],
                "rmse": ridge_results["rmse"],
                "r2": ridge_results["r2"],
                "weighted_training": True,
                "target": target,
                "engineered_features": True,
                "prior_features": True,
            },
            {
                "model": xgb_results["model"],
                "mae": xgb_results["mae"],
                "rmse": xgb_results["rmse"],
                "r2": xgb_results["r2"],
                "weighted_training": True,
                "target": target,
                "engineered_features": True,
                "prior_features": True,
            },
            {
                "model": lgbm_results["model"],
                "mae": lgbm_results["mae"],
                "rmse": lgbm_results["rmse"],
                "r2": lgbm_results["r2"],
                "weighted_training": True,
                "target": target,
                "engineered_features": True,
                "prior_features": True,
            },
        ]
    )

    results_df.to_csv(PREDICTION_DIR / "degradation_model_metrics.csv", index=False)

    all_results = [
        ("ridge_degradation_model.pkl", ridge_model, ridge_results),
        ("xgboost_degradation_model.pkl", xgb_model, xgb_results),
        ("lightgbm_degradation_model.pkl", lgbm_model, lgbm_results),
    ]

    best_model_name, best_model, best_results = min(
        all_results,
        key=lambda item: item[2]["mae"],
    )

    joblib.dump(best_model, MODEL_DIR / best_model_name)

    prediction_df = X_test.copy()
    prediction_df["actual_fuel_adjusted_degradation_delta"] = y_test.values
    prediction_df["sample_weight"] = weights_test.values

    prediction_df["ridge_prediction"] = ridge_results["predictions"]
    prediction_df["xgboost_prediction"] = xgb_results["predictions"]
    prediction_df["lightgbm_prediction"] = lgbm_results["predictions"]

    prediction_df["ridge_error"] = (
        prediction_df["actual_fuel_adjusted_degradation_delta"]
        - prediction_df["ridge_prediction"]
    ).abs()

    prediction_df["xgboost_error"] = (
        prediction_df["actual_fuel_adjusted_degradation_delta"]
        - prediction_df["xgboost_prediction"]
    ).abs()

    prediction_df["lightgbm_error"] = (
        prediction_df["actual_fuel_adjusted_degradation_delta"]
        - prediction_df["lightgbm_prediction"]
    ).abs()

    prediction_df.to_csv(
        PREDICTION_DIR / "degradation_model_predictions.csv",
        index=False,
    )

    print("\nBest model:")
    print(f"- {best_results['model']}")
    print(f"- MAE:  {best_results['mae']:.4f} seconds")
    print(f"- RMSE: {best_results['rmse']:.4f} seconds")
    print(f"- R²:   {best_results['r2']:.4f}")

    print("\nSaved files:")
    print(f"- {PREDICTION_DIR / 'degradation_model_metrics.csv'}")
    print(f"- {PREDICTION_DIR / 'degradation_model_predictions.csv'}")
    print(f"- {MODEL_DIR / best_model_name}")

    print("\nMLflow experiment:")
    print(f"- {EXPERIMENT_NAME}")
    print("- Open http://127.0.0.1:5000 to view runs")


if __name__ == "__main__":
    train_models()