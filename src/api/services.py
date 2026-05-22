from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.api.contracts import (
    CALIBRATED_REQUIRED_COLUMNS,
    LIVE_EVENT_COLUMNS,
    LIVE_EVENT_TYPES,
    STRATEGY_REQUIRED_COLUMNS,
)
from src.api.settings import get_settings
from src.api.storage import (
    CALIBRATED_DATASET,
    LIVE_DATASET,
    STRATEGY_DATASET,
    load_dataframe_dataset,
    save_dataframe_dataset,
    sync_csv_to_database,
)
from src.pipeline.generate_calibrated_predictions import (
    DEFAULT_BEST_MODEL_PATH,
    DEFAULT_HISTORICAL_CURVES_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CALIBRATED_OUTPUT_PATH,
    DEFAULT_PRACTICE_CSV_PATH,
    generate_calibrated_prediction_report,
)
from src.pipeline.generate_strategy_recommendations import (
    DEFAULT_CALIBRATED_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_STRATEGY_OUTPUT_PATH,
    DEFAULT_RACE_SETUP_PATH,
    generate_strategy_recommendations,
)
from src.pipeline.run_live_race_simulation import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_LIVE_OUTPUT_PATH,
    run_live_race_simulation,
)


CALIBRATED_PREDICTIONS_PATH = Path("data/predictions/calibrated_degradation_predictions.csv")
STRATEGY_RECOMMENDATIONS_PATH = Path("data/predictions/strategy_recommendations.csv")


def validate_dataframe_contract(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Required columns: {required_columns}"
        )


def _read_csv_with_contract(
    path: str | Path,
    required_columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} not found at {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_contract(df, required_columns, dataset_name)
    return df


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def dataframe_response(df: pd.DataFrame) -> dict[str, Any]:
    safe_df = df.where(pd.notna(df), None)
    return {
        "rows": safe_df.to_dict(orient="records"),
        "columns": safe_df.columns.tolist(),
        "row_count": int(len(safe_df)),
    }


def load_calibrated_predictions(
    path: str | Path = CALIBRATED_PREDICTIONS_PATH,
) -> pd.DataFrame:
    settings = get_settings()
    if settings.use_database:
        df = load_dataframe_dataset(CALIBRATED_DATASET)
        validate_dataframe_contract(
            df,
            CALIBRATED_REQUIRED_COLUMNS,
            "calibrated_degradation_predictions database rows",
        )
        return df

    return _read_csv_with_contract(
        path=path,
        required_columns=CALIBRATED_REQUIRED_COLUMNS,
        dataset_name="calibrated_degradation_predictions.csv",
    )


def load_strategy_recommendations(
    path: str | Path = STRATEGY_RECOMMENDATIONS_PATH,
) -> pd.DataFrame:
    settings = get_settings()
    if settings.use_database:
        strategy_df = load_dataframe_dataset(STRATEGY_DATASET)
        validate_dataframe_contract(
            strategy_df,
            STRATEGY_REQUIRED_COLUMNS,
            "strategy_recommendations database rows",
        )
    else:
        strategy_df = _read_csv_with_contract(
            path=path,
            required_columns=STRATEGY_REQUIRED_COLUMNS,
            dataset_name="strategy_recommendations.csv",
        )
    strategy_df = strategy_df.copy()
    strategy_df["is_recommended"] = _to_bool_series(strategy_df["is_recommended"])
    return strategy_df


def summarize_dashboard(
    calibrated_df: pd.DataFrame | None = None,
    strategy_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    calibrated = calibrated_df if calibrated_df is not None else load_calibrated_predictions()
    strategies = strategy_df if strategy_df is not None else load_strategy_recommendations()

    driver_count = (
        pd.concat([calibrated["driver"], strategies["driver"]], ignore_index=True)
        .dropna()
        .nunique()
    )
    team_count = (
        pd.concat([calibrated["team"], strategies["team"]], ignore_index=True)
        .dropna()
        .nunique()
    )

    return {
        "drivers": int(driver_count),
        "teams": int(team_count),
        "calibrated_rows": int(len(calibrated)),
        "strategy_rows": int(len(strategies)),
        "recommended_strategies": int(strategies["is_recommended"].sum()),
    }


def run_calibrated_predictions_pipeline(
    practice_csv_path: str | Path | None = None,
    historical_predictions_path: str | Path | None = None,
    best_model_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    resolved_output = Path(output_path or DEFAULT_CALIBRATED_OUTPUT_PATH)
    result = generate_calibrated_prediction_report(
        practice_csv_path=practice_csv_path or DEFAULT_PRACTICE_CSV_PATH,
        historical_predictions_path=historical_predictions_path or DEFAULT_HISTORICAL_CURVES_PATH,
        best_model_path=best_model_path or DEFAULT_BEST_MODEL_PATH,
        output_path=resolved_output,
    )
    if get_settings().use_database:
        save_dataframe_dataset(
            dataset=CALIBRATED_DATASET,
            df=result,
            source_path=resolved_output,
        )
    return result, resolved_output


def run_strategy_recommendations_pipeline(
    calibrated_predictions_path: str | Path | None = None,
    race_setup_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    resolved_output = Path(output_path or DEFAULT_STRATEGY_OUTPUT_PATH)
    result = generate_strategy_recommendations(
        calibrated_predictions_path=calibrated_predictions_path or DEFAULT_CALIBRATED_PATH,
        race_setup_path=race_setup_path or DEFAULT_RACE_SETUP_PATH,
        output_path=resolved_output,
    )
    if get_settings().use_database:
        save_dataframe_dataset(
            dataset=STRATEGY_DATASET,
            df=result,
            source_path=resolved_output,
        )
    return result, resolved_output


def write_live_events(events: list[dict[str, Any]], path: str | Path = DEFAULT_EVENTS_PATH) -> Path:
    normalized = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip().lower()
        if event_type not in LIVE_EVENT_TYPES:
            raise ValueError(f"Unsupported live event type: {event_type}")
        normalized.append(
            {
                "lap": int(event.get("lap", 1)),
                "event_type": event_type,
                "driver": str(event.get("driver", "")).strip().upper(),
                "target_driver": str(event.get("target_driver", "")).strip().upper(),
                "compound": str(event.get("compound", "")).strip().upper(),
                "notes": str(event.get("notes", "")).strip(),
            }
        )

    event_df = pd.DataFrame(normalized, columns=LIVE_EVENT_COLUMNS)
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(event_path, index=False)
    return event_path


def run_live_pipeline(
    race: str | None = None,
    setup_path: str | Path | None = None,
    output_path: str | Path | None = None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, Path]:
    events_path = write_live_events(events or [], path=DEFAULT_EVENTS_PATH)
    resolved_output = Path(output_path or DEFAULT_LIVE_OUTPUT_PATH)
    result = run_live_race_simulation(
        race=race,
        setup_path=setup_path,
        events_path=events_path,
        output_path=resolved_output,
    )
    if get_settings().use_database:
        save_dataframe_dataset(
            dataset=LIVE_DATASET,
            df=result,
            source_path=resolved_output,
        )
    return result, resolved_output


def storage_status() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "backend": settings.storage_backend,
        "database_configured": bool(settings.database_url),
        "database_active": settings.use_database,
    }


def sync_current_outputs_to_database() -> dict[str, int]:
    calibrated = sync_csv_to_database(
        dataset=CALIBRATED_DATASET,
        csv_path=CALIBRATED_PREDICTIONS_PATH,
        required_columns=CALIBRATED_REQUIRED_COLUMNS,
    )
    strategies = sync_csv_to_database(
        dataset=STRATEGY_DATASET,
        csv_path=STRATEGY_RECOMMENDATIONS_PATH,
        required_columns=STRATEGY_REQUIRED_COLUMNS,
    )
    return {
        "calibrated_rows": int(len(calibrated)),
        "strategy_rows": int(len(strategies)),
    }
