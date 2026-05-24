from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.api.database import db_connection, initialize_database


CALIBRATED_DATASET = "calibrated_degradation_predictions"
STRATEGY_DATASET = "strategy_recommendations"
LIVE_DATASET = "live_race_simulation"


def dataframe_to_json_rows(df: pd.DataFrame) -> list[dict]:
    raw_rows = df.astype(object).to_dict(orient="records")
    return [_sanitize_json_row(row) for row in raw_rows]


def _sanitize_json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_json_value(value) for key, value in row.items()}


def _sanitize_json_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    if pd.isna(value):
        return None

    return value


def save_dataframe_dataset(
    dataset: str,
    df: pd.DataFrame,
    source_path: str | Path | None = None,
) -> None:
    initialize_database()
    rows = dataframe_to_json_rows(df)

    with db_connection() as connection:
        connection.execute(text("delete from pitwall_table_rows where dataset = :dataset"), {"dataset": dataset})
        if not rows:
            return

        connection.execute(
            text(
                """
                insert into pitwall_table_rows (dataset, row_index, row_data, source_path)
                values (:dataset, :row_index, cast(:row_data as jsonb), :source_path)
                """
            ),
            [
                {
                    "dataset": dataset,
                    "row_index": index,
                    "row_data": json.dumps(row, allow_nan=False),
                    "source_path": str(source_path) if source_path is not None else None,
                }
                for index, row in enumerate(rows)
            ],
        )


def load_dataframe_dataset(dataset: str) -> pd.DataFrame:
    initialize_database()
    with db_connection() as connection:
        result = connection.execute(
            text(
                """
                select row_data
                from pitwall_table_rows
                where dataset = :dataset
                order by row_index
                """
            ),
            {"dataset": dataset},
        )
        rows = [dict(row[0]) for row in result]
    return pd.DataFrame(rows)


def sync_csv_to_database(
    dataset: str,
    csv_path: str | Path,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if required_columns:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

    save_dataframe_dataset(dataset=dataset, df=df, source_path=path)
    return df
