from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.api.database import db_connection, initialize_database


CALIBRATED_DATASET = "calibrated_degradation_predictions"
STRATEGY_DATASET = "strategy_recommendations"
LIVE_DATASET = "live_race_simulation"


def dataframe_to_json_rows(df: pd.DataFrame) -> list[dict]:
    safe_df = df.where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


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
                    "row_data": json.dumps(row),
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
