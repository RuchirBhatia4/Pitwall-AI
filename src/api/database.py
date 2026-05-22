from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from src.api.settings import get_settings


_ENGINE: Engine | None = None


def get_database_url() -> str | None:
    return get_settings().database_url


def get_engine() -> Engine:
    global _ENGINE

    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if _ENGINE is None:
        _ENGINE = create_engine(database_url, pool_pre_ping=True)
    return _ENGINE


@contextmanager
def db_connection() -> Iterator[Connection]:
    engine = get_engine()
    with engine.begin() as connection:
        yield connection


def initialize_database() -> None:
    """
    Creates the app storage tables when using a hosted Postgres database.

    Modeling outputs are intentionally stored as JSON rows in the first live
    backend version so new prediction/report columns do not require a database
    migration for every experiment.
    """
    with db_connection() as connection:
        connection.execute(
            text(
                """
                create table if not exists pitwall_table_rows (
                    id bigserial primary key,
                    dataset text not null,
                    row_index integer not null,
                    row_data jsonb not null,
                    source_path text,
                    created_at timestamptz not null default now()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                create index if not exists idx_pitwall_table_rows_dataset_created
                on pitwall_table_rows (dataset, created_at desc)
                """
            )
        )
