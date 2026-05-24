from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    database_url: str | None
    storage_backend: str
    cors_allow_origins: list[str]
    cors_allow_origin_regex: str | None

    @property
    def use_database(self) -> bool:
        return self.storage_backend == "database" and bool(self.database_url)


def _split_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_settings() -> ApiSettings:
    return ApiSettings(
        database_url=os.getenv("DATABASE_URL"),
        storage_backend=os.getenv("PITWALL_STORAGE_BACKEND", "csv").strip().lower(),
        cors_allow_origins=_split_csv_env(
            os.getenv(
                "CORS_ALLOW_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            )
        ),
        cors_allow_origin_regex=os.getenv(
            "CORS_ALLOW_ORIGIN_REGEX",
            r"https://.*\.vercel\.app",
        ),
    )
