from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    database_url: str | None
    storage_backend: str

    @property
    def use_database(self) -> bool:
        return self.storage_backend == "database" and bool(self.database_url)


def get_settings() -> ApiSettings:
    return ApiSettings(
        database_url=os.getenv("DATABASE_URL"),
        storage_backend=os.getenv("PITWALL_STORAGE_BACKEND", "csv").strip().lower(),
    )
