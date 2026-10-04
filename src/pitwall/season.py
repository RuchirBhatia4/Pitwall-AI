"""Season calendar and FastF1 cache helpers."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "fastf1" / "cache"
SEASON_DIR = PROJECT_ROOT / "data" / "season"

_cache_enabled = False


def enable_cache() -> None:
    global _cache_enabled
    if _cache_enabled:
        return
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    _cache_enabled = True


@lru_cache(maxsize=4)
def get_schedule(year: int = 2026) -> pd.DataFrame:
    enable_cache()
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule.reset_index(drop=True)
