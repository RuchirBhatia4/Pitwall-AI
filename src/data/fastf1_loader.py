from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd


CACHE_DIR = "data/raw/fastf1/cache"


def enable_fastf1_cache(cache_dir: str = CACHE_DIR) -> None:
    """
    Enables FastF1 caching so repeated data calls are faster.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def load_race_laps(
    season: int,
    race: str,
    session_type: str = "R",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads F1 lap data for a race using FastF1.

    Args:
        season: F1 season year, e.g. 2024.
        race: Race name, e.g. "Bahrain".
        session_type: "R" for race, "Q" for qualifying, "FP1", "FP2", "FP3".
        save_path: Optional path to save CSV.

    Returns:
        DataFrame containing lap-level race data.
    """
    enable_fastf1_cache()

    session = fastf1.get_session(season, race, session_type)
    session.load()

    laps = session.laps.copy()

    keep_cols = [
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Compound",
        "TyreLife",
        "Stint",
        "PitInTime",
        "PitOutTime",
        "TrackStatus",
        "Position",
        "Deleted",
    ]

    existing_cols = [col for col in keep_cols if col in laps.columns]
    df = laps[existing_cols].copy()

    df["season"] = season
    df["race"] = race
    df["session_type"] = session_type

    df["lap_time_seconds"] = df["LapTime"].dt.total_seconds()

    if "Sector1Time" in df.columns:
        df["sector1_seconds"] = df["Sector1Time"].dt.total_seconds()

    if "Sector2Time" in df.columns:
        df["sector2_seconds"] = df["Sector2Time"].dt.total_seconds()

    if "Sector3Time" in df.columns:
        df["sector3_seconds"] = df["Sector3Time"].dt.total_seconds()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    race_df = load_race_laps(
        season=2024,
        race="Bahrain",
        session_type="R",
        save_path="data/processed/2024_bahrain_race_laps_raw.csv",
    )

    print(race_df.head())
    print(f"Rows: {len(race_df)}")
    print(f"Columns: {race_df.columns.tolist()}")