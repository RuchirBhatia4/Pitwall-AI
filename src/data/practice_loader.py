from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REQUIRED_PRACTICE_COLUMNS = [
    "race",
    "session",
    "driver",
    "team",
    "compound",
    "lap_number",
    "stint_lap",
    "lap_time_seconds",
    "track_temp",
    "air_temp",
    "run_type",
    "clean_lap",
    "traffic_affected",
    "drs_used",
]

VALID_SESSIONS = {"FP1", "FP2", "FP3"}

# Modeling assumption:
# We only trust dedicated race-pace/long-run laps as a direct degradation signal.
DEFAULT_LONG_RUN_TYPES = {
    "LONG_RUN",
    "LONGRUN",
    "RACE_PACE",
    "RACEPACE",
    "RACE-PACE",
}


def _to_bool(series: pd.Series, default: bool = False) -> pd.Series:
    """
    Parses mixed boolean representations used in CSV exports.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    parsed = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    return parsed.fillna(default).astype(bool)


def validate_practice_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_PRACTICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Practice CSV is missing required columns: "
            f"{missing}. Required columns: {REQUIRED_PRACTICE_COLUMNS}"
        )


def normalize_practice_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes schema/casing/types for uploaded practice long-run data.
    """
    clean = df.copy()

    clean.columns = [col.strip().lower() for col in clean.columns]
    validate_practice_columns(clean)

    for col in ["lap_number", "stint_lap", "lap_time_seconds", "track_temp", "air_temp"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(
        subset=[
            "race",
            "session",
            "driver",
            "team",
            "compound",
            "lap_number",
            "stint_lap",
            "lap_time_seconds",
            "run_type",
        ]
    )

    clean["race"] = clean["race"].astype(str).str.strip()
    clean["session"] = clean["session"].astype(str).str.strip().str.upper()
    clean["driver"] = clean["driver"].astype(str).str.strip().str.upper()
    clean["team"] = clean["team"].astype(str).str.strip()
    clean["compound"] = clean["compound"].astype(str).str.strip().str.upper()
    clean["run_type"] = clean["run_type"].astype(str).str.strip().str.upper()

    clean["clean_lap"] = _to_bool(clean["clean_lap"], default=False)
    clean["traffic_affected"] = _to_bool(clean["traffic_affected"], default=False)
    clean["drs_used"] = _to_bool(clean["drs_used"], default=False)

    clean = clean[clean["session"].isin(VALID_SESSIONS)]

    clean = clean[(clean["lap_time_seconds"] > 0) & (clean["stint_lap"] > 0)]

    clean["lap_number"] = clean["lap_number"].astype(int)
    clean["stint_lap"] = clean["stint_lap"].astype(int)

    clean = clean.sort_values(
        ["race", "session", "driver", "team", "compound", "stint_lap"]
    ).reset_index(drop=True)

    return clean


def load_practice_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads one practice race-pace CSV and applies schema normalization.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    clean = normalize_practice_dataframe(df)
    clean["source_file"] = str(path)
    return clean


def load_practice_csvs(csv_paths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Loads multiple practice CSVs and concatenates them.
    """
    if not csv_paths:
        raise ValueError("No practice CSV paths were provided.")

    frames = [load_practice_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def load_practice_csvs_from_directory(
    directory: str | Path,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """
    Discovers practice CSVs in a directory and loads all matching files.
    """
    csv_files = sorted(Path(directory).glob(pattern))
    if not csv_files:
        raise ValueError(f"No practice CSV files found in {directory} matching {pattern}")

    return load_practice_csvs(csv_files)


def filter_clean_practice_laps(
    df: pd.DataFrame,
    allowed_run_types: Iterable[str] | None = None,
    allow_drs_laps: bool = False,
) -> pd.DataFrame:
    """
    Filters normalized practice laps to keep race-pace-quality signal only.

    Modeling assumptions:
    - Keep only race-pace/long-run rows (`run_type`).
    - Keep only marked `clean_lap` rows.
    - Remove traffic-affected rows.
    - By default, drop DRS-assisted rows for a cleaner pure-tire signal.
    """
    run_types = {item.strip().upper() for item in (allowed_run_types or DEFAULT_LONG_RUN_TYPES)}

    filtered = df.copy()
    filtered = filtered[filtered["run_type"].isin(run_types)]
    filtered = filtered[filtered["clean_lap"]]
    filtered = filtered[~filtered["traffic_affected"]]

    if not allow_drs_laps:
        filtered = filtered[~filtered["drs_used"]]

    return filtered.reset_index(drop=True)


def load_and_filter_practice_laps(
    csv_paths: Sequence[str | Path],
    allowed_run_types: Iterable[str] | None = None,
    allow_drs_laps: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper: load + normalize + clean-filter practice laps.
    """
    raw = load_practice_csvs(csv_paths)
    return filter_clean_practice_laps(
        raw,
        allowed_run_types=allowed_run_types,
        allow_drs_laps=allow_drs_laps,
    )


if __name__ == "__main__":
    sample_dir = Path("data/processed/practice")

    if sample_dir.exists():
        sample = load_practice_csvs_from_directory(sample_dir)
        filtered = filter_clean_practice_laps(sample)

        print("Loaded practice rows:", len(sample))
        print("Filtered clean race-pace rows:", len(filtered))
        print(filtered.head())
    else:
        print(
            "No practice directory found at data/processed/practice. "
            "Add CSVs there to test this loader."
        )
