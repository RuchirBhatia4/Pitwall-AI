from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live.race_state import LiveRaceState


REQUIRED_EVENT_COLUMNS = [
    "lap",
    "event_type",
    "driver",
    "target_driver",
    "compound",
    "notes",
]

SUPPORTED_EVENT_TYPES = {
    "safety_car_start",
    "safety_car_end",
    "pit_stop",
    "overtake",
    "retirement",
}


def _validate_event_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Live event CSV missing required columns: {missing}")


def load_live_events(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Live event file not found: {path}")

    df = pd.read_csv(path)
    _validate_event_columns(df)

    clean = df.copy()
    clean["lap"] = pd.to_numeric(clean["lap"], errors="coerce")
    clean = clean.dropna(subset=["lap", "event_type"]).copy()

    clean["lap"] = clean["lap"].astype(int)
    clean["event_type"] = clean["event_type"].astype(str).str.strip().str.lower()
    clean["driver"] = clean["driver"].fillna("").astype(str).str.strip().str.upper()
    clean["target_driver"] = clean["target_driver"].fillna("").astype(str).str.strip().str.upper()
    clean["compound"] = clean["compound"].fillna("").astype(str).str.strip().str.upper()
    clean["notes"] = clean["notes"].fillna("").astype(str)

    clean = clean[clean["event_type"].isin(SUPPORTED_EVENT_TYPES)].copy()
    clean = clean.sort_values(["lap", "event_type", "driver"]).reset_index(drop=True)

    return clean


def apply_events_for_lap(
    race_state: LiveRaceState,
    events_df: pd.DataFrame,
    lap: int,
) -> None:
    if events_df.empty:
        return

    lap_events = events_df[events_df["lap"] == int(lap)]
    if lap_events.empty:
        return

    event_priority = {
        "safety_car_start": 0,
        "retirement": 1,
        "overtake": 2,
        "pit_stop": 3,
        "safety_car_end": 4,
    }
    lap_events = lap_events.assign(
        _priority=lap_events["event_type"].map(event_priority).fillna(99)
    ).sort_values(["_priority", "driver", "target_driver"])

    for _, event in lap_events.iterrows():
        event_type = str(event["event_type"]).lower()
        driver = str(event["driver"]).upper()
        target_driver = str(event["target_driver"]).upper()
        compound = str(event["compound"]).upper()

        if event_type == "safety_car_start":
            race_state.set_under_safety_car(True)
            continue

        if event_type == "safety_car_end":
            race_state.set_under_safety_car(False)
            continue

        if event_type == "pit_stop":
            race_state.apply_pit_stop(driver=driver, compound=compound or None)
            continue

        if event_type == "overtake":
            race_state.apply_overtake(driver=driver, target_driver=target_driver)
            continue

        if event_type == "retirement":
            race_state.retire_driver(driver=driver)
            continue
