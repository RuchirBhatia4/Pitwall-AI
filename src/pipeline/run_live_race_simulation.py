from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.live.event_processor import load_live_events
from src.live.live_strategy_engine import LIVE_OUTPUT_COLUMNS, run_live_strategy_engine
from src.live.race_state import REQUIRED_SETUP_COLUMNS, build_race_state_from_setup
from src.live.scenario_simulator import compute_mandatory_stop_deadline_lap, is_dry_compound


DEFAULT_OFFICIAL_SETUP_PATH = Path("data/raw/race_setup/race_setup_official_seed.csv")
DEFAULT_SAMPLE_SETUP_PATH = Path("data/raw/race_setup/race_setup_sample.csv")
DEFAULT_EVENTS_PATH = Path("data/raw/race_control/live_race_input_sample.csv")
DEFAULT_OUTPUT_PATH = Path("data/predictions/live_race_simulation.csv")

# Decision defaults used when setup CSV does not include these assumptions.
DEFAULT_BASE_PACE_SECONDS = 92.0
DEFAULT_PIT_LOSS_GREEN_SECONDS = 22.0
DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS = 13.0
DEFAULT_OVERTAKING_DIFFICULTY = 0.70
DEFAULT_TRACK_POSITION_IMPORTANCE = 0.70
DEFAULT_MANDATORY_STOP_BUFFER_LAPS = 3

# Backward-compatible export used by pipeline tests/contracts.
OUTPUT_COLUMNS = LIVE_OUTPUT_COLUMNS


def _resolve_default_setup_path() -> tuple[Path, str]:
    if DEFAULT_OFFICIAL_SETUP_PATH.exists():
        return DEFAULT_OFFICIAL_SETUP_PATH, "official_seeded"

    if DEFAULT_SAMPLE_SETUP_PATH.exists():
        return DEFAULT_SAMPLE_SETUP_PATH, "sample_setup"

    raise FileNotFoundError(
        "No race setup file found. Expected either "
        f"{DEFAULT_OFFICIAL_SETUP_PATH} or {DEFAULT_SAMPLE_SETUP_PATH}."
    )


def _load_setup_df(path: str | Path, source_label: str) -> pd.DataFrame:
    setup_path = Path(path)
    if not setup_path.exists():
        raise FileNotFoundError(f"Race setup file not found: {setup_path}")

    df = pd.read_csv(setup_path)
    missing = [col for col in REQUIRED_SETUP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Race setup file missing required columns: {missing}")

    clean = df.copy()
    clean["race"] = clean["race"].astype(str).str.strip()
    clean["driver"] = clean["driver"].astype(str).str.strip().str.upper()
    clean["team"] = clean["team"].astype(str).str.strip()
    clean["starting_compound"] = clean["starting_compound"].astype(str).str.strip().str.upper()

    if "setup_source" not in clean.columns:
        clean["setup_source"] = source_label
    else:
        clean["setup_source"] = clean["setup_source"].fillna(source_label).astype(str).str.strip()

    numeric_defaults = {
        "race_laps": 57,
        "base_pace_seconds": DEFAULT_BASE_PACE_SECONDS,
        "pit_loss_green_seconds": DEFAULT_PIT_LOSS_GREEN_SECONDS,
        "pit_loss_safety_car_seconds": DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
        "overtaking_difficulty": DEFAULT_OVERTAKING_DIFFICULTY,
        "track_position_importance": DEFAULT_TRACK_POSITION_IMPORTANCE,
        "mandatory_stop_buffer_laps": DEFAULT_MANDATORY_STOP_BUFFER_LAPS,
    }

    for col, default in numeric_defaults.items():
        if col not in clean.columns:
            clean[col] = default
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(default)

    return clean


def _resolve_race_name(setup_df: pd.DataFrame, race_name: str | None = None) -> str:
    races = setup_df["race"].dropna().astype(str).str.strip().unique().tolist()
    if not races:
        raise ValueError("Setup CSV has no valid race values.")

    if race_name is None or not str(race_name).strip():
        return str(races[0])

    query = str(race_name).strip().lower()

    for candidate in races:
        if candidate.lower() == query:
            return str(candidate)

    for candidate in races:
        if query in candidate.lower() or candidate.lower() in query:
            return str(candidate)

    raise ValueError(f"Race '{race_name}' not found in setup file.")


def _build_driver_setup_map(race_setup_df: pd.DataFrame) -> dict[str, dict[str, float | str | int]]:
    mapping: dict[str, dict[str, float | str | int]] = {}

    for _, row in race_setup_df.iterrows():
        driver = str(row["driver"]).upper()
        race_laps = int(row["race_laps"])
        mandatory_stop_buffer_laps = int(row.get("mandatory_stop_buffer_laps", DEFAULT_MANDATORY_STOP_BUFFER_LAPS))
        mandatory_stop_deadline_lap = int(
            row.get(
                "mandatory_stop_deadline_lap",
                compute_mandatory_stop_deadline_lap(
                    race_laps=race_laps,
                    buffer_laps=mandatory_stop_buffer_laps,
                ),
            )
        )
        mapping[driver] = {
            "race_laps": race_laps,
            "starting_compound": str(row["starting_compound"]).upper(),
            "base_pace_seconds": float(row["base_pace_seconds"]),
            "pit_loss_green_seconds": float(row["pit_loss_green_seconds"]),
            "pit_loss_safety_car_seconds": float(row["pit_loss_safety_car_seconds"]),
            "overtaking_difficulty": float(row["overtaking_difficulty"]),
            "track_position_importance": float(row["track_position_importance"]),
            "is_dry_race": bool(is_dry_compound(str(row["starting_compound"]).upper())),
            "mandatory_stop_buffer_laps": int(mandatory_stop_buffer_laps),
            "mandatory_stop_deadline_lap": int(max(1, mandatory_stop_deadline_lap)),
        }

    return mapping


def run_live_race_simulation(
    race: str | None = None,
    setup_path: str | Path | None = None,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    if setup_path is None:
        resolved_setup_path, source_label = _resolve_default_setup_path()
    else:
        resolved_setup_path = Path(setup_path)
        source_label = "custom_setup"

    setup_df = _load_setup_df(path=resolved_setup_path, source_label=source_label)
    selected_race = _resolve_race_name(setup_df, race_name=race)

    race_setup = setup_df[
        setup_df["race"].astype(str).str.lower() == selected_race.lower()
    ].copy()
    if race_setup.empty:
        raise ValueError(f"No setup rows found for race '{selected_race}'.")

    row_setup_source = (
        race_setup["setup_source"].mode().iloc[0]
        if "setup_source" in race_setup.columns and not race_setup["setup_source"].empty
        else source_label
    )

    race_state = build_race_state_from_setup(
        setup_df=race_setup,
        race=selected_race,
        setup_source=str(row_setup_source),
    )
    events_df = load_live_events(events_path)
    driver_setup = _build_driver_setup_map(race_setup)

    result = run_live_strategy_engine(
        race_state=race_state,
        events_df=events_df,
        driver_setup=driver_setup,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    if result.empty:
        return pd.DataFrame(columns=LIVE_OUTPUT_COLUMNS)
    return result[LIVE_OUTPUT_COLUMNS]


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Run live race simulation with pit decision support, overtake probabilities, "
            "and projected finish outcomes."
        )
    )
    parser.add_argument("--race", default=None, help="Race name to simulate.")
    parser.add_argument(
        "--setup",
        default=None,
        help=(
            "Optional setup CSV path override. If omitted, pipeline prefers "
            "data/raw/race_setup/race_setup_official_seed.csv and falls back to "
            "data/raw/race_setup/race_setup_sample.csv."
        ),
    )
    parser.add_argument(
        "--events",
        default=str(DEFAULT_EVENTS_PATH),
        help="Live race-control event CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path for live race simulation rows.",
    )
    args = parser.parse_args()

    result = run_live_race_simulation(
        race=args.race,
        setup_path=args.setup,
        events_path=args.events,
        output_path=args.output,
    )

    print("Live race simulation generated.")
    print(f"- Rows: {len(result)}")
    print(f"- Output: {Path(args.output)}")
    if not result.empty:
        print(f"- Race: {result['race'].iloc[0]}")
        print(f"- Drivers tracked: {result['driver'].nunique()}")
        print(f"- Setup source: {result['setup_source'].mode().iloc[0]}")
        print(result.head(12))


if __name__ == "__main__":
    main()
