from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.simulation.race_simulator import (
    DEFAULT_BASE_PACE_SECONDS,
    DEFAULT_PIT_LOSS_GREEN_SECONDS,
    DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    DEFAULT_RACE_LAPS,
    is_valid_dry_strategy,
)
from src.simulation.strategy_optimizer import (
    apply_position_aware_ranking,
    evaluate_driver_strategies,
)


DEFAULT_CALIBRATED_PATH = Path("data/predictions/calibrated_degradation_predictions.csv")
DEFAULT_RACE_SETUP_PATH = Path("data/raw/race_setup/race_setup_sample.csv")
DEFAULT_OUTPUT_PATH = Path("data/predictions/strategy_recommendations.csv")

REQUIRED_CALIBRATED_COLUMNS = [
    "race",
    "driver",
    "team",
    "compound",
    "calibrated_deg_estimate",
    "practice_confidence",
]

REQUIRED_RACE_SETUP_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "base_pace_seconds",
    "race_laps",
    "pit_loss_green_seconds",
    "pit_loss_safety_car_seconds",
    "overtaking_difficulty",
    "safety_car_probability",
]

OPTIONAL_RACE_SETUP_COLUMNS = [
    "starting_compound",
    "track_position_importance",
]

OUTPUT_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "strategy",
    "pit_laps",
    "scenario_name",
    "expected_total_time",
    "time_delta_to_best",
    "risk_score",
    "is_recommended",
    "stops",
    "starting_compound",
    "best_safety_car_pit_lap",
    "safety_car_gain_seconds",
    "track_position_penalty",
    "pit_loss_used",
    "projected_finish_position",
    "position_delta_from_start",
    "position_outcome_score",
    "pit_recovery_probability",
    "expected_pit_cycle_position_loss",
    "recommendation_reason",
]


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def _load_calibrated_predictions(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Calibrated degradation file not found: {csv_path}. "
            "Run `python -m src.pipeline.generate_calibrated_predictions` first."
        )

    df = pd.read_csv(csv_path)
    _validate_required_columns(df, REQUIRED_CALIBRATED_COLUMNS, str(csv_path))

    normalized = df.copy()
    normalized["race"] = normalized["race"].astype(str).str.strip()
    normalized["driver"] = normalized["driver"].astype(str).str.strip().str.upper()
    normalized["team"] = normalized["team"].astype(str).str.strip()
    if "starting_compound" in normalized.columns:
        normalized["starting_compound"] = (
            normalized["starting_compound"].astype(str).str.strip().str.upper()
        )
    normalized["compound"] = normalized["compound"].astype(str).str.strip().str.upper()

    return normalized


def _load_race_setup(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Race setup file not found: {csv_path}. "
            "Provide a setup CSV with race/driver grid and simulation assumptions."
        )

    df = pd.read_csv(csv_path)
    _validate_required_columns(df, REQUIRED_RACE_SETUP_COLUMNS, str(csv_path))

    normalized = df.copy()
    normalized["race"] = normalized["race"].astype(str).str.strip()
    normalized["driver"] = normalized["driver"].astype(str).str.strip().str.upper()
    normalized["team"] = normalized["team"].astype(str).str.strip()

    numeric_cols = [
        "starting_position",
        "base_pace_seconds",
        "race_laps",
        "pit_loss_green_seconds",
        "pit_loss_safety_car_seconds",
        "overtaking_difficulty",
        "safety_car_probability",
    ]
    if "track_position_importance" in normalized.columns:
        numeric_cols.append("track_position_importance")

    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    return normalized


def generate_strategy_recommendations(
    calibrated_predictions_path: str | Path = DEFAULT_CALIBRATED_PATH,
    race_setup_path: str | Path = DEFAULT_RACE_SETUP_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    default_race_laps: int = DEFAULT_RACE_LAPS,
    default_pit_loss_green_seconds: float = DEFAULT_PIT_LOSS_GREEN_SECONDS,
    default_pit_loss_safety_car_seconds: float = DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    default_base_pace_seconds: float = DEFAULT_BASE_PACE_SECONDS,
) -> pd.DataFrame:
    calibrated_df = _load_calibrated_predictions(calibrated_predictions_path)
    setup_df = _load_race_setup(race_setup_path)

    rows = []
    for _, setup_row in setup_df.iterrows():
        race = str(setup_row["race"])
        driver = str(setup_row["driver"]).upper()
        team = str(setup_row["team"])

        driver_rows = calibrated_df[
            (calibrated_df["race"] == race)
            & (calibrated_df["driver"] == driver)
            & (calibrated_df["team"] == team)
        ].copy()

        driver_recommendations = evaluate_driver_strategies(
            race=race,
            driver=driver,
            team=team,
            starting_position=int(
                setup_row["starting_position"]
                if pd.notna(setup_row["starting_position"])
                else 10
            ),
            driver_rows=driver_rows,
            starting_compound=(
                str(setup_row["starting_compound"]).upper()
                if "starting_compound" in setup_row.index
                and pd.notna(setup_row["starting_compound"])
                and str(setup_row["starting_compound"]).strip()
                else None
            ),
            race_laps=int(
                setup_row["race_laps"]
                if pd.notna(setup_row["race_laps"])
                else default_race_laps
            ),
            pit_loss_green_seconds=float(
                setup_row["pit_loss_green_seconds"]
                if pd.notna(setup_row["pit_loss_green_seconds"])
                else default_pit_loss_green_seconds
            ),
            pit_loss_safety_car_seconds=float(
                setup_row["pit_loss_safety_car_seconds"]
                if pd.notna(setup_row["pit_loss_safety_car_seconds"])
                else default_pit_loss_safety_car_seconds
            ),
            base_pace_seconds=float(
                setup_row["base_pace_seconds"]
                if pd.notna(setup_row["base_pace_seconds"])
                else default_base_pace_seconds
            ),
            overtaking_difficulty=float(
                setup_row["overtaking_difficulty"]
                if pd.notna(setup_row["overtaking_difficulty"])
                else 0.7
            ),
            safety_car_probability=float(
                setup_row["safety_car_probability"]
                if pd.notna(setup_row["safety_car_probability"])
                else 0.5
            ),
        )
        rows.append(driver_recommendations)

    if not rows:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        result = pd.concat(rows, ignore_index=True)
        if result.empty:
            result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            result = apply_position_aware_ranking(result, setup_df)
            # Defensive guard: no invalid dry no-stop plans should survive to output.
            valid_mask = result.apply(
                lambda row: is_valid_dry_strategy(
                    strategy_compounds=str(row["strategy"]).split("-"),
                    pit_laps=[
                        int(token)
                        for token in str(row["pit_laps"]).split("-")
                        if str(token).strip()
                    ],
                ),
                axis=1,
            )
            result = result[valid_mask].reset_index(drop=True)
            result = result[OUTPUT_COLUMNS]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def main() -> None:
    parser = ArgumentParser(description="Generate strategy recommendations (Strategist Engine v2).")
    parser.add_argument(
        "--calibrated-predictions",
        default=str(DEFAULT_CALIBRATED_PATH),
        help="Path to calibrated degradation predictions CSV.",
    )
    parser.add_argument(
        "--race-setup",
        default=str(DEFAULT_RACE_SETUP_PATH),
        help="Path to race setup CSV with all drivers and scenario inputs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write strategy recommendations CSV.",
    )
    args = parser.parse_args()

    result = generate_strategy_recommendations(
        calibrated_predictions_path=args.calibrated_predictions,
        race_setup_path=args.race_setup,
        output_path=args.output,
    )

    print("Strategy recommendations generated.")
    print(f"- Rows: {len(result)}")
    print(f"- Output: {Path(args.output)}")

    if not result.empty:
        print(f"- Drivers: {result['driver'].nunique()}")
        print(f"- Scenarios: {result['scenario_name'].nunique()}")
        print(result.head(15))


if __name__ == "__main__":
    main()
