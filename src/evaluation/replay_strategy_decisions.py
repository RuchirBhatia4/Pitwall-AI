from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.pit_decision_engine import (
    build_initial_decision_context,
    evaluate_pit_decisions_for_lap,
)
from src.live.race_state import DriverRaceState, LiveRaceState
from src.pipeline.run_live_race_simulation import (
    DEFAULT_PIT_LOSS_GREEN_SECONDS,
    DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    DEFAULT_TRACK_POSITION_IMPORTANCE,
    _build_driver_setup_map,
)
from src.live.scenario_simulator import is_dry_compound


DEFAULT_CLEAN_LAPS_PATH = Path("data/processed/historical_laps_clean.csv")
DEFAULT_SETUP_PATH = Path("data/raw/race_setup/race_setup_2026_japan_actual.csv")
DEFAULT_OUTPUT_PATH = Path("reports/post_race_evaluations/japan_strategy_replay.csv")
DEFAULT_SUMMARY_PATH = Path("reports/post_race_evaluations/japan_strategy_replay_summary.csv")

OUTPUT_COLUMNS = [
    "race",
    "season",
    "lap",
    "driver",
    "team",
    "actual_position",
    "current_compound",
    "tire_age",
    "stops_made",
    "gap_to_leader_seconds",
    "gap_to_ahead_seconds",
    "gap_to_behind_seconds",
    "projected_total_time_if_pit_now",
    "projected_total_time_if_stay_out",
    "pit_now_gain_seconds",
    "projected_finish_if_pit_now",
    "projected_finish_if_stay_out",
    "overtake_probability_if_pit_now",
    "overtake_probability_if_stay_out",
    "best_action_now",
    "actual_first_pit_lap",
    "model_first_pit_now_lap",
    "confirmed_first_pit_call_lap",
    "first_pit_lap_error",
    "confirmed_first_pit_lap_error",
    "first_stop_signal_status",
    "confirmed_signal_status",
    "recommendation_reason",
]

SUMMARY_COLUMNS = [
    "race",
    "season",
    "driver",
    "team",
    "actual_first_pit_lap",
    "model_first_pit_now_lap",
    "confirmed_first_pit_call_lap",
    "first_pit_lap_error",
    "confirmed_first_pit_lap_error",
    "first_stop_signal_status",
    "confirmed_signal_status",
    "last_pre_stop_model_action",
    "actual_final_position",
    "model_projected_finish_at_signal",
    "mean_gap_to_ahead_seconds",
    "mean_pit_recovery_probability",
]


def _normalize_laps(df: pd.DataFrame, race: str, season: int) -> pd.DataFrame:
    required = [
        "season",
        "race",
        "Driver",
        "Team",
        "LapNumber",
        "Position",
        "Compound",
        "TyreLife",
        "Stint",
        "lap_time_seconds",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Clean lap data is missing required columns: {missing}")

    laps = df[
        (df["season"] == int(season))
        & (df["race"].astype(str).str.lower() == str(race).lower())
    ].copy()
    if laps.empty:
        raise ValueError(f"No clean lap rows found for {season} {race}.")

    laps["driver"] = laps["Driver"].astype(str).str.strip().str.upper()
    laps["team"] = laps["Team"].astype(str).str.strip()
    laps["lap"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["actual_position"] = pd.to_numeric(laps["Position"], errors="coerce")
    laps["current_compound"] = laps["Compound"].astype(str).str.strip().str.upper()
    laps["tire_age"] = pd.to_numeric(laps["TyreLife"], errors="coerce")
    laps["stint"] = pd.to_numeric(laps["Stint"], errors="coerce")
    laps["lap_time_seconds"] = pd.to_numeric(laps["lap_time_seconds"], errors="coerce")

    laps = laps.dropna(
        subset=[
            "driver",
            "team",
            "lap",
            "actual_position",
            "current_compound",
            "tire_age",
            "stint",
            "lap_time_seconds",
        ]
    ).copy()

    laps["lap"] = laps["lap"].astype(int)
    laps["actual_position"] = laps["actual_position"].astype(int)
    laps["tire_age"] = laps["tire_age"].astype(int)
    laps["stint"] = laps["stint"].astype(int)
    return laps.sort_values(["lap", "actual_position", "driver"]).reset_index(drop=True)


def add_gap_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Adds inferred race gaps from cumulative clean lap time.

    Modeling assumption:
    Cleaned lap files do not include official interval/gap telemetry, so replay
    uses cumulative observed lap time as an approximate gap signal. This is good
    enough to separate "car ahead is close" from "car ahead is far away", but it
    is not a replacement for official timing-loop intervals.
    """
    enriched = laps.copy()
    enriched["elapsed_time_seconds"] = (
        enriched.sort_values(["driver", "lap"])
        .groupby("driver")["lap_time_seconds"]
        .cumsum()
    )

    frames: list[pd.DataFrame] = []
    for _, lap_group in enriched.groupby("lap", sort=True):
        ordered = lap_group.sort_values(["actual_position", "elapsed_time_seconds"]).copy()
        leader_time = float(ordered["elapsed_time_seconds"].min())
        ordered["gap_to_leader_seconds"] = ordered["elapsed_time_seconds"] - leader_time
        ordered["gap_to_ahead_seconds"] = (
            ordered["elapsed_time_seconds"] - ordered["elapsed_time_seconds"].shift(1)
        )
        ordered["gap_to_behind_seconds"] = (
            ordered["elapsed_time_seconds"].shift(-1) - ordered["elapsed_time_seconds"]
        )
        ordered.loc[ordered["actual_position"] <= 1, "gap_to_ahead_seconds"] = 0.0
        ordered["gap_to_ahead_seconds"] = ordered["gap_to_ahead_seconds"].clip(lower=0.0)
        ordered["gap_to_behind_seconds"] = ordered["gap_to_behind_seconds"].fillna(999.0).clip(lower=0.0)
        frames.append(ordered)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["lap", "actual_position", "driver"]
    ).reset_index(drop=True)


def _build_setup_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    race = str(laps["race"].iloc[0])
    race_laps = int(laps["lap"].max())
    for (driver, team), group in laps.groupby(["driver", "team"], sort=True):
        ordered = group.sort_values("lap")
        first = ordered.iloc[0]
        pace_sample = ordered[
            (ordered["tire_age"].between(3, 12)) & ordered["lap_time_seconds"].notna()
        ]
        if pace_sample.empty:
            pace_sample = ordered[ordered["lap_time_seconds"].notna()]

        rows.append(
            {
                "race": race,
                "driver": driver,
                "team": team,
                "starting_position": int(first["actual_position"]),
                "starting_compound": str(first["current_compound"]).upper(),
                "race_laps": race_laps,
                "base_pace_seconds": float(pace_sample["lap_time_seconds"].quantile(0.30)),
                "pit_loss_green_seconds": DEFAULT_PIT_LOSS_GREEN_SECONDS,
                "pit_loss_safety_car_seconds": DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
                "overtaking_difficulty": 0.78,
                "track_position_importance": DEFAULT_TRACK_POSITION_IMPORTANCE,
                "setup_source": "post_race_replay_clean_laps",
            }
        )

    setup = pd.DataFrame(rows).sort_values(["starting_position", "driver"]).reset_index(drop=True)
    setup["starting_position"] = range(1, len(setup) + 1)
    return setup


def _load_or_build_setup(setup_path: str | Path | None, laps: pd.DataFrame) -> pd.DataFrame:
    if setup_path is not None and Path(setup_path).exists():
        setup = pd.read_csv(setup_path)
    else:
        setup = _build_setup_from_laps(laps)

    setup = setup.copy()
    setup["race"] = setup["race"].astype(str).str.strip()
    setup["driver"] = setup["driver"].astype(str).str.strip().str.upper()
    setup["team"] = setup["team"].astype(str).str.strip()
    setup["starting_compound"] = setup["starting_compound"].astype(str).str.strip().str.upper()

    numeric_defaults = {
        "starting_position": 10,
        "race_laps": int(laps["lap"].max()),
        "base_pace_seconds": float(laps["lap_time_seconds"].median()),
        "pit_loss_green_seconds": DEFAULT_PIT_LOSS_GREEN_SECONDS,
        "pit_loss_safety_car_seconds": DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
        "overtaking_difficulty": 0.78,
        "track_position_importance": DEFAULT_TRACK_POSITION_IMPORTANCE,
    }
    for col, default in numeric_defaults.items():
        if col not in setup.columns:
            setup[col] = default
        setup[col] = pd.to_numeric(setup[col], errors="coerce").fillna(default)

    return setup


def _actual_first_pit_laps(laps: pd.DataFrame) -> dict[str, int]:
    pit_laps: dict[str, int] = {}
    for driver, group in laps.groupby("driver"):
        stint_starts = group.groupby("stint")["lap"].min().sort_index()
        if len(stint_starts) <= 1:
            continue
        first_new_stint_lap = int(stint_starts.iloc[1])
        pit_laps[str(driver).upper()] = max(1, first_new_stint_lap - 1)
    return pit_laps


def _final_positions(laps: pd.DataFrame) -> dict[str, int]:
    positions: dict[str, int] = {}
    for driver, group in laps.groupby("driver"):
        last = group.sort_values("lap").iloc[-1]
        positions[str(driver).upper()] = int(last["actual_position"])
    return positions


def _compounds_used_through_lap(laps: pd.DataFrame, lap: int) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    past = laps[laps["lap"] <= int(lap)].copy()
    for driver, group in past.groupby("driver"):
        used[str(driver).upper()] = {
            str(compound).upper()
            for compound in group["current_compound"].dropna().astype(str)
            if str(compound).strip()
        }
    return used


def _seed_context_from_observed_state(
    *,
    context: dict[str, dict[str, Any]],
    compounds_used: dict[str, set[str]],
) -> None:
    for driver, entry in context.items():
        used = compounds_used.get(driver, set())
        if not used:
            continue

        is_dry_race = all(is_dry_compound(compound) for compound in used)
        dry_used = {compound for compound in used if is_dry_compound(compound)}
        entry["compounds_used"] = used
        entry["is_dry_race"] = is_dry_race
        entry["mandatory_stop_pending"] = bool(is_dry_race and len(dry_used) < 2)


def _state_from_lap_group(
    *,
    race: str,
    season: int,
    race_laps: int,
    setup_source: str,
    lap: int,
    lap_group: pd.DataFrame,
) -> LiveRaceState:
    drivers: dict[str, DriverRaceState] = {}
    for _, row in lap_group.iterrows():
        driver = str(row["driver"]).upper()
        drivers[driver] = DriverRaceState(
            race=str(race),
            driver=driver,
            team=str(row["team"]),
            current_position=int(row["actual_position"]),
            current_compound=str(row["current_compound"]).upper(),
            tire_age=int(row["tire_age"]),
            stops_made=max(0, int(row["stint"]) - 1),
            in_race=True,
        )

    return LiveRaceState(
        race=str(race),
        lap=int(lap),
        race_laps=int(race_laps),
        under_safety_car=False,
        setup_source=str(setup_source),
        drivers=drivers,
    )


def replay_strategy_decisions(
    *,
    clean_laps_path: str | Path = DEFAULT_CLEAN_LAPS_PATH,
    setup_path: str | Path | None = DEFAULT_SETUP_PATH,
    race: str = "Japan",
    season: int = 2026,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_laps = pd.read_csv(clean_laps_path)
    laps = _normalize_laps(clean_laps, race=race, season=season)
    laps = add_gap_features(laps)
    setup = _load_or_build_setup(setup_path=setup_path, laps=laps)

    race_laps = int(setup["race_laps"].max()) if not setup.empty else int(laps["lap"].max())
    setup_source = (
        str(setup["setup_source"].mode().iloc[0])
        if "setup_source" in setup.columns and not setup["setup_source"].empty
        else "post_race_replay"
    )
    base_driver_setup = _build_driver_setup_map(setup)
    actual_first_pits = _actual_first_pit_laps(laps)
    final_positions = _final_positions(laps)

    rows: list[dict[str, Any]] = []
    first_model_signal: dict[str, int] = {}
    confirmed_model_signal: dict[str, int] = {}
    projected_finish_at_signal: dict[str, int] = {}
    projected_finish_at_confirmed_signal: dict[str, int] = {}
    last_pre_stop_action: dict[str, str] = {}
    pit_now_streak: dict[str, int] = {}

    for lap, lap_group in laps.groupby("lap", sort=True):
        race_state = _state_from_lap_group(
            race=race,
            season=season,
            race_laps=race_laps,
            setup_source=setup_source,
            lap=int(lap),
            lap_group=lap_group,
        )
        context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=base_driver_setup,
        )
        _seed_context_from_observed_state(
            context=context,
            compounds_used=_compounds_used_through_lap(laps, int(lap)),
        )
        driver_setup = {driver: dict(values) for driver, values in base_driver_setup.items()}
        for _, gap_row in lap_group.iterrows():
            driver = str(gap_row["driver"]).upper()
            if driver in driver_setup:
                driver_setup[driver]["gap_to_ahead_seconds"] = float(gap_row["gap_to_ahead_seconds"])
                driver_setup[driver]["gap_to_behind_seconds"] = float(gap_row["gap_to_behind_seconds"])

        decisions = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=context,
        )

        for _, row in lap_group.iterrows():
            driver = str(row["driver"]).upper()
            if driver not in decisions:
                continue
            decision = decisions[driver]
            actual_first = actual_first_pits.get(driver)
            is_before_or_at_actual_first_stop = (
                actual_first is None or int(lap) <= int(actual_first)
            )
            if is_before_or_at_actual_first_stop:
                last_pre_stop_action[driver] = str(decision["best_action_now"])
                if decision["best_action_now"] == "pit_now":
                    pit_now_streak[driver] = pit_now_streak.get(driver, 0) + 1
                else:
                    pit_now_streak[driver] = 0

            if (
                is_before_or_at_actual_first_stop
                and decision["best_action_now"] == "pit_now"
                and driver not in first_model_signal
            ):
                first_model_signal[driver] = int(lap)
                projected_finish_at_signal[driver] = int(decision["projected_finish_if_pit_now"])

            if (
                is_before_or_at_actual_first_stop
                and pit_now_streak.get(driver, 0) >= 2
                and driver not in confirmed_model_signal
            ):
                confirmed_model_signal[driver] = int(lap)
                projected_finish_at_confirmed_signal[driver] = int(
                    decision["projected_finish_if_pit_now"]
                )

            model_first = first_model_signal.get(driver)
            confirmed_first = confirmed_model_signal.get(driver)
            if model_first is not None and actual_first is not None:
                signal_status = "pre_stop_signal"
            elif actual_first is not None:
                signal_status = "no_signal_before_actual_stop"
            else:
                signal_status = "no_actual_stop_reference"
            if confirmed_first is not None and actual_first is not None:
                confirmed_status = "confirmed_pre_stop_signal"
            elif actual_first is not None:
                confirmed_status = "no_confirmed_signal_before_actual_stop"
            else:
                confirmed_status = "no_actual_stop_reference"
            rows.append(
                {
                    "race": race,
                    "season": int(season),
                    "lap": int(lap),
                    "driver": driver,
                    "team": str(row["team"]),
                    "actual_position": int(row["actual_position"]),
                    "current_compound": str(row["current_compound"]).upper(),
                    "tire_age": int(row["tire_age"]),
                    "stops_made": max(0, int(row["stint"]) - 1),
                    "gap_to_leader_seconds": float(row["gap_to_leader_seconds"]),
                    "gap_to_ahead_seconds": float(row["gap_to_ahead_seconds"]),
                    "gap_to_behind_seconds": float(row["gap_to_behind_seconds"]),
                    "projected_total_time_if_pit_now": float(
                        decision["projected_total_time_if_pit_now"]
                    ),
                    "projected_total_time_if_stay_out": float(
                        decision["projected_total_time_if_stay_out"]
                    ),
                    "pit_now_gain_seconds": float(decision["pit_now_gain_seconds"]),
                    "projected_finish_if_pit_now": int(decision["projected_finish_if_pit_now"]),
                    "projected_finish_if_stay_out": int(decision["projected_finish_if_stay_out"]),
                    "overtake_probability_if_pit_now": float(
                        decision["overtake_probability_if_pit_now"]
                    ),
                    "overtake_probability_if_stay_out": float(
                        decision["overtake_probability_if_stay_out"]
                    ),
                    "best_action_now": str(decision["best_action_now"]),
                    "actual_first_pit_lap": actual_first,
                    "model_first_pit_now_lap": model_first,
                    "confirmed_first_pit_call_lap": confirmed_first,
                    "first_pit_lap_error": (
                        int(model_first) - int(actual_first)
                        if model_first is not None and actual_first is not None
                        else pd.NA
                    ),
                    "confirmed_first_pit_lap_error": (
                        int(confirmed_first) - int(actual_first)
                        if confirmed_first is not None and actual_first is not None
                        else pd.NA
                    ),
                    "first_stop_signal_status": signal_status,
                    "confirmed_signal_status": confirmed_status,
                    "recommendation_reason": str(decision["recommendation_reason"]),
                }
            )

    replay = pd.DataFrame(rows)
    if replay.empty:
        replay = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        replay = replay[OUTPUT_COLUMNS].copy()

    summary_rows: list[dict[str, Any]] = []
    for driver, group in replay.groupby("driver", sort=True):
        actual_first = actual_first_pits.get(driver)
        model_first = first_model_signal.get(driver)
        confirmed_first = confirmed_model_signal.get(driver)
        signal_rows = group[group["lap"] == model_first] if model_first is not None else pd.DataFrame()
        signal_finish = (
            int(signal_rows["projected_finish_if_pit_now"].iloc[0])
            if not signal_rows.empty
            else projected_finish_at_signal.get(driver)
        )
        confirmed_signal_rows = (
            group[group["lap"] == confirmed_first] if confirmed_first is not None else pd.DataFrame()
        )
        confirmed_signal_finish = (
            int(confirmed_signal_rows["projected_finish_if_pit_now"].iloc[0])
            if not confirmed_signal_rows.empty
            else projected_finish_at_confirmed_signal.get(driver)
        )
        summary_rows.append(
            {
                "race": race,
                "season": int(season),
                "driver": driver,
                "team": str(group["team"].iloc[0]),
                "actual_first_pit_lap": actual_first,
                "model_first_pit_now_lap": model_first,
                "confirmed_first_pit_call_lap": confirmed_first,
                "first_pit_lap_error": (
                    int(model_first) - int(actual_first)
                    if model_first is not None and actual_first is not None
                    else pd.NA
                ),
                "confirmed_first_pit_lap_error": (
                    int(confirmed_first) - int(actual_first)
                    if confirmed_first is not None and actual_first is not None
                    else pd.NA
                ),
                "first_stop_signal_status": (
                    "pre_stop_signal"
                    if model_first is not None and actual_first is not None
                    else (
                        "no_signal_before_actual_stop"
                        if actual_first is not None
                        else "no_actual_stop_reference"
                    )
                ),
                "confirmed_signal_status": (
                    "confirmed_pre_stop_signal"
                    if confirmed_first is not None and actual_first is not None
                    else (
                        "no_confirmed_signal_before_actual_stop"
                        if actual_first is not None
                        else "no_actual_stop_reference"
                    )
                ),
                "last_pre_stop_model_action": last_pre_stop_action.get(driver),
                "actual_final_position": final_positions.get(driver),
                "model_projected_finish_at_signal": confirmed_signal_finish or signal_finish,
                "mean_gap_to_ahead_seconds": float(group["gap_to_ahead_seconds"].mean()),
                "mean_pit_recovery_probability": float(
                    group["overtake_probability_if_pit_now"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        summary = summary[SUMMARY_COLUMNS].sort_values(
            ["actual_final_position", "driver"],
            na_position="last",
        ).reset_index(drop=True)

    output_file = Path(output_path)
    summary_file = Path(summary_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output_file, index=False)
    summary.to_csv(summary_file, index=False)

    return replay, summary


def main() -> None:
    parser = ArgumentParser(description="Replay post-race strategy decisions with inferred gaps.")
    parser.add_argument("--clean-laps", default=str(DEFAULT_CLEAN_LAPS_PATH))
    parser.add_argument("--setup", default=str(DEFAULT_SETUP_PATH))
    parser.add_argument("--race", default="Japan")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    args = parser.parse_args()

    replay, summary = replay_strategy_decisions(
        clean_laps_path=args.clean_laps,
        setup_path=args.setup,
        race=args.race,
        season=args.season,
        output_path=args.output,
        summary_path=args.summary_output,
    )

    print("Strategy replay generated.")
    print(f"- Replay rows: {len(replay)}")
    print(f"- Summary rows: {len(summary)}")
    print(f"- Output: {Path(args.output)}")
    print(f"- Summary: {Path(args.summary_output)}")
    if not summary.empty:
        valid_errors = pd.to_numeric(summary["first_pit_lap_error"], errors="coerce").dropna()
        if not valid_errors.empty:
            print(f"- Median abs first-pit error: {valid_errors.abs().median():.2f} laps")
        confirmed_errors = pd.to_numeric(
            summary["confirmed_first_pit_lap_error"],
            errors="coerce",
        ).dropna()
        if not confirmed_errors.empty:
            print(
                "- Median abs confirmed first-pit error: "
                f"{confirmed_errors.abs().median():.2f} laps"
            )
        print(summary.head(12))


if __name__ == "__main__":
    main()
