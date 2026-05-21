from typing import Any

import pandas as pd

from src.live.overtake_model import estimate_recovery_probability
from src.live.scenario_simulator import estimate_pit_cycle_position_loss
from src.simulation.race_simulator import (
    DEFAULT_BASE_PACE_SECONDS,
    DEFAULT_PIT_LOSS_GREEN_SECONDS,
    DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    DEFAULT_RACE_LAPS,
    estimate_strategy_risk_score,
    is_valid_dry_strategy,
    simulate_strategy_scenario,
)
from src.simulation.safety_car import get_scenario_names
from src.simulation.strategy_generator import generate_legal_strategy_candidates


FALLBACK_COMPOUND_DEGRADATION = {
    "SOFT": 0.120,
    "MEDIUM": 0.080,
    "HARD": 0.055,
}


def build_compound_degradation_map(
    driver_rows: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """
    Builds compound-level degradation map for one race+driver.

    Rules:
    - Use calibrated values where available.
    - Fill missing compounds with fixed fallback estimates.
    - If no calibrated rows exist, use fallback map for all compounds.
    """
    deg_map: dict[str, float] = {}
    confidence = 0.0

    if not driver_rows.empty:
        grouped = (
            driver_rows.groupby("compound", as_index=False)["calibrated_deg_estimate"]
            .median()
        )
        for _, row in grouped.iterrows():
            compound = str(row["compound"]).upper()
            deg_map[compound] = float(row["calibrated_deg_estimate"])

        if "practice_confidence" in driver_rows.columns and driver_rows["practice_confidence"].notna().any():
            confidence = float(driver_rows["practice_confidence"].max())

    for compound, fallback in FALLBACK_COMPOUND_DEGRADATION.items():
        deg_map.setdefault(compound, float(fallback))

    return deg_map, confidence


def rank_strategy_results(strategy_df: pd.DataFrame) -> pd.DataFrame:
    if strategy_df.empty:
        return strategy_df.copy()

    ranked = strategy_df.sort_values(
        ["scenario_name", "expected_total_time", "risk_score", "strategy"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    ranked["time_delta_to_best"] = ranked.groupby("scenario_name")["expected_total_time"].transform(
        lambda s: s - float(s.min())
    )

    ranked["is_recommended"] = False
    best_idx = ranked.groupby("scenario_name", as_index=False).head(1).index
    ranked.loc[best_idx, "is_recommended"] = True

    return ranked


def _estimate_strategy_recovery_probability(
    *,
    expected_drop_positions: int,
    starting_position: int,
    field_size: int,
    strategy_compounds: list[str],
    pit_laps: str,
    race_laps: int,
    overtaking_difficulty: float,
    track_position_importance: float,
) -> float:
    if expected_drop_positions <= 0:
        return 0.90

    laps = [
        int(token)
        for token in str(pit_laps).split("-")
        if str(token).strip().isdigit()
    ]
    first_pit_lap = laps[0] if laps else max(1, int(race_laps) // 2)
    remaining_after_first_stop = max(1, int(race_laps) - int(first_pit_lap))
    recovery_window = min(1.0, remaining_after_first_stop / max(1.0, float(race_laps)))

    uses_soft_after_first_stop = any(compound == "SOFT" for compound in strategy_compounds[1:])
    freshness_advantage = 0.18 + (0.12 if uses_soft_after_first_stop else 0.0)
    pace_delta_seconds = 0.20 + (0.10 if uses_soft_after_first_stop else 0.0)

    base_probability = 0.46 + (0.18 * recovery_window)
    return estimate_recovery_probability(
        base_overtake_probability=base_probability,
        expected_positions_to_recover=int(expected_drop_positions),
        current_position=int(starting_position),
        field_size=int(field_size),
        pace_delta_seconds=pace_delta_seconds,
        freshness_advantage=freshness_advantage,
        overtaking_difficulty=float(overtaking_difficulty),
        track_position_importance=float(track_position_importance),
    )


def apply_position_aware_ranking(
    strategy_df: pd.DataFrame,
    setup_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ranks candidates by projected race outcome, not solo lap-time sum alone.

    Modeling assumptions:
    - Candidate total time is still the primary pace signal.
    - Projected finish compares a candidate against each rival driver's best
      strategy in the same scenario.
    - Pit-cycle losses are only partially recoverable, depending on overtaking
      difficulty, track-position importance, tire freshness, and laps remaining.
    """
    if strategy_df.empty:
        return strategy_df.copy()

    ranked = strategy_df.copy()
    setup_lookup = (
        setup_df.assign(driver=setup_df["driver"].astype(str).str.strip().str.upper())
        .set_index("driver")
        .to_dict("index")
    )
    field_size = max(1, ranked["driver"].nunique())

    projected_finish_positions: list[int] = []
    position_deltas: list[int] = []
    outcome_scores: list[float] = []
    recovery_probabilities: list[float] = []
    expected_drop_positions_values: list[int] = []

    scenario_best_by_driver = (
        ranked.groupby(["scenario_name", "driver"])["expected_total_time"].min().to_dict()
    )

    for _, row in ranked.iterrows():
        driver = str(row["driver"]).upper()
        scenario_name = str(row["scenario_name"])
        setup = setup_lookup.get(driver, {})
        race_laps = int(setup.get("race_laps", DEFAULT_RACE_LAPS))
        overtaking_difficulty = float(
            row.get("overtaking_difficulty", setup.get("overtaking_difficulty", 0.7))
        )
        track_position_importance = float(setup.get("track_position_importance", 0.60))
        starting_position = int(row["starting_position"])
        pit_loss_used = float(row["pit_loss_used"])
        strategy_compounds = [
            compound.strip().upper()
            for compound in str(row["strategy"]).split("-")
            if compound.strip()
        ]

        rival_times = {
            other_driver: time
            for (other_scenario, other_driver), time in scenario_best_by_driver.items()
            if other_scenario == scenario_name and other_driver != driver
        }
        candidate_times = {**rival_times, driver: float(row["expected_total_time"])}
        time_rank = sorted(candidate_times, key=lambda code: (candidate_times[code], code)).index(driver) + 1

        expected_drop_positions = estimate_pit_cycle_position_loss(
            current_position=starting_position,
            pit_loss_seconds=pit_loss_used,
            overtaking_difficulty=overtaking_difficulty,
            track_position_importance=track_position_importance,
            field_size=field_size,
        )
        recovery_probability = _estimate_strategy_recovery_probability(
            expected_drop_positions=expected_drop_positions,
            starting_position=starting_position,
            field_size=field_size,
            strategy_compounds=strategy_compounds,
            pit_laps=str(row["pit_laps"]),
            race_laps=race_laps,
            overtaking_difficulty=overtaking_difficulty,
            track_position_importance=track_position_importance,
        )
        unrecovered_positions = max(
            0,
            int(round(float(expected_drop_positions) * (1.0 - recovery_probability))),
        )

        # Front-running cars on hard-to-pass tracks get extra credit for plans
        # that protect position, while slow total-time ranks still pull them down.
        track_protection_credit = 0
        if starting_position <= 6 and overtaking_difficulty >= 0.75 and track_position_importance >= 0.70:
            track_protection_credit = 1

        projected_finish = max(
            1,
            min(
                field_size,
                int(round((0.70 * float(time_rank)) + (0.30 * float(starting_position)))
                    + unrecovered_positions
                    - track_protection_credit),
            ),
        )
        position_delta = int(projected_finish - starting_position)

        outcome_score = (
            float(projected_finish)
            + (0.020 * float(row["time_delta_to_best"]))
            + (0.25 * float(row["risk_score"]))
            + (0.10 * max(0, position_delta))
        )

        projected_finish_positions.append(int(projected_finish))
        position_deltas.append(int(position_delta))
        outcome_scores.append(float(outcome_score))
        recovery_probabilities.append(float(recovery_probability))
        expected_drop_positions_values.append(int(expected_drop_positions))

    ranked["projected_finish_position"] = projected_finish_positions
    ranked["position_delta_from_start"] = position_deltas
    ranked["position_outcome_score"] = outcome_scores
    ranked["pit_recovery_probability"] = recovery_probabilities
    ranked["expected_pit_cycle_position_loss"] = expected_drop_positions_values

    ranked = ranked.sort_values(
        [
            "driver",
            "scenario_name",
            "position_outcome_score",
            "projected_finish_position",
            "expected_total_time",
            "risk_score",
            "strategy",
        ],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)

    ranked["is_recommended"] = False
    best_idx = ranked.groupby(["driver", "scenario_name"], as_index=False).head(1).index
    ranked.loc[best_idx, "is_recommended"] = True
    ranked["recommendation_reason"] = ranked.apply(_build_recommendation_reason, axis=1)
    return ranked


def _build_recommendation_reason(row: pd.Series) -> str:
    scenario_name = str(row["scenario_name"])
    is_recommended = bool(row["is_recommended"])
    stops = int(row["stops"])
    safety_gain = float(row["safety_car_gain_seconds"])
    best_sc_lap = row["best_safety_car_pit_lap"]
    starting_position = int(row["starting_position"])
    track_penalty = float(row["track_position_penalty"])
    projected_finish = row.get("projected_finish_position")
    recovery_probability = row.get("pit_recovery_probability")

    reasons: list[str] = []

    if is_recommended and scenario_name == "no_safety_car" and stops == 1:
        reasons.append("Best no-safety-car race time with one-stop strategy.")
    elif is_recommended and stops == 2:
        reasons.append(
            "Two-stop is competitive because lower tire degradation offsets extra pit loss."
        )
    elif is_recommended:
        reasons.append("Best expected race time for this scenario.")
    else:
        reasons.append("Alternative scenario plan benchmark.")

    if scenario_name != "no_safety_car" and safety_gain > 0.0 and pd.notna(best_sc_lap):
        reasons.append(
            "Safety car window creates a cheaper stop opportunity around "
            f"lap {int(float(best_sc_lap))}."
        )

    # Modeling assumption:
    # Poorer grid slots with meaningful track penalty favor conservative
    # track-position protection choices.
    if starting_position >= 8 and track_penalty > 5.0:
        reasons.append("High starting position penalty makes track position protection important.")

    if pd.notna(projected_finish):
        reasons.append(f"Projected finish: P{int(projected_finish)}.")

    if pd.notna(recovery_probability):
        reasons.append(f"Pit-cycle recovery probability: {float(recovery_probability):.2f}.")

    return " ".join(reasons)


def evaluate_driver_strategies(
    *,
    race: str,
    driver: str,
    team: str,
    starting_position: int,
    driver_rows: pd.DataFrame,
    starting_compound: str | None = None,
    race_laps: int = DEFAULT_RACE_LAPS,
    pit_loss_green_seconds: float = DEFAULT_PIT_LOSS_GREEN_SECONDS,
    pit_loss_safety_car_seconds: float = DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    base_pace_seconds: float | None = None,
    overtaking_difficulty: float = 0.7,
    safety_car_probability: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluates legal strategy candidates across multiple safety-car scenarios.
    """
    base_pace = (
        float(base_pace_seconds)
        if base_pace_seconds is not None
        else float(DEFAULT_BASE_PACE_SECONDS)
    )

    deg_map, practice_confidence = build_compound_degradation_map(driver_rows)
    scenarios = get_scenario_names()
    candidates = generate_legal_strategy_candidates(
        race_laps=race_laps,
        starting_compound=starting_compound,
    )

    rows: list[dict[str, Any]] = []

    for scenario_name in scenarios:
        for candidate in candidates:
            strategy = candidate["strategy"]
            compounds = candidate["compounds"]
            pit_options = candidate["pit_lap_options"]
            if not pit_options:
                continue

            best_result: dict[str, Any] | None = None
            for pit_laps in pit_options:
                if not is_valid_dry_strategy(
                    strategy_compounds=compounds,
                    pit_laps=pit_laps,
                ):
                    # Skip invalid dry no-stop / single-compound plans.
                    continue

                sim = simulate_strategy_scenario(
                    race_laps=int(race_laps),
                    strategy_compounds=compounds,
                    pit_laps=pit_laps,
                    base_pace_seconds=base_pace,
                    pit_loss_green_seconds=float(pit_loss_green_seconds),
                    pit_loss_safety_car_seconds=float(pit_loss_safety_car_seconds),
                    compound_degradation_map=deg_map,
                    fallback_degradation=FALLBACK_COMPOUND_DEGRADATION["MEDIUM"],
                    starting_position=int(starting_position),
                    overtaking_difficulty=float(overtaking_difficulty),
                    scenario_name=scenario_name,
                )

                if (
                    best_result is None
                    or float(sim["expected_total_time"]) < float(best_result["expected_total_time"])
                ):
                    best_result = sim

            if best_result is None:
                continue

            stops = int(candidate["stops"])
            risk_score = estimate_strategy_risk_score(
                strategy_compounds=compounds,
                num_stops=stops,
                practice_confidence=practice_confidence,
                overtaking_difficulty=float(overtaking_difficulty),
                safety_car_probability=float(safety_car_probability),
            )

            rows.append(
                {
                    "race": race,
                    "driver": driver,
                    "team": team,
                    "starting_position": int(starting_position),
                    "strategy": strategy,
                    "pit_laps": str(best_result["pit_laps"]),
                    "scenario_name": scenario_name,
                    "expected_total_time": float(best_result["expected_total_time"]),
                    "risk_score": float(risk_score),
                    "stops": stops,
                    "starting_compound": str(candidate["starting_compound"]),
                    "best_safety_car_pit_lap": best_result["best_safety_car_pit_lap"],
                    "safety_car_gain_seconds": float(best_result["safety_car_gain_seconds"]),
                    "track_position_penalty": float(best_result["track_position_penalty"]),
                    "pit_loss_used": float(best_result["pit_loss_used"]),
                    "overtaking_difficulty": float(overtaking_difficulty),
                }
            )

    evaluated = pd.DataFrame(rows)
    ranked = rank_strategy_results(evaluated)

    if ranked.empty:
        return ranked

    ranked["recommendation_reason"] = ranked.apply(_build_recommendation_reason, axis=1)

    return ranked[
        [
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
            "recommendation_reason",
        ]
    ]
