from typing import Mapping, Sequence

import numpy as np

from src.simulation.safety_car import (
    best_window_lap,
    get_safety_car_window,
    lap_in_window,
)
from src.simulation.strategy_generator import (
    DEFAULT_MAX_STINT_LAPS_BY_COMPOUND,
    MIN_FIRST_STINT_LAPS,
    MIN_LAST_STINT_LAPS,
    MIN_SECOND_STOP_GAP,
)
from src.simulation.track_position import estimate_track_position_penalty_per_lap


DEFAULT_BASE_PACE_SECONDS = 92.0
DEFAULT_PIT_LOSS_SECONDS = 22.0
DEFAULT_PIT_LOSS_GREEN_SECONDS = DEFAULT_PIT_LOSS_SECONDS
DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS = 13.0
DEFAULT_RACE_LAPS = 57

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def _sanitize_degradation_estimate(value: float) -> float:
    numeric = float(value)
    return max(0.0, numeric)


def _is_dry_race_strategy(strategy_compounds: Sequence[str]) -> bool:
    compounds = [str(compound).upper() for compound in strategy_compounds]
    return bool(compounds) and all(compound in DRY_COMPOUNDS for compound in compounds)


def _has_mandatory_dry_compound_change(strategy_compounds: Sequence[str]) -> bool:
    dry_used = {str(compound).upper() for compound in strategy_compounds if str(compound).upper() in DRY_COMPOUNDS}
    return len(dry_used) >= 2


def _validate_dry_strategy_rule(
    *,
    strategy_compounds: Sequence[str],
    pit_laps: Sequence[int],
) -> None:
    # FIA dry-race assumption in this simulator:
    # at least one stop and at least two distinct dry compounds.
    if not _is_dry_race_strategy(strategy_compounds):
        return

    if len(pit_laps) == 0:
        raise ValueError("Invalid dry strategy: no-stop single-compound runs are not allowed.")

    if not _has_mandatory_dry_compound_change(strategy_compounds):
        raise ValueError(
            "Invalid dry strategy: must use at least two distinct dry compounds."
        )


def is_valid_dry_strategy(
    *,
    strategy_compounds: Sequence[str],
    pit_laps: Sequence[int],
) -> bool:
    try:
        _validate_dry_strategy_rule(
            strategy_compounds=strategy_compounds,
            pit_laps=pit_laps,
        )
    except ValueError:
        return False
    return True


def _stint_lengths_from_pit_laps(race_laps: int, pit_laps: Sequence[int]) -> list[int]:
    pits = sorted(int(lap) for lap in pit_laps)
    boundaries = [0] + pits + [int(race_laps)]
    return [boundaries[idx + 1] - boundaries[idx] for idx in range(len(boundaries) - 1)]


def _is_valid_pit_schedule(
    race_laps: int,
    pit_laps: Sequence[int],
    strategy_compounds: Sequence[str] | None = None,
) -> bool:
    pits = sorted(int(lap) for lap in pit_laps)
    if pits != list(pit_laps):
        return False
    if len(set(pits)) != len(pits):
        return False

    if any(lap < MIN_FIRST_STINT_LAPS for lap in pits):
        return False
    if pits and (race_laps - pits[-1] < MIN_LAST_STINT_LAPS):
        return False

    if len(pits) == 2 and (pits[1] - pits[0] < MIN_SECOND_STOP_GAP):
        return False

    if any(lap >= race_laps for lap in pits):
        return False

    if strategy_compounds is not None:
        boundaries = [0] + pits + [int(race_laps)]
        stint_lengths = [
            boundaries[idx + 1] - boundaries[idx]
            for idx in range(len(boundaries) - 1)
        ]
        compounds = [str(compound).upper() for compound in strategy_compounds]
        if len(stint_lengths) != len(compounds):
            return False

        # Modeling assumption:
        # Strategies should stay inside plausible compound working ranges.
        # The lap-time model is linear, so this guard represents tire cliff,
        # graining, overheating, and drivability constraints that are not yet
        # directly learned from practice/race data.
        for compound, stint_length in zip(compounds, stint_lengths):
            max_laps = DEFAULT_MAX_STINT_LAPS_BY_COMPOUND.get(compound)
            if max_laps is not None and int(stint_length) > int(max_laps):
                return False

    return True


def _estimate_traffic_penalty_seconds(
    race_laps: int,
    pit_laps: Sequence[int],
) -> float:
    """
    Modeling assumption:
    - Early pit stops increase traffic exposure.
    - Very short stints increase rejoin traffic risk.
    """
    pits = sorted(int(lap) for lap in pit_laps)
    stint_lengths = _stint_lengths_from_pit_laps(race_laps, pits)

    traffic = 0.0
    for pit in pits:
        if pit <= 12:
            traffic += 1.25
        elif pit <= 20:
            traffic += 0.70

    for stint in stint_lengths:
        if stint < 10:
            traffic += (10 - stint) * 0.20

    return float(traffic)


def _compute_lap_time_total(
    *,
    race_laps: int,
    strategy_compounds: Sequence[str],
    pit_laps: Sequence[int],
    base_pace_seconds: float,
    pit_loss_green_seconds: float,
    pit_loss_safety_car_seconds: float,
    compound_degradation_map: Mapping[str, float],
    fallback_degradation: float,
    starting_position: int,
    overtaking_difficulty: float,
    scenario_name: str,
) -> dict[str, float]:
    num_stints = len(strategy_compounds)
    expected_stops = max(0, num_stints - 1)

    pits = [int(lap) for lap in pit_laps]
    if len(pits) != expected_stops:
        raise ValueError(
            f"Invalid pit_laps length for strategy {strategy_compounds}. "
            f"Expected {expected_stops}, got {len(pits)}."
        )

    if not _is_valid_pit_schedule(
        race_laps=race_laps,
        pit_laps=pits,
        strategy_compounds=strategy_compounds,
    ):
        raise ValueError(f"Invalid pit schedule for {race_laps} laps: {pits}")

    track_position_penalty_per_lap = estimate_track_position_penalty_per_lap(
        starting_position=int(starting_position),
        overtaking_difficulty=float(overtaking_difficulty),
    )

    scenario_window = get_safety_car_window(scenario_name)

    total_time = 0.0
    pit_loss_used = 0.0

    pit_set = set(pits)
    stint_index = 0
    tyre_age = 0

    for lap in range(1, int(race_laps) + 1):
        compound = strategy_compounds[stint_index].upper()
        raw_deg = float(compound_degradation_map.get(compound, fallback_degradation))
        deg_per_lap = _sanitize_degradation_estimate(raw_deg)

        lap_time = (
            float(base_pace_seconds)
            + (deg_per_lap * float(tyre_age))
            + track_position_penalty_per_lap
        )

        total_time += float(lap_time)
        tyre_age += 1

        if lap in pit_set:
            pit_loss = (
                float(pit_loss_safety_car_seconds)
                if lap_in_window(lap, scenario_window)
                else float(pit_loss_green_seconds)
            )
            total_time += pit_loss
            pit_loss_used += pit_loss

            stint_index += 1
            tyre_age = 0

    traffic_penalty = _estimate_traffic_penalty_seconds(race_laps=race_laps, pit_laps=pits)
    total_time += traffic_penalty

    return {
        "expected_total_time": float(total_time),
        "pit_loss_used": float(pit_loss_used),
        "track_position_penalty": float(track_position_penalty_per_lap * race_laps),
        "traffic_penalty": float(traffic_penalty),
    }


def _nearest_window_candidate(
    *,
    race_laps: int,
    pit_laps: Sequence[int],
    strategy_compounds: Sequence[str],
    window: tuple[int, int] | None,
) -> list[int]:
    if window is None:
        return [int(lap) for lap in pit_laps]

    pits = [int(lap) for lap in pit_laps]
    if not pits:
        return pits

    # If an existing stop is already in the safety-car window, keep schedule.
    if any(lap_in_window(lap, window) for lap in pits):
        return pits

    start, end = window
    center = (start + end) / 2.0

    target_index = min(range(len(pits)), key=lambda idx: abs(pits[idx] - center))

    feasible_candidates: list[list[int]] = []
    for candidate_lap in range(int(start), int(end) + 1):
        proposal = pits.copy()
        proposal[target_index] = int(candidate_lap)
        proposal = sorted(proposal)
        if _is_valid_pit_schedule(
            race_laps=race_laps,
            pit_laps=proposal,
            strategy_compounds=strategy_compounds,
        ):
            feasible_candidates.append(proposal)

    if not feasible_candidates:
        return pits

    return min(
        feasible_candidates,
        key=lambda proposal: abs(proposal[target_index] - pits[target_index]),
    )


def simulate_strategy_scenario(
    *,
    race_laps: int,
    strategy_compounds: Sequence[str],
    pit_laps: Sequence[int],
    base_pace_seconds: float,
    pit_loss_green_seconds: float,
    pit_loss_safety_car_seconds: float,
    compound_degradation_map: Mapping[str, float],
    fallback_degradation: float,
    starting_position: int,
    overtaking_difficulty: float,
    scenario_name: str,
) -> dict[str, float | int | str]:
    _validate_dry_strategy_rule(
        strategy_compounds=strategy_compounds,
        pit_laps=pit_laps,
    )

    if scenario_name == "no_safety_car":
        baseline = _compute_lap_time_total(
            race_laps=race_laps,
            strategy_compounds=strategy_compounds,
            pit_laps=pit_laps,
            base_pace_seconds=base_pace_seconds,
            pit_loss_green_seconds=pit_loss_green_seconds,
            pit_loss_safety_car_seconds=pit_loss_safety_car_seconds,
            compound_degradation_map=compound_degradation_map,
            fallback_degradation=fallback_degradation,
            starting_position=starting_position,
            overtaking_difficulty=overtaking_difficulty,
            scenario_name="no_safety_car",
        )

        return {
            **baseline,
            "scenario_name": scenario_name,
            "best_safety_car_pit_lap": np.nan,
            "safety_car_gain_seconds": 0.0,
            "pit_laps": "-".join(str(int(lap)) for lap in pit_laps),
        }

    baseline_no_sc = _compute_lap_time_total(
        race_laps=race_laps,
        strategy_compounds=strategy_compounds,
        pit_laps=pit_laps,
        base_pace_seconds=base_pace_seconds,
        pit_loss_green_seconds=pit_loss_green_seconds,
        pit_loss_safety_car_seconds=pit_loss_safety_car_seconds,
        compound_degradation_map=compound_degradation_map,
        fallback_degradation=fallback_degradation,
        starting_position=starting_position,
        overtaking_difficulty=overtaking_difficulty,
        scenario_name="no_safety_car",
    )

    scenario_window = get_safety_car_window(scenario_name)

    original_scenario = _compute_lap_time_total(
        race_laps=race_laps,
        strategy_compounds=strategy_compounds,
        pit_laps=pit_laps,
        base_pace_seconds=base_pace_seconds,
        pit_loss_green_seconds=pit_loss_green_seconds,
        pit_loss_safety_car_seconds=pit_loss_safety_car_seconds,
        compound_degradation_map=compound_degradation_map,
        fallback_degradation=fallback_degradation,
        starting_position=starting_position,
        overtaking_difficulty=overtaking_difficulty,
        scenario_name=scenario_name,
    )

    adjusted_pits = _nearest_window_candidate(
        race_laps=race_laps,
        pit_laps=pit_laps,
        strategy_compounds=strategy_compounds,
        window=scenario_window,
    )

    adjusted_scenario = _compute_lap_time_total(
        race_laps=race_laps,
        strategy_compounds=strategy_compounds,
        pit_laps=adjusted_pits,
        base_pace_seconds=base_pace_seconds,
        pit_loss_green_seconds=pit_loss_green_seconds,
        pit_loss_safety_car_seconds=pit_loss_safety_car_seconds,
        compound_degradation_map=compound_degradation_map,
        fallback_degradation=fallback_degradation,
        starting_position=starting_position,
        overtaking_difficulty=overtaking_difficulty,
        scenario_name=scenario_name,
    )

    if adjusted_scenario["expected_total_time"] < original_scenario["expected_total_time"]:
        chosen = adjusted_scenario
        chosen_pits = adjusted_pits
    else:
        chosen = original_scenario
        chosen_pits = [int(lap) for lap in pit_laps]

    best_pit = best_window_lap(chosen_pits, scenario_window)

    return {
        **chosen,
        "scenario_name": scenario_name,
        "best_safety_car_pit_lap": (int(best_pit) if best_pit is not None else np.nan),
        "safety_car_gain_seconds": float(
            baseline_no_sc["expected_total_time"] - chosen["expected_total_time"]
        ),
        "pit_laps": "-".join(str(int(lap)) for lap in chosen_pits),
    }


def simulate_strategy_time(
    *,
    race_laps: int,
    strategy_compounds: Sequence[str],
    pit_laps: Sequence[int],
    base_pace_seconds: float,
    pit_loss_seconds: float,
    compound_degradation_map: Mapping[str, float],
    fallback_degradation: float,
) -> float:
    """
    Backward-compatible wrapper for basic simulation calls.
    """
    _validate_dry_strategy_rule(
        strategy_compounds=strategy_compounds,
        pit_laps=pit_laps,
    )

    result = _compute_lap_time_total(
        race_laps=race_laps,
        strategy_compounds=strategy_compounds,
        pit_laps=pit_laps,
        base_pace_seconds=base_pace_seconds,
        pit_loss_green_seconds=pit_loss_seconds,
        pit_loss_safety_car_seconds=pit_loss_seconds,
        compound_degradation_map=compound_degradation_map,
        fallback_degradation=fallback_degradation,
        starting_position=10,
        overtaking_difficulty=0.6,
        scenario_name="no_safety_car",
    )
    return float(result["expected_total_time"])


def estimate_strategy_risk_score(
    *,
    strategy_compounds: Sequence[str],
    num_stops: int,
    practice_confidence: float,
    overtaking_difficulty: float,
    safety_car_probability: float,
) -> float:
    confidence = float(np.clip(practice_confidence, 0.0, 1.0))
    overtaking = float(np.clip(overtaking_difficulty, 0.0, 1.0))
    sc_prob = float(np.clip(safety_car_probability, 0.0, 1.0))

    has_soft = any(compound.upper() == "SOFT" for compound in strategy_compounds)

    risk = (
        0.45 * (1.0 - confidence)
        + 0.25 * min(1.0, float(num_stops) / 2.0)
        + 0.15 * (1.0 if has_soft else 0.0)
        + 0.10 * overtaking
        + 0.05 * sc_prob
    )
    return float(np.clip(risk, 0.0, 1.0))
