from __future__ import annotations


COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

COMPOUND_DEGRADATION = {
    "SOFT": 0.125,
    "MEDIUM": 0.085,
    "HARD": 0.060,
}

COMPOUND_PACE_OFFSET = {
    "SOFT": -0.18,
    "MEDIUM": 0.00,
    "HARD": 0.14,
}

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

TIRE_CLIFF_LAPS = {
    "SOFT": 16,
    "MEDIUM": 26,
    "HARD": 36,
}

# Live-enforcement defaults for the FIA dry-race compound-change rule.
MANDATORY_STOP_BUFFER_LAPS = 3
INVALID_DRY_NO_STOP_PENALTY_SECONDS = 120.0


def is_dry_compound(compound: str) -> bool:
    return str(compound).upper() in DRY_COMPOUNDS


def get_tire_cliff_lap(compound: str) -> int:
    return int(TIRE_CLIFF_LAPS.get(str(compound).upper(), TIRE_CLIFF_LAPS["MEDIUM"]))


def compute_mandatory_stop_deadline_lap(race_laps: int, buffer_laps: int = MANDATORY_STOP_BUFFER_LAPS) -> int:
    """
    Deadline assumption:
    - In dry races, compulsory stop should be completed before the final few laps.
    - Default: race_laps - 3.
    """
    laps = max(1, int(race_laps))
    buffer = max(1, int(buffer_laps))
    return max(1, laps - buffer)


def estimate_cliff_and_puncture_penalty(
    *,
    compound: str,
    start_tire_age: int,
    laps_to_run: int,
) -> tuple[float, float]:
    """
    Modeling assumption:
    - Performance drops sharply after the compound cliff.
    - Risk of puncture/failure increases materially deep into the cliff zone.
    """
    cliff_lap = get_tire_cliff_lap(compound)
    start_age = max(0, int(start_tire_age))
    total_laps = max(0, int(laps_to_run))

    cliff_penalty = 0.0
    puncture_penalty = 0.0

    for offset in range(total_laps):
        age = start_age + offset
        if age <= cliff_lap:
            continue

        over_cliff = age - cliff_lap
        cliff_penalty += 0.65 + (0.14 * float(over_cliff))

        # Beyond ~6 laps over cliff, puncture/failure exposure ramps up quickly.
        if over_cliff > 6:
            puncture_over = over_cliff - 6
            puncture_penalty += 0.40 + (0.22 * float(puncture_over))

    puncture_penalty = min(float(puncture_penalty), 18.0)
    return float(cliff_penalty), float(puncture_penalty)


def estimate_ideal_pit_lap(starting_compound: str, race_laps: int) -> int:
    compound = str(starting_compound).upper()
    if compound == "SOFT":
        ratio = 0.26
    elif compound == "HARD":
        ratio = 0.56
    else:
        ratio = 0.43

    return max(6, min(int(round(race_laps * ratio)), max(6, race_laps - 6)))


def build_pit_window_string(ideal_pit_lap: int, race_laps: int) -> str:
    start = max(1, ideal_pit_lap - 2)
    end = min(race_laps, ideal_pit_lap + 2)
    return f"{start}-{end}"


def choose_best_next_compound(remaining_laps: int, current_compound: str) -> str:
    if remaining_laps <= 10:
        return "SOFT"
    if remaining_laps <= 22:
        return "MEDIUM"

    current = str(current_compound).upper()
    if current == "SOFT":
        return "HARD"
    return "MEDIUM"


def estimate_single_lap_pace(
    *,
    base_pace_seconds: float,
    compound: str,
    tire_age: int,
) -> float:
    cpd = str(compound).upper()
    deg = COMPOUND_DEGRADATION.get(cpd, COMPOUND_DEGRADATION["MEDIUM"])
    offset = COMPOUND_PACE_OFFSET.get(cpd, 0.0)
    return float(base_pace_seconds + offset + (deg * max(0, int(tire_age))))


def estimate_pit_cycle_position_loss(
    *,
    current_position: int,
    pit_loss_seconds: float,
    overtaking_difficulty: float,
    track_position_importance: float,
    field_size: int,
) -> int:
    # Lightweight heuristic for expected places dropped around a green/SC stop.
    raw_drop = (float(pit_loss_seconds) / 8.0) * (
        1.0 + (0.35 * float(overtaking_difficulty)) + (0.25 * float(track_position_importance))
    )
    rounded = int(round(raw_drop))
    max_drop = max(0, int(field_size) - int(current_position))
    return max(0, min(rounded, max_drop))


def estimate_rejoin_penalty(current_position: int, overtaking_difficulty: float) -> float:
    # Backmarkers lose less from rejoin traffic than front-runners rejoining into packs.
    position_factor = max(0.0, (10 - min(current_position, 10)) / 10)
    return (0.5 + position_factor) * float(overtaking_difficulty)


def estimate_track_position_benefit(
    current_position: int,
    track_position_importance: float,
    overtaking_difficulty: float,
) -> float:
    if current_position > 8:
        return 0.0

    position_weight = (9 - current_position) / 8
    return position_weight * track_position_importance * overtaking_difficulty * 2.0


def _stint_time(
    *,
    base_pace_seconds: float,
    compound: str,
    start_tire_age: int,
    laps: int,
) -> float:
    if laps <= 0:
        return 0.0

    cpd = str(compound).upper()
    deg = COMPOUND_DEGRADATION.get(cpd, COMPOUND_DEGRADATION["MEDIUM"])
    offset = COMPOUND_PACE_OFFSET.get(cpd, 0.0)

    laps_float = float(laps)
    start_age = float(max(0, start_tire_age))

    # Sum of linear degradation over tire ages from start_age to end of stint.
    degradation_area = (laps_float * start_age) + ((laps_float - 1.0) * laps_float * 0.5)

    return (base_pace_seconds + offset) * laps_float + (deg * degradation_area)


def estimate_total_time_if_pit_now(
    *,
    base_pace_seconds: float,
    race_laps: int,
    current_lap: int,
    current_position: int,
    pit_loss_green_seconds: float,
    pit_loss_safety_car_seconds: float,
    overtaking_difficulty: float,
    under_safety_car: bool,
    next_compound: str,
) -> float:
    remaining_laps = max(0, race_laps - current_lap + 1)
    pit_loss = pit_loss_safety_car_seconds if under_safety_car else pit_loss_green_seconds

    return (
        _stint_time(
            base_pace_seconds=base_pace_seconds,
            compound=next_compound,
            start_tire_age=0,
            laps=remaining_laps,
        )
        + pit_loss
        + estimate_rejoin_penalty(current_position=current_position, overtaking_difficulty=overtaking_difficulty)
    )


def estimate_total_time_if_stay_out(
    *,
    base_pace_seconds: float,
    race_laps: int,
    current_lap: int,
    current_position: int,
    current_compound: str,
    tire_age: int,
    pit_loss_green_seconds: float,
    overtaking_difficulty: float,
    track_position_importance: float,
    ideal_pit_lap: int,
    next_compound: str,
) -> float:
    remaining_laps = max(0, race_laps - current_lap + 1)
    if remaining_laps == 0:
        return 0.0

    # "Stay out" means no immediate stop this lap. Earliest normal stop is next lap.
    ideal_offset = int(ideal_pit_lap) - int(current_lap)
    if remaining_laps <= 1:
        laps_until_ideal = remaining_laps
    else:
        laps_until_ideal = max(1, min(remaining_laps, ideal_offset))

    if laps_until_ideal >= remaining_laps:
        no_pit_total = _stint_time(
            base_pace_seconds=base_pace_seconds,
            compound=current_compound,
            start_tire_age=tire_age,
            laps=remaining_laps,
        )
        cliff_penalty, puncture_penalty = estimate_cliff_and_puncture_penalty(
            compound=current_compound,
            start_tire_age=tire_age,
            laps_to_run=remaining_laps,
        )

        # Legacy pressure penalty retained, but reduced because cliff model now carries most risk.
        tire_age_pressure_penalty = max(0.0, (tire_age - 14) * 0.20)
        return no_pit_total + tire_age_pressure_penalty + cliff_penalty + puncture_penalty

    continuation = _stint_time(
        base_pace_seconds=base_pace_seconds,
        compound=current_compound,
        start_tire_age=tire_age,
        laps=laps_until_ideal,
    )
    post_pit_laps = max(0, remaining_laps - laps_until_ideal)
    post_pit = _stint_time(
        base_pace_seconds=base_pace_seconds,
        compound=next_compound,
        start_tire_age=0,
        laps=post_pit_laps,
    )

    track_position_benefit = estimate_track_position_benefit(
        current_position=current_position,
        track_position_importance=track_position_importance,
        overtaking_difficulty=overtaking_difficulty,
    )
    cliff_penalty, puncture_penalty = estimate_cliff_and_puncture_penalty(
        compound=current_compound,
        start_tire_age=tire_age,
        laps_to_run=laps_until_ideal,
    )
    tire_age_pressure_penalty = max(0.0, (tire_age - 14) * 0.20)

    return (
        continuation
        + pit_loss_green_seconds
        + post_pit
        + tire_age_pressure_penalty
        + cliff_penalty
        + puncture_penalty
        - track_position_benefit
    )


def estimate_total_time_if_stay_out_with_compulsory_stop(
    *,
    base_pace_seconds: float,
    race_laps: int,
    current_lap: int,
    current_position: int,
    current_compound: str,
    tire_age: int,
    pit_loss_green_seconds: float,
    overtaking_difficulty: float,
    track_position_importance: float,
    compulsory_stop_lap: int,
    next_compound: str,
) -> float:
    """
    Mandatory-stop enforcement path:
    - Driver stays out for this lap.
    - A compulsory dry stop is projected before race end.
    - If no legal compulsory stop can be performed, treat no-stop finish as invalid.
    """
    remaining_laps = max(0, race_laps - current_lap + 1)
    if remaining_laps == 0:
        return 0.0

    stop_lap = max(int(current_lap) + 1, min(int(compulsory_stop_lap), int(race_laps)))
    laps_before_stop = max(0, stop_lap - int(current_lap))

    if laps_before_stop >= remaining_laps:
        no_pit_total = _stint_time(
            base_pace_seconds=base_pace_seconds,
            compound=current_compound,
            start_tire_age=tire_age,
            laps=remaining_laps,
        )
        cliff_penalty, puncture_penalty = estimate_cliff_and_puncture_penalty(
            compound=current_compound,
            start_tire_age=tire_age,
            laps_to_run=remaining_laps,
        )
        tire_age_pressure_penalty = max(0.0, (tire_age - 14) * 0.20)
        return (
            no_pit_total
            + tire_age_pressure_penalty
            + cliff_penalty
            + puncture_penalty
            + INVALID_DRY_NO_STOP_PENALTY_SECONDS
        )

    continuation = _stint_time(
        base_pace_seconds=base_pace_seconds,
        compound=current_compound,
        start_tire_age=tire_age,
        laps=laps_before_stop,
    )
    cliff_penalty, puncture_penalty = estimate_cliff_and_puncture_penalty(
        compound=current_compound,
        start_tire_age=tire_age,
        laps_to_run=laps_before_stop,
    )
    post_pit_laps = max(0, int(race_laps) - int(stop_lap) + 1)
    post_pit = _stint_time(
        base_pace_seconds=base_pace_seconds,
        compound=next_compound,
        start_tire_age=0,
        laps=post_pit_laps,
    )

    # Staying out for track position still has value, but it fades as the compulsory stop approaches.
    pre_stop_ratio = float(laps_before_stop) / max(1.0, float(remaining_laps))
    track_position_benefit = estimate_track_position_benefit(
        current_position=current_position,
        track_position_importance=track_position_importance,
        overtaking_difficulty=overtaking_difficulty,
    ) * pre_stop_ratio
    tire_age_pressure_penalty = max(0.0, (tire_age - 14) * 0.20)
    rejoin_penalty = estimate_rejoin_penalty(
        current_position=current_position,
        overtaking_difficulty=overtaking_difficulty,
    )

    return (
        continuation
        + pit_loss_green_seconds
        + post_pit
        + tire_age_pressure_penalty
        + cliff_penalty
        + puncture_penalty
        + rejoin_penalty
        - track_position_benefit
    )
