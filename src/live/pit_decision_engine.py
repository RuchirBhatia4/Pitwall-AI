from __future__ import annotations

from typing import Any

from src.live.overtake_model import (
    estimate_freshness_advantage,
    estimate_overtake_probability,
    estimate_recovery_probability,
)
from src.live.race_state import DriverRaceState, LiveRaceState
from src.live.scenario_simulator import (
    build_pit_window_string,
    choose_best_next_compound,
    compute_mandatory_stop_deadline_lap,
    estimate_ideal_pit_lap,
    estimate_pit_cycle_position_loss,
    estimate_single_lap_pace,
    estimate_total_time_if_pit_now,
    estimate_total_time_if_stay_out,
    estimate_total_time_if_stay_out_with_compulsory_stop,
    get_tire_cliff_lap,
    INVALID_DRY_NO_STOP_PENALTY_SECONDS,
    is_dry_compound,
)


def _rank_position(times_by_driver: dict[str, float], driver: str) -> int:
    ranking = sorted(times_by_driver.items(), key=lambda item: (item[1], item[0]))
    for idx, (driver_code, _) in enumerate(ranking, start=1):
        if driver_code == driver:
            return idx
    return len(ranking)


def _clamp_position(position: int, field_size: int) -> int:
    return max(1, min(int(position), int(field_size)))


def _high_wear_threshold(compound: str) -> int:
    return max(8, get_tire_cliff_lap(compound) - 4)


def _resolve_mandatory_stop_deadline_lap(setup: dict[str, Any]) -> int:
    if "mandatory_stop_deadline_lap" in setup:
        return max(1, int(setup["mandatory_stop_deadline_lap"]))
    race_laps = int(setup["race_laps"])
    buffer_laps = int(setup.get("mandatory_stop_buffer_laps", 3))
    return compute_mandatory_stop_deadline_lap(race_laps=race_laps, buffer_laps=buffer_laps)


def _mandatory_stop_pending(compounds_used: set[str], is_dry_race: bool) -> bool:
    if not is_dry_race:
        return False
    dry_used = {compound for compound in compounds_used if is_dry_compound(compound)}
    return len(dry_used) < 2


def _pick_mandatory_next_compound(current_compound: str, remaining_laps: int) -> str:
    current = str(current_compound).upper()
    if remaining_laps <= 10 and current != "SOFT":
        return "SOFT"
    if remaining_laps >= 18 and current != "HARD":
        return "HARD"
    if current != "MEDIUM":
        return "MEDIUM"
    if current != "HARD":
        return "HARD"
    return "SOFT"


def _blend_finish_projection(
    *,
    baseline_position_rank: int,
    position_based_projection: int,
    confidence: float,
    field_size: int,
) -> int:
    # Blend pure time-rank projection with position-cycle projection to avoid unrealistic extremes.
    weight_time_rank = max(0.25, min(0.55, 0.30 + (0.30 * (1.0 - float(confidence)))))
    blended = (weight_time_rank * float(baseline_position_rank)) + (
        (1.0 - weight_time_rank) * float(position_based_projection)
    )
    return _clamp_position(int(round(blended)), field_size)


def _prefer_stay_out_for_track_position(
    *,
    current_position: int,
    overtaking_difficulty: float,
    track_position_importance: float,
    pit_now_gain_seconds: float,
    overtake_probability_if_pit_now: float,
) -> bool:
    if current_position > 6:
        return False

    if overtaking_difficulty < 0.80 or track_position_importance < 0.80:
        return False

    return (pit_now_gain_seconds < 1.2) or (overtake_probability_if_pit_now < 0.42)


def _estimate_undercut_cover_pressure(
    *,
    setup: dict[str, Any],
    tire_age: int,
    current_compound: str,
    current_lap: int,
    ideal_pit_lap: int,
) -> float:
    """
    Estimates strategic pressure to pit before tires are fully exhausted.

    Modeling assumption:
    - A close car ahead creates undercut opportunity.
    - A close car behind creates cover pressure.
    - Pressure matters most as the stint approaches its normal pit window.
    """
    gap_ahead = setup.get("gap_to_ahead_seconds")
    gap_behind = setup.get("gap_to_behind_seconds")

    try:
        gap_ahead_value = float(gap_ahead)
    except (TypeError, ValueError):
        gap_ahead_value = float("nan")

    try:
        gap_behind_value = float(gap_behind)
    except (TypeError, ValueError):
        gap_behind_value = float("nan")

    cliff_lap = get_tire_cliff_lap(current_compound)
    tire_window_factor = max(0.0, min(1.0, (int(tire_age) - max(8, cliff_lap - 12)) / 8.0))
    pit_window_factor = max(0.0, min(1.0, 1.0 - (abs(int(current_lap) - int(ideal_pit_lap)) / 8.0)))

    pressure = 0.0
    if gap_ahead_value == gap_ahead_value and 0.0 < gap_ahead_value <= 3.0:
        # Smaller gaps to the car ahead make the undercut more valuable.
        pressure += (3.0 - gap_ahead_value) / 3.0 * 1.8

    if gap_behind_value == gap_behind_value and 0.0 < gap_behind_value <= 2.5:
        # Close cars behind can force a cover stop before pure tire math says to pit.
        pressure += (2.5 - gap_behind_value) / 2.5 * 2.2

    return float(pressure * max(0.35, tire_window_factor) * max(0.45, pit_window_factor))


def _has_undercut_cover_trigger(
    *,
    setup: dict[str, Any],
    tire_age: int,
    current_compound: str,
    pit_now_gain_seconds: float,
    projected_finish_if_pit_now: int,
    projected_finish_if_stay_out: int,
    overtake_probability_if_pit_now: float,
) -> bool:
    gap_ahead = setup.get("gap_to_ahead_seconds")
    gap_behind = setup.get("gap_to_behind_seconds")
    try:
        gap_ahead_value = float(gap_ahead)
    except (TypeError, ValueError):
        gap_ahead_value = float("nan")
    try:
        gap_behind_value = float(gap_behind)
    except (TypeError, ValueError):
        gap_behind_value = float("nan")

    close_ahead = gap_ahead_value == gap_ahead_value and 0.0 < gap_ahead_value <= 3.0
    close_behind = gap_behind_value == gap_behind_value and 0.0 < gap_behind_value <= 2.5
    if not (close_ahead or close_behind):
        return False

    # Mediums and softs need strategic windows before the cliff, not only after.
    cliff_lap = get_tire_cliff_lap(current_compound)
    if int(tire_age) < max(12, cliff_lap - 11):
        return False

    projected_position_cost = int(projected_finish_if_pit_now) - int(projected_finish_if_stay_out)
    if projected_position_cost > 1:
        return False

    return bool(
        float(pit_now_gain_seconds) > -6.5
        and float(overtake_probability_if_pit_now) >= 0.50
    )


def _build_reason(
    *,
    best_action: str,
    under_safety_car: bool,
    high_wear: bool,
    mandatory_stop_pending: bool,
    mandatory_stop_enforced: bool,
    forced_compulsory_stop_lap: int | None,
    illegal_no_stop_completion: bool,
    tire_cliff_risk: bool,
    puncture_risk: bool,
    prefer_stay_out_position: bool,
    pit_now_gain_seconds: float,
    overtake_probability_if_pit_now: float,
    overtake_probability_if_stay_out: float,
    pitted_early_recomputed: bool,
    undercut_cover_pressure_seconds: float,
    undercut_cover_trigger: bool,
) -> str:
    if illegal_no_stop_completion and best_action == "pit_now":
        reason = (
            "Mandatory dry compound-change rule is unresolved and finishing this lap without stopping is illegal."
        )
    elif mandatory_stop_enforced and best_action == "pit_now":
        reason = (
            "Mandatory dry stop must be completed now; delaying only shifts a compulsory stop into the projection."
        )
    elif mandatory_stop_enforced and best_action == "stay_out":
        forced_lap = forced_compulsory_stop_lap if forced_compulsory_stop_lap is not None else "soon"
        reason = (
            f"Projected finish includes a forced compulsory stop around lap {forced_lap} to satisfy the dry rule."
        )
    elif puncture_risk and best_action == "pit_now":
        reason = "Tire-life risk is critical and puncture exposure is rising, so pit now is safer and faster."
    elif tire_cliff_risk and best_action == "pit_now":
        reason = "Current tire is beyond the cliff window, so staying out carries steep pace loss."
    elif under_safety_car and best_action == "pit_now":
        reason = (
            "Safety-car stop is attractive because reduced pit loss offsets likely rejoin traffic."
        )
    elif undercut_cover_trigger and best_action == "pit_now":
        reason = "Pit now is preferred to attack/cover nearby traffic before the window closes."
    elif best_action == "pit_now" and overtake_probability_if_pit_now >= 0.55:
        reason = "Pit now is faster and positions are recoverable."
    elif high_wear and best_action == "pit_now":
        reason = "Current tire age is high, and delaying the stop risks a bigger pace drop."
    elif prefer_stay_out_position and best_action == "stay_out":
        reason = (
            "Stay out protects track position on a difficult overtaking track while pit recovery is uncertain."
        )
    elif best_action == "stay_out" and (
        overtake_probability_if_pit_now + 0.08 < overtake_probability_if_stay_out
    ):
        reason = "Stay out is preferred for now because pit-cycle recovery odds are limited."
    elif best_action == "pit_now":
        reason = "Pitting now gives the stronger projected race outcome."
    else:
        reason = "Staying out gives the stronger projected race outcome."

    if pitted_early_recomputed:
        reason += " Projections were recomputed after an earlier-than-ideal stop."

    reason += (
        f" Net pit-now gain: {pit_now_gain_seconds:.2f}s, "
        f"pit recovery probability: {overtake_probability_if_pit_now:.2f}, "
        f"stay-out overtake probability: {overtake_probability_if_stay_out:.2f}."
    )
    if undercut_cover_pressure_seconds > 0.0:
        reason += f" Undercut/cover pressure: {undercut_cover_pressure_seconds:.2f}s."
    return reason


def _estimate_pit_recovery_probability(
    *,
    state: DriverRaceState,
    setup: dict[str, Any],
    rejoin_reference_state: DriverRaceState | None,
    next_compound: str,
    expected_drop_positions: int,
    field_size: int,
    pit_lap_pace: float,
) -> float:
    overtaking_difficulty = float(setup["overtaking_difficulty"])
    track_position_importance = float(setup["track_position_importance"])

    if expected_drop_positions <= 0:
        return 0.90

    rival_compound = "MEDIUM"
    rival_tire_age = 10
    rival_lap_pace = float(setup["base_pace_seconds"]) + 0.20
    if rejoin_reference_state is None:
        base_probability = 0.46 - (overtaking_difficulty * 0.12) - (track_position_importance * 0.10)
    else:
        rival_compound = rejoin_reference_state.current_compound
        rival_tire_age = rejoin_reference_state.tire_age
        rival_lap_pace = estimate_single_lap_pace(
            base_pace_seconds=float(setup["base_pace_seconds"]) + 0.20,
            compound=rival_compound,
            tire_age=rival_tire_age,
        )

    freshness_advantage = estimate_freshness_advantage(
        attacker_compound=next_compound,
        attacker_tire_age=0,
        defender_compound=rival_compound,
        defender_tire_age=rival_tire_age,
    )

    pace_delta_seconds = float(rival_lap_pace - pit_lap_pace)
    if rejoin_reference_state is not None:
        base_probability = estimate_overtake_probability(
            pace_delta_seconds=pace_delta_seconds,
            tire_age_delta=float(rival_tire_age),
            freshness_advantage=freshness_advantage,
            overtaking_difficulty=overtaking_difficulty,
            track_position_importance=track_position_importance,
        )

    return estimate_recovery_probability(
        base_overtake_probability=base_probability,
        expected_positions_to_recover=int(expected_drop_positions),
        current_position=int(state.current_position),
        field_size=int(field_size),
        pace_delta_seconds=pace_delta_seconds,
        freshness_advantage=freshness_advantage,
        overtaking_difficulty=overtaking_difficulty,
        track_position_importance=track_position_importance,
    )


def _estimate_stay_out_overtake_probability(
    *,
    state: DriverRaceState,
    setup: dict[str, Any],
    ahead_state: DriverRaceState | None,
    current_lap_pace: float,
) -> float:
    overtaking_difficulty = float(setup["overtaking_difficulty"])
    track_position_importance = float(setup["track_position_importance"])

    if ahead_state is None:
        # Leader case: use a conservative "ability to keep/pull away" proxy.
        base = 0.52 - (max(0, state.tire_age - 10) * 0.01)
        base += track_position_importance * 0.08
        base += overtaking_difficulty * 0.05
        return max(0.05, min(0.95, base))

    # Estimate pace of the car ahead with slight fallback offset.
    rival_lap_pace = estimate_single_lap_pace(
        base_pace_seconds=float(setup["base_pace_seconds"]) + 0.18,
        compound=ahead_state.current_compound,
        tire_age=ahead_state.tire_age,
    )

    freshness_advantage = estimate_freshness_advantage(
        attacker_compound=state.current_compound,
        attacker_tire_age=state.tire_age,
        defender_compound=ahead_state.current_compound,
        defender_tire_age=ahead_state.tire_age,
    )

    return estimate_overtake_probability(
        pace_delta_seconds=(rival_lap_pace - current_lap_pace),
        tire_age_delta=float(ahead_state.tire_age - state.tire_age),
        freshness_advantage=freshness_advantage,
        overtaking_difficulty=overtaking_difficulty,
        track_position_importance=track_position_importance,
    )


def _evaluate_driver_pair(
    *,
    state: DriverRaceState,
    race_state: LiveRaceState,
    setup: dict[str, Any],
    ideal_pit_lap: int,
    pitted_early_recomputed: bool,
    mandatory_stop_pending: bool,
    mandatory_stop_deadline_lap: int,
    is_dry_race: bool,
    active_states_by_position: dict[int, DriverRaceState],
    field_size: int,
) -> dict[str, Any]:
    current_lap = int(race_state.lap)
    race_laps = int(setup["race_laps"])
    remaining_laps = max(0, race_laps - current_lap + 1)
    overtaking_difficulty = float(setup["overtaking_difficulty"])
    track_position_importance = float(setup["track_position_importance"])

    next_compound = choose_best_next_compound(
        remaining_laps=remaining_laps,
        current_compound=state.current_compound,
    )
    if mandatory_stop_pending and is_dry_compound(state.current_compound):
        next_compound = _pick_mandatory_next_compound(
            current_compound=state.current_compound,
            remaining_laps=remaining_laps,
        )

    pit_now_total = estimate_total_time_if_pit_now(
        base_pace_seconds=float(setup["base_pace_seconds"]),
        race_laps=race_laps,
        current_lap=current_lap,
        current_position=int(state.current_position),
        pit_loss_green_seconds=float(setup["pit_loss_green_seconds"]),
        pit_loss_safety_car_seconds=float(setup["pit_loss_safety_car_seconds"]),
        overtaking_difficulty=overtaking_difficulty,
        under_safety_car=bool(race_state.under_safety_car),
        next_compound=next_compound,
    )

    stay_out_total = estimate_total_time_if_stay_out(
        base_pace_seconds=float(setup["base_pace_seconds"]),
        race_laps=race_laps,
        current_lap=current_lap,
        current_position=int(state.current_position),
        current_compound=state.current_compound,
        tire_age=int(state.tire_age),
        pit_loss_green_seconds=float(setup["pit_loss_green_seconds"]),
        overtaking_difficulty=overtaking_difficulty,
        track_position_importance=track_position_importance,
        ideal_pit_lap=int(ideal_pit_lap),
        next_compound=next_compound,
    )
    undercut_cover_pressure = _estimate_undercut_cover_pressure(
        setup=setup,
        tire_age=int(state.tire_age),
        current_compound=state.current_compound,
        current_lap=current_lap,
        ideal_pit_lap=int(ideal_pit_lap),
    )
    pit_now_total -= undercut_cover_pressure

    cliff_lap = get_tire_cliff_lap(state.current_compound)
    tire_cliff_risk = int(state.tire_age) >= cliff_lap
    puncture_risk = int(state.tire_age) >= (cliff_lap + 6)
    mandatory_stop_enforced = bool(
        mandatory_stop_pending and int(current_lap) >= int(mandatory_stop_deadline_lap)
    )
    forced_compulsory_stop_lap: int | None = None
    illegal_no_stop_completion = bool(mandatory_stop_pending and remaining_laps <= 1)
    if mandatory_stop_enforced and not illegal_no_stop_completion:
        forced_compulsory_stop_lap = min(
            race_laps,
            max(current_lap + 1, int(mandatory_stop_deadline_lap) + 1),
        )
        stay_out_total = estimate_total_time_if_stay_out_with_compulsory_stop(
            base_pace_seconds=float(setup["base_pace_seconds"]),
            race_laps=race_laps,
            current_lap=current_lap,
            current_position=int(state.current_position),
            current_compound=state.current_compound,
            tire_age=int(state.tire_age),
            pit_loss_green_seconds=float(setup["pit_loss_green_seconds"]),
            overtaking_difficulty=overtaking_difficulty,
            track_position_importance=track_position_importance,
            compulsory_stop_lap=forced_compulsory_stop_lap,
            next_compound=next_compound,
        )
    if illegal_no_stop_completion:
        # Keep this very high so end-of-race no-stop completion on one dry compound is never treated as valid.
        stay_out_total += INVALID_DRY_NO_STOP_PENALTY_SECONDS

    pit_now_gain = float(stay_out_total - pit_now_total)
    stay_out_gain = float(-pit_now_gain)

    pit_loss_seconds = (
        float(setup["pit_loss_safety_car_seconds"])
        if race_state.under_safety_car
        else float(setup["pit_loss_green_seconds"])
    )

    expected_drop_positions = estimate_pit_cycle_position_loss(
        current_position=int(state.current_position),
        pit_loss_seconds=pit_loss_seconds,
        overtaking_difficulty=overtaking_difficulty,
        track_position_importance=track_position_importance,
        field_size=int(field_size),
    )
    rejoin_position = _clamp_position(int(state.current_position) + expected_drop_positions, field_size)

    rejoin_reference_state = active_states_by_position.get(rejoin_position)
    if rejoin_reference_state is not None and rejoin_reference_state.driver == state.driver:
        rejoin_reference_state = active_states_by_position.get(
            _clamp_position(rejoin_position + 1, field_size)
        )

    current_lap_pace = estimate_single_lap_pace(
        base_pace_seconds=float(setup["base_pace_seconds"]),
        compound=state.current_compound,
        tire_age=int(state.tire_age),
    )
    pit_lap_pace = estimate_single_lap_pace(
        base_pace_seconds=float(setup["base_pace_seconds"]),
        compound=next_compound,
        tire_age=0,
    )

    overtake_probability_if_pit_now = _estimate_pit_recovery_probability(
        state=state,
        setup=setup,
        rejoin_reference_state=rejoin_reference_state,
        next_compound=next_compound,
        expected_drop_positions=expected_drop_positions,
        field_size=field_size,
        pit_lap_pace=pit_lap_pace,
    )

    ahead_state = active_states_by_position.get(_clamp_position(int(state.current_position) - 1, field_size))
    if ahead_state is not None and ahead_state.driver == state.driver:
        ahead_state = None

    overtake_probability_if_stay_out = _estimate_stay_out_overtake_probability(
        state=state,
        setup=setup,
        ahead_state=ahead_state,
        current_lap_pace=current_lap_pace,
    )

    high_wear = int(state.tire_age) >= _high_wear_threshold(state.current_compound)
    prefer_stay_out_position = _prefer_stay_out_for_track_position(
        current_position=int(state.current_position),
        overtaking_difficulty=overtaking_difficulty,
        track_position_importance=track_position_importance,
        pit_now_gain_seconds=pit_now_gain,
        overtake_probability_if_pit_now=overtake_probability_if_pit_now,
    )
    return {
        "projected_total_time_if_pit_now": float(pit_now_total),
        "projected_total_time_if_stay_out": float(stay_out_total),
        "pit_now_gain_seconds": float(pit_now_gain),
        "stay_out_gain_seconds": float(stay_out_gain),
        "best_next_compound": str(next_compound),
        "best_pit_window": build_pit_window_string(int(ideal_pit_lap), race_laps),
        "overtake_probability_if_pit_now": float(overtake_probability_if_pit_now),
        "overtake_probability_if_stay_out": float(overtake_probability_if_stay_out),
        "_expected_drop_positions": int(expected_drop_positions),
        "_high_wear": high_wear,
        "_mandatory_stop_pending": bool(mandatory_stop_pending),
        "_mandatory_stop_enforced": mandatory_stop_enforced,
        "_mandatory_stop_deadline_lap": int(mandatory_stop_deadline_lap),
        "_forced_compulsory_stop_lap": forced_compulsory_stop_lap,
        "_illegal_no_stop_completion": illegal_no_stop_completion,
        "_tire_cliff_risk": tire_cliff_risk,
        "_puncture_risk": puncture_risk,
        "_prefer_stay_out_position": prefer_stay_out_position,
        "_pitted_early_recomputed": pitted_early_recomputed,
        "_undercut_cover_pressure_seconds": float(undercut_cover_pressure),
    }


def evaluate_pit_decisions_for_lap(
    *,
    race_state: LiveRaceState,
    driver_setup: dict[str, dict[str, Any]],
    decision_context: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    active_states = list(race_state.iter_active())
    field_size = len(active_states)
    active_by_position = {state.current_position: state for state in active_states}

    decisions: dict[str, dict[str, Any]] = {}

    for state in active_states:
        setup = driver_setup[state.driver]
        context = decision_context[state.driver]

        ideal_pit_lap = int(context["ideal_pit_lap"])
        pitted_early_recomputed = bool(context.get("pitted_early_recomputed", False))
        mandatory_stop_pending = bool(context.get("mandatory_stop_pending", False))
        mandatory_stop_deadline_lap = int(
            context.get("mandatory_stop_deadline_lap", _resolve_mandatory_stop_deadline_lap(setup))
        )
        is_dry_race = bool(context.get("is_dry_race", False))

        decisions[state.driver] = _evaluate_driver_pair(
            state=state,
            race_state=race_state,
            setup=setup,
            ideal_pit_lap=ideal_pit_lap,
            pitted_early_recomputed=pitted_early_recomputed,
            mandatory_stop_pending=mandatory_stop_pending,
            mandatory_stop_deadline_lap=mandatory_stop_deadline_lap,
            is_dry_race=is_dry_race,
            active_states_by_position=active_by_position,
            field_size=field_size,
        )

    stay_out_totals = {
        driver: float(item["projected_total_time_if_stay_out"])
        for driver, item in decisions.items()
    }

    for state in active_states:
        driver = state.driver
        item = decisions[driver]
        setup = driver_setup[driver]

        pit_projection = dict(stay_out_totals)
        pit_projection[driver] = float(item["projected_total_time_if_pit_now"])

        base_finish_if_stay_out = _rank_position(stay_out_totals, driver=driver)
        base_finish_if_pit_now = _rank_position(pit_projection, driver=driver)

        expected_drop_positions = int(item.get("_expected_drop_positions", 0))
        overtake_probability_if_pit_now = float(item["overtake_probability_if_pit_now"])
        overtake_probability_if_stay_out = float(item["overtake_probability_if_stay_out"])
        remaining_laps = max(0, int(setup["race_laps"]) - int(race_state.lap) + 1)
        recovery_window_factor = min(1.0, remaining_laps / max(1.0, float(setup["race_laps"])))

        recovered_positions = int(
            round(
                expected_drop_positions
                * overtake_probability_if_pit_now
                * (0.55 + (0.45 * recovery_window_factor))
            )
        )
        net_drop_positions = max(0, expected_drop_positions - recovered_positions)
        position_based_finish_if_pit = _clamp_position(
            int(state.current_position) + net_drop_positions,
            field_size,
        )
        projected_finish_if_pit_now = _blend_finish_projection(
            baseline_position_rank=base_finish_if_pit_now,
            position_based_projection=position_based_finish_if_pit,
            confidence=overtake_probability_if_pit_now,
            field_size=field_size,
        )

        wear_threshold = _high_wear_threshold(state.current_compound)
        wear_factor = max(0.0, (int(state.tire_age) - wear_threshold) / 6.0)
        recovery_gap = max(0.0, 0.54 - overtake_probability_if_stay_out) * 1.8
        projected_losses_if_stay = int(round(wear_factor + recovery_gap))

        if (
            int(state.current_position) <= 6
            and float(setup["overtaking_difficulty"]) >= 0.80
            and float(setup["track_position_importance"]) >= 0.80
        ):
            projected_losses_if_stay = max(0, projected_losses_if_stay - 1)

        position_based_finish_if_stay = _clamp_position(
            int(state.current_position) + projected_losses_if_stay,
            field_size,
        )
        projected_finish_if_stay_out = _blend_finish_projection(
            baseline_position_rank=base_finish_if_stay_out,
            position_based_projection=position_based_finish_if_stay,
            confidence=overtake_probability_if_stay_out,
            field_size=field_size,
        )

        item["projected_finish_if_stay_out"] = int(projected_finish_if_stay_out)
        item["projected_finish_if_pit_now"] = int(projected_finish_if_pit_now)

        prefer_stay_out_position = bool(item["_prefer_stay_out_position"])
        high_wear = bool(item["_high_wear"])
        mandatory_stop_pending = bool(item.get("_mandatory_stop_pending", False))
        mandatory_stop_enforced = bool(item.get("_mandatory_stop_enforced", False))
        forced_compulsory_stop_lap = item.get("_forced_compulsory_stop_lap")
        illegal_no_stop_completion = bool(item.get("_illegal_no_stop_completion", False))
        tire_cliff_risk = bool(item.get("_tire_cliff_risk", False))
        puncture_risk = bool(item.get("_puncture_risk", False))
        pit_now_gain = float(item["pit_now_gain_seconds"])
        undercut_cover_pressure = float(item.get("_undercut_cover_pressure_seconds", 0.0))
        undercut_cover_trigger = _has_undercut_cover_trigger(
            setup=setup,
            tire_age=int(state.tire_age),
            current_compound=state.current_compound,
            pit_now_gain_seconds=pit_now_gain,
            projected_finish_if_pit_now=projected_finish_if_pit_now,
            projected_finish_if_stay_out=projected_finish_if_stay_out,
            overtake_probability_if_pit_now=overtake_probability_if_pit_now,
        )

        if illegal_no_stop_completion:
            projected_finish_if_stay_out = int(field_size)
            item["projected_finish_if_stay_out"] = int(projected_finish_if_stay_out)

        if illegal_no_stop_completion:
            best_action = "pit_now"
        elif mandatory_stop_enforced:
            best_action = "pit_now"
        elif puncture_risk and pit_now_gain > -8.0:
            best_action = "pit_now"
        elif tire_cliff_risk and pit_now_gain > -4.0:
            best_action = "pit_now"
        elif race_state.under_safety_car and pit_now_gain > -0.75:
            best_action = "pit_now"
        elif undercut_cover_trigger:
            best_action = "pit_now"
        elif prefer_stay_out_position and projected_finish_if_stay_out <= projected_finish_if_pit_now:
            best_action = "stay_out"
        elif (
            undercut_cover_pressure >= 1.0
            and pit_now_gain > -3.5
            and overtake_probability_if_pit_now >= 0.35
        ):
            best_action = "pit_now"
        elif high_wear and (
            (pit_now_gain > -1.0) or (projected_finish_if_pit_now < projected_finish_if_stay_out)
        ):
            best_action = "pit_now"
        elif projected_finish_if_pit_now + (0 if pit_now_gain > 0 else 1) < projected_finish_if_stay_out:
            best_action = "pit_now"
        elif pit_now_gain > 0 and (
            overtake_probability_if_pit_now >= max(0.25, overtake_probability_if_stay_out - 0.20)
        ):
            best_action = "pit_now"
        else:
            best_action = "stay_out"

        item["best_action_now"] = best_action
        item["recommendation_reason"] = _build_reason(
            best_action=best_action,
            under_safety_car=bool(race_state.under_safety_car),
            high_wear=high_wear,
            mandatory_stop_pending=mandatory_stop_pending,
            mandatory_stop_enforced=mandatory_stop_enforced,
            forced_compulsory_stop_lap=(
                int(forced_compulsory_stop_lap)
                if forced_compulsory_stop_lap is not None
                else None
            ),
            illegal_no_stop_completion=illegal_no_stop_completion,
            tire_cliff_risk=tire_cliff_risk,
            puncture_risk=puncture_risk,
            prefer_stay_out_position=prefer_stay_out_position,
            pit_now_gain_seconds=pit_now_gain,
            overtake_probability_if_pit_now=overtake_probability_if_pit_now,
            overtake_probability_if_stay_out=overtake_probability_if_stay_out,
            pitted_early_recomputed=bool(item["_pitted_early_recomputed"]),
            undercut_cover_pressure_seconds=undercut_cover_pressure,
            undercut_cover_trigger=undercut_cover_trigger,
        )

        # Remove helper fields from output payload.
        item.pop("_expected_drop_positions", None)
        item.pop("_high_wear", None)
        item.pop("_mandatory_stop_pending", None)
        item.pop("_mandatory_stop_enforced", None)
        item.pop("_mandatory_stop_deadline_lap", None)
        item.pop("_forced_compulsory_stop_lap", None)
        item.pop("_illegal_no_stop_completion", None)
        item.pop("_tire_cliff_risk", None)
        item.pop("_puncture_risk", None)
        item.pop("_prefer_stay_out_position", None)
        item.pop("_pitted_early_recomputed", None)
        item.pop("_undercut_cover_pressure_seconds", None)

    return decisions


def build_initial_decision_context(
    *,
    race_state: LiveRaceState,
    driver_setup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for state in race_state.iter_active():
        setup = driver_setup[state.driver]
        starting_compound = str(setup["starting_compound"]).upper()
        compounds_used = {starting_compound}
        is_dry_race = bool(setup.get("is_dry_race", is_dry_compound(starting_compound)))
        mandatory_stop_deadline_lap = _resolve_mandatory_stop_deadline_lap(setup)
        original_ideal = estimate_ideal_pit_lap(
            starting_compound=starting_compound,
            race_laps=int(setup["race_laps"]),
        )
        context[state.driver] = {
            "original_ideal_pit_lap": int(original_ideal),
            "ideal_pit_lap": int(original_ideal),
            "last_stops_made": int(state.stops_made),
            "pitted_early_recomputed": False,
            "compounds_used": compounds_used,
            "is_dry_race": is_dry_race,
            "mandatory_stop_deadline_lap": int(mandatory_stop_deadline_lap),
            "mandatory_stop_pending": _mandatory_stop_pending(compounds_used, is_dry_race),
        }

    return context


def sync_decision_context_after_events(
    *,
    race_state: LiveRaceState,
    decision_context: dict[str, dict[str, Any]],
    driver_setup: dict[str, dict[str, Any]],
) -> None:
    current_lap = int(race_state.lap)

    for state in race_state.iter_active():
        entry = decision_context[state.driver]
        setup = driver_setup[state.driver]

        previous_stops = int(entry.get("last_stops_made", 0))
        current_stops = int(state.stops_made)
        compounds_used = {
            str(compound).upper()
            for compound in entry.get("compounds_used", {str(setup["starting_compound"]).upper()})
            if str(compound).strip()
        }
        compounds_used.add(str(state.current_compound).upper())
        is_dry_race = bool(entry.get("is_dry_race", True))
        if not is_dry_compound(state.current_compound):
            is_dry_race = False

        if current_stops > previous_stops:
            original_ideal = int(entry["original_ideal_pit_lap"])
            if current_lap < original_ideal:
                entry["pitted_early_recomputed"] = True

            remaining_laps = max(0, int(setup["race_laps"]) - current_lap)
            entry["ideal_pit_lap"] = min(
                int(setup["race_laps"]),
                current_lap + max(8, remaining_laps // 2),
            )

        entry["last_stops_made"] = current_stops
        entry["compounds_used"] = compounds_used
        entry["is_dry_race"] = is_dry_race
        entry["mandatory_stop_deadline_lap"] = int(_resolve_mandatory_stop_deadline_lap(setup))
        entry["mandatory_stop_pending"] = _mandatory_stop_pending(compounds_used, is_dry_race)
