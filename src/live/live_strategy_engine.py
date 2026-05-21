from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.event_processor import apply_events_for_lap
from src.live.pit_decision_engine import (
    build_initial_decision_context,
    evaluate_pit_decisions_for_lap,
    sync_decision_context_after_events,
)
from src.live.race_state import LiveRaceState
from src.live.scenario_simulator import (
    choose_best_next_compound,
    compute_mandatory_stop_deadline_lap,
    is_dry_compound,
)


LIVE_OUTPUT_COLUMNS = [
    "race",
    "lap",
    "driver",
    "team",
    "current_position",
    "current_compound",
    "tire_age",
    "stops_made",
    "under_safety_car",
    "setup_source",
    "projected_total_time_if_pit_now",
    "projected_total_time_if_stay_out",
    "pit_now_gain_seconds",
    "stay_out_gain_seconds",
    "best_next_compound",
    "best_pit_window",
    "projected_finish_if_pit_now",
    "projected_finish_if_stay_out",
    "overtake_probability_if_pit_now",
    "overtake_probability_if_stay_out",
    "best_action_now",
    "recommendation_reason",
]


def _choose_forced_mandatory_compound(current_compound: str, remaining_laps: int) -> str:
    current = str(current_compound).upper()
    candidate = choose_best_next_compound(
        remaining_laps=max(1, int(remaining_laps)),
        current_compound=current,
    )
    candidate = str(candidate).upper()
    if is_dry_compound(candidate) and candidate != current:
        return candidate

    for fallback in ("HARD", "MEDIUM", "SOFT"):
        if fallback != current:
            return fallback
    return "MEDIUM"


def _apply_compulsory_mandatory_stops(
    *,
    race_state: LiveRaceState,
    driver_setup: dict[str, dict[str, Any]],
    decision_context: dict[str, dict[str, Any]],
) -> list[str]:
    current_lap = int(race_state.lap)
    forced_drivers: list[str] = []

    for state in list(race_state.iter_active()):
        setup = driver_setup[state.driver]
        context = decision_context[state.driver]
        mandatory_stop_pending = bool(context.get("mandatory_stop_pending", False))
        is_dry_race = bool(context.get("is_dry_race", False))
        deadline_lap = int(
            context.get(
                "mandatory_stop_deadline_lap",
                setup.get(
                    "mandatory_stop_deadline_lap",
                    compute_mandatory_stop_deadline_lap(
                        race_laps=int(setup["race_laps"]),
                        buffer_laps=int(setup.get("mandatory_stop_buffer_laps", 3)),
                    ),
                ),
            )
        )

        if not mandatory_stop_pending or not is_dry_race:
            continue
        if current_lap < deadline_lap:
            continue

        remaining_laps = max(0, int(setup["race_laps"]) - current_lap + 1)
        forced_compound = _choose_forced_mandatory_compound(
            current_compound=state.current_compound,
            remaining_laps=remaining_laps,
        )
        race_state.apply_pit_stop(driver=state.driver, compound=forced_compound)
        forced_drivers.append(state.driver)

    return forced_drivers


def run_live_strategy_engine(
    *,
    race_state: LiveRaceState,
    events_df: pd.DataFrame,
    driver_setup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, str | int | float | bool]] = []
    for state in race_state.iter_active():
        setup = dict(driver_setup[state.driver])
        if "mandatory_stop_deadline_lap" not in setup:
            setup["mandatory_stop_deadline_lap"] = compute_mandatory_stop_deadline_lap(
                race_laps=int(setup["race_laps"]),
                buffer_laps=int(setup.get("mandatory_stop_buffer_laps", 3)),
            )
            driver_setup[state.driver] = setup

    decision_context = build_initial_decision_context(
        race_state=race_state,
        driver_setup=driver_setup,
    )

    for lap in range(1, int(race_state.race_laps) + 1):
        race_state.lap = lap

        apply_events_for_lap(race_state=race_state, events_df=events_df, lap=lap)
        sync_decision_context_after_events(
            race_state=race_state,
            decision_context=decision_context,
            driver_setup=driver_setup,
        )
        forced_compulsory_stops = _apply_compulsory_mandatory_stops(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )
        if forced_compulsory_stops:
            sync_decision_context_after_events(
                race_state=race_state,
                decision_context=decision_context,
                driver_setup=driver_setup,
            )

        active_states = list(race_state.iter_active())
        if not active_states:
            break

        decisions = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )

        for state in active_states:
            state_row = race_state.to_output_row(state)
            # Copy to avoid mutating cached decision payload while applying display guardrails.
            decision_row = dict(decisions[state.driver])
            # Guardrail to keep live overtake probabilities in a realistic display range.
            decision_row["overtake_probability_if_pit_now"] = max(
                0.05,
                min(0.95, float(decision_row["overtake_probability_if_pit_now"])),
            )
            decision_row["overtake_probability_if_stay_out"] = max(
                0.05,
                min(0.95, float(decision_row["overtake_probability_if_stay_out"])),
            )
            rows.append({**state_row, **decision_row})

        # Keep minimal state progression behavior intact.
        for state in race_state.iter_active():
            state.tire_age += 1

    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=LIVE_OUTPUT_COLUMNS)

    return output[LIVE_OUTPUT_COLUMNS].copy()
