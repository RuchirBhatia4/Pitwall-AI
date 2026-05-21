from typing import Sequence


SAFETY_CAR_WINDOWS = {
    "no_safety_car": None,
    "early_safety_car": (8, 15),
    "mid_safety_car": (20, 32),
    "late_safety_car": (38, 48),
}


def get_scenario_names() -> list[str]:
    return list(SAFETY_CAR_WINDOWS.keys())


def get_safety_car_window(scenario_name: str) -> tuple[int, int] | None:
    if scenario_name not in SAFETY_CAR_WINDOWS:
        raise ValueError(f"Unknown safety-car scenario '{scenario_name}'.")
    return SAFETY_CAR_WINDOWS[scenario_name]


def lap_in_window(lap: int, window: tuple[int, int] | None) -> bool:
    if window is None:
        return False
    start, end = window
    return int(start) <= int(lap) <= int(end)


def best_window_lap(pit_laps: Sequence[int], window: tuple[int, int] | None) -> int | None:
    if window is None:
        return None

    in_window = [int(lap) for lap in pit_laps if lap_in_window(int(lap), window)]
    if not in_window:
        return None

    center = (window[0] + window[1]) / 2.0
    return min(in_window, key=lambda lap: abs(lap - center))
