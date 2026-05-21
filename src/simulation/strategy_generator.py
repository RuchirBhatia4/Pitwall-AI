from typing import Any


LEGAL_STRATEGIES = [
    "SOFT-HARD",
    "SOFT-MEDIUM",
    "MEDIUM-HARD",
    "HARD-MEDIUM",
    "SOFT-MEDIUM-HARD",
    "SOFT-HARD-SOFT",
    "MEDIUM-HARD-SOFT",
    "MEDIUM-HARD-MEDIUM",
    "HARD-MEDIUM-SOFT",
    "SOFT-MEDIUM-SOFT",
]

# Modeling assumption:
# These are conservative race-planning stint limits, not hard FIA rules. They
# keep the optimizer from treating linear degradation as if tires can be pushed
# indefinitely without a cliff, graining, thermal drop-off, or drivability loss.
DEFAULT_MAX_STINT_LAPS_BY_COMPOUND = {
    "SOFT": 25,
    "MEDIUM": 30,
    "HARD": 42,
}

ONE_STOP_WINDOWS = {
    "SOFT-HARD": (8, 25),
    "SOFT-MEDIUM": (8, 22),
    "MEDIUM-HARD": (15, 35),
    "HARD-MEDIUM": (25, 45),
}

TWO_STOP_FIRST_WINDOW = (8, 28)
TWO_STOP_SECOND_WINDOW = (25, 50)

MIN_FIRST_STINT_LAPS = 6
MIN_LAST_STINT_LAPS = 6
MIN_SECOND_STOP_GAP = 8


def parse_strategy_compounds(strategy: str) -> list[str]:
    return [compound.strip().upper() for compound in strategy.split("-") if compound.strip()]


def _clamp_window(window: tuple[int, int], race_laps: int) -> tuple[int, int] | None:
    start, end = window
    start = max(int(start), MIN_FIRST_STINT_LAPS)
    end = min(int(end), int(race_laps) - MIN_LAST_STINT_LAPS)

    if start > end:
        return None

    return (start, end)


def _within_compound_stint_limits(
    *,
    race_laps: int,
    compounds: list[str],
    pit_laps: list[int],
    max_stint_laps_by_compound: dict[str, int] | None = None,
) -> bool:
    max_stint_laps = max_stint_laps_by_compound or DEFAULT_MAX_STINT_LAPS_BY_COMPOUND
    pits = sorted(int(lap) for lap in pit_laps)
    boundaries = [0] + pits + [int(race_laps)]
    stint_lengths = [
        boundaries[idx + 1] - boundaries[idx]
        for idx in range(len(boundaries) - 1)
    ]

    if len(stint_lengths) != len(compounds):
        return False

    for compound, stint_length in zip(compounds, stint_lengths):
        limit = max_stint_laps.get(str(compound).upper())
        if limit is not None and int(stint_length) > int(limit):
            return False

    return True


def generate_one_stop_pit_lap_options(
    strategy: str,
    race_laps: int,
    max_stint_laps_by_compound: dict[str, int] | None = None,
) -> list[list[int]]:
    window = ONE_STOP_WINDOWS.get(strategy)
    if window is None:
        raise ValueError(f"No one-stop pit window configured for strategy '{strategy}'.")

    clamped = _clamp_window(window, race_laps)
    if clamped is None:
        return []

    start, end = clamped
    compounds = parse_strategy_compounds(strategy)
    return [
        [lap]
        for lap in range(start, end + 1)
        if _within_compound_stint_limits(
            race_laps=race_laps,
            compounds=compounds,
            pit_laps=[lap],
            max_stint_laps_by_compound=max_stint_laps_by_compound,
        )
    ]


def generate_two_stop_pit_lap_options(
    race_laps: int,
    compounds: list[str] | None = None,
    first_window: tuple[int, int] = TWO_STOP_FIRST_WINDOW,
    second_window: tuple[int, int] = TWO_STOP_SECOND_WINDOW,
    max_stint_laps_by_compound: dict[str, int] | None = None,
) -> list[list[int]]:
    first_clamped = _clamp_window(first_window, race_laps)
    second_clamped = _clamp_window(second_window, race_laps)

    if first_clamped is None or second_clamped is None:
        return []

    first_start, first_end = first_clamped
    second_start, second_end = second_clamped

    options: list[list[int]] = []
    for first_stop in range(first_start, first_end + 1):
        second_min = max(second_start, first_stop + MIN_SECOND_STOP_GAP)
        for second_stop in range(second_min, second_end + 1):
            if first_stop < MIN_FIRST_STINT_LAPS:
                continue
            if second_stop - first_stop < MIN_SECOND_STOP_GAP:
                continue
            if race_laps - second_stop < MIN_LAST_STINT_LAPS:
                continue
            if compounds is not None and not _within_compound_stint_limits(
                race_laps=race_laps,
                compounds=compounds,
                pit_laps=[first_stop, second_stop],
                max_stint_laps_by_compound=max_stint_laps_by_compound,
            ):
                continue
            options.append([first_stop, second_stop])

    return options


def _default_option(pit_lap_options: list[list[int]]) -> list[int]:
    if not pit_lap_options:
        return []
    return pit_lap_options[len(pit_lap_options) // 2]


def generate_legal_strategy_candidates(
    race_laps: int,
    strategies: list[str] | None = None,
    starting_compound: str | None = None,
    max_stint_laps_by_compound: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Generates legal strategy candidates with dynamic pit-lap options.
    """
    strategy_list = strategies or LEGAL_STRATEGIES
    required_start = str(starting_compound).strip().upper() if starting_compound else None
    candidates: list[dict[str, Any]] = []

    for strategy in strategy_list:
        compounds = parse_strategy_compounds(strategy)
        if required_start and compounds and compounds[0] != required_start:
            continue

        num_stops = max(0, len(compounds) - 1)

        if num_stops == 1:
            pit_lap_options = generate_one_stop_pit_lap_options(
                strategy,
                race_laps,
                max_stint_laps_by_compound=max_stint_laps_by_compound,
            )
        elif num_stops == 2:
            pit_lap_options = generate_two_stop_pit_lap_options(
                race_laps,
                compounds=compounds,
                max_stint_laps_by_compound=max_stint_laps_by_compound,
            )
        else:
            pit_lap_options = [[]]

        candidates.append(
            {
                "strategy": strategy,
                "compounds": compounds,
                "pit_laps": _default_option(pit_lap_options),
                "pit_lap_options": pit_lap_options,
                "stops": num_stops,
                "starting_compound": compounds[0] if compounds else "UNKNOWN",
            }
        )

    return candidates
