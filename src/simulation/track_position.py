def estimate_track_position_penalty_per_lap(
    starting_position: int,
    overtaking_difficulty: float,
) -> float:
    """
    Simple first-order track-position penalty model.
    """
    return float(starting_position) * float(overtaking_difficulty) * 0.015


def estimate_track_position_penalty_total(
    starting_position: int,
    overtaking_difficulty: float,
    race_laps: int,
) -> float:
    return estimate_track_position_penalty_per_lap(
        starting_position=starting_position,
        overtaking_difficulty=overtaking_difficulty,
    ) * float(race_laps)
