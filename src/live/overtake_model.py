from __future__ import annotations


REALISTIC_PROB_MIN = 0.05
REALISTIC_PROB_MAX = 0.95


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _compound_freshness(compound: str, tire_age: int) -> float:
    cpd = str(compound).upper()
    age = max(0, int(tire_age))

    if cpd == "SOFT":
        base = 1.00
    elif cpd == "HARD":
        base = 0.86
    else:
        base = 0.93

    freshness_decay = _clamp(age / 30.0, 0.0, 1.0)
    return _clamp(base - (0.58 * freshness_decay), 0.10, 1.15)


def estimate_freshness_advantage(
    *,
    attacker_compound: str,
    attacker_tire_age: int,
    defender_compound: str,
    defender_tire_age: int,
) -> float:
    attacker_freshness = _compound_freshness(attacker_compound, attacker_tire_age)
    defender_freshness = _compound_freshness(defender_compound, defender_tire_age)
    return float(attacker_freshness - defender_freshness)


def estimate_overtake_probability(
    *,
    pace_delta_seconds: float,
    tire_age_delta: float,
    freshness_advantage: float,
    overtaking_difficulty: float,
    track_position_importance: float,
) -> float:
    # Positive pace_delta_seconds means attacker is faster than defender.
    normalized_pace = _clamp(float(pace_delta_seconds), -1.5, 1.5) / 1.5
    normalized_tire_age = _clamp(float(tire_age_delta), -20.0, 20.0) / 20.0
    normalized_freshness = _clamp(float(freshness_advantage), -1.0, 1.0)

    score = 0.50
    score += normalized_pace * 0.22
    score += normalized_tire_age * 0.12
    score += normalized_freshness * 0.20
    score -= _clamp(float(overtaking_difficulty), 0.0, 1.0) * 0.16
    score -= _clamp(float(track_position_importance), 0.0, 1.0) * 0.10

    return _clamp(score, REALISTIC_PROB_MIN, REALISTIC_PROB_MAX)


def estimate_recovery_probability(
    *,
    base_overtake_probability: float,
    expected_positions_to_recover: int,
    current_position: int,
    field_size: int,
    pace_delta_seconds: float,
    freshness_advantage: float,
    overtaking_difficulty: float,
    track_position_importance: float,
) -> float:
    positions = max(0, int(expected_positions_to_recover))
    field = max(2, int(field_size))
    current = _clamp(int(current_position), 1, field)

    pressure = positions / max(1.0, float(field - 1))
    front_runner_context = 1.0 - ((current - 1.0) / max(1.0, float(field - 1)))

    adjusted = _clamp(float(base_overtake_probability), REALISTIC_PROB_MIN, REALISTIC_PROB_MAX)
    adjusted += (_clamp(float(pace_delta_seconds), -1.2, 1.2) / 1.2) * 0.08
    adjusted += _clamp(float(freshness_advantage), -1.0, 1.0) * 0.10
    adjusted += front_runner_context * 0.06
    adjusted -= pressure * 0.22
    adjusted -= _clamp(float(overtaking_difficulty), 0.0, 1.0) * 0.14
    adjusted -= _clamp(float(track_position_importance), 0.0, 1.0) * 0.10

    return _clamp(adjusted, REALISTIC_PROB_MIN, REALISTIC_PROB_MAX)
