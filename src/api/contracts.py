from __future__ import annotations

CALIBRATED_REQUIRED_COLUMNS = [
    "race",
    "driver",
    "team",
    "compound",
    "historical_deg_estimate",
    "practice_deg_estimate",
    "practice_confidence",
    "calibrated_deg_estimate",
    "clean_laps_used",
    "sessions_used",
    "calibration_method",
]

STRATEGY_REQUIRED_COLUMNS = [
    "race",
    "driver",
    "team",
    "strategy",
    "pit_laps",
    "expected_total_time",
    "time_delta_to_best",
    "risk_score",
    "is_recommended",
]

LIVE_EVENT_COLUMNS = [
    "lap",
    "event_type",
    "driver",
    "target_driver",
    "compound",
    "notes",
]

LIVE_EVENT_TYPES = [
    "safety_car_start",
    "safety_car_end",
    "pit_stop",
    "overtake",
    "retirement",
]
