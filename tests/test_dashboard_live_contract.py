import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.streamlit_app import (
    LIVE_MISSING_DATA_INSTRUCTIONS,
    LIVE_REQUIRED_COLUMNS,
    get_live_missing_message,
    load_live_simulation_data,
    validate_dataframe_contract,
)
from src.pipeline.run_live_race_simulation import OUTPUT_COLUMNS


class TestDashboardLiveContract(unittest.TestCase):
    def test_live_required_columns_match_expected_contract(self):
        expected = [
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
        self.assertEqual(LIVE_REQUIRED_COLUMNS, expected)

    def test_live_contract_stable_against_pipeline_output_columns(self):
        self.assertEqual(LIVE_REQUIRED_COLUMNS, OUTPUT_COLUMNS)

    def test_load_live_simulation_data_validates_contract(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "live_race_simulation.csv"

            row = {column: 0 for column in LIVE_REQUIRED_COLUMNS}
            row.update(
                {
                    "race": "Miami",
                    "lap": 12,
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "current_position": 1,
                    "current_compound": "MEDIUM",
                    "tire_age": 14,
                    "stops_made": 1,
                    "under_safety_car": False,
                    "setup_source": "standings_seeded",
                    "best_next_compound": "HARD",
                    "best_pit_window": "18-22",
                    "best_action_now": "stay_out",
                    "recommendation_reason": "Stay out protects track position.",
                }
            )
            pd.DataFrame([row]).to_csv(output_path, index=False)

            loaded = load_live_simulation_data(live_output_path=output_path)
            validate_dataframe_contract(loaded, LIVE_REQUIRED_COLUMNS, "live_race_simulation.csv")
            self.assertEqual(loaded.columns.tolist(), LIVE_REQUIRED_COLUMNS)

    def test_missing_live_output_message_contains_commands(self):
        with TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "does_not_exist.csv"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_live_simulation_data(live_output_path=missing_path)

            message = str(ctx.exception)
            self.assertIn(get_live_missing_message(), message)
            self.assertIn(
                "python -m src.data.fetch_official_f1_setup --race Miami --season 2026",
                message,
            )
            self.assertIn(
                "venv/bin/python -m src.pipeline.run_live_race_simulation",
                message,
            )
            self.assertIn("fetch_official_f1_setup", LIVE_MISSING_DATA_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
