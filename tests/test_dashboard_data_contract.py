import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.streamlit_app import (
    CALIBRATED_REQUIRED_COLUMNS,
    STRATEGY_REQUIRED_COLUMNS,
    load_dashboard_data,
    validate_dataframe_contract,
)


class TestDashboardDataContract(unittest.TestCase):
    def test_calibrated_contract_columns_match_expected(self):
        expected = [
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
        self.assertEqual(CALIBRATED_REQUIRED_COLUMNS, expected)

    def test_strategy_contract_columns_match_expected(self):
        expected = [
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
        self.assertEqual(STRATEGY_REQUIRED_COLUMNS, expected)

    def test_validate_dataframe_contract_passes_with_required_columns(self):
        df = pd.DataFrame([{col: 1 for col in CALIBRATED_REQUIRED_COLUMNS}])
        validate_dataframe_contract(
            df,
            CALIBRATED_REQUIRED_COLUMNS,
            "calibrated_degradation_predictions.csv",
        )

    def test_validate_dataframe_contract_fails_when_columns_missing(self):
        bad_df = pd.DataFrame([{"race": "Australia", "driver": "VER"}])
        with self.assertRaises(ValueError):
            validate_dataframe_contract(
                bad_df,
                STRATEGY_REQUIRED_COLUMNS,
                "strategy_recommendations.csv",
            )

    def test_load_dashboard_data_contract_validation(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            calibrated_path = root / "calibrated_degradation_predictions.csv"
            strategy_path = root / "strategy_recommendations.csv"

            calibrated_df = pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "compound": "MEDIUM",
                        "historical_deg_estimate": 0.08,
                        "practice_deg_estimate": 0.09,
                        "practice_confidence": 0.75,
                        "calibrated_deg_estimate": 0.085,
                        "clean_laps_used": 12,
                        "sessions_used": 3,
                        "calibration_method": "practice_confidence_weighted_slope_blend_v1",
                    }
                ]
            )
            calibrated_df.to_csv(calibrated_path, index=False)

            strategy_df = pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "strategy": "MEDIUM-HARD",
                        "pit_laps": "28",
                        "expected_total_time": 5260.0,
                        "time_delta_to_best": 0.0,
                        "risk_score": 0.24,
                        "is_recommended": True,
                    }
                ]
            )
            strategy_df.to_csv(strategy_path, index=False)

            loaded_calibrated, loaded_strategy = load_dashboard_data(
                calibrated_path=calibrated_path,
                strategy_path=strategy_path,
            )

            self.assertEqual(loaded_calibrated.columns.tolist(), CALIBRATED_REQUIRED_COLUMNS)
            self.assertEqual(loaded_strategy.columns.tolist(), STRATEGY_REQUIRED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
