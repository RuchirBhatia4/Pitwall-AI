import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from fastapi.testclient import TestClient

from src.api.contracts import CALIBRATED_REQUIRED_COLUMNS, STRATEGY_REQUIRED_COLUMNS
from src.api.main import app
from src.api.services import (
    dataframe_response,
    load_calibrated_predictions,
    load_strategy_recommendations,
    summarize_dashboard,
)


class TestApiContract(unittest.TestCase):
    def test_health_endpoint(self):
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "pitwall-api"})

    def test_storage_status_defaults_to_csv(self):
        client = TestClient(app)

        response = client.get("/api/storage/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "csv")
        self.assertFalse(payload["database_active"])

    def test_dataframe_response_is_stable(self):
        df = pd.DataFrame([{"driver": "VER", "practice_confidence": 0.75}])

        payload = dataframe_response(df)

        self.assertEqual(payload["columns"], ["driver", "practice_confidence"])
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["driver"], "VER")

    def test_dashboard_summary_counts_outputs(self):
        calibrated = pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "MEDIUM",
                },
                {
                    "race": "Australia",
                    "driver": "LEC",
                    "team": "Ferrari",
                    "compound": "HARD",
                },
            ]
        )
        strategies = pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "strategy": "MEDIUM-HARD",
                    "is_recommended": True,
                },
                {
                    "race": "Australia",
                    "driver": "LEC",
                    "team": "Ferrari",
                    "strategy": "HARD-MEDIUM",
                    "is_recommended": False,
                },
            ]
        )

        summary = summarize_dashboard(calibrated_df=calibrated, strategy_df=strategies)

        self.assertEqual(summary["drivers"], 2)
        self.assertEqual(summary["teams"], 2)
        self.assertEqual(summary["calibrated_rows"], 2)
        self.assertEqual(summary["strategy_rows"], 2)
        self.assertEqual(summary["recommended_strategies"], 1)

    def test_csv_loaders_validate_contracts(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            calibrated_path = root / "calibrated.csv"
            strategy_path = root / "strategy.csv"

            pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "compound": "MEDIUM",
                        "historical_deg_estimate": 0.12,
                        "practice_deg_estimate": 0.14,
                        "practice_confidence": 0.8,
                        "calibrated_deg_estimate": 0.13,
                        "clean_laps_used": 10,
                        "sessions_used": 2,
                        "calibration_method": "practice_confidence_weighted_slope_blend_v1",
                    }
                ],
                columns=CALIBRATED_REQUIRED_COLUMNS,
            ).to_csv(calibrated_path, index=False)

            pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "strategy": "MEDIUM-HARD",
                        "pit_laps": "28",
                        "expected_total_time": 5260.0,
                        "time_delta_to_best": 0.0,
                        "risk_score": 0.2,
                        "is_recommended": "true",
                    }
                ],
                columns=STRATEGY_REQUIRED_COLUMNS,
            ).to_csv(strategy_path, index=False)

            calibrated = load_calibrated_predictions(calibrated_path)
            strategies = load_strategy_recommendations(strategy_path)

            self.assertEqual(calibrated.columns.tolist(), CALIBRATED_REQUIRED_COLUMNS)
            self.assertEqual(strategies.columns.tolist(), STRATEGY_REQUIRED_COLUMNS)
            self.assertTrue(bool(strategies.loc[0, "is_recommended"]))


if __name__ == "__main__":
    unittest.main()
