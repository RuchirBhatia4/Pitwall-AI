import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import joblib
import numpy as np
import pandas as pd

from src.pipeline.generate_calibrated_predictions import (
    DEFAULT_PRACTICE_CSV_PATH,
    generate_calibrated_prediction_report,
    summarize_calibration_report,
)


class DummyBestDegradationModel:
    """
    Minimal pickleable model object with sklearn-like predict API.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Linear-in-tyre-life prediction so slope extraction is deterministic.
        tyre_life = pd.to_numeric(X["TyreLife"], errors="coerce").fillna(0).to_numpy(dtype=float)
        return (0.20 * tyre_life) + 0.50


class TestGenerateCalibratedPredictions(unittest.TestCase):
    def _build_historical_curves(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "Driver": "VER",
                    "Team": "Red Bull Racing",
                    "Compound": "MEDIUM",
                    "TyreLife": tyre_life,
                    "predicted_degradation_seconds": 0.10 * (tyre_life - 1),
                }
                for tyre_life in range(1, 6)
            ]
            + [
                {
                    "race": "Australia",
                    "Driver": "BOT",
                    "Team": "Kick Sauber",
                    "Compound": "MEDIUM",
                    "TyreLife": tyre_life,
                    "predicted_degradation_seconds": 0.09 * (tyre_life - 1),
                }
                for tyre_life in range(1, 6)
            ]
        )

    def _expected_output_columns(self) -> list[str]:
        return [
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

    def test_pipeline_works_with_model_present(self):
        self.assertEqual(
            str(DEFAULT_PRACTICE_CSV_PATH),
            "data/raw/practice_uploads/practice_long_run_sample.csv",
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            historical_path = tmp_dir_path / "historical_curves.csv"
            output_path = tmp_dir_path / "calibrated_report.csv"
            model_path = tmp_dir_path / "best_degradation_model.pkl"

            historical = self._build_historical_curves()
            historical.to_csv(historical_path, index=False)
            joblib.dump(DummyBestDegradationModel(), model_path)

            report = generate_calibrated_prediction_report(
                practice_csv_path=DEFAULT_PRACTICE_CSV_PATH,
                historical_predictions_path=historical_path,
                best_model_path=model_path,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertFalse(report.empty)
            self.assertEqual(report.columns.tolist(), self._expected_output_columns())

            # With dummy model, historical slope should be approximately 0.2.
            ver_row = report[report["driver"] == "VER"].iloc[0]
            self.assertTrue(np.isclose(float(ver_row["historical_deg_estimate"]), 0.2, atol=1e-6))

    def test_pipeline_works_when_model_missing_and_falls_back(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            historical_path = tmp_dir_path / "historical_curves.csv"
            output_path = tmp_dir_path / "calibrated_report.csv"
            missing_model_path = tmp_dir_path / "missing_best_model.pkl"

            historical = self._build_historical_curves()
            historical.to_csv(historical_path, index=False)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                report = generate_calibrated_prediction_report(
                    practice_csv_path=DEFAULT_PRACTICE_CSV_PATH,
                    historical_predictions_path=historical_path,
                    best_model_path=missing_model_path,
                    output_path=output_path,
                )

            self.assertTrue(output_path.exists())
            self.assertFalse(report.empty)
            self.assertEqual(report.columns.tolist(), self._expected_output_columns())

            self.assertTrue(
                any("Best degradation model not found" in str(w.message) for w in captured)
            )

            # Fallback historical curve for BOT should remain unchanged by no-practice path.
            bot_row = report[report["driver"] == "BOT"].iloc[0]
            self.assertEqual(bot_row["clean_laps_used"], 0)
            self.assertEqual(bot_row["sessions_used"], 0)
            self.assertEqual(
                bot_row["calibrated_deg_estimate"],
                bot_row["historical_deg_estimate"],
            )

    def test_output_columns_remain_unchanged(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            historical_path = tmp_dir_path / "historical_curves.csv"
            output_path = tmp_dir_path / "calibrated_report.csv"

            historical = self._build_historical_curves()
            historical.to_csv(historical_path, index=False)

            report = generate_calibrated_prediction_report(
                practice_csv_path=DEFAULT_PRACTICE_CSV_PATH,
                historical_predictions_path=historical_path,
                best_model_path=tmp_dir_path / "missing_model.pkl",
                output_path=output_path,
            )

            self.assertEqual(report.columns.tolist(), self._expected_output_columns())
            saved = pd.read_csv(output_path)
            self.assertEqual(saved.columns.tolist(), self._expected_output_columns())

    def test_existing_calibration_expectations_still_hold(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            historical_path = tmp_dir_path / "historical_curves.csv"
            output_path = tmp_dir_path / "calibrated_report.csv"

            historical = self._build_historical_curves()
            historical.to_csv(historical_path, index=False)

            report = generate_calibrated_prediction_report(
                practice_csv_path=DEFAULT_PRACTICE_CSV_PATH,
                historical_predictions_path=historical_path,
                best_model_path=tmp_dir_path / "missing_model.pkl",
                output_path=output_path,
            )

            ver_row = report[report["driver"] == "VER"].iloc[0]
            self.assertGreater(ver_row["clean_laps_used"], 0)
            self.assertGreater(ver_row["sessions_used"], 0)
            self.assertGreater(ver_row["practice_confidence"], 0.0)
            self.assertTrue(
                ver_row["calibration_method"].startswith(
                    "practice_confidence_weighted_slope_blend"
                )
            )

            bot_row = report[report["driver"] == "BOT"].iloc[0]
            self.assertEqual(bot_row["clean_laps_used"], 0)
            self.assertEqual(bot_row["sessions_used"], 0)
            self.assertEqual(bot_row["practice_confidence"], 0.0)
            self.assertEqual(
                bot_row["calibrated_deg_estimate"],
                bot_row["historical_deg_estimate"],
            )
            self.assertEqual(
                bot_row["calibration_method"],
                "historical_only_no_practice_data",
            )

            self.assertTrue(
                np.isclose(
                    float(report.loc[report["driver"] == "BOT", "calibrated_deg_estimate"].iloc[0]),
                    float(report.loc[report["driver"] == "BOT", "historical_deg_estimate"].iloc[0]),
                )
            )

    def test_summary_reporting_metrics(self):
        report = pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "compound": "MEDIUM",
                    "historical_deg_estimate": 0.12,
                    "practice_deg_estimate": 0.14,
                    "practice_confidence": 0.90,
                    "calibrated_deg_estimate": 0.13,
                    "clean_laps_used": 8,
                    "sessions_used": 2,
                    "calibration_method": "practice_confidence_weighted_slope_blend_v1",
                },
                {
                    "race": "Australia",
                    "driver": "LEC",
                    "team": "Ferrari",
                    "compound": "HARD",
                    "historical_deg_estimate": 0.10,
                    "practice_deg_estimate": 0.08,
                    "practice_confidence": 0.20,
                    "calibrated_deg_estimate": 0.09,
                    "clean_laps_used": 3,
                    "sessions_used": 1,
                    "calibration_method": "practice_confidence_weighted_slope_blend_v1",
                },
            ]
        )

        practice_df = pd.DataFrame(
            [
                {"session": "FP3"},
                {"session": "FP2"},
                {"session": "FP3"},
            ]
        )
        summary = summarize_calibration_report(report, practice_df=practice_df)

        self.assertEqual(summary["drivers_covered"], "LEC, VER")
        self.assertEqual(summary["compounds_covered"], "HARD, MEDIUM")
        self.assertEqual(summary["sessions_covered"], "FP2, FP3")
        self.assertTrue(np.isclose(summary["average_practice_confidence"], 0.55))
        self.assertEqual(
            summary["highest_confidence_row"],
            "Australia | VER | Red Bull Racing | MEDIUM | conf=0.900",
        )
        self.assertEqual(
            summary["lowest_confidence_row"],
            "Australia | LEC | Ferrari | HARD | conf=0.200",
        )


if __name__ == "__main__":
    unittest.main()
