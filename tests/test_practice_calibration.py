import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.practice_loader import (
    filter_clean_practice_laps,
    load_practice_csv,
)
from src.models.practice_calibration import (
    aggregate_practice_degradation,
    blend_practice_with_historical_predictions,
    calibrate_predictions_from_practice_csvs,
    estimate_session_degradation,
)


SAMPLE_CSV_PATH = Path("data/processed/practice/practice_long_run_sample.csv")


class TestPracticeCalibration(unittest.TestCase):
    def test_loader_and_filtering_on_sample(self):
        raw = load_practice_csv(SAMPLE_CSV_PATH)

        self.assertIn("source_file", raw.columns)
        self.assertIn("driver", raw.columns)
        self.assertIn("compound", raw.columns)
        self.assertTrue((raw["driver"].str.upper() == raw["driver"]).all())
        self.assertTrue((raw["compound"].str.upper() == raw["compound"]).all())

        filtered = filter_clean_practice_laps(raw)

        # 24 total rows in sample, 4 intentionally noisy rows removed.
        self.assertEqual(len(filtered), 20)
        self.assertTrue(filtered["clean_lap"].all())
        self.assertFalse(filtered["traffic_affected"].any())
        self.assertFalse(filtered["drs_used"].any())

    def test_session_slope_and_confidence_estimation(self):
        filtered = filter_clean_practice_laps(load_practice_csv(SAMPLE_CSV_PATH))
        session_estimates = estimate_session_degradation(filtered, min_clean_laps=3)

        # Expected clean groups in the sample:
        # VER FP1/FP2/FP3 on MEDIUM and LEC FP2 on HARD.
        self.assertEqual(len(session_estimates), 4)

        self.assertTrue((session_estimates["degradation_slope_per_lap"] > 0).all())
        self.assertTrue(((session_estimates["confidence"] >= 0) & (session_estimates["confidence"] <= 1)).all())

        ver = session_estimates[
            (session_estimates["driver"] == "VER")
            & (session_estimates["compound"] == "MEDIUM")
        ]

        fp1_conf = ver.loc[ver["session"] == "FP1", "confidence"].iloc[0]
        fp3_conf = ver.loc[ver["session"] == "FP3", "confidence"].iloc[0]

        # FP3 receives higher relevance weight than FP1 in this calibration design.
        self.assertGreater(fp3_conf, fp1_conf)

    def test_blend_applies_practice_signal_and_preserves_missing_groups(self):
        filtered = filter_clean_practice_laps(load_practice_csv(SAMPLE_CSV_PATH))
        session_estimates = estimate_session_degradation(filtered, min_clean_laps=3)
        practice_aggregate = aggregate_practice_degradation(session_estimates)

        historical = pd.DataFrame(
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
                    "Driver": "HAM",
                    "Team": "Ferrari",
                    "Compound": "MEDIUM",
                    "TyreLife": tyre_life,
                    "predicted_degradation_seconds": 0.09 * (tyre_life - 1),
                }
                for tyre_life in range(1, 6)
            ]
        )

        blended = blend_practice_with_historical_predictions(
            historical_predictions=historical,
            practice_aggregate=practice_aggregate,
            prediction_col="predicted_degradation_seconds",
            tyre_life_col="TyreLife",
            output_col="practice_calibrated_prediction",
            max_practice_blend_weight=0.75,
        )

        self.assertIn("practice_calibrated_prediction", blended.columns)
        self.assertIn("practice_weight", blended.columns)

        ver_last = blended[
            (blended["Driver"] == "VER")
            & (blended["TyreLife"] == 5)
        ].iloc[0]

        # Practice sample implies steeper degradation than the historical base curve.
        self.assertGreater(
            ver_last["practice_calibrated_prediction"],
            ver_last["predicted_degradation_seconds"],
        )

        ham_last = blended[
            (blended["Driver"] == "HAM")
            & (blended["TyreLife"] == 5)
        ].iloc[0]

        # No HAM practice data in sample: output should remain identical to historical value.
        self.assertTrue(
            np.isclose(
                ham_last["practice_calibrated_prediction"],
                ham_last["predicted_degradation_seconds"],
            )
        )

    def test_end_to_end_csv_calibration(self):
        historical = pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "Driver": "VER",
                    "Team": "Red Bull Racing",
                    "Compound": "MEDIUM",
                    "TyreLife": tyre_life,
                    "predicted_degradation_seconds": 0.12 * (tyre_life - 1),
                }
                for tyre_life in range(1, 6)
            ]
        )

        calibrated_df, session_df, aggregate_df = calibrate_predictions_from_practice_csvs(
            practice_csv_paths=[SAMPLE_CSV_PATH],
            historical_predictions=historical,
        )

        self.assertFalse(calibrated_df.empty)
        self.assertFalse(session_df.empty)
        self.assertFalse(aggregate_df.empty)
        self.assertIn("practice_calibrated_prediction", calibrated_df.columns)


if __name__ == "__main__":
    unittest.main()
