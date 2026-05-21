import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd

from src.pipeline.generate_strategy_summary import (
    OUTPUT_COLUMNS,
    generate_strategy_summary,
)


class TestStrategySummary(unittest.TestCase):
    def _sample_strategy_rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "starting_position": 1,
                    "strategy": "MEDIUM-HARD",
                    "pit_laps": "30",
                    "scenario_name": "no_safety_car",
                    "expected_total_time": 5310.0,
                    "is_recommended": False,
                    "stops": 1,
                    "safety_car_gain_seconds": 0.0,
                    "recommendation_reason": "Alternative scenario plan benchmark.",
                },
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "starting_position": 1,
                    "strategy": "HARD-MEDIUM",
                    "pit_laps": "29",
                    "scenario_name": "no_safety_car",
                    "expected_total_time": 5305.0,
                    "is_recommended": True,
                    "stops": 1,
                    "safety_car_gain_seconds": 0.0,
                    "recommendation_reason": "Best no-safety-car race time with one-stop strategy.",
                },
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "starting_position": 1,
                    "strategy": "SOFT-HARD",
                    "pit_laps": "15",
                    "scenario_name": "early_safety_car",
                    "expected_total_time": 5292.0,
                    "is_recommended": True,
                    "stops": 1,
                    "safety_car_gain_seconds": 13.2,
                    "recommendation_reason": "Safety car window creates a cheaper stop opportunity around lap 15.",
                },
                {
                    "race": "Australia",
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "starting_position": 1,
                    "strategy": "MEDIUM-HARD-MEDIUM",
                    "pit_laps": "20-33",
                    "scenario_name": "mid_safety_car",
                    "expected_total_time": 5295.0,
                    "is_recommended": False,
                    "stops": 2,
                    "safety_car_gain_seconds": 7.0,
                    "recommendation_reason": "Alternative scenario plan benchmark.",
                },
                {
                    "race": "Australia",
                    "driver": "HAM",
                    "team": "Mercedes",
                    "starting_position": 5,
                    "strategy": "MEDIUM-HARD",
                    "pit_laps": "28",
                    "scenario_name": "no_safety_car",
                    "expected_total_time": 5350.0,
                    "is_recommended": True,
                    "stops": 1,
                    "safety_car_gain_seconds": 0.0,
                    "recommendation_reason": "Best expected race time for this scenario.",
                },
                {
                    "race": "Australia",
                    "driver": "HAM",
                    "team": "Mercedes",
                    "starting_position": 5,
                    "strategy": "SOFT-MEDIUM",
                    "pit_laps": "15",
                    "scenario_name": "early_safety_car",
                    "expected_total_time": 5362.0,
                    "is_recommended": False,
                    "stops": 1,
                    "safety_car_gain_seconds": -1.5,
                    "recommendation_reason": "Alternative scenario plan benchmark.",
                },
            ]
        )

    def test_strategy_summary_contract_and_one_row_per_driver(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source = tmp / "strategy_recommendations.csv"
            output = tmp / "strategy_summary.csv"

            self._sample_strategy_rows().to_csv(source, index=False)

            summary = generate_strategy_summary(
                strategy_recommendations_path=source,
                output_path=output,
            )

            self.assertTrue(output.exists())
            self.assertEqual(summary.columns.tolist(), OUTPUT_COLUMNS)
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary["driver"].nunique(), 2)

            saved = pd.read_csv(output)
            self.assertEqual(saved.columns.tolist(), OUTPUT_COLUMNS)
            self.assertEqual(len(saved), 2)

    def test_no_safety_car_and_best_overall_logic(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source = tmp / "strategy_recommendations.csv"
            output = tmp / "strategy_summary.csv"

            self._sample_strategy_rows().to_csv(source, index=False)

            summary = generate_strategy_summary(
                strategy_recommendations_path=source,
                output_path=output,
            )

            ver = summary[summary["driver"] == "VER"].iloc[0]

            # no_safety_car selection must come from scenario=no_safety_car and is_recommended=True
            self.assertEqual(ver["best_no_safety_car_strategy"], "HARD-MEDIUM")
            self.assertEqual(ver["best_no_safety_car_pit_laps"], "29")
            self.assertTrue(np.isclose(float(ver["best_no_safety_car_time"]), 5305.0))

            # best_overall should be min expected_total_time across all scenarios
            self.assertEqual(ver["best_overall_strategy"], "SOFT-HARD")
            self.assertEqual(ver["best_overall_scenario"], "early_safety_car")
            self.assertEqual(ver["best_overall_pit_laps"], "15")
            self.assertTrue(np.isclose(float(ver["best_overall_time"]), 5292.0))

            # derived fields from best overall strategy
            self.assertEqual(int(ver["recommended_stop_count"]), 1)
            self.assertEqual(ver["starting_compound_recommendation"], "SOFT")

            # safety car opportunity should be scenario with max safety gain
            self.assertEqual(ver["best_safety_car_opportunity"], "early_safety_car")
            self.assertTrue(np.isclose(float(ver["max_safety_car_gain_seconds"]), 13.2))

            ham = summary[summary["driver"] == "HAM"].iloc[0]
            self.assertEqual(ham["best_safety_car_opportunity"], "early_safety_car")
            self.assertTrue(np.isclose(float(ham["max_safety_car_gain_seconds"]), -1.5))

    def test_training_and_holdout_scripts_unchanged(self):
        repo_root = Path(__file__).resolve().parents[1]
        tracked_files = [
            "src/models/train_degradation.py",
            "src/models/train_degradation_holdout.py",
            "src/models/train_degradation_best.py",
        ]

        try:
            for rel_path in tracked_files:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD", "--", rel_path],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    self.skipTest("git status check unavailable in this environment")
                self.assertEqual(
                    result.stdout.strip(),
                    "",
                    msg=f"Expected no local changes in {rel_path}",
                )
        except FileNotFoundError:
            self.skipTest("git is not available in this environment")


if __name__ == "__main__":
    unittest.main()
