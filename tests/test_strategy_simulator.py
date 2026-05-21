import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.pipeline.generate_strategy_recommendations import (
    OUTPUT_COLUMNS,
    generate_strategy_recommendations,
)
from src.simulation.race_simulator import simulate_strategy_scenario
from src.simulation.strategy_generator import (
    DEFAULT_MAX_STINT_LAPS_BY_COMPOUND,
    LEGAL_STRATEGIES,
    MIN_FIRST_STINT_LAPS,
    MIN_LAST_STINT_LAPS,
    MIN_SECOND_STOP_GAP,
    generate_legal_strategy_candidates,
)
from src.simulation.strategy_optimizer import (
    FALLBACK_COMPOUND_DEGRADATION,
    build_compound_degradation_map,
)


class TestStrategySimulatorV2(unittest.TestCase):
    def test_strategy_generation_returns_required_legal_strategies(self):
        candidates = generate_legal_strategy_candidates(race_laps=57)

        self.assertEqual(len(candidates), len(LEGAL_STRATEGIES))
        self.assertEqual([c["strategy"] for c in candidates], LEGAL_STRATEGIES)

    def test_one_stop_strategies_use_dynamic_pit_windows_and_stint_limits(self):
        candidates = generate_legal_strategy_candidates(race_laps=57)
        by_strategy = {candidate["strategy"]: candidate for candidate in candidates}

        soft_hard_options = by_strategy["SOFT-HARD"]["pit_lap_options"]
        self.assertGreater(len(soft_hard_options), 1)

        pit_laps = [option[0] for option in soft_hard_options]
        self.assertEqual(min(pit_laps), 15)
        self.assertEqual(max(pit_laps), 25)
        self.assertNotEqual(sorted(set(pit_laps)), [28])

        medium_hard_options = by_strategy["MEDIUM-HARD"]["pit_lap_options"]
        medium_hard_pits = [option[0] for option in medium_hard_options]
        self.assertLessEqual(
            max(medium_hard_pits),
            DEFAULT_MAX_STINT_LAPS_BY_COMPOUND["MEDIUM"],
        )

    def test_two_stop_strategies_produce_valid_pit_pairs(self):
        candidates = generate_legal_strategy_candidates(race_laps=57)
        two_stop = [item for item in candidates if item["stops"] == 2]

        self.assertTrue(two_stop)

        for candidate in two_stop:
            for pit1, pit2 in candidate["pit_lap_options"]:
                self.assertGreaterEqual(pit1, MIN_FIRST_STINT_LAPS)
                self.assertGreaterEqual(pit2 - pit1, MIN_SECOND_STOP_GAP)
                self.assertGreaterEqual(57 - pit2, MIN_LAST_STINT_LAPS)
                self.assertGreaterEqual(pit1, 8)
                self.assertLessEqual(pit1, 28)
                self.assertGreaterEqual(pit2, 25)
                self.assertLessEqual(pit2, 50)

    def test_strategy_generation_can_filter_by_actual_starting_compound(self):
        candidates = generate_legal_strategy_candidates(
            race_laps=57,
            starting_compound="MEDIUM",
        )

        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate["starting_compound"] == "MEDIUM" for candidate in candidates)
        )

    def test_safety_car_scenario_uses_cheaper_pit_loss_inside_window(self):
        no_sc = simulate_strategy_scenario(
            race_laps=40,
            strategy_compounds=["SOFT", "HARD"],
            pit_laps=[10],
            base_pace_seconds=92.0,
            pit_loss_green_seconds=22.0,
            pit_loss_safety_car_seconds=12.0,
            compound_degradation_map={"SOFT": 0.12, "HARD": 0.06},
            fallback_degradation=0.08,
            starting_position=4,
            overtaking_difficulty=0.7,
            scenario_name="no_safety_car",
        )

        early_sc = simulate_strategy_scenario(
            race_laps=40,
            strategy_compounds=["SOFT", "HARD"],
            pit_laps=[10],
            base_pace_seconds=92.0,
            pit_loss_green_seconds=22.0,
            pit_loss_safety_car_seconds=12.0,
            compound_degradation_map={"SOFT": 0.12, "HARD": 0.06},
            fallback_degradation=0.08,
            starting_position=4,
            overtaking_difficulty=0.7,
            scenario_name="early_safety_car",
        )

        self.assertLess(float(early_sc["pit_loss_used"]), float(no_sc["pit_loss_used"]))
        self.assertTrue(np.isclose(float(early_sc["pit_loss_used"]), 12.0))
        self.assertGreater(float(early_sc["safety_car_gain_seconds"]), 0.0)

    def test_fallback_degradation_estimates_for_missing_practice(self):
        empty = pd.DataFrame(
            columns=["compound", "calibrated_deg_estimate", "practice_confidence"]
        )
        deg_map, confidence = build_compound_degradation_map(empty)

        self.assertEqual(confidence, 0.0)
        self.assertEqual(deg_map, FALLBACK_COMPOUND_DEGRADATION)

        partial = pd.DataFrame(
            [
                {
                    "compound": "MEDIUM",
                    "calibrated_deg_estimate": 0.091,
                    "practice_confidence": 0.74,
                }
            ]
        )
        partial_map, partial_conf = build_compound_degradation_map(partial)

        self.assertTrue(np.isclose(partial_map["MEDIUM"], 0.091))
        self.assertTrue(np.isclose(partial_map["SOFT"], FALLBACK_COMPOUND_DEGRADATION["SOFT"]))
        self.assertTrue(np.isclose(partial_map["HARD"], FALLBACK_COMPOUND_DEGRADATION["HARD"]))
        self.assertTrue(np.isclose(partial_conf, 0.74))

    def test_dry_no_stop_run_is_invalid(self):
        with self.assertRaises(ValueError):
            simulate_strategy_scenario(
                race_laps=57,
                strategy_compounds=["MEDIUM"],
                pit_laps=[],
                base_pace_seconds=92.0,
                pit_loss_green_seconds=22.0,
                pit_loss_safety_car_seconds=13.0,
                compound_degradation_map={"MEDIUM": 0.08},
                fallback_degradation=0.08,
                starting_position=5,
                overtaking_difficulty=0.7,
                scenario_name="no_safety_car",
            )

    def test_pipeline_outputs_all_drivers_and_required_contract(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            calibrated_path = tmp / "calibrated.csv"
            setup_path = tmp / "race_setup.csv"
            output_path = tmp / "strategy_recommendations.csv"

            calibrated_df = pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "compound": "MEDIUM",
                        "calibrated_deg_estimate": 0.082,
                        "practice_confidence": 0.80,
                    },
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "compound": "HARD",
                        "calibrated_deg_estimate": 0.060,
                        "practice_confidence": 0.80,
                    },
                ]
            )
            calibrated_df.to_csv(calibrated_path, index=False)

            setup_df = pd.DataFrame(
                [
                    {
                        "race": "Australia",
                        "driver": "VER",
                        "team": "Red Bull Racing",
                        "starting_compound": "MEDIUM",
                        "starting_position": 1,
                        "base_pace_seconds": 91.8,
                        "race_laps": 35,
                        "pit_loss_green_seconds": 21.5,
                        "pit_loss_safety_car_seconds": 12.8,
                        "overtaking_difficulty": 0.75,
                        "safety_car_probability": 0.45,
                    },
                    {
                        "race": "Australia",
                        "driver": "PER",
                        "team": "Red Bull Racing",
                        "starting_compound": "HARD",
                        "starting_position": 8,
                        "base_pace_seconds": 92.2,
                        "race_laps": 35,
                        "pit_loss_green_seconds": 21.5,
                        "pit_loss_safety_car_seconds": 12.8,
                        "overtaking_difficulty": 0.75,
                        "safety_car_probability": 0.45,
                    },
                    {
                        "race": "Australia",
                        "driver": "GAS",
                        "team": "Alpine",
                        "starting_compound": "SOFT",
                        "starting_position": 10,
                        "base_pace_seconds": 92.8,
                        "race_laps": 35,
                        "pit_loss_green_seconds": 21.5,
                        "pit_loss_safety_car_seconds": 12.8,
                        "overtaking_difficulty": 0.75,
                        "safety_car_probability": 0.45,
                    },
                ]
            )
            setup_df.to_csv(setup_path, index=False)

            result = generate_strategy_recommendations(
                calibrated_predictions_path=calibrated_path,
                race_setup_path=setup_path,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertFalse(result.empty)
            self.assertEqual(result.columns.tolist(), OUTPUT_COLUMNS)

            # All setup drivers must receive recommendations even if no calibrated rows exist.
            self.assertEqual(set(result["driver"].unique().tolist()), {"VER", "PER", "GAS"})

            # Each driver + scenario should have exactly one recommended strategy.
            recommended_counts = (
                result.groupby(["driver", "scenario_name"], as_index=False)["is_recommended"].sum()
            )
            self.assertTrue((recommended_counts["is_recommended"] == 1).all())

            # Ensure fallback-only drivers still get simulated outputs.
            fallback_driver = result[result["driver"] == "GAS"]
            self.assertFalse(fallback_driver.empty)
            self.assertTrue(fallback_driver["expected_total_time"].notna().all())

            # Setup starting compound should constrain each driver's strategy set.
            self.assertTrue(
                (result[result["driver"] == "VER"]["starting_compound"] == "MEDIUM").all()
            )
            self.assertTrue(
                (result[result["driver"] == "PER"]["starting_compound"] == "HARD").all()
            )

            saved = pd.read_csv(output_path)
            self.assertEqual(saved.columns.tolist(), OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
