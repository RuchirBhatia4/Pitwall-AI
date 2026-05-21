import unittest

import numpy as np

from src.simulation.race_simulator import simulate_strategy_scenario


class TestStrategyScenarios(unittest.TestCase):
    def test_safety_car_can_improve_strategy_by_shifting_stop_into_window(self):
        # Original pit at lap 22 is outside the early safety-car window (8-15).
        # With near-flat degradation, moving into the window should be cheaper.
        scenario = simulate_strategy_scenario(
            race_laps=40,
            strategy_compounds=["MEDIUM", "HARD"],
            pit_laps=[22],
            base_pace_seconds=92.0,
            pit_loss_green_seconds=22.0,
            pit_loss_safety_car_seconds=12.0,
            compound_degradation_map={"MEDIUM": 0.0, "HARD": 0.0},
            fallback_degradation=0.0,
            starting_position=7,
            overtaking_difficulty=0.7,
            scenario_name="early_safety_car",
        )

        self.assertGreater(float(scenario["safety_car_gain_seconds"]), 0.0)
        self.assertTrue(np.isfinite(float(scenario["expected_total_time"])))
        self.assertTrue(np.isfinite(float(scenario["pit_loss_used"])))

        sc_lap = scenario["best_safety_car_pit_lap"]
        self.assertTrue(np.isfinite(float(sc_lap)))
        self.assertGreaterEqual(int(sc_lap), 8)
        self.assertLessEqual(int(sc_lap), 15)


if __name__ == "__main__":
    unittest.main()
