import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.live.pit_decision_engine import (
    build_initial_decision_context,
    evaluate_pit_decisions_for_lap,
)
from src.live.race_state import build_race_state_from_setup
from src.pipeline.run_live_race_simulation import run_live_race_simulation


class TestLivePitDecisions(unittest.TestCase):
    def _setup_df(
        self,
        *,
        overtaking_difficulty: float = 0.65,
        track_position_importance: float = 0.60,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race": "TestGP",
                    "driver": "AAA",
                    "team": "Team A",
                    "starting_position": 1,
                    "starting_compound": "MEDIUM",
                    "race_laps": 40,
                    "base_pace_seconds": 92.0,
                    "pit_loss_green_seconds": 22.0,
                    "pit_loss_safety_car_seconds": 12.0,
                    "overtaking_difficulty": overtaking_difficulty,
                    "track_position_importance": track_position_importance,
                },
                {
                    "race": "TestGP",
                    "driver": "BBB",
                    "team": "Team B",
                    "starting_position": 2,
                    "starting_compound": "HARD",
                    "race_laps": 40,
                    "base_pace_seconds": 92.2,
                    "pit_loss_green_seconds": 22.0,
                    "pit_loss_safety_car_seconds": 12.0,
                    "overtaking_difficulty": overtaking_difficulty,
                    "track_position_importance": track_position_importance,
                },
            ]
        )

    def _driver_setup(self, setup_df: pd.DataFrame) -> dict[str, dict[str, float | str | int]]:
        mapping: dict[str, dict[str, float | str | int]] = {}
        for _, row in setup_df.iterrows():
            driver = str(row["driver"]).upper()
            mapping[driver] = {
                "race_laps": int(row["race_laps"]),
                "starting_compound": str(row["starting_compound"]).upper(),
                "base_pace_seconds": float(row["base_pace_seconds"]),
                "pit_loss_green_seconds": float(row["pit_loss_green_seconds"]),
                "pit_loss_safety_car_seconds": float(row["pit_loss_safety_car_seconds"]),
                "overtaking_difficulty": float(row["overtaking_difficulty"]),
                "track_position_importance": float(row["track_position_importance"]),
            }
        return mapping

    def test_safety_car_pit_uses_reduced_pit_loss(self):
        setup_df = self._setup_df()
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 10

        race_state.set_under_safety_car(False)
        green = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        race_state.set_under_safety_car(True)
        safety_car = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertLess(
            float(safety_car["projected_total_time_if_pit_now"]),
            float(green["projected_total_time_if_pit_now"]),
        )

    def test_decision_context_tracks_mandatory_stop_fields(self):
        setup_df = self._setup_df()
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=self._driver_setup(setup_df),
        )
        aaa = context["AAA"]
        self.assertIn("compounds_used", aaa)
        self.assertIn("mandatory_stop_pending", aaa)
        self.assertIn("mandatory_stop_deadline_lap", aaa)
        self.assertEqual(int(aaa["mandatory_stop_deadline_lap"]), 37)

    def test_high_tire_age_makes_pit_now_more_attractive(self):
        setup_df = self._setup_df(overtaking_difficulty=0.60, track_position_importance=0.55)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 15
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None

        driver.tire_age = 6
        low_age = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        driver.tire_age = 22
        high_age = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertGreater(
            float(high_age["pit_now_gain_seconds"]),
            float(low_age["pit_now_gain_seconds"]),
        )
        self.assertEqual(high_age["best_action_now"], "pit_now")

    def test_high_track_position_importance_can_make_stay_out_better(self):
        setup_df = self._setup_df(overtaking_difficulty=0.95, track_position_importance=0.95)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 18
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.tire_age = 12

        decision = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertEqual(decision["best_action_now"], "stay_out")

    def test_output_contract_includes_new_columns(self):
        required_columns = [
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

        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            setup_path = tmp / "race_setup_sample.csv"
            events_path = tmp / "live_events.csv"
            output_path = tmp / "live_output.csv"

            self._setup_df().to_csv(setup_path, index=False)
            pd.DataFrame(
                [
                    {
                        "lap": 2,
                        "event_type": "pit_stop",
                        "driver": "AAA",
                        "target_driver": "",
                        "compound": "SOFT",
                        "notes": "Box",
                    }
                ]
            ).to_csv(events_path, index=False)

            result = run_live_race_simulation(
                race="TestGP",
                setup_path=setup_path,
                events_path=events_path,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertFalse(result.empty)
            self.assertEqual(result.columns.tolist(), required_columns)

    def test_mandatory_stop_pending_favors_pit_late_in_race(self):
        setup_df = self._setup_df(overtaking_difficulty=0.75, track_position_importance=0.70)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 37  # Mandatory-stop deadline lap in a 40-lap run (race_laps - 3).
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.tire_age = 20

        decision = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertEqual(decision["best_action_now"], "pit_now")
        self.assertIn("mandatory dry", decision["recommendation_reason"].lower())

    def test_projected_stay_out_includes_mandatory_stop_enforcement_near_end(self):
        setup_df = self._setup_df(overtaking_difficulty=0.70, track_position_importance=0.65)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )
        race_state.lap = 37  # Deadline lap in a 40-lap race when using race_laps - 3.
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.tire_age = 15

        # Keep ideal stop beyond race distance to isolate mandatory-stop enforcement effect.
        decision_context["AAA"]["ideal_pit_lap"] = 45
        decision_context["AAA"]["mandatory_stop_deadline_lap"] = 37
        decision_context["AAA"]["mandatory_stop_pending"] = True
        enforced = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        decision_context["AAA"]["mandatory_stop_pending"] = False
        non_enforced = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertGreater(
            float(enforced["projected_total_time_if_stay_out"]),
            float(non_enforced["projected_total_time_if_stay_out"]) + 10.0,
        )
        self.assertIn("mandatory", enforced["recommendation_reason"].lower())

    def test_staying_out_beyond_tire_cliff_gets_worse(self):
        setup_df = self._setup_df(overtaking_difficulty=0.70, track_position_importance=0.60)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 24
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.current_compound = "SOFT"

        driver.tire_age = 14
        before_cliff = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        driver.tire_age = 22
        after_cliff = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertGreater(
            float(after_cliff["projected_total_time_if_stay_out"]),
            float(before_cliff["projected_total_time_if_stay_out"]),
        )
        self.assertIn("tire", after_cliff["recommendation_reason"].lower())

    def test_close_gap_pressure_can_trigger_undercut_or_cover_stop(self):
        setup_df = self._setup_df(overtaking_difficulty=0.55, track_position_importance=0.55)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 16
        driver = race_state.get_driver("BBB")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.current_position = 2
        driver.current_compound = "MEDIUM"
        driver.tire_age = 16

        driver_setup["BBB"]["gap_to_ahead_seconds"] = 1.0
        driver_setup["BBB"]["gap_to_behind_seconds"] = 1.2
        decision_context["BBB"]["ideal_pit_lap"] = 20

        decision = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["BBB"]

        self.assertEqual(decision["best_action_now"], "pit_now")
        self.assertIn("attack/cover", decision["recommendation_reason"])

    def test_illegal_dry_no_stop_completion_not_valid_projected_outcome(self):
        setup_df = self._setup_df(overtaking_difficulty=0.70, track_position_importance=0.70)
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 40
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.tire_age = 24

        decision = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertEqual(decision["best_action_now"], "pit_now")
        self.assertGreater(
            float(decision["projected_total_time_if_stay_out"]),
            float(decision["projected_total_time_if_pit_now"]) + 30.0,
        )
        self.assertGreaterEqual(
            int(decision["projected_finish_if_stay_out"]),
            int(decision["projected_finish_if_pit_now"]),
        )
        self.assertIn("mandatory dry compound-change rule", decision["recommendation_reason"].lower())


if __name__ == "__main__":
    unittest.main()
