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


class TestLiveFinishProjection(unittest.TestCase):
    def _setup_df(
        self,
        *,
        overtaking_difficulty: float,
        track_position_importance: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race": "TestGP",
                    "driver": "AAA",
                    "team": "Team A",
                    "starting_position": 1,
                    "starting_compound": "MEDIUM",
                    "race_laps": 45,
                    "base_pace_seconds": 92.0,
                    "pit_loss_green_seconds": 22.5,
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
                    "race_laps": 45,
                    "base_pace_seconds": 92.2,
                    "pit_loss_green_seconds": 22.5,
                    "pit_loss_safety_car_seconds": 12.0,
                    "overtaking_difficulty": overtaking_difficulty,
                    "track_position_importance": track_position_importance,
                },
                {
                    "race": "TestGP",
                    "driver": "CCC",
                    "team": "Team C",
                    "starting_position": 3,
                    "starting_compound": "HARD",
                    "race_laps": 45,
                    "base_pace_seconds": 92.4,
                    "pit_loss_green_seconds": 22.5,
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

    def _large_field_setup_df(
        self,
        *,
        race_laps: int = 57,
        overtaking_difficulty: float = 0.70,
        track_position_importance: float = 0.70,
    ) -> pd.DataFrame:
        rows = []
        for idx in range(1, 23):
            rows.append(
                {
                    "race": "LargeFieldGP",
                    "driver": f"D{idx:02d}",
                    "team": f"Team {idx:02d}",
                    "starting_position": idx,
                    "starting_compound": "MEDIUM" if idx % 2 else "HARD",
                    "race_laps": race_laps,
                    "base_pace_seconds": 91.9 + (idx * 0.04),
                    "pit_loss_green_seconds": 22.0,
                    "pit_loss_safety_car_seconds": 13.0,
                    "overtaking_difficulty": overtaking_difficulty,
                    "track_position_importance": track_position_importance,
                }
            )
        return pd.DataFrame(rows)

    def test_high_track_position_importance_can_make_stay_out_finish_better(self):
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

        race_state.lap = 19
        driver = race_state.get_driver("AAA")
        self.assertIsNotNone(driver)
        assert driver is not None
        driver.tire_age = 12

        decisions = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )
        aaa = decisions["AAA"]

        self.assertLessEqual(
            int(aaa["projected_finish_if_stay_out"]),
            int(aaa["projected_finish_if_pit_now"]),
        )
        self.assertEqual(aaa["best_action_now"], "stay_out")

    def test_front_runner_early_pit_does_not_always_become_last_place(self):
        setup_df = self._large_field_setup_df(
            overtaking_difficulty=0.70,
            track_position_importance=0.70,
        )
        race_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="LargeFieldGP",
            setup_source="sample_setup",
        )
        driver_setup = self._driver_setup(setup_df)
        decision_context = build_initial_decision_context(
            race_state=race_state,
            driver_setup=driver_setup,
        )

        race_state.lap = 1
        leader = race_state.get_driver("D01")
        self.assertIsNotNone(leader)
        assert leader is not None
        leader.tire_age = 1

        decisions = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )
        leader_decision = decisions["D01"]

        self.assertLess(int(leader_decision["projected_finish_if_pit_now"]), 22)

    def test_strong_fresh_tire_advantage_improves_pit_now_recovery_probability(self):
        setup_df = self._setup_df(overtaking_difficulty=0.55, track_position_importance=0.55)

        worn_field_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        worn_driver_setup = self._driver_setup(setup_df)
        worn_context = build_initial_decision_context(
            race_state=worn_field_state,
            driver_setup=worn_driver_setup,
        )
        worn_field_state.lap = 9
        worn_aaa = worn_field_state.get_driver("AAA")
        worn_bbb = worn_field_state.get_driver("BBB")
        worn_ccc = worn_field_state.get_driver("CCC")
        self.assertIsNotNone(worn_aaa)
        self.assertIsNotNone(worn_bbb)
        self.assertIsNotNone(worn_ccc)
        assert worn_aaa is not None
        assert worn_bbb is not None
        assert worn_ccc is not None
        worn_aaa.tire_age = 10
        worn_bbb.tire_age = 20
        worn_ccc.tire_age = 18

        worn_decision = evaluate_pit_decisions_for_lap(
            race_state=worn_field_state,
            driver_setup=worn_driver_setup,
            decision_context=worn_context,
        )["AAA"]

        fresh_field_state = build_race_state_from_setup(
            setup_df=setup_df,
            race="TestGP",
            setup_source="sample_setup",
        )
        fresh_driver_setup = self._driver_setup(setup_df)
        fresh_context = build_initial_decision_context(
            race_state=fresh_field_state,
            driver_setup=fresh_driver_setup,
        )
        fresh_field_state.lap = 9
        fresh_aaa = fresh_field_state.get_driver("AAA")
        fresh_bbb = fresh_field_state.get_driver("BBB")
        fresh_ccc = fresh_field_state.get_driver("CCC")
        self.assertIsNotNone(fresh_aaa)
        self.assertIsNotNone(fresh_bbb)
        self.assertIsNotNone(fresh_ccc)
        assert fresh_aaa is not None
        assert fresh_bbb is not None
        assert fresh_ccc is not None
        fresh_aaa.tire_age = 10
        fresh_bbb.tire_age = 4
        fresh_ccc.tire_age = 3

        fresh_decision = evaluate_pit_decisions_for_lap(
            race_state=fresh_field_state,
            driver_setup=fresh_driver_setup,
            decision_context=fresh_context,
        )["AAA"]

        self.assertGreater(
            float(worn_decision["overtake_probability_if_pit_now"]),
            float(fresh_decision["overtake_probability_if_pit_now"]),
        )

    def test_recommendation_reason_changes_with_overtake_tradeoff(self):
        hard_setup = self._setup_df(overtaking_difficulty=0.95, track_position_importance=0.95)
        hard_state = build_race_state_from_setup(
            setup_df=hard_setup,
            race="TestGP",
            setup_source="sample_setup",
        )
        hard_driver_setup = self._driver_setup(hard_setup)
        hard_context = build_initial_decision_context(
            race_state=hard_state,
            driver_setup=hard_driver_setup,
        )
        hard_state.lap = 20
        hard_driver = hard_state.get_driver("AAA")
        self.assertIsNotNone(hard_driver)
        assert hard_driver is not None
        hard_driver.tire_age = 12

        hard_decision = evaluate_pit_decisions_for_lap(
            race_state=hard_state,
            driver_setup=hard_driver_setup,
            decision_context=hard_context,
        )["AAA"]

        easy_setup = self._setup_df(overtaking_difficulty=0.35, track_position_importance=0.35)
        easy_state = build_race_state_from_setup(
            setup_df=easy_setup,
            race="TestGP",
            setup_source="sample_setup",
        )
        easy_driver_setup = self._driver_setup(easy_setup)
        easy_context = build_initial_decision_context(
            race_state=easy_state,
            driver_setup=easy_driver_setup,
        )
        easy_state.lap = 20
        easy_driver = easy_state.get_driver("AAA")
        self.assertIsNotNone(easy_driver)
        assert easy_driver is not None
        easy_driver.tire_age = 21

        easy_decision = evaluate_pit_decisions_for_lap(
            race_state=easy_state,
            driver_setup=easy_driver_setup,
            decision_context=easy_context,
        )["AAA"]

        self.assertIn("difficult overtaking track", hard_decision["recommendation_reason"])
        self.assertNotEqual(
            hard_decision["recommendation_reason"],
            easy_decision["recommendation_reason"],
        )
        self.assertTrue(
            ("Pit now is faster and positions are recoverable" in easy_decision["recommendation_reason"])
            or ("Current tire age is high" in easy_decision["recommendation_reason"])
            or ("Pitting now gives the stronger projected race outcome" in easy_decision["recommendation_reason"])
            or ("pit-cycle recovery odds are limited" in easy_decision["recommendation_reason"])
        )

    def test_output_contract_includes_overtake_probability_columns(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            setup_path = tmp / "race_setup_sample.csv"
            events_path = tmp / "live_events.csv"
            output_path = tmp / "live_output.csv"

            self._setup_df(
                overtaking_difficulty=0.70,
                track_position_importance=0.70,
            ).to_csv(setup_path, index=False)
            pd.DataFrame(
                [
                    {
                        "lap": 3,
                        "event_type": "pit_stop",
                        "driver": "AAA",
                        "target_driver": "",
                        "compound": "HARD",
                        "notes": "Early stop",
                    }
                ]
            ).to_csv(events_path, index=False)

            output = run_live_race_simulation(
                race="TestGP",
                setup_path=setup_path,
                events_path=events_path,
                output_path=output_path,
            )
            self.assertIn("overtake_probability_if_pit_now", output.columns.tolist())
            self.assertIn("overtake_probability_if_stay_out", output.columns.tolist())

    def test_recommendation_reason_mentions_tire_life_risk_beyond_cliff(self):
        setup_df = self._setup_df(overtaking_difficulty=0.65, track_position_importance=0.60)
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

        race_state.lap = 27
        aaa = race_state.get_driver("AAA")
        self.assertIsNotNone(aaa)
        assert aaa is not None
        aaa.current_compound = "SOFT"
        aaa.tire_age = 24

        decision = evaluate_pit_decisions_for_lap(
            race_state=race_state,
            driver_setup=driver_setup,
            decision_context=decision_context,
        )["AAA"]

        self.assertEqual(decision["best_action_now"], "pit_now")
        reason = decision["recommendation_reason"].lower()
        self.assertTrue(("tire-life risk" in reason) or ("tire" in reason))

    def test_final_lap_dry_drivers_have_mandatory_stop_enforced_in_state(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            setup_path = tmp / "race_setup_sample.csv"
            events_path = tmp / "live_events.csv"
            output_path = tmp / "live_output.csv"

            self._setup_df(
                overtaking_difficulty=0.70,
                track_position_importance=0.70,
            ).to_csv(setup_path, index=False)
            pd.DataFrame(columns=["lap", "event_type", "driver", "target_driver", "compound", "notes"]).to_csv(
                events_path,
                index=False,
            )

            output = run_live_race_simulation(
                race="TestGP",
                setup_path=setup_path,
                events_path=events_path,
                output_path=output_path,
            )

            final_lap = int(output["lap"].max())
            final_rows = output[output["lap"] == final_lap].copy()
            zero_stop_rows = final_rows[final_rows["stops_made"] == 0].copy()
            self.assertTrue(zero_stop_rows.empty)
            self.assertTrue((final_rows["stops_made"] >= 1).all())

            deadline_lap = int(final_lap - 3)
            at_or_after_deadline = output[output["lap"] >= deadline_lap].copy()
            self.assertFalse(at_or_after_deadline.empty)
            final_stop_counts = (
                output.sort_values(["driver", "lap"])
                .groupby("driver", as_index=False)
                .tail(1)[["driver", "stops_made"]]
            )
            self.assertTrue(
                (final_stop_counts["stops_made"] >= 1).all()
            )


if __name__ == "__main__":
    unittest.main()
