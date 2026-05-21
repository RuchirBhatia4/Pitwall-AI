import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluation.replay_strategy_decisions import (
    OUTPUT_COLUMNS,
    SUMMARY_COLUMNS,
    add_gap_features,
    replay_strategy_decisions,
)


class TestStrategyReplay(unittest.TestCase):
    def _clean_laps(self) -> pd.DataFrame:
        rows = []
        for lap in range(1, 7):
            rows.append(
                {
                    "season": 2026,
                    "race": "ReplayGP",
                    "Driver": "AAA",
                    "Team": "Team A",
                    "LapNumber": lap,
                    "Position": 1,
                    "Compound": "MEDIUM" if lap <= 3 else "HARD",
                    "TyreLife": lap if lap <= 3 else lap - 3,
                    "Stint": 1 if lap <= 3 else 2,
                    "lap_time_seconds": 90.0 + (lap * 0.10),
                }
            )
            rows.append(
                {
                    "season": 2026,
                    "race": "ReplayGP",
                    "Driver": "BBB",
                    "Team": "Team B",
                    "LapNumber": lap,
                    "Position": 2,
                    "Compound": "MEDIUM",
                    "TyreLife": lap,
                    "Stint": 1,
                    "lap_time_seconds": 91.0 + (lap * 0.10),
                }
            )
        return pd.DataFrame(rows)

    def _setup(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race": "ReplayGP",
                    "driver": "AAA",
                    "team": "Team A",
                    "starting_position": 1,
                    "starting_compound": "MEDIUM",
                    "race_laps": 6,
                    "base_pace_seconds": 90.0,
                    "pit_loss_green_seconds": 20.0,
                    "pit_loss_safety_car_seconds": 12.0,
                    "overtaking_difficulty": 0.70,
                    "track_position_importance": 0.70,
                },
                {
                    "race": "ReplayGP",
                    "driver": "BBB",
                    "team": "Team B",
                    "starting_position": 2,
                    "starting_compound": "MEDIUM",
                    "race_laps": 6,
                    "base_pace_seconds": 91.0,
                    "pit_loss_green_seconds": 20.0,
                    "pit_loss_safety_car_seconds": 12.0,
                    "overtaking_difficulty": 0.70,
                    "track_position_importance": 0.70,
                },
            ]
        )

    def test_gap_features_are_added_from_cumulative_lap_time(self):
        clean = self._clean_laps().rename(
            columns={
                "Driver": "driver",
                "Team": "team",
                "LapNumber": "lap",
                "Position": "actual_position",
                "Compound": "current_compound",
                "TyreLife": "tire_age",
                "Stint": "stint",
            }
        )
        enriched = add_gap_features(clean)

        lap_two = enriched[enriched["lap"] == 2].sort_values("actual_position")
        leader = lap_two.iloc[0]
        second = lap_two.iloc[1]

        self.assertEqual(float(leader["gap_to_leader_seconds"]), 0.0)
        self.assertGreater(float(second["gap_to_leader_seconds"]), 0.0)
        self.assertGreater(float(second["gap_to_ahead_seconds"]), 0.0)

    def test_replay_outputs_contract_and_summary(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            clean_path = tmp / "clean_laps.csv"
            setup_path = tmp / "setup.csv"
            output_path = tmp / "replay.csv"
            summary_path = tmp / "summary.csv"

            self._clean_laps().to_csv(clean_path, index=False)
            self._setup().to_csv(setup_path, index=False)

            replay, summary = replay_strategy_decisions(
                clean_laps_path=clean_path,
                setup_path=setup_path,
                race="ReplayGP",
                season=2026,
                output_path=output_path,
                summary_path=summary_path,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(replay.columns.tolist(), OUTPUT_COLUMNS)
            self.assertEqual(summary.columns.tolist(), SUMMARY_COLUMNS)
            self.assertFalse(replay.empty)
            self.assertEqual(set(summary["driver"].tolist()), {"AAA", "BBB"})

            aaa = summary[summary["driver"] == "AAA"].iloc[0]
            self.assertEqual(int(aaa["actual_first_pit_lap"]), 3)

    def test_confirmed_signal_requires_consecutive_pit_now_laps(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            clean_path = tmp / "clean_laps.csv"
            setup_path = tmp / "setup.csv"
            output_path = tmp / "replay.csv"
            summary_path = tmp / "summary.csv"

            clean = self._clean_laps()
            clean.loc[
                (clean["Driver"] == "AAA") & (clean["LapNumber"] >= 5),
                "Compound",
            ] = "MEDIUM"
            clean.loc[
                (clean["Driver"] == "AAA") & (clean["LapNumber"] >= 5),
                "Stint",
            ] = 1
            clean.loc[
                (clean["Driver"] == "AAA") & (clean["LapNumber"] >= 5),
                "TyreLife",
            ] = clean.loc[
                (clean["Driver"] == "AAA") & (clean["LapNumber"] >= 5),
                "LapNumber",
            ]
            clean.to_csv(clean_path, index=False)
            self._setup().to_csv(setup_path, index=False)

            replay, summary = replay_strategy_decisions(
                clean_laps_path=clean_path,
                setup_path=setup_path,
                race="ReplayGP",
                season=2026,
                output_path=output_path,
                summary_path=summary_path,
            )

            self.assertIn("confirmed_first_pit_call_lap", replay.columns)
            self.assertIn("confirmed_signal_status", summary.columns)
            for _, row in summary.dropna(subset=["confirmed_first_pit_call_lap"]).iterrows():
                driver_rows = replay[
                    (replay["driver"] == row["driver"])
                    & (replay["lap"].isin(
                        [
                            int(row["confirmed_first_pit_call_lap"]) - 1,
                            int(row["confirmed_first_pit_call_lap"]),
                        ]
                    ))
                ].sort_values("lap")
                self.assertEqual(driver_rows["best_action_now"].tolist(), ["pit_now", "pit_now"])


if __name__ == "__main__":
    unittest.main()
