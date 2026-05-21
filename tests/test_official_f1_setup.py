import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import pandas as pd

from src.data.fetch_official_f1_setup import (
    OUTPUT_COLUMNS,
    generate_official_seeded_race_setup,
)


class TestOfficialF1SetupSeed(unittest.TestCase):
    def _standings_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "standing_position": 1,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team": "Red Bull Racing",
                    "points": 99.0,
                },
                {
                    "standing_position": 2,
                    "driver_name": "Charles Leclerc",
                    "driver_code": "LEC",
                    "team": "Ferrari",
                    "points": 88.0,
                },
                {
                    "standing_position": 3,
                    "driver_name": "Lewis Hamilton",
                    "driver_code": "HAM",
                    "team": "Ferrari",
                    "points": 73.0,
                },
                {
                    "standing_position": 4,
                    "driver_name": "Lando Norris",
                    "driver_code": "NOR",
                    "team": "McLaren",
                    "points": 61.0,
                },
            ]
        )

    def _lineup_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"team": "Ferrari", "driver_name": "Charles Leclerc"},
                {"team": "Ferrari", "driver_name": "Lewis Hamilton"},
                {"team": "Red Bull Racing", "driver_name": "Max Verstappen"},
                {"team": "McLaren", "driver_name": "Lando Norris"},
            ]
        )

    def _schedule_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "race_slug": "australia",
                    "race_name": "Australia",
                    "race_result_url": "https://www.formula1.com/en/results/2026/races/1279/australia/race-result",
                    "race_laps": 58,
                }
            ]
        )

    @patch("src.data.fetch_official_f1_setup.fetch_official_starting_grid")
    @patch("src.data.fetch_official_f1_setup.fetch_race_schedule")
    @patch("src.data.fetch_official_f1_setup.fetch_official_team_lineup")
    @patch("src.data.fetch_official_f1_setup.fetch_driver_standings")
    def test_contract_and_official_grid_source(
        self,
        mock_fetch_standings,
        mock_fetch_lineup,
        mock_fetch_schedule,
        mock_fetch_grid,
    ):
        mock_fetch_standings.return_value = self._standings_df()
        mock_fetch_lineup.return_value = self._lineup_df()
        mock_fetch_schedule.return_value = self._schedule_df()
        mock_fetch_grid.return_value = pd.DataFrame(
            [
                {
                    "grid_position": 1,
                    "driver_name": "Lando Norris",
                    "driver_code": "NOR",
                    "team": "McLaren",
                },
                {
                    "grid_position": 2,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team": "Red Bull Racing",
                },
                {
                    "grid_position": 3,
                    "driver_name": "Lewis Hamilton",
                    "driver_code": "HAM",
                    "team": "Ferrari",
                },
                {
                    "grid_position": 4,
                    "driver_name": "Charles Leclerc",
                    "driver_code": "LEC",
                    "team": "Ferrari",
                },
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "race_setup_official_seed.csv"
            setup = generate_official_seeded_race_setup(
                race="Australia",
                season=2026,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(setup.columns.tolist(), OUTPUT_COLUMNS)
            self.assertTrue((setup["setup_source"] == "official_grid").all())

            expected_order = {"NOR": 1, "VER": 2, "HAM": 3, "LEC": 4}
            actual_order = dict(zip(setup["driver"], setup["starting_position"]))
            self.assertEqual(actual_order, expected_order)

            saved = pd.read_csv(output_path)
            self.assertEqual(saved.columns.tolist(), OUTPUT_COLUMNS)

    @patch("src.data.fetch_official_f1_setup.fetch_official_starting_grid")
    @patch("src.data.fetch_official_f1_setup.fetch_race_schedule")
    @patch("src.data.fetch_official_f1_setup.fetch_official_team_lineup")
    @patch("src.data.fetch_official_f1_setup.fetch_driver_standings")
    def test_fallback_to_standings_seeded_when_grid_missing(
        self,
        mock_fetch_standings,
        mock_fetch_lineup,
        mock_fetch_schedule,
        mock_fetch_grid,
    ):
        mock_fetch_standings.return_value = self._standings_df()
        mock_fetch_lineup.return_value = self._lineup_df()
        mock_fetch_schedule.return_value = self._schedule_df()
        mock_fetch_grid.return_value = pd.DataFrame(
            columns=["grid_position", "driver_name", "driver_code", "team"]
        )

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "race_setup_official_seed.csv"

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                setup = generate_official_seeded_race_setup(
                    race="Australia",
                    season=2026,
                    output_path=output_path,
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(setup.columns.tolist(), OUTPUT_COLUMNS)
            self.assertTrue((setup["setup_source"] == "standings_seeded").all())

            expected_order = {"VER": 1, "LEC": 2, "HAM": 3, "NOR": 4}
            actual_order = dict(zip(setup["driver"], setup["starting_position"]))
            self.assertEqual(actual_order, expected_order)

            self.assertTrue(
                any("Official grid is not available yet" in str(w.message) for w in captured)
            )


if __name__ == "__main__":
    unittest.main()
