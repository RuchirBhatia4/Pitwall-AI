import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import src.pipeline.run_live_race_simulation as live_pipeline
from src.pipeline.run_live_race_simulation import OUTPUT_COLUMNS


class TestLiveRaceSimulationPipeline(unittest.TestCase):
    def _setup_df(
        self,
        *,
        race: str = "TestGP",
        include_setup_source: bool = False,
        setup_source_value: str = "official_grid",
    ) -> pd.DataFrame:
        rows = [
            {
                "race": race,
                "driver": "AAA",
                "team": "Team A",
                "starting_position": 1,
                "starting_compound": "MEDIUM",
                "race_laps": 5,
            },
            {
                "race": race,
                "driver": "BBB",
                "team": "Team B",
                "starting_position": 2,
                "starting_compound": "HARD",
                "race_laps": 5,
            },
        ]
        df = pd.DataFrame(rows)
        if include_setup_source:
            df["setup_source"] = setup_source_value
        return df

    def _events_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "lap": 1,
                    "event_type": "safety_car_start",
                    "driver": "",
                    "target_driver": "",
                    "compound": "",
                    "notes": "SC deployed",
                },
                {
                    "lap": 2,
                    "event_type": "pit_stop",
                    "driver": "AAA",
                    "target_driver": "",
                    "compound": "SOFT",
                    "notes": "Box",
                },
                {
                    "lap": 3,
                    "event_type": "overtake",
                    "driver": "BBB",
                    "target_driver": "AAA",
                    "compound": "",
                    "notes": "Pass",
                },
                {
                    "lap": 4,
                    "event_type": "retirement",
                    "driver": "AAA",
                    "target_driver": "",
                    "compound": "",
                    "notes": "DNF",
                },
                {
                    "lap": 5,
                    "event_type": "safety_car_end",
                    "driver": "",
                    "target_driver": "",
                    "compound": "",
                    "notes": "SC in",
                },
            ]
        )

    def test_official_setup_is_preferred_when_present(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            official_path = tmp / "race_setup_official_seed.csv"
            sample_path = tmp / "race_setup_sample.csv"
            events_path = tmp / "live_events.csv"
            output_path = tmp / "live_output.csv"

            self._setup_df(include_setup_source=True, setup_source_value="official_grid").to_csv(
                official_path,
                index=False,
            )
            self._setup_df(include_setup_source=False, setup_source_value="sample_setup").to_csv(
                sample_path,
                index=False,
            )
            self._events_df().to_csv(events_path, index=False)

            with patch.object(live_pipeline, "DEFAULT_OFFICIAL_SETUP_PATH", official_path), patch.object(
                live_pipeline,
                "DEFAULT_SAMPLE_SETUP_PATH",
                sample_path,
            ):
                result = live_pipeline.run_live_race_simulation(
                    race="TestGP",
                    events_path=events_path,
                    output_path=output_path,
                )

            self.assertFalse(result.empty)
            self.assertTrue((result["setup_source"] == "official_grid").all())

    def test_setup_fallback_and_output_contract(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            official_path = tmp / "missing_official.csv"
            sample_path = tmp / "race_setup_sample.csv"
            events_path = tmp / "live_events.csv"
            output_path = tmp / "live_output.csv"

            self._setup_df(include_setup_source=False).to_csv(sample_path, index=False)
            self._events_df().to_csv(events_path, index=False)

            with patch.object(live_pipeline, "DEFAULT_OFFICIAL_SETUP_PATH", official_path), patch.object(
                live_pipeline,
                "DEFAULT_SAMPLE_SETUP_PATH",
                sample_path,
            ):
                result = live_pipeline.run_live_race_simulation(
                    race=None,
                    events_path=events_path,
                    output_path=output_path,
                )

            self.assertTrue(output_path.exists())
            self.assertFalse(result.empty)
            self.assertEqual(result.columns.tolist(), OUTPUT_COLUMNS)
            self.assertTrue((result["setup_source"] == "sample_setup").all())

            saved = pd.read_csv(output_path)
            self.assertEqual(saved.columns.tolist(), OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
