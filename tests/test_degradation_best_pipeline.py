import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluation.backtest_2026_holdouts import (
    BACKTEST_REQUIRED_COLUMNS,
    run_2026_holdout_backtest,
)
from src.models.train_degradation_best import (
    BEST_METRIC_COLUMNS,
    DRY_COMPOUNDS,
    load_training_data,
    train_best_model,
)


def _build_synthetic_training_data(path: Path) -> pd.DataFrame:
    rows = []

    race_calendar = [
        (2025, "Australia"),
        (2025, "Japan"),
        (2026, "Australia"),
        (2026, "China"),
        (2026, "Japan"),
    ]
    drivers = [
        ("VER", "Red Bull Racing"),
        ("LEC", "Ferrari"),
    ]
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    race_max_lap = 58

    for season, race in race_calendar:
        race_shift = {"Australia": 0.05, "China": 0.07, "Japan": 0.10}.get(race, 0.08)
        for driver, team in drivers:
            driver_shift = 0.02 if driver == "VER" else 0.06
            for stint in [1, 2]:
                for compound in compounds:
                    compound_slope = {
                        "SOFT": 0.13,
                        "MEDIUM": 0.09,
                        "HARD": 0.07,
                        "INTERMEDIATE": 0.05,
                        "WET": 0.04,
                    }[compound]
                    for tyre_life in [1, 2, 3]:
                        lap_number = (stint - 1) * 20 + tyre_life
                        race_phase = lap_number / race_max_lap
                        stint_progress = tyre_life / 3.0

                        target = (
                            driver_shift
                            + race_shift
                            + (0.02 if season == 2026 else 0.06)
                            + compound_slope * tyre_life
                        )

                        rows.append(
                            {
                                "season": season,
                                "race": race,
                                "Driver": driver,
                                "Team": team,
                                "Compound": compound,
                                "TyreLife": tyre_life,
                                "Stint": stint,
                                "LapNumber": lap_number,
                                "race_phase": race_phase,
                                "stint_progress": stint_progress,
                                "tyre_life_squared": tyre_life ** 2,
                                "lap_number_squared": lap_number ** 2,
                                "is_first_stint": 1 if stint == 1 else 0,
                                "is_second_stint": 1 if stint == 2 else 0,
                                "is_late_race": 1 if race_phase >= 0.66 else 0,
                                "is_early_race": 1 if race_phase <= 0.33 else 0,
                                "soft_tyre_life": tyre_life if compound == "SOFT" else 0,
                                "medium_tyre_life": tyre_life if compound == "MEDIUM" else 0,
                                "hard_tyre_life": tyre_life if compound == "HARD" else 0,
                                "fuel_adjusted_degradation_delta": target,
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


class TestDegradationBestPipeline(unittest.TestCase):
    def test_dry_filter_excludes_intermediate_and_wet(self):
        with TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "historical_laps_clean.csv"
            _build_synthetic_training_data(data_path)

            dry_df = load_training_data(path=data_path, dry_only=True)
            compounds = set(dry_df["Compound"].unique().tolist())

            self.assertTrue(compounds.issubset(DRY_COMPOUNDS))
            self.assertNotIn("INTERMEDIATE", compounds)
            self.assertNotIn("WET", compounds)

    def test_best_model_metrics_file_created_with_expected_columns(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_path = tmp / "historical_laps_clean.csv"
            model_path = tmp / "best_degradation_model.pkl"
            metrics_path = tmp / "best_degradation_model_metrics.csv"
            _build_synthetic_training_data(data_path)

            train_best_model(
                data_path=data_path,
                model_output_path=model_path,
                metrics_output_path=metrics_path,
                xgb_search_space=[
                    {
                        "n_estimators": 20,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    }
                ],
                lgbm_search_space=[
                    {
                        "n_estimators": 20,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "num_leaves": 15,
                    }
                ],
                log_to_mlflow=False,
                random_state=7,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())

            metrics_df = pd.read_csv(metrics_path)
            for col in BEST_METRIC_COLUMNS:
                self.assertIn(col, metrics_df.columns)

            self.assertTrue((metrics_df["is_best_model"].astype(bool).sum()) >= 1)

    def test_backtest_output_has_required_columns(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            data_path = tmp / "historical_laps_clean.csv"
            csv_path = tmp / "2026_holdout_backtest.csv"
            md_path = tmp / "2026_holdout_backtest.md"
            _build_synthetic_training_data(data_path)

            backtest_df = run_2026_holdout_backtest(
                data_path=data_path,
                output_csv_path=csv_path,
                output_md_path=md_path,
                xgb_search_space=[
                    {
                        "n_estimators": 20,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    }
                ],
                lgbm_search_space=[
                    {
                        "n_estimators": 20,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "num_leaves": 15,
                    }
                ],
                random_state=7,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())

            saved_df = pd.read_csv(csv_path)
            for col in BACKTEST_REQUIRED_COLUMNS:
                self.assertIn(col, saved_df.columns)

            self.assertGreaterEqual(len(backtest_df), 1)
            self.assertEqual(
                set(saved_df["best_model_type"].unique().tolist()).issubset(
                    {"XGBoost", "LightGBM"}
                ),
                True,
            )

            markdown_text = md_path.read_text(encoding="utf-8")
            self.assertIn("historical-only pre-practice performance", markdown_text)


if __name__ == "__main__":
    unittest.main()
