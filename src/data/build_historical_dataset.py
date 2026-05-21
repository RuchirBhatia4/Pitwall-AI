from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.fastf1_loader import load_race_laps
from src.data.cleaning import clean_laps_for_degradation


OUTPUT_DIR = Path("data/processed")
RAW_OUTPUT_DIR = OUTPUT_DIR / "historical_raw"
CLEAN_OUTPUT_DIR = OUTPUT_DIR / "historical_clean"


# -------------------------------------------------------------------
# DATASET STRATEGY
# -------------------------------------------------------------------
# We prioritize 2026 completed races first because this project is meant
# to become a live/current-season F1 strategy predictor.
#
# As more 2026 races happen, keep adding them here.
#
# IMPORTANT:
# FastF1 can only load race/session data once that race/session data exists.
# Future races will fail or return unavailable data, so do not add future
# races until after the race weekend is complete.
# -------------------------------------------------------------------


COMPLETED_2026_RACES: List[Dict] = [
    {"season": 2026, "race": "Australia"},
    {"season": 2026, "race": "China"},
    {"season": 2026, "race": "Japan"},
]


# These older races are used as historical backup.
# They help the model learn broader tire behavior, track patterns,
# compound effects, and team/driver tendencies.
#
# Once more 2026 races finish, we will keep reducing the dependency on
# older seasons or use older seasons only for priors/track similarity.
HISTORICAL_BACKUP_RACES: List[Dict] = [
    # 2025 races
    {"season": 2025, "race": "Australia"},
    {"season": 2025, "race": "China"},
    {"season": 2025, "race": "Japan"},
    {"season": 2025, "race": "Bahrain"},
    {"season": 2025, "race": "Saudi Arabia"},
    {"season": 2025, "race": "Miami"},
    {"season": 2025, "race": "Monaco"},
    {"season": 2025, "race": "Spain"},
    {"season": 2025, "race": "Canada"},
    {"season": 2025, "race": "Austria"},

    # 2024 races
    {"season": 2024, "race": "Bahrain"},
    {"season": 2024, "race": "Saudi Arabia"},
    {"season": 2024, "race": "Australia"},
    {"season": 2024, "race": "Japan"},
    {"season": 2024, "race": "China"},
    {"season": 2024, "race": "Miami"},
    {"season": 2024, "race": "Emilia Romagna"},
    {"season": 2024, "race": "Monaco"},
    {"season": 2024, "race": "Canada"},
    {"season": 2024, "race": "Spain"},
]


RACES_TO_LOAD: List[Dict] = COMPLETED_2026_RACES + HISTORICAL_BACKUP_RACES


def safe_filename(season: int, race: str) -> str:
    race_clean = (
        race.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    return f"{season}_{race_clean}_race"


def build_historical_dataset() -> pd.DataFrame:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_clean_laps = []
    failed_races = []

    for item in RACES_TO_LOAD:
        season = item["season"]
        race = item["race"]

        print("\n" + "=" * 90)
        print(f"Loading {season} {race} GP race data")
        print("=" * 90)

        raw_path = RAW_OUTPUT_DIR / f"{safe_filename(season, race)}_raw.csv"
        clean_path = CLEAN_OUTPUT_DIR / f"{safe_filename(season, race)}_clean.csv"

        try:
            raw_df = load_race_laps(
                season=season,
                race=race,
                session_type="R",
                save_path=str(raw_path),
            )

            clean_df = clean_laps_for_degradation(raw_df)
            clean_df.to_csv(clean_path, index=False)

            all_clean_laps.append(clean_df)

            print(f"Loaded successfully: {season} {race}")
            print(f"Raw rows:   {len(raw_df)}")
            print(f"Clean rows: {len(clean_df)}")
            print(f"Saved raw:   {raw_path}")
            print(f"Saved clean: {clean_path}")

        except Exception as e:
            print(f"FAILED: {season} {race}")
            print(f"Reason: {e}")
            failed_races.append(
                {
                    "season": season,
                    "race": race,
                    "error": str(e),
                }
            )

    if not all_clean_laps:
        raise RuntimeError("No races were successfully loaded. Check FastF1 data availability.")

    historical_df = pd.concat(all_clean_laps, ignore_index=True)

    combined_output_path = OUTPUT_DIR / "historical_laps_clean.csv"
    historical_df.to_csv(combined_output_path, index=False)

    failed_output_path = OUTPUT_DIR / "failed_races.csv"
    if failed_races:
        pd.DataFrame(failed_races).to_csv(failed_output_path, index=False)

    print("\n" + "=" * 90)
    print("Historical dataset complete")
    print("=" * 90)

    print(f"Total clean rows: {len(historical_df)}")
    print(f"Saved combined dataset: {combined_output_path}")

    print("\nRows by season:")
    print(historical_df.groupby("season").size())

    print("\nRows by race:")
    print(historical_df.groupby(["season", "race"]).size())

    print("\nCompounds:")
    print(historical_df["Compound"].value_counts())

    if failed_races:
        print("\nSome races failed to load.")
        print(f"Failure log saved to: {failed_output_path}")

    return historical_df


if __name__ == "__main__":
    build_historical_dataset()