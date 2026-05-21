import pandas as pd


def clean_laps_for_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans lap data for tire degradation modeling.

    Removes:
    - missing lap times
    - pit-in laps
    - pit-out laps
    - deleted laps
    - extreme outlier laps

    Creates:
    - stint_baseline_lap_time
    - lap_time_delta_from_stint_start
    - normalized_lap_time
    - fuel_adjusted_lap_time
    - stint_baseline_fuel_adjusted_lap_time
    - fuel_adjusted_degradation_delta
    - race_phase
    - stint_progress
    - nonlinear tire/race features

    The fuel-adjusted target is important because race laps naturally get faster
    as fuel burns off. Without this correction, early stint deltas can look
    strongly negative and confuse tire degradation modeling.
    """

    clean = df.copy()

    required_cols = ["lap_time_seconds", "Compound", "TyreLife", "Stint"]
    clean = clean.dropna(subset=required_cols)

    # Remove pit-in laps
    if "PitInTime" in clean.columns:
        clean = clean[clean["PitInTime"].isna()]

    # Remove pit-out laps
    if "PitOutTime" in clean.columns:
        clean = clean[clean["PitOutTime"].isna()]

    # Remove deleted laps
    if "Deleted" in clean.columns:
        clean = clean[clean["Deleted"] != True]

    # Basic lap-time sanity check
    clean = clean[clean["lap_time_seconds"] > 0]

    numeric_cols = ["lap_time_seconds", "TyreLife", "Stint", "LapNumber"]

    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(
        subset=[
            "lap_time_seconds",
            "TyreLife",
            "Stint",
            "LapNumber",
        ]
    )

    # Make sure season exists and is numeric if present.
    if "season" in clean.columns:
        clean["season"] = pd.to_numeric(clean["season"], errors="coerce")
        clean = clean.dropna(subset=["season"])
        clean["season"] = clean["season"].astype(int)

    # Make sure race/driver/team/compound are strings.
    if "race" in clean.columns:
        clean["race"] = clean["race"].astype(str)

    if "Driver" in clean.columns:
        clean["Driver"] = clean["Driver"].astype(str)

    if "Team" in clean.columns:
        clean["Team"] = clean["Team"].astype(str)

    if "Compound" in clean.columns:
        clean["Compound"] = clean["Compound"].astype(str).str.upper()

    # ------------------------------------------------------------
    # Remove very large outlier laps per driver/race
    # ------------------------------------------------------------
    group_cols = ["season", "race", "Driver"]

    clean["driver_median_lap"] = clean.groupby(group_cols)["lap_time_seconds"].transform(
        "median"
    )

    clean["abs_delta_from_median"] = (
        clean["lap_time_seconds"] - clean["driver_median_lap"]
    ).abs()

    # Removes very slow laps caused by traffic, safety cars, incidents, etc.
    # This is a simple first version. Later we can add TrackStatus-specific logic.
    clean = clean[clean["abs_delta_from_median"] < 8.0]

    # ------------------------------------------------------------
    # Sort before creating stint-level features
    # ------------------------------------------------------------
    clean = clean.sort_values(
        ["season", "race", "Driver", "Stint", "LapNumber"]
    ).reset_index(drop=True)

    stint_group_cols = ["season", "race", "Driver", "Stint"]

    # ------------------------------------------------------------
    # Original stint-normalized target
    # ------------------------------------------------------------
    clean["stint_baseline_lap_time"] = clean.groupby(stint_group_cols)[
        "lap_time_seconds"
    ].transform("first")

    clean["lap_time_delta_from_stint_start"] = (
        clean["lap_time_seconds"] - clean["stint_baseline_lap_time"]
    )

    # Useful for later experiments.
    clean["normalized_lap_time"] = (
        clean["lap_time_seconds"] - clean["driver_median_lap"]
    )

    # ------------------------------------------------------------
    # Fuel-adjusted degradation target
    # ------------------------------------------------------------
    # Later race laps are naturally faster because the car gets lighter.
    # This simple correction adds back an estimated fuel-burn advantage.
    #
    # Example:
    # Lap 10 receives +0.60 sec correction if correction = 0.06 sec/lap.
    #
    # This is not perfect, but it reduces unrealistic large negative deltas.
    # Later we can tune this value by season, track, or learn it from data.
    FUEL_BURN_CORRECTION_PER_LAP = 0.06

    clean["fuel_adjusted_lap_time"] = (
        clean["lap_time_seconds"] + clean["LapNumber"] * FUEL_BURN_CORRECTION_PER_LAP
    )

    clean["stint_baseline_fuel_adjusted_lap_time"] = clean.groupby(stint_group_cols)[
        "fuel_adjusted_lap_time"
    ].transform("first")

    clean["fuel_adjusted_degradation_delta"] = (
        clean["fuel_adjusted_lap_time"]
        - clean["stint_baseline_fuel_adjusted_lap_time"]
    )

    # ------------------------------------------------------------
    # Race phase and nonlinear degradation features
    # ------------------------------------------------------------
    clean["race_max_lap"] = clean.groupby(["season", "race"])["LapNumber"].transform(
        "max"
    )

    clean["race_phase"] = clean["LapNumber"] / clean["race_max_lap"]

    clean["stint_max_tyre_life"] = clean.groupby(stint_group_cols)[
        "TyreLife"
    ].transform("max")

    clean["stint_progress"] = clean["TyreLife"] / clean["stint_max_tyre_life"]

    clean["tyre_life_squared"] = clean["TyreLife"] ** 2
    clean["lap_number_squared"] = clean["LapNumber"] ** 2

    clean["is_first_stint"] = (clean["Stint"] == 1).astype(int)
    clean["is_second_stint"] = (clean["Stint"] == 2).astype(int)
    clean["is_late_race"] = (clean["race_phase"] >= 0.66).astype(int)
    clean["is_early_race"] = (clean["race_phase"] <= 0.33).astype(int)

    # Compound-specific tire-age interactions.
    clean["is_soft"] = (clean["Compound"] == "SOFT").astype(int)
    clean["is_medium"] = (clean["Compound"] == "MEDIUM").astype(int)
    clean["is_hard"] = (clean["Compound"] == "HARD").astype(int)

    clean["soft_tyre_life"] = clean["is_soft"] * clean["TyreLife"]
    clean["medium_tyre_life"] = clean["is_medium"] * clean["TyreLife"]
    clean["hard_tyre_life"] = clean["is_hard"] * clean["TyreLife"]

    # ------------------------------------------------------------
    # Remove tiny stints with weak degradation signal
    # ------------------------------------------------------------
    clean["stint_lap_count"] = clean.groupby(stint_group_cols)["LapNumber"].transform(
        "count"
    )

    clean = clean[clean["stint_lap_count"] >= 3]

    clean = clean.reset_index(drop=True)

    return clean


if __name__ == "__main__":
    raw_path = "data/processed/2024_bahrain_race_laps_raw.csv"
    output_path = "data/processed/2024_bahrain_race_laps_clean.csv"

    df = pd.read_csv(raw_path)
    clean_df = clean_laps_for_degradation(df)

    clean_df.to_csv(output_path, index=False)

    print(clean_df.head())
    print(f"Raw rows: {len(df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Saved to: {output_path}")

    if "lap_time_delta_from_stint_start" in clean_df.columns:
        print("\nOriginal target summary:")
        print(clean_df["lap_time_delta_from_stint_start"].describe())

    if "fuel_adjusted_degradation_delta" in clean_df.columns:
        print("\nFuel-adjusted target summary:")
        print(clean_df["fuel_adjusted_degradation_delta"].describe())

    engineered_cols = [
        "race_phase",
        "stint_progress",
        "tyre_life_squared",
        "lap_number_squared",
        "is_first_stint",
        "is_second_stint",
        "is_late_race",
        "is_early_race",
        "soft_tyre_life",
        "medium_tyre_life",
        "hard_tyre_life",
    ]

    print("\nEngineered feature check:")
    print(clean_df[engineered_cols].head())