from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = "data/models/xgboost_degradation_model.pkl"
OUTPUT_DIR = Path("data/predictions/degradation_curves")


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def build_prediction_grid(
    season: int,
    race: str,
    driver: str,
    team: str,
    compound: str,
    stint: int,
    start_lap: int,
    max_tyre_life: int = 25,
) -> pd.DataFrame:
    """
    Creates synthetic rows to predict degradation as tire age increases.

    The trained model now predicts:
    lap_time_delta_from_stint_start
    """
    rows = []

    for tyre_life in range(1, max_tyre_life + 1):
        rows.append(
            {
                "season": season,
                "race": race,
                "Driver": driver,
                "Team": team,
                "Compound": compound,
                "TyreLife": tyre_life,
                "Stint": stint,
                "LapNumber": start_lap + tyre_life - 1,
            }
        )

    return pd.DataFrame(rows)


def predict_curve(
    model,
    season: int,
    race: str,
    driver: str,
    team: str,
    compound: str,
    stint: int = 1,
    start_lap: int = 1,
    max_tyre_life: int = 25,
) -> pd.DataFrame:
    grid = build_prediction_grid(
        season=season,
        race=race,
        driver=driver,
        team=team,
        compound=compound,
        stint=stint,
        start_lap=start_lap,
        max_tyre_life=max_tyre_life,
    )

    grid["predicted_degradation_seconds"] = model.predict(grid)

    return grid


def plot_curve(curve_df: pd.DataFrame, driver: str, compound: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        curve_df["TyreLife"],
        curve_df["predicted_degradation_seconds"],
        marker="o",
    )

    plt.title(f"Predicted Tire Degradation: {driver} on {compound}")
    plt.xlabel("Tire Age / TyreLife")
    plt.ylabel("Predicted Degradation From Stint Start (seconds)")
    plt.grid(True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_sample_curves():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    # Use races/drivers that exist in your current historical dataset.
    examples = [
        {
            "season": 2026,
            "race": "Australia",
            "driver": "VER",
            "team": "Red Bull Racing",
            "compound": "MEDIUM",
            "stint": 1,
            "start_lap": 1,
        },
        {
            "season": 2026,
            "race": "Australia",
            "driver": "LEC",
            "team": "Ferrari",
            "compound": "MEDIUM",
            "stint": 1,
            "start_lap": 1,
        },
        {
            "season": 2026,
            "race": "Australia",
            "driver": "NOR",
            "team": "McLaren",
            "compound": "MEDIUM",
            "stint": 1,
            "start_lap": 1,
        },
        {
            "season": 2026,
            "race": "Australia",
            "driver": "HAM",
            "team": "Ferrari",
            "compound": "MEDIUM",
            "stint": 1,
            "start_lap": 1,
        },
    ]

    all_curves = []

    for item in examples:
        curve_df = predict_curve(
            model=model,
            season=item["season"],
            race=item["race"],
            driver=item["driver"],
            team=item["team"],
            compound=item["compound"],
            stint=item["stint"],
            start_lap=item["start_lap"],
            max_tyre_life=25,
        )

        all_curves.append(curve_df)

        csv_path = OUTPUT_DIR / f"{item['driver']}_{item['compound']}_curve.csv"
        png_path = OUTPUT_DIR / f"{item['driver']}_{item['compound']}_curve.png"

        curve_df.to_csv(csv_path, index=False)
        plot_curve(curve_df, item["driver"], item["compound"], png_path)

        print(f"Saved curve for {item['driver']} {item['compound']}:")
        print(f"- {csv_path}")
        print(f"- {png_path}")

    combined = pd.concat(all_curves, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "combined_degradation_curves.csv", index=False)

    print("\nSaved combined curves:")
    print(f"- {OUTPUT_DIR / 'combined_degradation_curves.csv'}")


if __name__ == "__main__":
    generate_sample_curves()