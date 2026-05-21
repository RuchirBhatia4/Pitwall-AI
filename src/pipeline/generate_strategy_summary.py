from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_STRATEGY_RECOMMENDATIONS_PATH = Path("data/predictions/strategy_recommendations.csv")
DEFAULT_OUTPUT_PATH = Path("data/predictions/strategy_summary.csv")

REQUIRED_INPUT_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "strategy",
    "pit_laps",
    "scenario_name",
    "expected_total_time",
    "is_recommended",
    "stops",
    "safety_car_gain_seconds",
    "recommendation_reason",
]

OUTPUT_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "best_no_safety_car_strategy",
    "best_no_safety_car_pit_laps",
    "best_no_safety_car_time",
    "best_overall_strategy",
    "best_overall_scenario",
    "best_overall_pit_laps",
    "best_overall_time",
    "best_safety_car_opportunity",
    "max_safety_car_gain_seconds",
    "recommended_stop_count",
    "starting_compound_recommendation",
    "recommendation_reason",
]


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Strategy recommendations CSV is missing required columns: "
            f"{missing}. Required columns: {required_columns}"
        )


def _load_strategy_recommendations(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Strategy recommendations file not found: {csv_path}. "
            "Run `python -m src.pipeline.generate_strategy_recommendations` first."
        )

    df = pd.read_csv(csv_path)
    _validate_columns(df, REQUIRED_INPUT_COLUMNS)

    normalized = df.copy()
    normalized["race"] = normalized["race"].astype(str).str.strip()
    normalized["driver"] = normalized["driver"].astype(str).str.strip().str.upper()
    normalized["team"] = normalized["team"].astype(str).str.strip()
    normalized["scenario_name"] = normalized["scenario_name"].astype(str).str.strip()

    numeric_cols = [
        "starting_position",
        "expected_total_time",
        "safety_car_gain_seconds",
        "stops",
    ]
    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if normalized["is_recommended"].dtype != bool:
        normalized["is_recommended"] = (
            normalized["is_recommended"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    return normalized


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    ranked = df.sort_values(
        ["expected_total_time", "strategy", "pit_laps"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _format_safety_car_opportunity(row: pd.Series | None) -> str:
    if row is None:
        return "none"
    return str(row["scenario_name"])


def _infer_stop_count(best_overall: pd.Series) -> int:
    stops = best_overall.get("stops", float("nan"))
    if pd.notna(stops):
        return int(stops)

    strategy = str(best_overall["strategy"])
    compounds = [item for item in strategy.split("-") if item.strip()]
    return max(0, len(compounds) - 1)


def _infer_starting_compound(strategy: str) -> str:
    items = [item.strip().upper() for item in str(strategy).split("-") if item.strip()]
    return items[0] if items else "UNKNOWN"


def _build_recommendation_reason(
    *,
    best_overall: pd.Series,
    best_no_sc: pd.Series,
    best_safety_row: pd.Series | None,
    max_safety_gain: float,
) -> str:
    strategy = str(best_overall["strategy"])
    pit_laps = str(best_overall["pit_laps"])
    scenario = str(best_overall["scenario_name"])

    if scenario == "no_safety_car":
        base = f"Primary call: {strategy} ({pit_laps}) for a clean race."
    else:
        scenario_text = scenario.replace("_", " ")
        base = f"Primary call: {strategy} ({pit_laps}) if {scenario_text} develops."

    if best_safety_row is not None:
        safety_scenario = str(best_safety_row["scenario_name"]).replace("_", " ")
        return f"{base} Max safety-car upside: {max_safety_gain:.2f}s in {safety_scenario}."

    no_sc_strategy = str(best_no_sc["strategy"])
    if no_sc_strategy != strategy:
        return f"{base} Baseline no-safety-car option: {no_sc_strategy}."

    return base


def build_strategy_summary(strategy_df: pd.DataFrame) -> pd.DataFrame:
    if strategy_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict[str, str | int | float]] = []

    group_cols = ["race", "driver", "team", "starting_position"]
    for keys, group in strategy_df.groupby(group_cols, dropna=False):
        race, driver, team, starting_position = keys
        group = group.copy().reset_index(drop=True)

        no_sc_recommended = group[
            (group["scenario_name"] == "no_safety_car")
            & (group["is_recommended"])
        ].copy()
        if not no_sc_recommended.empty:
            best_no_sc = _pick_best_row(no_sc_recommended)
        else:
            no_sc = group[group["scenario_name"] == "no_safety_car"].copy()
            best_no_sc = _pick_best_row(no_sc) if not no_sc.empty else _pick_best_row(group)

        best_overall = _pick_best_row(group)

        safety_rows = group[group["scenario_name"] != "no_safety_car"].copy()
        if safety_rows.empty:
            max_gain = 0.0
            best_safety = None
        else:
            max_gain = float(
                pd.to_numeric(safety_rows["safety_car_gain_seconds"], errors="coerce")
                .fillna(0.0)
                .max()
            )
            best_safety_candidates = safety_rows[
                pd.to_numeric(
                    safety_rows["safety_car_gain_seconds"],
                    errors="coerce",
                ).fillna(0.0) == max_gain
            ]
            best_safety = _pick_best_row(best_safety_candidates)

        concise_reason = _build_recommendation_reason(
            best_overall=best_overall,
            best_no_sc=best_no_sc,
            best_safety_row=best_safety,
            max_safety_gain=max_gain,
        )

        rows.append(
            {
                "race": str(race),
                "driver": str(driver),
                "team": str(team),
                "starting_position": int(starting_position) if pd.notna(starting_position) else 0,
                "best_no_safety_car_strategy": str(best_no_sc["strategy"]),
                "best_no_safety_car_pit_laps": str(best_no_sc["pit_laps"]),
                "best_no_safety_car_time": float(best_no_sc["expected_total_time"]),
                "best_overall_strategy": str(best_overall["strategy"]),
                "best_overall_scenario": str(best_overall["scenario_name"]),
                "best_overall_pit_laps": str(best_overall["pit_laps"]),
                "best_overall_time": float(best_overall["expected_total_time"]),
                "best_safety_car_opportunity": _format_safety_car_opportunity(best_safety),
                "max_safety_car_gain_seconds": float(max_gain),
                "recommended_stop_count": _infer_stop_count(best_overall),
                "starting_compound_recommendation": _infer_starting_compound(
                    best_overall["strategy"]
                ),
                "recommendation_reason": concise_reason,
            }
        )

    summary = pd.DataFrame(rows)
    return summary[OUTPUT_COLUMNS].sort_values(["race", "starting_position", "driver"]).reset_index(drop=True)


def generate_strategy_summary(
    strategy_recommendations_path: str | Path = DEFAULT_STRATEGY_RECOMMENDATIONS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    strategy_df = _load_strategy_recommendations(strategy_recommendations_path)
    summary = build_strategy_summary(strategy_df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False)

    return summary


def main() -> None:
    parser = ArgumentParser(description="Generate concise per-driver strategy summary report.")
    parser.add_argument(
        "--strategy-recommendations",
        default=str(DEFAULT_STRATEGY_RECOMMENDATIONS_PATH),
        help="Path to strategy recommendations CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for concise strategy summary CSV.",
    )
    args = parser.parse_args()

    summary = generate_strategy_summary(
        strategy_recommendations_path=args.strategy_recommendations,
        output_path=args.output,
    )

    print("Strategy summary generated.")
    print(f"- Rows: {len(summary)}")
    print(f"- Output: {Path(args.output)}")
    print(summary.head(10))


if __name__ == "__main__":
    main()
