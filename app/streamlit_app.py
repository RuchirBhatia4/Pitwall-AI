from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_live_race_simulation import (
    OUTPUT_COLUMNS as LIVE_REQUIRED_COLUMNS,
    run_live_race_simulation,
)

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - handled at runtime in Streamlit.
    st = None


CALIBRATED_PREDICTIONS_PATH = Path("data/predictions/calibrated_degradation_predictions.csv")
STRATEGY_RECOMMENDATIONS_PATH = Path("data/predictions/strategy_recommendations.csv")
LIVE_SIMULATION_PATH = Path("data/predictions/live_race_simulation.csv")

LIVE_EVENTS_PATH = Path("data/raw/race_control/live_race_input_sample.csv")
LIVE_OFFICIAL_SETUP_PATH = Path("data/raw/race_setup/race_setup_official_seed.csv")
LIVE_SAMPLE_SETUP_PATH = Path("data/raw/race_setup/race_setup_sample.csv")

CALIBRATED_REQUIRED_COLUMNS = [
    "race",
    "driver",
    "team",
    "compound",
    "historical_deg_estimate",
    "practice_deg_estimate",
    "practice_confidence",
    "calibrated_deg_estimate",
    "clean_laps_used",
    "sessions_used",
    "calibration_method",
]

STRATEGY_REQUIRED_COLUMNS = [
    "race",
    "driver",
    "team",
    "strategy",
    "pit_laps",
    "expected_total_time",
    "time_delta_to_best",
    "risk_score",
    "is_recommended",
]

MISSING_DATA_INSTRUCTIONS = (
    "python -m src.pipeline.generate_calibrated_predictions\n"
    "python -m src.pipeline.generate_strategy_recommendations"
)

LIVE_MISSING_DATA_INSTRUCTIONS = (
    "python -m src.data.fetch_official_f1_setup --race Miami --season 2026\n"
    "venv/bin/python -m src.pipeline.run_live_race_simulation"
)

LIVE_EVENT_COLUMNS = [
    "lap",
    "event_type",
    "driver",
    "target_driver",
    "compound",
    "notes",
]

LIVE_EVENT_TYPES = [
    "safety_car_start",
    "safety_car_end",
    "pit_stop",
    "overtake",
    "retirement",
]


def validate_dataframe_contract(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Required columns: {required_columns}"
        )


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def load_dashboard_data(
    calibrated_path: Path = CALIBRATED_PREDICTIONS_PATH,
    strategy_path: Path = STRATEGY_RECOMMENDATIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated_df = pd.read_csv(calibrated_path)
    strategy_df = pd.read_csv(strategy_path)

    validate_dataframe_contract(
        calibrated_df,
        CALIBRATED_REQUIRED_COLUMNS,
        "calibrated_degradation_predictions.csv",
    )
    validate_dataframe_contract(
        strategy_df,
        STRATEGY_REQUIRED_COLUMNS,
        "strategy_recommendations.csv",
    )

    strategy_df["is_recommended"] = _to_bool_series(strategy_df["is_recommended"])
    return calibrated_df, strategy_df


def apply_filters(
    calibrated_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    selected_drivers: list[str],
    selected_teams: list[str],
    selected_compounds: list[str],
    selected_strategies: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = calibrated_df.copy()
    strat = strategy_df.copy()

    if selected_drivers:
        cal = cal[cal["driver"].isin(selected_drivers)]
        strat = strat[strat["driver"].isin(selected_drivers)]

    if selected_teams:
        cal = cal[cal["team"].isin(selected_teams)]
        strat = strat[strat["team"].isin(selected_teams)]

    if selected_compounds:
        cal = cal[cal["compound"].isin(selected_compounds)]

    if selected_strategies:
        strat = strat[strat["strategy"].isin(selected_strategies)]

    return cal.reset_index(drop=True), strat.reset_index(drop=True)


def _summary_metrics(calibrated_df: pd.DataFrame, strategy_df: pd.DataFrame) -> dict[str, int]:
    driver_count = (
        pd.concat([calibrated_df["driver"], strategy_df["driver"]], ignore_index=True)
        .dropna()
        .nunique()
    )
    team_count = (
        pd.concat([calibrated_df["team"], strategy_df["team"]], ignore_index=True)
        .dropna()
        .nunique()
    )

    return {
        "drivers": int(driver_count),
        "teams": int(team_count),
        "calibrated_rows": int(len(calibrated_df)),
        "strategy_rows": int(len(strategy_df)),
        "recommended_strategies": int(strategy_df["is_recommended"].sum()),
    }


def get_live_missing_message() -> str:
    return (
        "Live simulation output is missing. Generate official setup seed and run live simulation first."
    )


def _show_missing_file_message() -> None:
    st.error(
        "Required prediction files were not found. Generate calibrated and strategy outputs first."
    )
    st.code(MISSING_DATA_INSTRUCTIONS)


def _show_missing_live_message() -> None:
    st.warning(get_live_missing_message())
    st.code(LIVE_MISSING_DATA_INSTRUCTIONS)


def _resolve_live_setup_path(prefer_official: bool = True) -> tuple[Path | None, str]:
    if prefer_official and LIVE_OFFICIAL_SETUP_PATH.exists():
        return LIVE_OFFICIAL_SETUP_PATH, "official_seeded"

    if LIVE_SAMPLE_SETUP_PATH.exists():
        return LIVE_SAMPLE_SETUP_PATH, "sample_setup (fallback)"

    if LIVE_OFFICIAL_SETUP_PATH.exists():
        return LIVE_OFFICIAL_SETUP_PATH, "official_seeded"

    return None, "missing"


def _read_race_options(setup_path: Path | None) -> list[str]:
    if setup_path is None or not setup_path.exists():
        return []
    setup_df = pd.read_csv(setup_path)
    if "race" not in setup_df.columns:
        return []
    return sorted(setup_df["race"].dropna().astype(str).str.strip().unique().tolist())


def _build_events_dataframe(
    manual_events: list[dict[str, str | int]],
    safety_car_start_lap: int | None = None,
    safety_car_end_lap: int | None = None,
) -> pd.DataFrame:
    events = []
    for event in manual_events:
        normalized = {
            "lap": int(event.get("lap", 1)),
            "event_type": str(event.get("event_type", "")).strip().lower(),
            "driver": str(event.get("driver", "")).strip().upper(),
            "target_driver": str(event.get("target_driver", "")).strip().upper(),
            "compound": str(event.get("compound", "")).strip().upper(),
            "notes": str(event.get("notes", "")).strip(),
        }
        if normalized["event_type"] in LIVE_EVENT_TYPES:
            events.append(normalized)

    if safety_car_start_lap is not None:
        events.append(
            {
                "lap": int(safety_car_start_lap),
                "event_type": "safety_car_start",
                "driver": "",
                "target_driver": "",
                "compound": "",
                "notes": "Safety car deployed from dashboard control",
            }
        )
    if safety_car_end_lap is not None:
        events.append(
            {
                "lap": int(safety_car_end_lap),
                "event_type": "safety_car_end",
                "driver": "",
                "target_driver": "",
                "compound": "",
                "notes": "Safety car ended from dashboard control",
            }
        )

    if not events:
        return pd.DataFrame(columns=LIVE_EVENT_COLUMNS)

    event_df = pd.DataFrame(events)
    validate_dataframe_contract(
        event_df,
        LIVE_EVENT_COLUMNS,
        "live_race_input_sample.csv",
    )
    event_df = event_df.sort_values(["lap", "event_type", "driver"]).reset_index(drop=True)
    return event_df


def _write_live_events_file(events_df: pd.DataFrame, events_path: Path = LIVE_EVENTS_PATH) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(events_path, index=False)


def load_live_simulation_data(
    live_output_path: Path = LIVE_SIMULATION_PATH,
) -> pd.DataFrame:
    if not live_output_path.exists():
        raise FileNotFoundError(
            f"{get_live_missing_message()}\n{LIVE_MISSING_DATA_INSTRUCTIONS}"
        )

    live_df = pd.read_csv(live_output_path)
    validate_dataframe_contract(live_df, LIVE_REQUIRED_COLUMNS, "live_race_simulation.csv")
    return live_df


def _render_overview_tab(calibrated_df: pd.DataFrame, strategy_df: pd.DataFrame) -> None:
    st.sidebar.header("Overview Filters")

    driver_options = sorted(
        pd.concat([calibrated_df["driver"], strategy_df["driver"]], ignore_index=True)
        .dropna()
        .unique()
        .tolist()
    )
    team_options = sorted(
        pd.concat([calibrated_df["team"], strategy_df["team"]], ignore_index=True)
        .dropna()
        .unique()
        .tolist()
    )
    compound_options = sorted(calibrated_df["compound"].dropna().unique().tolist())
    strategy_options = sorted(strategy_df["strategy"].dropna().unique().tolist())

    selected_drivers = st.sidebar.multiselect(
        "Driver Filter",
        options=driver_options,
        default=driver_options,
    )
    selected_teams = st.sidebar.multiselect(
        "Team Filter",
        options=team_options,
        default=team_options,
    )
    selected_compounds = st.sidebar.multiselect(
        "Compound Filter (Degradation Table)",
        options=compound_options,
        default=compound_options,
    )
    selected_strategies = st.sidebar.multiselect(
        "Strategy Filter (Strategy Table)",
        options=strategy_options,
        default=strategy_options,
    )

    cal_filtered, strat_filtered = apply_filters(
        calibrated_df=calibrated_df,
        strategy_df=strategy_df,
        selected_drivers=selected_drivers,
        selected_teams=selected_teams,
        selected_compounds=selected_compounds,
        selected_strategies=selected_strategies,
    )

    metrics = _summary_metrics(cal_filtered, strat_filtered)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Drivers", metrics["drivers"])
    metric_cols[1].metric("Teams", metrics["teams"])
    metric_cols[2].metric("Calibrated Rows", metrics["calibrated_rows"])
    metric_cols[3].metric("Strategy Rows", metrics["strategy_rows"])
    metric_cols[4].metric("Recommended Strategies", metrics["recommended_strategies"])

    st.subheader("Calibrated Degradation Predictions")
    st.dataframe(cal_filtered, use_container_width=True)

    st.subheader("Strategy Recommendations")
    st.dataframe(strat_filtered, use_container_width=True)

    st.subheader("Recommended Strategies Only")
    recommended_only = strat_filtered[strat_filtered["is_recommended"]].reset_index(drop=True)
    if recommended_only.empty:
        st.info("No recommended strategies for the current filter selection.")
    else:
        st.dataframe(recommended_only, use_container_width=True)


def _render_live_tab() -> None:
    st.subheader("Live Race Strategy")
    st.write(
        "Run the live race simulation pipeline with custom race-control events and view "
        "real-time strategist recommendations."
    )

    if "live_manual_events" not in st.session_state:
        st.session_state["live_manual_events"] = []

    # Default behavior: prefer official setup when available, otherwise fallback to sample.
    setup_path, setup_source = _resolve_live_setup_path(prefer_official=True)

    setup_col, button_col = st.columns([3, 2])
    setup_col.info(f"Current setup source: `{setup_source}`")

    if button_col.button("Load Official Seeded Setup If Available"):
        if LIVE_OFFICIAL_SETUP_PATH.exists():
            setup_path, setup_source = _resolve_live_setup_path(prefer_official=True)
            st.success("Official seeded setup loaded.")
        else:
            st.warning(
                "Official seeded setup file not found. Fallback sample setup will be used if available."
            )

    if setup_path is None:
        _show_missing_live_message()
        return

    race_options = _read_race_options(setup_path)
    if not race_options:
        st.error(f"No race options found in setup file: {setup_path}")
        return

    selected_race = st.selectbox("Race Selector", options=race_options, index=0)

    st.markdown("**Safety Car Controls**")
    sc_enabled = st.checkbox("Enable safety car window", value=False)
    sc_col1, sc_col2 = st.columns(2)
    sc_start_lap = sc_col1.number_input("Safety car start lap", min_value=1, value=12, step=1)
    sc_end_lap = sc_col2.number_input("Safety car end lap", min_value=1, value=16, step=1)

    if not sc_enabled:
        sc_start = None
        sc_end = None
    else:
        sc_start = int(sc_start_lap)
        sc_end = int(sc_end_lap)
        if sc_end <= sc_start:
            st.warning("Safety car end lap should be greater than start lap.")

    st.markdown("**Manual Event Controls**")
    with st.form("live_manual_event_form"):
        form_col1, form_col2, form_col3 = st.columns(3)
        event_lap = form_col1.number_input("lap", min_value=1, value=10, step=1)
        event_type = form_col2.selectbox("event_type", options=LIVE_EVENT_TYPES, index=2)
        event_driver = form_col3.text_input("driver", value="").strip().upper()

        form_col4, form_col5, form_col6 = st.columns(3)
        event_target_driver = form_col4.text_input("target_driver", value="").strip().upper()
        event_compound = form_col5.text_input("compound", value="").strip().upper()
        event_notes = form_col6.text_input("notes", value="dashboard event").strip()

        add_event = st.form_submit_button("Add Event")

    if add_event:
        st.session_state["live_manual_events"].append(
            {
                "lap": int(event_lap),
                "event_type": str(event_type),
                "driver": str(event_driver),
                "target_driver": str(event_target_driver),
                "compound": str(event_compound),
                "notes": str(event_notes),
            }
        )
        st.success("Event added.")

    events_preview = _build_events_dataframe(
        manual_events=st.session_state["live_manual_events"],
        safety_car_start_lap=sc_start,
        safety_car_end_lap=sc_end,
    )
    st.write("Events queued for simulation:")
    st.dataframe(events_preview, use_container_width=True)

    event_button_col1, event_button_col2 = st.columns(2)
    if event_button_col1.button("Clear Manual Events"):
        st.session_state["live_manual_events"] = []
        st.info("Manual events cleared.")
    if event_button_col2.button("Remove Last Event"):
        if st.session_state["live_manual_events"]:
            st.session_state["live_manual_events"].pop()
            st.info("Removed last manual event.")

    run_live = st.button("Run Live Simulation")
    if run_live:
        _write_live_events_file(events_preview, events_path=LIVE_EVENTS_PATH)
        try:
            run_live_race_simulation(
                race=selected_race,
                setup_path=setup_path,
                events_path=LIVE_EVENTS_PATH,
                output_path=LIVE_SIMULATION_PATH,
            )
            st.success("Live simulation updated.")
        except Exception as exc:
            st.error(f"Live simulation failed: {exc}")
            return

    if not LIVE_SIMULATION_PATH.exists():
        _show_missing_live_message()
        return

    try:
        live_df = load_live_simulation_data()
    except Exception as exc:
        st.error(f"Could not load live simulation output: {exc}")
        st.code(LIVE_MISSING_DATA_INSTRUCTIONS)
        return

    live_df = live_df[live_df["race"].astype(str) == str(selected_race)].copy()
    if live_df.empty:
        st.warning("No live simulation rows found for the selected race.")
        return

    st.markdown("**Live Filters**")
    lf_col1, lf_col2, lf_col3, lf_col4 = st.columns(4)
    live_driver_options = sorted(live_df["driver"].dropna().unique().tolist())
    live_team_options = sorted(live_df["team"].dropna().unique().tolist())
    live_compound_options = sorted(live_df["current_compound"].dropna().unique().tolist())
    live_action_options = sorted(live_df["best_action_now"].dropna().unique().tolist())

    selected_live_drivers = lf_col1.multiselect(
        "driver",
        options=live_driver_options,
        default=live_driver_options,
    )
    selected_live_teams = lf_col2.multiselect(
        "team",
        options=live_team_options,
        default=live_team_options,
    )
    selected_live_compounds = lf_col3.multiselect(
        "current compound",
        options=live_compound_options,
        default=live_compound_options,
    )
    selected_live_actions = lf_col4.multiselect(
        "best_action_now",
        options=live_action_options,
        default=live_action_options,
    )

    filtered_live = live_df.copy()
    if selected_live_drivers:
        filtered_live = filtered_live[filtered_live["driver"].isin(selected_live_drivers)]
    if selected_live_teams:
        filtered_live = filtered_live[filtered_live["team"].isin(selected_live_teams)]
    if selected_live_compounds:
        filtered_live = filtered_live[filtered_live["current_compound"].isin(selected_live_compounds)]
    if selected_live_actions:
        filtered_live = filtered_live[filtered_live["best_action_now"].isin(selected_live_actions)]

    latest_lap = int(filtered_live["lap"].max())
    latest_rows = filtered_live[filtered_live["lap"] == latest_lap].copy()
    latest_rows = latest_rows.sort_values("current_position").reset_index(drop=True)

    live_metric_cols = st.columns(5)
    live_metric_cols[0].metric("Current Lap", latest_lap)
    live_metric_cols[1].metric("Active Drivers", int(latest_rows["driver"].nunique()))
    live_metric_cols[2].metric(
        "Recommended Pit Now",
        int((latest_rows["best_action_now"] == "pit_now").sum()),
    )
    live_metric_cols[3].metric(
        "Recommended Stay Out",
        int((latest_rows["best_action_now"] == "stay_out").sum()),
    )
    sc_active = bool(latest_rows["under_safety_car"].fillna(False).any())
    live_metric_cols[4].metric("Safety Car Active", "Yes" if sc_active else "No")

    st.subheader("Current Running Order (Latest Lap)")
    running_order_cols = [
        "driver",
        "team",
        "current_position",
        "current_compound",
        "tire_age",
        "stops_made",
        "best_action_now",
        "projected_finish_if_pit_now",
        "projected_finish_if_stay_out",
        "pit_now_gain_seconds",
        "stay_out_gain_seconds",
        "overtake_probability_if_pit_now",
        "overtake_probability_if_stay_out",
        "recommendation_reason",
    ]
    st.dataframe(latest_rows[running_order_cols], use_container_width=True)

    st.subheader("Live Race Strategy Table")
    st.dataframe(filtered_live, use_container_width=True)


def run_dashboard() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install Streamlit and run with "
            "`streamlit run app/streamlit_app.py`."
        )

    st.set_page_config(page_title="PitWall AI", layout="wide")
    st.title("PitWall AI")
    st.write(
        "Formula 1 tire degradation and race strategy intelligence dashboard. "
        "Use the overview and live tabs to analyze recommendations."
    )

    overview_tab, live_tab = st.tabs(["Model Overview", "Live Race Strategy"])

    with overview_tab:
        if not CALIBRATED_PREDICTIONS_PATH.exists() or not STRATEGY_RECOMMENDATIONS_PATH.exists():
            _show_missing_file_message()
        else:
            try:
                calibrated_df, strategy_df = load_dashboard_data()
                _render_overview_tab(calibrated_df=calibrated_df, strategy_df=strategy_df)
            except Exception as exc:
                st.error(f"Could not load dashboard data: {exc}")
                st.code(MISSING_DATA_INSTRUCTIONS)

    with live_tab:
        _render_live_tab()


if __name__ == "__main__":
    run_dashboard()
