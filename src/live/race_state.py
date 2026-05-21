from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_SETUP_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "starting_compound",
    "race_laps",
]


@dataclass
class DriverRaceState:
    race: str
    driver: str
    team: str
    current_position: int
    current_compound: str
    tire_age: int
    stops_made: int
    in_race: bool


@dataclass
class LiveRaceState:
    race: str
    lap: int
    race_laps: int
    under_safety_car: bool
    setup_source: str
    drivers: dict[str, DriverRaceState]

    def get_driver(self, driver: str) -> DriverRaceState | None:
        return self.drivers.get(str(driver).strip().upper())

    def iter_active(self) -> Iterable[DriverRaceState]:
        active = [driver_state for driver_state in self.drivers.values() if driver_state.in_race]
        return sorted(active, key=lambda item: item.current_position)

    def recompute_positions(self) -> None:
        for idx, state in enumerate(self.iter_active(), start=1):
            state.current_position = idx

    def set_under_safety_car(self, enabled: bool) -> None:
        self.under_safety_car = bool(enabled)

    def apply_pit_stop(self, driver: str, compound: str | None = None) -> None:
        state = self.get_driver(driver)
        if state is None or not state.in_race:
            return

        if compound and str(compound).strip():
            state.current_compound = str(compound).strip().upper()

        state.tire_age = 0
        state.stops_made += 1

    def apply_overtake(self, driver: str, target_driver: str) -> None:
        attacker = self.get_driver(driver)
        defender = self.get_driver(target_driver)
        if attacker is None or defender is None:
            return
        if not attacker.in_race or not defender.in_race:
            return

        if attacker.current_position > defender.current_position:
            attacker.current_position, defender.current_position = (
                defender.current_position,
                attacker.current_position,
            )
            self.recompute_positions()

    def retire_driver(self, driver: str) -> None:
        state = self.get_driver(driver)
        if state is None:
            return

        state.in_race = False
        self.recompute_positions()

    def to_output_row(self, state: DriverRaceState) -> dict[str, str | int | bool]:
        return {
            "race": self.race,
            "lap": int(self.lap),
            "driver": state.driver,
            "team": state.team,
            "current_position": int(state.current_position),
            "current_compound": state.current_compound,
            "tire_age": int(state.tire_age),
            "stops_made": int(state.stops_made),
            "under_safety_car": bool(self.under_safety_car),
            "setup_source": self.setup_source,
        }


def _validate_setup_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_SETUP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Setup CSV missing required columns: {missing}")


def build_race_state_from_setup(
    setup_df: pd.DataFrame,
    race: str,
    setup_source: str,
) -> LiveRaceState:
    _validate_setup_columns(setup_df)

    race_rows = setup_df[
        setup_df["race"].astype(str).str.strip().str.lower() == str(race).strip().lower()
    ].copy()
    if race_rows.empty:
        raise ValueError(f"No setup rows found for race '{race}'.")

    race_rows["driver"] = race_rows["driver"].astype(str).str.strip().str.upper()
    race_rows["team"] = race_rows["team"].astype(str).str.strip()
    race_rows["starting_compound"] = race_rows["starting_compound"].astype(str).str.strip().str.upper()

    race_rows["starting_position"] = pd.to_numeric(
        race_rows["starting_position"], errors="coerce"
    )
    race_rows["race_laps"] = pd.to_numeric(race_rows["race_laps"], errors="coerce")

    race_rows = race_rows.dropna(
        subset=["driver", "team", "starting_position", "starting_compound", "race_laps"]
    ).copy()

    race_rows["starting_position"] = race_rows["starting_position"].astype(int)
    race_rows["race_laps"] = race_rows["race_laps"].astype(int)

    drivers: dict[str, DriverRaceState] = {}
    for _, row in race_rows.sort_values("starting_position").iterrows():
        driver = str(row["driver"]).upper()
        drivers[driver] = DriverRaceState(
            race=str(row["race"]).strip(),
            driver=driver,
            team=str(row["team"]).strip(),
            current_position=int(row["starting_position"]),
            current_compound=str(row["starting_compound"]).upper(),
            tire_age=0,
            stops_made=0,
            in_race=True,
        )

    if not drivers:
        raise ValueError(f"No valid driver rows found for race '{race}'.")

    race_laps = int(race_rows["race_laps"].max())

    return LiveRaceState(
        race=str(race).strip(),
        lap=0,
        race_laps=race_laps,
        under_safety_car=False,
        setup_source=str(setup_source),
        drivers=drivers,
    )
