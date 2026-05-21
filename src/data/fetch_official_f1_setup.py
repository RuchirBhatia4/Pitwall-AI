from argparse import ArgumentParser
from pathlib import Path
import re
import unicodedata
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_SEASON = 2026
DEFAULT_OUTPUT_PATH = Path("data/raw/race_setup/race_setup_official_seed.csv")
DEFAULT_STARTING_COMPOUND = "MEDIUM"
DEFAULT_BASE_PACE_SECONDS = 92.0
DEFAULT_RACE_LAPS = 57
DEFAULT_PIT_LOSS_GREEN_SECONDS = 22.0
DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS = 13.0
DEFAULT_OVERTAKING_DIFFICULTY = 0.72
DEFAULT_TRACK_POSITION_IMPORTANCE = 0.78
DEFAULT_SAFETY_CAR_PROBABILITY = 0.45
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = "Mozilla/5.0 (PitWall-AI data ingest)"

F1_BASE_URL = "https://www.formula1.com"
TEAMS_URL = f"{F1_BASE_URL}/en/teams"

OUTPUT_COLUMNS = [
    "race",
    "driver",
    "team",
    "starting_position",
    "starting_compound",
    "base_pace_seconds",
    "race_laps",
    "pit_loss_green_seconds",
    "pit_loss_safety_car_seconds",
    "overtaking_difficulty",
    "track_position_importance",
    "safety_car_probability",
    "setup_source",
]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _title_from_slug(slug: str) -> str:
    tokens = [item for item in str(slug).replace("-", " ").split() if item]
    return " ".join(token.capitalize() for token in tokens)


def _build_driver_code(driver_name: str) -> str:
    letters_only = re.sub(r"[^A-Za-z]", "", str(driver_name)).upper()
    if len(letters_only) >= 3:
        return letters_only[:3]
    return (letters_only + "XXX")[:3]


def _parse_driver_cell(value: str) -> tuple[str, str]:
    cell = str(value).strip()
    match = re.match(r"^(?P<name>.+?)\s(?P<code>[A-Z]{3})$", cell)
    if match:
        return match.group("name").strip(), match.group("code").strip()
    return cell, _build_driver_code(cell)


def _get_html(url: str, session: requests.Session | None = None) -> str:
    http = session or requests.Session()
    response = http.get(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def fetch_driver_standings(
    season: int = DEFAULT_SEASON,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    standings_url = f"{F1_BASE_URL}/en/results/{season}/drivers"
    html = _get_html(standings_url, session=session)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find driver standings table on Formula1.com.")

    rows = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
        if len(cells) < 5:
            continue

        pos_raw, driver_raw, _, team, points_raw = cells[:5]
        position = pd.to_numeric(pos_raw, errors="coerce")
        if pd.isna(position):
            continue

        driver_name, driver_code = _parse_driver_cell(driver_raw)
        points = pd.to_numeric(points_raw, errors="coerce")

        rows.append(
            {
                "standing_position": int(position),
                "driver_name": driver_name,
                "driver_code": driver_code,
                "team": str(team).strip(),
                "points": float(points) if pd.notna(points) else 0.0,
            }
        )

    if not rows:
        raise ValueError("No rows parsed from Formula1.com driver standings.")

    standings = pd.DataFrame(rows)
    standings = standings.sort_values(["standing_position", "driver_name"]).reset_index(drop=True)
    return standings


def fetch_official_team_lineup(
    session: requests.Session | None = None,
) -> pd.DataFrame:
    html = _get_html(TEAMS_URL, session=session)
    soup = BeautifulSoup(html, "html.parser")

    # Pull candidate driver names from visible driver links.
    candidate_driver_names: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if not href.startswith("/en/drivers/") or href.endswith("/hall-of-fame"):
            continue
        text = " ".join(anchor.stripped_strings).strip()
        if not text or len(text.split()) < 2:
            continue
        if text not in candidate_driver_names:
            candidate_driver_names.append(text)

    team_card_text_by_href: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if not href.startswith("/en/teams/") or href == "/en/teams":
            continue

        text = " ".join(anchor.stripped_strings).strip()
        if not text:
            continue

        # The longer variant includes both team + listed drivers.
        if len(text) > len(team_card_text_by_href.get(href, "")):
            team_card_text_by_href[href] = text

    rows: list[dict[str, str]] = []
    for href, team_text in sorted(team_card_text_by_href.items()):
        matched_names = [
            name for name in candidate_driver_names
            if re.search(rf"\b{re.escape(name)}\b", team_text)
        ]

        if not matched_names:
            continue

        matched_names = sorted(
            set(matched_names),
            key=lambda item: team_text.find(item),
        )
        first_driver_idx = min(team_text.find(name) for name in matched_names)
        inferred_team_name = team_text[:first_driver_idx].strip()

        if not inferred_team_name:
            slug = href.split("/")[-1]
            inferred_team_name = _title_from_slug(slug)

        for driver_name in matched_names:
            rows.append(
                {
                    "team": inferred_team_name,
                    "driver_name": driver_name,
                }
            )

    lineup = pd.DataFrame(rows).drop_duplicates(subset=["team", "driver_name"]).reset_index(drop=True)
    if lineup.empty:
        raise ValueError("No team-driver lineup rows parsed from Formula1.com teams page.")

    return lineup


def fetch_race_schedule(
    season: int = DEFAULT_SEASON,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    races_url = f"{F1_BASE_URL}/en/results/{season}/races"
    html = _get_html(races_url, session=session)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find race schedule/results table on Formula1.com.")

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        race_anchor = tds[0].find("a", href=True)
        if race_anchor is None:
            continue

        href = str(race_anchor["href"])
        match = re.match(
            rf"^/en/results/{season}/races/(?P<round_id>\d+)/(?P<race_slug>[^/]+)/race-result$",
            href,
        )
        if match is None:
            continue

        race_slug = match.group("race_slug")
        laps_raw = tds[4].get_text(" ", strip=True)
        laps = pd.to_numeric(laps_raw, errors="coerce")

        rows.append(
            {
                "race_slug": race_slug,
                "race_name": _title_from_slug(race_slug),
                "race_result_url": f"{F1_BASE_URL}{href}",
                "race_laps": int(laps) if pd.notna(laps) else DEFAULT_RACE_LAPS,
            }
        )

    schedule = pd.DataFrame(rows).drop_duplicates(subset=["race_slug"]).reset_index(drop=True)
    return schedule


def _select_race_entry(race_query: str, schedule_df: pd.DataFrame) -> pd.Series | None:
    if schedule_df.empty:
        return None

    query = _normalize_space(race_query)
    if not query:
        return None

    scored: list[tuple[int, int, pd.Series]] = []
    for idx, row in schedule_df.reset_index(drop=True).iterrows():
        race_name = _normalize_space(row["race_name"])
        race_slug = _normalize_space(str(row["race_slug"]).replace("-", " "))

        score = 0
        if query == race_name or query == race_slug:
            score = 3
        elif query in race_name or query in race_slug:
            score = 2
        elif race_name in query or race_slug in query:
            score = 1

        if score > 0:
            scored.append((score, -idx, row))

    if not scored:
        return None

    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return scored[0][2]


def fetch_official_starting_grid(
    race_result_url: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    grid_url = str(race_result_url).replace("/race-result", "/starting-grid")
    html = _get_html(grid_url, session=session)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame(columns=["grid_position", "driver_name", "driver_code", "team"])

    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue

        pos_raw = cells[0]
        driver_raw = cells[2]
        team = cells[3]

        grid_pos = pd.to_numeric(pos_raw, errors="coerce")
        if pd.isna(grid_pos):
            continue

        driver_name, driver_code = _parse_driver_cell(driver_raw)

        rows.append(
            {
                "grid_position": int(grid_pos),
                "driver_name": driver_name,
                "driver_code": driver_code,
                "team": str(team).strip(),
            }
        )

    return pd.DataFrame(rows)


def _lookup_standings_row(
    standings_df: pd.DataFrame,
    driver_name: str,
    team_name: str,
) -> pd.Series | None:
    name_key = _normalize_text(driver_name)
    team_key = _normalize_text(team_name)

    candidates = standings_df[standings_df["_name_key"] == name_key]
    if candidates.empty:
        return None

    team_match = candidates[candidates["_team_key"] == team_key]
    if not team_match.empty:
        return team_match.iloc[0]

    return candidates.iloc[0]


def _lookup_grid_row(
    grid_df: pd.DataFrame,
    driver_name: str,
    driver_code: str,
    team_name: str,
) -> pd.Series | None:
    if grid_df.empty:
        return None

    name_key = _normalize_text(driver_name)
    code_key = _normalize_text(driver_code)
    team_key = _normalize_text(team_name)

    team_rows = grid_df[grid_df["_team_key"] == team_key]
    if not team_rows.empty:
        by_code = team_rows[team_rows["_code_key"] == code_key]
        if not by_code.empty:
            return by_code.iloc[0]

        by_name = team_rows[team_rows["_name_key"] == name_key]
        if not by_name.empty:
            return by_name.iloc[0]

    by_code_global = grid_df[grid_df["_code_key"] == code_key]
    if not by_code_global.empty:
        return by_code_global.iloc[0]

    by_name_global = grid_df[grid_df["_name_key"] == name_key]
    if not by_name_global.empty:
        return by_name_global.iloc[0]

    return None


def _assign_starting_positions(seed_df: pd.DataFrame, position_col: str) -> pd.Series:
    sortable = seed_df.copy()
    sortable[position_col] = pd.to_numeric(sortable[position_col], errors="coerce")

    # Modeling assumption:
    # If any seeded positions are missing or duplicated (e.g., lineup updates
    # outpace standings/grid availability), we preserve relative ordering and
    # then assign a clean contiguous 1..N starting_position sequence.
    sortable = sortable.sort_values(
        by=[position_col, "standing_position", "team", "driver"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    sortable["starting_position"] = range(1, len(sortable) + 1)
    return sortable[["driver", "starting_position"]].set_index("driver")["starting_position"]


def build_official_seeded_setup(
    race: str,
    standings_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
    grid_df: pd.DataFrame | None,
    race_laps: int,
    setup_source: str,
    starting_compound: str = DEFAULT_STARTING_COMPOUND,
    base_pace_seconds: float = DEFAULT_BASE_PACE_SECONDS,
    pit_loss_green_seconds: float = DEFAULT_PIT_LOSS_GREEN_SECONDS,
    pit_loss_safety_car_seconds: float = DEFAULT_PIT_LOSS_SAFETY_CAR_SECONDS,
    overtaking_difficulty: float = DEFAULT_OVERTAKING_DIFFICULTY,
    track_position_importance: float = DEFAULT_TRACK_POSITION_IMPORTANCE,
    safety_car_probability: float = DEFAULT_SAFETY_CAR_PROBABILITY,
) -> pd.DataFrame:
    standings = standings_df.copy()
    standings["_name_key"] = standings["driver_name"].map(_normalize_text)
    standings["_team_key"] = standings["team"].map(_normalize_text)

    parsed_grid = (grid_df.copy() if grid_df is not None else pd.DataFrame())
    if not parsed_grid.empty:
        parsed_grid["_name_key"] = parsed_grid["driver_name"].map(_normalize_text)
        parsed_grid["_team_key"] = parsed_grid["team"].map(_normalize_text)
        parsed_grid["_code_key"] = parsed_grid["driver_code"].map(_normalize_text)

    rows = []
    for _, lineup_row in lineup_df.iterrows():
        team = str(lineup_row["team"]).strip()
        driver_name = str(lineup_row["driver_name"]).strip()

        standing_row = _lookup_standings_row(standings, driver_name=driver_name, team_name=team)
        if standing_row is not None:
            driver_code = str(standing_row["driver_code"])
            standing_position = int(standing_row["standing_position"])
        else:
            driver_code = _build_driver_code(driver_name)
            standing_position = None

        grid_position = None
        if setup_source == "official_grid" and not parsed_grid.empty:
            grid_row = _lookup_grid_row(
                parsed_grid,
                driver_name=driver_name,
                driver_code=driver_code,
                team_name=team,
            )
            if grid_row is not None and pd.notna(grid_row.get("grid_position")):
                grid_position = int(grid_row["grid_position"])

        position_seed = grid_position if grid_position is not None else standing_position

        rows.append(
            {
                "race": str(race).strip(),
                "driver": driver_code,
                "team": team,
                "standing_position": standing_position,
                "grid_position": grid_position,
                "position_seed": position_seed,
            }
        )

    setup = pd.DataFrame(rows)
    if setup.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    position_lookup = _assign_starting_positions(setup, position_col="position_seed")
    setup["starting_position"] = setup["driver"].map(position_lookup).astype(int)

    setup["starting_compound"] = str(starting_compound).strip().upper()
    setup["base_pace_seconds"] = float(base_pace_seconds)
    setup["race_laps"] = int(race_laps)
    setup["pit_loss_green_seconds"] = float(pit_loss_green_seconds)
    setup["pit_loss_safety_car_seconds"] = float(pit_loss_safety_car_seconds)
    setup["overtaking_difficulty"] = float(overtaking_difficulty)
    setup["track_position_importance"] = float(track_position_importance)
    setup["safety_car_probability"] = float(safety_car_probability)
    setup["setup_source"] = str(setup_source)

    setup = setup.sort_values(["starting_position", "driver"]).reset_index(drop=True)
    return setup[OUTPUT_COLUMNS]


def generate_official_seeded_race_setup(
    race: str,
    season: int = DEFAULT_SEASON,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    standings_df = fetch_driver_standings(season=season, session=session)
    lineup_df = fetch_official_team_lineup(session=session)

    schedule_df = fetch_race_schedule(season=season, session=session)
    race_entry = _select_race_entry(race_query=race, schedule_df=schedule_df)

    resolved_race_name = str(race).strip()
    race_laps = DEFAULT_RACE_LAPS
    race_result_url = ""

    if race_entry is not None:
        resolved_race_name = str(race_entry["race_name"])
        race_laps = int(race_entry.get("race_laps", DEFAULT_RACE_LAPS))
        race_result_url = str(race_entry.get("race_result_url", ""))

    grid_df = pd.DataFrame()
    setup_source = "standings_seeded"
    grid_warning_emitted = False

    if race_result_url:
        try:
            grid_df = fetch_official_starting_grid(race_result_url=race_result_url, session=session)
        except Exception as exc:
            warnings.warn(
                "Official grid is not available yet for "
                f"{resolved_race_name} ({season}); falling back to standings_seeded. "
                f"Reason: {exc}",
                UserWarning,
            )
            grid_df = pd.DataFrame()
            grid_warning_emitted = True

    if grid_df.empty and not grid_warning_emitted:
        warnings.warn(
            "Official grid is not available yet for "
            f"{resolved_race_name} ({season}); falling back to standings_seeded.",
            UserWarning,
        )
    else:
        setup_source = "official_grid"

    setup = build_official_seeded_setup(
        race=resolved_race_name,
        standings_df=standings_df,
        lineup_df=lineup_df,
        grid_df=grid_df,
        race_laps=race_laps,
        setup_source=setup_source,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup.to_csv(output_file, index=False)

    return setup


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Fetch official Formula1.com lineup + standings and build a seeded "
            "race setup CSV for strategy simulation."
        )
    )
    parser.add_argument(
        "--race",
        required=True,
        help="Requested race name/track text used to seed the setup CSV.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help="F1 season used for official standings/schedule fetch.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path for race setup seeding.",
    )
    args = parser.parse_args()

    setup = generate_official_seeded_race_setup(
        race=args.race,
        season=args.season,
        output_path=args.output,
    )

    print("Official race setup seed generated.")
    print(f"- Rows: {len(setup)}")
    print(f"- Drivers: {setup['driver'].nunique()}")
    print(f"- Teams: {setup['team'].nunique()}")
    print(f"- Setup source: {setup['setup_source'].iloc[0] if not setup.empty else 'N/A'}")
    if not setup.empty and setup["setup_source"].iloc[0] == "standings_seeded":
        print("- Official grid unavailable; used standings-seeded starting positions.")
    print(f"- Output: {Path(args.output)}")
    print(setup.head(12))


if __name__ == "__main__":
    main()
