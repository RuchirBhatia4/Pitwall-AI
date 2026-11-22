"""Keep the published season current without anyone running the build.

    python -m src.pitwall.auto_update due        # what is waiting (stdlib only: no installs needed)
    python -m src.pitwall.auto_update build      # rebuild it, check it, record it

Run on a schedule by .github/workflows/update-season.yml, which commits and
pushes what `build` leaves behind; Render and Vercel redeploy from main.

A step is due when a session of a race weekend ended long enough ago for
OpenF1 to publish it and the build has not taken it in yet
(data/season/2026/updates.json). Practice and qualifying rebuild the weekend's
pre-race page, the race adds its results, and one more rebuild a few hours
before the start picks up grid penalties. Nothing is published while a session is
on (see on_air). `build` rebuilds only that round,
reading earlier rounds from their saved files, then checks the result before
anything is published:
  * the session's data was there (OpenF1 can publish a session late; then the
    step waits for the next run);
  * no file other than this round's own changed (walk-forward: a new round
    never changes an earlier one);
  * the round has what the session should add (a grid after qualifying,
    results after the race), and the API serves it.
A failed check exits non-zero with the files restored, so nothing is pushed
and GitHub reports the failed run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "season" / "2026"
STATE = OUT / "updates.json"
# After a session starts: how long until OpenF1 has all of it (sessions run
# about an hour, the race about two, then ~30 min before OpenF1 publishes).
READY_AFTER = {"Race": timedelta(hours=3, minutes=15)}
READY_DEFAULT = timedelta(hours=1, minutes=45)
# Grid penalties land after qualifying: one more pre-race build this long before
# the start, but never later (a deploy restarts the live feed).
GRID_REFRESH = timedelta(hours=3)
GRID_STEP = "Grid"
# Only weekends this recent are looked at (a late OpenF1 upload still gets in).
LOOKBACK = timedelta(days=5)
# Nothing is published while a session is on: a deploy restarts the server,
# which drops the live feed and every visitor's live page. From 30 min before
# the start; a race can run 3 h with a red flag.
ON_AIR_BEFORE = timedelta(minutes=30)
ON_AIR = {"Race": timedelta(hours=3), "Sprint": timedelta(hours=1, minutes=30)}
ON_AIR_DEFAULT = timedelta(hours=1, minutes=30)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _sessions(entry: dict) -> dict[str, datetime]:
    return {k: datetime.fromisoformat(v) for k, v in (entry.get("sessions") or {}).items() if v}


def on_air(now: datetime, season: dict) -> str | None:
    """The session being run now (or about to start), if any."""
    for r in season["rounds"]:
        for name, t in _sessions(r).items():
            if t - ON_AIR_BEFORE <= now <= t + ON_AIR.get(name, ON_AIR_DEFAULT):
                return f"R{r['round']} {name}"
    return None


def due(now: datetime | None = None) -> list[tuple[int, str]]:
    """(round, step) waiting to be built, oldest first. Steps are session names,
    plus GRID_STEP for the pre-race grid refresh. Nothing while a session is on."""
    now = now or _now()
    season = json.loads((OUT / "season.json").read_text())
    if live := on_air(now, season):
        print(f"{live} is on: holding until it ends", file=sys.stderr, flush=True)
        return []
    done = _state()
    out = []
    for r in season["rounds"]:
        sessions = _sessions(r)
        if not sessions or not now - LOOKBACK <= max(sessions.values()) + READY_AFTER["Race"] <= now + LOOKBACK:
            continue
        have = set(done.get(str(r["round"]), []))
        steps = [(t + READY_AFTER.get(name, READY_DEFAULT), name) for name, t in sessions.items()]
        if "Race" in sessions and "Qualifying" in sessions:
            steps.append((sessions["Race"] - GRID_REFRESH, GRID_STEP))
        for ready, name in sorted(steps):
            if name in have or ready > now:
                continue
            if name == GRID_STEP and now >= sessions["Race"] - timedelta(minutes=30):
                continue  # too close to the start to redeploy
            out.append((r["round"], name))
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _changed() -> set[str]:
    out = _git("status", "--porcelain", "--untracked-files=all", "--", str(OUT.relative_to(ROOT)))
    return {line[3:].strip() for line in out.splitlines() if line.strip()}


def _restore() -> None:
    rel = str(OUT.relative_to(ROOT))
    _git("checkout", "--", rel)
    _git("clean", "-fdq", "--", rel)


def _session_published(rnd: int, name: str) -> bool:
    from src.pitwall.race_data import load_session

    if name == GRID_STEP:
        return True
    try:
        return len(load_session(2026, rnd, name).laps) > 0
    except Exception as exc:
        print(f"R{rnd} {name}: not published yet ({type(exc).__name__}: {exc})", flush=True)
        return False


def _check_round(rnd: int, steps: list[str]) -> list[str]:
    """What is wrong with the rebuilt round, if anything."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    problems = []
    data = json.loads((OUT / f"r{rnd:02d}.json").read_text())
    if "Race" in steps:
        if not data.get("overview"):
            problems.append("no race results after the race")
        elif not (data.get("metrics") or {}).get("wet") and not data.get("metrics"):
            problems.append("no scores for a dry race")
    elif any(s in steps for s in ("Qualifying", GRID_STEP)) and len(data.get("drivers") or []) < 18:
        problems.append(f"only {len(data.get('drivers') or [])} drivers after qualifying")
    client = TestClient(app)
    for path in ("/api/season", f"/api/races/{rnd}"):
        res = client.get(path)
        if res.status_code != 200:
            problems.append(f"{path} answered {res.status_code}")
    return problems


def build(dry_run: bool = False) -> int:
    from src.pitwall import build_season

    waiting = due()
    if not waiting:
        print("nothing to build", flush=True)
        return 0
    rnd = waiting[0][0]
    steps = [s for r, s in waiting if r == rnd]
    ready = [s for s in steps if _session_published(rnd, s)]
    if not ready:
        print(f"R{rnd}: {', '.join(steps)} not on OpenF1 yet; next run", flush=True)
        return 0
    if _changed():
        print(f"data/season has uncommitted changes, not building: {sorted(_changed())}", flush=True)
        return 1
    print(f"R{rnd}: building with {', '.join(ready)}", flush=True)
    build_season.main(only=rnd)
    allowed = {f"data/season/2026/{p}" for p in (f"r{rnd:02d}.json", "season.json", "backtest.json", f"fits/r{rnd:02d}.json",
                                                 f"replay/r{rnd:02d}.pkl.gz", "updates.json")}
    unexpected = sorted(_changed() - allowed)
    problems = [f"files of other rounds changed: {unexpected}"] if unexpected else []
    problems += _check_round(rnd, ready)
    if problems or dry_run:
        for p in problems:
            print("CHECK FAILED:", p, flush=True)
        _restore()
        return 1 if problems else 0
    state = _state()
    state[str(rnd)] = sorted(set(state.get(str(rnd), [])) | set(ready))
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    event = json.loads((OUT / f"r{rnd:02d}.json").read_text()).get("event", f"Round {rnd}")
    (ROOT / ".update-message").write_text(f"Data: R{rnd} {event} after {', '.join(ready)}\n")
    print(f"R{rnd}: built and checked", flush=True)
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "due"
    if cmd == "due":
        waiting = due()
        print(" ".join(f"R{r}:{s.replace(' ', '_')}" for r, s in waiting) or "none")
        return 0
    if cmd == "build":
        return build(dry_run="--dry-run" in argv)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
