"""Warm the FastF1 cache for every completed 2026 session.

Run once (or after each race weekend):
    python -m src.pitwall.prefetch
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from src.pitwall.season import enable_cache, get_schedule


def main(year: int = 2026) -> None:
    enable_cache()
    import fastf1

    schedule = get_schedule(year)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name = event[f"Session{i}"]
            start = event[f"Session{i}DateUtc"]
            if not name or pd.isna(start) or start > now - pd.Timedelta(hours=2):
                continue
            t0 = time.time()
            try:
                session = fastf1.get_session(year, int(event["RoundNumber"]), name)
                session.load(telemetry=False, weather=True, messages=name == "Race")
                print(f"R{event['RoundNumber']:02d} {name:<18} laps={len(session.laps):4d} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:  # keep going; one bad session should not stop the warm-up
                print(f"R{event['RoundNumber']:02d} {name:<18} FAILED {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
