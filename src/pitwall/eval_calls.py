"""Score the live pit-wall calls on finished races, lap by lap.

For every driver and every lap of a replay, run the live engine and record the
call. A "BOX, BOX" is *false* if the driver did not pit on that lap or the
next one; a green-flag stop is *anticipated* if the engine said BOX_NOW,
BOX_SOON or WINDOW_OPEN in the three laps before it (needs --every 1). Wet
races are where the dry-tyre maths used to break.

    python -m src.pitwall.eval_calls 4 12        # Miami (wet), Dutch (wet)
    python -m src.pitwall.eval_calls 4 --every 2 # faster: every 2nd lap
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from src.pitwall.api import weekend_for
from src.pitwall.live_engine import analyse
from src.pitwall.live_sources import ReplaySource


def score_race(rnd: int, every: int = 1, sims: int = 60) -> dict:
    src = ReplaySource(rnd)
    weekend = weekend_for(rnd)
    pits = {d: {int(r.lap) for r in g.itertuples() if r.pit_in} for d, g in src.laps.groupby("driver")}
    calls, false_box, examples = Counter(), 0, []
    ready: dict[str, set[int]] = {}
    gate: list[dict] = []  # every lap the optimiser wanted a stop *this* lap
    for lap in range(2, src.total_laps, every):
        state = src.state(lap)
        for car in state["cars"]:
            if car["retired"] or car["laps_completed"] >= src.total_laps:
                continue
            a = analyse(state, weekend, car["driver"], sims=sims)
            act = a.get("call", {}).get("action", "ERROR")
            calls[act] += 1
            cur = car["laps_completed"] + 1
            c = a.get("call", {})
            if c.get("wait_cost") is not None or (act == "BOX_NOW" and c.get("wait_cost") is None):
                gate.append({"wait": c.get("wait_cost"), "p": c.get("box_win_probability"), "ts": state["track_status"],
                             "true": bool({cur, cur + 1} & pits.get(car["driver"], set())), "act": act})
            if act in ("BOX_NOW", "BOX_SOON", "WINDOW_OPEN"):
                ready.setdefault(car["driver"], set()).add(cur)
            if act == "BOX_NOW" and not ({cur, cur + 1} & pits.get(car["driver"], set())):
                false_box += 1
                if len(examples) < 8:
                    examples.append({"lap": cur, "driver": car["driver"], "compound": car["compound"], "headline": a["call"]["headline"]})
    green = [(d, p) for d, ps in pits.items() for p in ps if p not in src.neutral_laps and 2 < p < src.total_laps]
    caught = sum(1 for d, p in green if ready.get(d, set()) & set(range(p - 3, p + 1)))
    return {"round": rnd, "event": state["event"], "calls": dict(calls), "box_now": calls["BOX_NOW"],
            "false_box_now": false_box, "false_box_rate": round(false_box / max(1, calls["BOX_NOW"]), 3),
            "real_stops": sum(len(v) for v in pits.values()), "green_stops": len(green),
            "anticipated": caught if every == 1 else None, "examples": examples, "gate": gate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", type=int, nargs="+")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--gate-out", help="write every would-box-now lap (wait cost, P(best), outcome) to <prefix>_rNN.json")
    args = ap.parse_args()
    for rnd in args.rounds:
        r = score_race(rnd, args.every, args.sims)
        if args.gate_out:
            with open(f"{args.gate_out}_r{rnd:02d}.json", "w") as f:
                json.dump(r["gate"], f)
        r.pop("gate")
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
