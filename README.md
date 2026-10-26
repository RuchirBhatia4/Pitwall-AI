# PitWall AI

**A Formula 1 race-strategy engine and live pit-wall tracker for the 2026 season.** Tyre-degradation modelling with quantified uncertainty, exact strategy optimisation, Monte Carlo risk, a full-field race simulation with learned overtaking, a hybrid physics + ML model of team behaviour, and a live tracker that makes the pit call for any driver during a race.

### ▶ Live site: **[pitwall-ai-f1.vercel.app](https://pitwall-ai-f1.vercel.app)**

| | |
|---|---|
| Season & strategy predictor | [pitwall-ai-f1.vercel.app](https://pitwall-ai-f1.vercel.app) |
| Live pit wall | [pitwall-ai-f1.vercel.app/live](https://pitwall-ai-f1.vercel.app/live) |
| Method & validation | [pitwall-ai-f1.vercel.app/methodology](https://pitwall-ai-f1.vercel.app/methodology) |
| API (FastAPI) | [pitwall-ai-2i89.onrender.com/health](https://pitwall-ai-2i89.onrender.com/health) |

Next.js frontend · FastAPI backend · FastF1 / F1 live timing / OpenF1 data · validated walk-forward on every 2026 race.

---

## What it does

| | |
|---|---|
| **Live pit wall** (`/live`) | Connects to F1's live-timing feed (free; F1TV login optional), OpenF1, a replay of any 2026 race, or manual input. For the chosen driver, every few seconds: **BOX / STAY OUT** call with target lap and next tyre, rest-of-race plan, all alternatives with P(best) and tail risk, rejoin position if they box now, undercut threat/opportunity, Safety-Car "cheap stop" detection, live-updated tyre wear with a drift alert, predicted next stops for every rival, and **where each option finishes** in a full-field simulation of the rest of the race from the current gaps and tyres. |
| **Strategy predictor** (`/race/[round]`) | For every 2026 round: the **race-simulation pick** (best expected finishing position, with points odds), the **physics-fastest plan** and the **most likely team call** per driver, a **predicted finishing order** with win / podium / points probabilities (with stop-count and start-tyre probabilities), pit windows, an 80% conformal interval for the first stop, team-specific tyre curves, the whole grid's strategies, a what-if builder, and — for finished races — what actually happened and the hindsight optimum. |
| **Ask the pit wall** (chat) | Questions like *"What happens to Leclerc if there's a Safety Car on lap 20?"*. With an `ANTHROPIC_API_KEY`, an LLM answers through tool calls into the engine (it can only quote numbers the engine returned). Without a key it still answers in deterministic engine mode. |
| **Method & validation** (`/methodology`) | The model, the pipeline, every ML technique and why it is there, the honest scorecard, and known limitations. |

## The model

Lap time for driver *d* on compound *c*, tyre age *a*, race lap *n*:

```
t = base_d + offset_c + wear_{c,team} · a · (1 + φ · fuel_remaining(n)) + q · a² − fuel_burn · n + ε
```

1. **Season prior** – the model is fitted (robust least squares) to every *earlier* 2026 race. Fuel burn is identifiable because tyre age resets at each stop while lap number does not (fitted 0.04–0.07 s/lap). Between-race spread becomes the prior uncertainty.
2. **Weekend evidence** – long runs detected in FP/sprint sessions (Theil–Sen slopes), plus short-run compound pace gaps. Practice ≠ race, so the **practice→race transfer ratio is learned** from earlier weekends (≈0.4–0.56×) and its residual error becomes the observation noise.
3. **Hierarchical Bayesian blend** – conjugate Gaussian updates per compound; team effects shrunk toward the field by evidence; **isotonic projection** enforces softer = faster-when-new and faster-wearing.
4. **Revealed-preference calibration (inverse optimisation)** – soft-tyre penalty, fuel-wear sensitivity φ and a per-stop track-position cost are chosen so the optimiser best reproduces what teams actually did in earlier rounds.
5. **Exact optimisation** – backward dynamic programming over every 0–3-stop compound sequence and every pit lap, honouring the two-compound rule and stint-life limits (~10 ms). Pit windows = laps within 2 s of the optimum.
6. **Monte Carlo risk** – Safety Cars / VSCs from per-lap hazards (the 2026 season rate blended with the circuit's own 2024–25 record, worth ~120 laps of season data) plus posterior parameter draws, with a reactive pit-wall policy (stop under a neutralisation inside the window). An upcoming race's pit-lane loss blends the circuit's 2024–25 median with the 2026 season median (2:1), which cut the pit-loss error on 2026 races from 2.7 to 2.2 s. Reports P(best), expected regret, CVaR-90, and the probability that 1, 2 or 3 stops is fastest.
7. **Hybrid behaviour model** – a walk-forward multinomial logistic regression predicts stop count and starting tyre from physics features (1- vs 2-stop gap, optimal stop lap, wear, grid slot) and the weekend's share of practice long-run laps on each compound (teams run their race simulations on the tyres they plan to race; +13 points on stop count walk-forward); the optimiser times the plan inside that class. Practice is kept out of the first-stop timing model, where it made errors worse. "Fastest" and "likely" are shown separately.
8. **Personal (per-driver) wear** – every driver has their own wear multiplier vs the field, used in their plan and in rivals' predictions. Variance components were measured on 2026 data (persistent driver effect sd ≈ 0.25–0.34, per-stint noise ≈ 0.44–0.55, race-to-race variation ≈ 0.30), and the per-stint weight learned on rounds 1–8 beat the field average on rounds 9–15 (MAE 0.531 vs 0.557; raw per-driver numbers did worse, 0.564). Pre-race, a driver's history and practice did **not** predict their race wear better than the field (0.146 field vs 0.168 with practice), so practice is recorded per driver but not used to move the estimate; personalisation comes from the driver's own race stints.
9. **Weekend severity** – compounds with no practice long runs (often the Hard) take their wear from the weekend's overall severity × the compound's typical ratio, instead of a low-wear season average.
10. **Full-field race simulation** (`race_sim.py`) – the core of team strategy tools: every car simulated lap by lap with its race pace, tyres, pit stops (stationary-time noise, slow stops), Safety Cars / VSCs (stops brought forward, field bunched at the restart) and traffic. A car within 1 s of the one ahead passes with probability sigmoid(a_track + b·pace advantage), **learned from every close-following lap of earlier 2026 races** (about 2,400 before Austin), with each circuit's difficulty taken from its 2024–25 races and shrunk toward the 2026 average (equal-pace pass chance per lap: Monaco ~0%, Singapore ~7%, Monza ~21%). Cars start from the official grid (OpenF1, penalties included) once it is published, else qualifying order; race pace comes from a walk-forward ridge model on qualifying position (so a fast car moved back by a penalty is still fast) and recency-weighted form. Like a pit wall, it only ranks strategies a team would realistically consider: stop counts and start tyres with at least 10% probability under the behaviour model. Each driver's candidate plans (fastest few, best per stop count and start tyre, first stop moved ±3/±6 laps) are ranked by **expected finishing position** against rivals' predicted plans, with common random numbers so plan differences are not noise. Live, it restarts from the current gaps, tyre ages and each rival's predicted remaining plan (~0.1 s per lap). It uses the soft-corrected tyre model but not the per-stop track-position cost, because track position is simulated.
11. **Live loop** – every lap: field-wide wear updates from green-flag laps (laps 1–3 excluded, stints ≥ 6 laps, conflict-aware prior widening so a wrong pre-race estimate is overruled within ~10 laps — checked on Barcelona and Madrid replays), personal wear multipliers for every car from all of its own stints, pit-loss re-estimation from observed stops, re-optimisation from the car's current tyre state, rejoin/undercut checks, drift monitoring.

## Validation (walk-forward, no leakage)

Every round is predicted using only information available before that race. Scope: dry 2026 races from round 4, classified finishers, **210 driver-races** (through round 18, Austin).

| Metric | Race simulation | Hybrid (physics + behaviour ML) | Physics optimiser only | Naive baseline* |
|---|---|---|---|---|
| Number of stops correct | **51.4%** | 43.8% | 38.6% | 34.3% |
| Compounds used correct | 40.0% | 38.6% | **40.5%** | 33.3% |
| Starting tyre correct | 55.7% | 56.2% | 31.4% | **58.6%** |
| First-stop lap, mean abs. error | 8.7 laps | 10.8 laps | **8.5 laps** | 9.9 laps |
| Finishing order, rank correlation with the result | **0.85** | — | — | grid order 0.81 |
| 80% conformal interval coverage (first stop) | — | — | 86.7% | target 80% |

\*Baseline: the most common strategy of earlier dry 2026 races with the median first-stop fraction.

On the same 210 driver-races, the race simulation gets the stop count right where the physics optimiser does not 47 times, and the reverse 20 times (McNemar p = 0.001); the starting tyre 52 times vs once (p < 0.001); compounds and first-stop timing are level (2 vs 3, p = 1; 8.7 vs 8.5 laps, Wilcoxon p = 0.58). Against the naive baseline it is ahead on stops (54 vs 18, p < 0.001), compounds (22 vs 8, p = 0.02) and first-stop timing (8.7 vs 9.9 laps, p = 0.009), and slightly behind on the starting tyre (2 vs 8, p = 0.11). After Austin, three changes were made together and checked walk-forward: the plausible-strategy shortlist, practice long runs as a behaviour feature, and the official starting grid. They moved the race simulation from 45.0% to 51.4% on stops, 34.4% to 55.7% on the starting tyre, 39.2% to 40.0% on compounds and 9.6 to 8.7 laps on the first stop; the hybrid gained on stops (31.0% → 43.8%) and the starting tyre (52.4% → 56.2%) but its first stop is 0.3 laps further off. Before the race simulation was adopted, seven variants were compared walk-forward: pure-physics vs soft-corrected tyre model, with and without first-stop shifts and the fuel-wear term, circuit overtaking history, and a second round of best responses. The chosen one was the a priori design; the others differed by a few points either way. Adding the circuits' 2024–25 Safety Car record to the hazards moved the physics optimiser from 34.6% to 42.4% on compounds (15 driver-races gained, none lost; p < 0.001) and from 8.9 to 8.8 laps on the first stop (p = 0.001), and left the other metrics unchanged. Weighting recent races more when learning team behaviour was also tried and did not help, so it is not used.

**Data source:** rounds 1–15 come from FastF1. In October 2026 F1's timing archive started requiring a login, so later rounds come from OpenF1 (`src/pitwall/openf1_data.py`), mapped to the same tables. On rounds 10 and 15, where both exist, the two give identical lap times, safety-car/VSC laps, pit loss and results; OpenF1 occasionally records a used tyre set as new, which moves some fitted wear rates slightly.

**Reading this honestly:** predicting what teams will do from public data is hard — incidents, team orders, covering and track position drive many calls. Simulating track position makes the stop-count and starting-tyre calls clearly better; the physics optimiser still times the first stop marginally best, and the naive baseline is still marginally best on the starting tyre. The race simulation still fits a Hard into more plans than teams do (78% of plans vs 69% of real races; the physics optimiser 86%), because 2026 softs and mediums are lasting longer than the season's wear model expects. Austin showed this most clearly: the predicted finishing order was close (rank correlation 0.97), but the wear model's soft-tyre correction (soft ×1.6 the medium's wear, learned from team choices) is not physical — Austin's race laps show soft, medium and hard wearing at about the same rate (~0.11 s/lap) — so plans leaned to Hard-heavy one-stops while Leclerc's two-stop was the standout strategy of the race. The engine's main value is **decision support** — what is fastest from the current state, how risky, and how that changes lap by lap — which the live pit wall and what-if tools expose. The original project's XGBoost degradation model (random-split R² 0.66, future-race R² < 0) is kept in `src/models/` as the reason the new engine is evaluated walk-forward only.

**Live calls, scored lap by lap** (`python -m src.pitwall.eval_calls <rounds>`): every driver, every lap of a replay, through the live engine. A "BOX, BOX" is false if the car did not pit that lap or the next.

| | False "BOX, BOX" (old → new) | Green-flag stops flagged in advance (old → new) |
|---|---|---|
| Tuning races R4 (wet), R7, R12 (wet), R15 | 1,100 → 60 | 98 / 105 → 94 / 105 |
| Held-out races R9, R10, R13, R14 (tested once) | 425 → 24 | 30 / 56 → 30 / 56 |

Austin (R18, scored after the race): 5 "BOX, BOX" calls, 3 of them false (Stroll staying out on softs, laps 37–39), and 29 of 31 green-flag stops flagged in advance. Its race-control messages about a slippery track sector (on a dry track) had been read as wet-track warnings and held 107 calls inside pit windows; only rain / wet / intermediate messages count now (`tests/test_wet_calls.py`).

The dry wear model times stops to a window, not a lap (no green-flag threshold reached 2-lap precision above ~0.35), so under green the call is **"pit window open — box by lap N"** and "BOX, BOX" is kept for a closing window (waiting costs ≥ 3 s), a confident SC/VSC stop, or a wet/slick crossover. In wet or changing conditions (rain that moves lap times, cars on intermediates, race-control wet-track messages, a field drying > 1.2 %/lap) the affected laps are kept out of the wear model, dry stops are held, cars that ran wet tyres are freed from the two-compound rule, and slick ↔ intermediate switches are called from who is faster on each tyre right now. No 2026 race has used intermediates yet, so that path is covered by tests (`tests/test_wet_calls.py`), not by a replay.

Backtests are logged to MLflow (`mlflow ui`, experiment `pitwall_strategy_backtest`).

## Architecture

```
FastF1 (practice / sprint / quali / race)          F1 live timing (SignalR)  OpenF1  replay  manual
        │                                                     │
        ▼                                                     ▼
src/pitwall/race_data.py ─► tyre_model.py ─► weekend.py    live_sources.py  (one RaceState format)
                                 │               │                │
                                 ▼               ▼                ▼
                          optimizer.py (DP + Monte Carlo) ◄── live_engine.py (per-lap call)
                                 │                                 │
   track_history.py ─► race_sim.py (full-field simulation) ◄──────┘
                                 │
            build_season.py (walk-forward backtest) ─► behaviour_model.py (hybrid ML)
                                 │
                       data/season/2026/*.json
                                 │
                 src/pitwall/api.py + chat.py  (FastAPI, mounted in src/api/main.py)
                                 │   /api/*
                                 ▼
                 frontend/ (Next.js 16, App Router, Tailwind, Recharts)
```

## Run it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Download/cache the 2026 sessions (FastF1 cache first, OpenF1 for anything F1's archive no longer serves)
python -m src.pitwall.prefetch
# 2. Build every round + walk-forward backtest + hybrid behaviour model
python -m src.pitwall.build_season
# 3. API
uvicorn src.api.main:app --host 127.0.0.1 --port 8010
```

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000  (proxies /api/* to PITWALL_API_URL, default http://127.0.0.1:8010)
```

Next.js 16 needs Node ≥ 20.9.

### Race day

1. Open **[Live pit wall](https://pitwall-ai-f1.vercel.app/live)** (or `localhost:3000/live`) → **F1 live timing (free)** → **Connect**. It auto-reconnects and records the raw feed to `data/season/live_rXX_*.txt`.
   Each visitor has their own session (source, race, replay clock); real live feeds are shared read-only per race, so nobody can switch what someone else is watching.
2. Optional: `python -m src.pitwall.f1tv_login` once to use your F1TV subscription for the authenticated feed; or set `OPENF1_USERNAME`/`OPENF1_PASSWORD` (OpenF1 sponsor tier) and pick **OpenF1**.
3. Pick a driver. If a feed fails, **Manual input** always works (lap, tyre, age, gaps from the TV graphics).
4. After the race: `python -m src.pitwall.build_season` adds it to the season and the backtest.

### Tests

```bash
python -m pytest -q        # engine (DP vs brute force, SC logic, Bayesian updates, conformal, live-feed reducer) + API + original pipeline
```

## Configuration

See `.env.example`. Key variables: `PITWALL_API_URL` (frontend → API), `ANTHROPIC_API_KEY` (LLM chat, optional), `PITWALL_CHAT_MODEL`, `OPENF1_USERNAME`/`OPENF1_PASSWORD`, `CORS_ALLOW_ORIGINS`.

## Repository map

- `src/pitwall/` – the strategy engine, live tracker, API and chat (this README).
- `src/models/`, `src/pipeline/`, `src/simulation/`, `src/live/` – the original ML pipeline (XGBoost/LightGBM degradation models, Japan holdout, practice calibration, first simulator), kept for reference and its tests.
- `frontend/` – Next.js app. `data/season/2026/` – built artefacts the API serves (small JSON, committed).
- `DEPLOYMENT.md` – Render (API) + Vercel (frontend).

## Known limitations

- The race simulation has no retirements, no lapped-car traffic and no reactive undercut covering between rivals (a second round of best responses was tried and did not help); overtaking depends only on pace difference and the circuit.
- Wet-tyre strategy is not modelled; wet races are excluded from dry metrics.
- Tyre wear is linear + quadratic with a fuel-load interaction; cliffs are approximated through stint-life limits and the soft-tyre penalty.
- Sepang (Round 16) has no recent F1 data: pit loss starts at the 2026 season median and is re-estimated live from observed stops.
- The unauthenticated live-timing feed carries timing, tyres and race control but not car telemetry, and F1 can change its access rules.

---
Independent project, not affiliated with Formula 1, the FIA or any team.
