# PitWall AI

**A Formula 1 race-strategy engine and live pit-wall tracker for the 2026 season.** Tyre-degradation modelling with quantified uncertainty, exact strategy optimisation, Monte Carlo risk, a hybrid physics + ML model of team behaviour, and a live tracker that makes the pit call for any driver during a race.

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
| **Live pit wall** (`/live`) | Connects to F1's live-timing feed (free; F1TV login optional), OpenF1, a replay of any 2026 race, or manual input. For the chosen driver, every few seconds: **BOX / STAY OUT** call with target lap and next tyre, rest-of-race plan, all alternatives with P(best) and tail risk, rejoin position if they box now, undercut threat/opportunity, Safety-Car "cheap stop" detection, live-updated tyre wear with a drift alert, and predicted next stops for every rival. |
| **Strategy predictor** (`/race/[round]`) | For every 2026 round: the **physics-fastest plan** and the **most likely team call** per driver (with stop-count and start-tyre probabilities), pit windows, an 80% conformal interval for the first stop, team-specific tyre curves, the whole grid's strategies, a what-if builder, and — for finished races — what actually happened and the hindsight optimum. |
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
6. **Monte Carlo risk** – Safety Cars / VSCs from per-lap hazards plus posterior parameter draws, with a reactive pit-wall policy (stop under a neutralisation inside the window). Reports P(best), expected regret, CVaR-90, and the probability that 1, 2 or 3 stops is fastest.
7. **Hybrid behaviour model** – a walk-forward multinomial logistic regression predicts stop count and starting tyre from physics features (1- vs 2-stop gap, optimal stop lap, wear, grid slot); the optimiser times the plan inside that class. "Fastest" and "likely" are shown separately.
8. **Personal (per-driver) wear** – every driver has their own wear multiplier vs the field, used in their plan and in rivals' predictions. Variance components were measured on 2026 data (persistent driver effect sd ≈ 0.25–0.34, per-stint noise ≈ 0.44–0.55, race-to-race variation ≈ 0.30), and the per-stint weight learned on rounds 1–8 beat the field average on rounds 9–15 (MAE 0.531 vs 0.557; raw per-driver numbers did worse, 0.564). Pre-race, a driver's history and practice did **not** predict their race wear better than the field (0.146 field vs 0.168 with practice), so practice is recorded per driver but not used to move the estimate; personalisation comes from the driver's own race stints.
9. **Weekend severity** – compounds with no practice long runs (often the Hard) take their wear from the weekend's overall severity × the compound's typical ratio, instead of a low-wear season average.
10. **Live loop** – every lap: field-wide wear updates from green-flag laps (laps 1–3 excluded, stints ≥ 6 laps, conflict-aware prior widening so a wrong pre-race estimate is overruled within ~10 laps — checked on Barcelona and Madrid replays), personal wear multipliers for every car from all of its own stints, pit-loss re-estimation from observed stops, re-optimisation from the car's current tyre state, rejoin/undercut checks, drift monitoring.

## Validation (walk-forward, no leakage)

Every round is predicted using only information available before that race. Scope: dry 2026 races from round 4, classified finishers, **173 driver-races**.

| Metric | Hybrid (physics + behaviour ML) | Physics optimiser only | Naive baseline* |
|---|---|---|---|
| Number of stops correct | 28.9% | **33.5%** | 27.7% |
| Compounds used correct | **43.9%** | 38.2% | 38.2% |
| Starting tyre correct | 59.0% | 38.2% | **64.2%** |
| First-stop lap, mean abs. error | 11.0 laps | **9.2 laps** | 10.3 laps |
| 80% conformal interval coverage (first stop) | — | 85.0% | target 80% |

\*Baseline: the most common strategy of earlier dry 2026 races with the median first-stop fraction.

**Reading this honestly:** predicting what teams will do from public data is hard — incidents, team orders, covering and track position drive many calls. The physics optimiser now times the first stop better than the baseline and is best on stop count; the hybrid model is best on compound choice; the naive baseline is still best on the starting tyre. The engine's main value is **decision support** — what is fastest from the current state, how risky, and how that changes lap by lap — which the live pit wall and what-if tools expose. The original project's XGBoost degradation model (random-split R² 0.66, future-race R² < 0) is kept in `src/models/` as the reason the new engine is evaluated walk-forward only.

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

# 1. Download/cache the 2026 sessions (FastF1 rate-limits itself to 500 calls/h; re-run if it stops)
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

- Track position is a per-stop cost plus live rejoin/undercut checks, not a full multi-car overtaking simulation.
- Wet-tyre strategy is not modelled; wet races are excluded from dry metrics.
- Tyre wear is linear + quadratic with a fuel-load interaction; cliffs are approximated through stint-life limits and the soft-tyre penalty.
- Sepang (Round 16) has no recent F1 data: pit loss starts at the 2026 season median and is re-estimated live from observed stops.
- The unauthenticated live-timing feed carries timing, tyres and race control but not car telemetry, and F1 can change its access rules.

---
Independent project, not affiliated with Formula 1, the FIA or any team.
