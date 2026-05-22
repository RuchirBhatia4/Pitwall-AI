# PitWall AI
**An end-to-end ML systems project for Formula 1 tire degradation forecasting and race strategy recommendation under real-world uncertainty.**

## 1. Project Title
PitWall AI

## 2. One-Line Summary
PitWall AI combines historical race data, practice-session calibration, and strategy simulation to produce data-driven, transparently-evaluated pit strategy recommendations.

## 3. Problem Statement
Formula 1 race strategy decisions depend heavily on tire degradation, but degradation behavior changes by driver, team, compound, stint phase, and weekend context. PitWall AI is built to answer a practical ML engineering question:

> Can we build a robust system that predicts degradation and strategy outcomes honestly enough to be useful for future-race decisions, not just retrospective fit?

## 4. Why F1 Tire Degradation Prediction Is Hard
- Tire behavior is nonlinear and context-dependent.
- Weekend-specific setup and track evolution shift degradation away from historical averages.
- Label quality is noisy due to traffic, incidents, race control events, and pit cycle effects.
- Distribution shift is large in regulation transitions (especially 2026).
- Strategy quality depends on compounding model uncertainty across multiple stints and pit windows.

## 5. System Architecture
```text
FastF1 race data
  -> cleaning + feature engineering
  -> degradation model training
  -> future-race holdout evaluation
  -> practice race-pace calibration
  -> strategy generation + simulation + optimization
  -> FastAPI backend
  -> React frontend for analysis/recommendations
```

Main code areas:
- `src/data/` (ingestion, cleaning, historical dataset build)
- `src/models/` (training, holdout, calibration)
- `src/evaluation/` (holdout error analysis)
- `src/simulation/` (strategy generator/simulator/optimizer)
- `src/pipeline/` (end-to-end prediction/recommendation scripts)
- `src/api/` (FastAPI service layer and HTTP contracts)
- `frontend/` (React/Vite dashboard)

## 6. Current Pipeline
1. Build historical lap dataset from FastF1 races.
2. Clean laps and engineer degradation-relevant features.
3. Train degradation models (Ridge, XGBoost, LightGBM).
4. Evaluate with both random split and future-race holdout.
5. Ingest FP1/FP2/FP3 long-run practice data and calibrate degradation.
6. Simulate legal strategy candidates and rank recommendations.
7. Surface outputs through a backend API and React frontend for filtered inspection.

## 7. Data Sources
- FastF1 race lap data (historical races across 2024, 2025, and completed 2026 rounds).
- User-uploaded practice race-pace CSVs (FP1/FP2/FP3 long runs).
- Derived outputs in `data/processed/` and `data/predictions/`.

## 8. Tire Degradation Modeling
Current target:
- `fuel_adjusted_degradation_delta`

Why this target:
- Raw lap time is not comparable across circuits.
- Simple stint delta is distorted by fuel burn.
- Fuel-adjusted degradation partially corrects late-stint fuel effects.

Modeling details:
- Engineered race-phase and nonlinear tire features.
- Sample weighting by regulation relevance.
- Prior features experiment (driver/team priors) retained as a non-default branch.

## 9. 2026 Regulation-Aware Weighting
PitWall AI intentionally emphasizes current-regulation data:
- 2026 rows: weight `5.0`
- 2025 rows: weight `1.5`
- 2024 or older rows: weight `1.0`

Rationale:
- 2026 is primary signal under the new regulation era.
- Older seasons are useful priors, not equal truth.

## 10. Future-Race Holdout Validation
The project uses a dedicated **future-race holdout** setup (2026 Japan) to measure realistic generalization.

This is treated as the primary performance signal for credibility, while random split is retained for model development diagnostics.

## 11. Practice Race-Pace Calibration
Practice calibration pipeline includes:
- Practice CSV loading and schema validation.
- Clean long-run filtering.
- Session-level degradation slope estimation.
- Confidence scoring from clean-lap count + fit quality.
- Blending practice signal with historical model predictions.

This is expected to be the **major real-world accuracy improvement path** because it injects current-weekend context missing from historical-only modeling.

## 12. Strategy Simulator
Strategy Simulator MVP supports:
- Legal candidate generation:
  - `MEDIUM-HARD`
  - `HARD-MEDIUM`
  - `SOFT-HARD`
  - `MEDIUM-HARD-SOFT`
  - `SOFT-MEDIUM-SOFT`
- Race-time simulation using:
  - `base_pace_seconds`
  - compound degradation estimate
  - `pit_loss_seconds`
  - `race_laps`
  - `pit_laps`
- Ranking outputs:
  - `expected_total_time`
  - `time_delta_to_best`
  - `risk_score`
  - `is_recommended`

## 13. Product App
The non-Streamlit application provides:
- Calibrated degradation table
- Strategy recommendation table
- Recommended-only strategy section
- Filters for driver/team/compound/strategy
- Summary metrics (drivers, teams, row counts, recommended count)
- Pipeline triggers for calibration and strategy generation

Backend entrypoint:
- `src/api/main.py`

Frontend entrypoint:
- `frontend/src/main.jsx`

## 14. Current Validation Results
### Random split (engineered features)
- **XGBoost MAE ~1.02 sec**
- **R² ~0.66**

### 2026 Japan future-race holdout (engineered features, without priors)
- **XGBoost MAE ~1.6566 sec**
- **R² ~-0.0418**

### Driver/team prior experiment
- **XGBoost MAE ~1.6956 sec**
- **R² ~-0.0989**

Interpretation:
- Priors slightly worsened future-race generalization in this setup.
- Priors are kept as an experiment, not the current best production path.
- Random split is materially easier than future-race holdout and should not be interpreted as deployment-level performance.

## 15. Known Limitations
- Historical-only modeling remains weak on unseen future races.
- Holdout error concentration persists in specific driver/team/compound regimes.
- Practice ingestion and signal extraction are MVP-level and can be deepened.
- Strategy simulator is deterministic/heuristic today (not yet uncertainty-aware Monte Carlo).
- This is **not** a perfect race predictor; it is an evolving ML systems platform with honest evaluation.

## 16. How To Run Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.data.build_historical_dataset
python -m src.models.train_degradation
python -m src.models.train_degradation_holdout
python -m src.evaluation.analyze_holdout_errors
python -m src.pipeline.generate_calibrated_predictions
python -m src.pipeline.generate_strategy_recommendations

uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

cd frontend
npm install
npm run dev
```

Open the frontend at `http://127.0.0.1:5173`.

## 17. Test Commands
```bash
venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## 18. Deployment
This repository is structured as a split frontend/backend app:
- FastAPI serves pipeline actions and prediction outputs.
- React/Vite serves the dashboard UI.
- Supabase/Postgres can be used as the live storage backend for deployment.

The app expects generated CSV outputs under `data/predictions/` and sample setup/control inputs under `data/raw/race_setup/` and `data/raw/race_control/`. FastF1 cache files, local virtual environments, and MLflow tracking artifacts are intentionally ignored because they are large local runtime artifacts.

See `DEPLOYMENT.md` for Supabase, Render, Vercel, and custom-domain setup.

## 19. Roadmap
- Improve practice calibration with richer session/run segmentation and weather/context controls.
- Add uncertainty-aware degradation estimation (intervals, not only point predictions).
- Add probabilistic strategy simulation and scenario sweeps.
- Add tighter post-race evaluation loops for calibration and strategy outcomes.
- Introduce stronger driver/team base pace modeling from same-weekend evidence.
- Extend dashboard UX for comparison views, error diagnostics, and decision auditability.

---
PitWall AI is positioned as an **ML systems engineering project with honest validation**, not as a claim of perfect race prediction accuracy.
