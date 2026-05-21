# PitWall AI — Codex Project Context

## Project overview

PitWall AI is a Formula 1 tire degradation and race strategy intelligence system.

The long-term goal is to build a race-strategy platform that:
1. extracts historical race data using FastF1,
2. cleans lap/stint/tire data,
3. predicts driver/team-specific tire degradation,
4. accounts for the 2026 regulation reset,
5. ingests FP1/FP2/FP3 race-pace long-run data,
6. calibrates degradation predictions using practice data,
7. simulates likely and optimal pit strategies,
8. evaluates predictions against actual race outcomes.

## Current project stage

We are currently in the tire-degradation modeling phase.

Implemented so far:
- FastF1 race data extraction
- historical multi-race dataset builder
- lap cleaning pipeline
- fuel-adjusted degradation target
- race-phase and nonlinear tire features
- regulation-aware sample weighting
- random train/test model training
- 2026 Japan future-race holdout validation
- MLflow experiment tracking
- holdout error analysis
- driver/team degradation prior experiment

## Important modeling context

2026 is treated as a regulation-reset season.

The model should not treat 2024, 2025, and 2026 equally.

Current regulation-aware weights:
- 2026 rows: weight 5.0
- 2025 rows: weight 1.5
- 2024 or older rows: weight 1.0

The reasoning is:
- 2026 is the primary truth for current-season prediction.
- 2025 is a recent historical prior.
- 2024 is a weaker historical prior.

## Current target variable

The current modeling target is:

fuel_adjusted_degradation_delta

This is preferred over raw lap_time_seconds and over lap_time_delta_from_stint_start.

Reason:
- Raw lap time is not comparable across circuits.
- Simple stint delta is distorted by fuel burn.
- Fuel-adjusted delta partially corrects later-lap fuel burn advantage.

Current fuel correction:
- FUEL_BURN_CORRECTION_PER_LAP = 0.06 seconds/lap

This is in:
src/data/cleaning.py

## Current model features

Base features:
- season
- race
- Driver
- Team
- Compound
- TyreLife
- Stint
- LapNumber
- race_phase
- stint_progress
- tyre_life_squared
- lap_number_squared
- is_first_stint
- is_second_stint
- is_late_race
- is_early_race
- soft_tyre_life
- medium_tyre_life
- hard_tyre_life

Prior features experiment:
- driver_compound_mean_deg
- team_compound_mean_deg
- team_stint_mean_deg
- compound_mean_deg
- driver_mean_deg
- team_mean_deg

The prior features did not improve the 2026 Japan holdout. Keep them as an experiment, but do not assume they are part of the best model path.

## Current best results

Random split with engineered features:
- best model: XGBoost
- MAE about 1.02 sec
- R² about 0.66

2026 Japan future-race holdout before prior features:
- best model: XGBoost
- MAE about 1.6566 sec
- R² about -0.0418

2026 Japan future-race holdout after prior features:
- best model: XGBoost
- MAE about 1.6956 sec
- R² about -0.0989

Conclusion:
- Priors slightly worsened the holdout.
- Best current holdout path is fuel-adjusted + race-phase/nonlinear features, without priors.
- Historical-only prediction is limited.
- The next major improvement should be practice race-pace calibration.

## Known failure patterns

From 2026 Japan holdout error analysis, the model struggles most with:
- MEDIUM tires
- first stint
- specific driver/team/compound pairs
- SAI + Williams + MEDIUM
- OCO + Haas + HARD
- HAD + Red Bull + HARD
- COL + MEDIUM
- BOR + MEDIUM

This indicates that historical-only data is missing current-weekend setup and practice race-pace information.

## Next intended step

Build practice race-pace calibration.

Create modules for:
- src/data/practice_loader.py
- src/models/practice_calibration.py

The practice calibration system should:
1. load user-uploaded FP1/FP2/FP3 long-run race-pace CSVs,
2. filter clean race-pace laps,
3. estimate degradation slope per driver/team/compound/session,
4. compute clean lap count and confidence,
5. blend practice degradation with historical model prediction,
6. later feed calibrated degradation into strategy simulation.

Suggested practice CSV columns:
- race
- session
- driver
- team
- compound
- lap_number
- stint_lap
- lap_time_seconds
- track_temp
- air_temp
- run_type
- clean_lap
- traffic_affected
- drs_used

## Important files

Data:
- src/data/fastf1_loader.py
- src/data/build_historical_dataset.py
- src/data/cleaning.py

Features:
- src/features/degradation_priors.py

Models:
- src/models/train_degradation.py
- src/models/train_degradation_holdout.py
- src/models/predict_degradation_curves.py

Evaluation:
- src/evaluation/analyze_holdout_errors.py

Outputs:
- data/processed/historical_laps_clean.csv
- data/predictions/degradation_model_metrics.csv
- data/predictions/degradation_model_predictions.csv
- data/predictions/holdout_2026_japan_metrics.csv
- data/predictions/holdout_2026_japan_predictions.csv
- reports/post_race_evaluations/

## Commands

Rebuild historical dataset:
python -m src.data.build_historical_dataset

Train random split model:
python -m src.models.train_degradation

Train 2026 Japan holdout model:
python -m src.models.train_degradation_holdout

Analyze holdout errors:
python -m src.evaluation.analyze_holdout_errors

Generate degradation curves:
python -m src.models.predict_degradation_curves

Run MLflow UI:
mlflow ui --host 127.0.0.1 --port 5000

## Development rules for Codex

1. Do not delete working experiment files unless explicitly asked.
2. Do not remove MLflow logging.
3. Do not remove the holdout validation workflow.
4. Do not claim random-split results are the true prediction performance.
5. Prefer future-race holdout metrics for serious evaluation.
6. If modifying target variables, update both training files and the error analyzer.
7. If adding new features, update:
   - cleaning.py if features are generated during cleaning,
   - train_degradation.py,
   - train_degradation_holdout.py,
   - analyze_holdout_errors.py if needed.
8. Keep model outputs saved to data/predictions/.
9. Keep trained models saved to data/models/.
10. Add clear comments for every new modeling assumption.

## Current best modeling interpretation

The model is not expected to perfectly predict unseen 2026 races from historical data alone.

The project should be positioned as:
- honest ML pipeline,
- future-race validation,
- regulation-aware modeling,
- practice-session calibration,
- strategy simulation,
- post-race evaluation.

The next meaningful improvement should come from FP1/FP2/FP3 race-pace calibration rather than more historical-only feature engineering.