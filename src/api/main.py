from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.settings import get_settings
from src.api.schemas import (
    DashboardSummary,
    GenerateCalibratedRequest,
    GenerateStrategiesRequest,
    HealthResponse,
    PipelineRunResponse,
    RunLiveSimulationRequest,
    StorageStatusResponse,
    SyncOutputsResponse,
    TableResponse,
)
from src.api.services import (
    dataframe_response,
    load_calibrated_predictions,
    load_strategy_recommendations,
    run_calibrated_predictions_pipeline,
    run_live_pipeline,
    run_strategy_recommendations_pipeline,
    storage_status,
    summarize_dashboard,
    sync_current_outputs_to_database,
)


app = FastAPI(
    title="PitWall AI API",
    version="0.1.0",
    description="Backend API for degradation calibration and race strategy workflows.",
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="pitwall-api")


@app.get("/api/storage/status", response_model=StorageStatusResponse)
def get_storage_status() -> dict:
    return storage_status()


@app.post("/api/storage/sync-current-outputs", response_model=SyncOutputsResponse)
def sync_current_outputs() -> dict[str, int]:
    try:
        return sync_current_outputs_to_database()
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.get("/api/calibrated-degradation", response_model=TableResponse)
def get_calibrated_degradation() -> dict:
    try:
        return dataframe_response(load_calibrated_predictions())
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.get("/api/strategy-recommendations", response_model=TableResponse)
def get_strategy_recommendations() -> dict:
    try:
        return dataframe_response(load_strategy_recommendations())
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.get("/api/dashboard-summary", response_model=DashboardSummary)
def get_dashboard_summary() -> dict[str, int]:
    try:
        return summarize_dashboard()
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.post("/api/pipeline/calibrated-predictions", response_model=PipelineRunResponse)
def generate_calibrated_predictions(request: GenerateCalibratedRequest) -> PipelineRunResponse:
    try:
        result, output_path = run_calibrated_predictions_pipeline(
            practice_csv_path=request.practice_csv_path,
            historical_predictions_path=request.historical_predictions_path,
            best_model_path=request.best_model_path,
            output_path=request.output_path,
        )
        return PipelineRunResponse(
            output_path=str(output_path),
            row_count=int(len(result)),
            columns=result.columns.tolist(),
        )
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.post("/api/pipeline/strategy-recommendations", response_model=PipelineRunResponse)
def generate_strategies(request: GenerateStrategiesRequest) -> PipelineRunResponse:
    try:
        result, output_path = run_strategy_recommendations_pipeline(
            calibrated_predictions_path=request.calibrated_predictions_path,
            race_setup_path=request.race_setup_path,
            output_path=request.output_path,
        )
        return PipelineRunResponse(
            output_path=str(output_path),
            row_count=int(len(result)),
            columns=result.columns.tolist(),
        )
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)


@app.post("/api/live/simulation", response_model=PipelineRunResponse)
def run_live_simulation(request: RunLiveSimulationRequest) -> PipelineRunResponse:
    try:
        result, output_path = run_live_pipeline(
            race=request.race,
            setup_path=request.setup_path,
            output_path=request.output_path,
            events=[event.model_dump() for event in request.events],
        )
        return PipelineRunResponse(
            output_path=str(output_path),
            row_count=int(len(result)),
            columns=result.columns.tolist(),
        )
    except Exception as exc:  # pragma: no cover - mapped at API boundary.
        _raise_http_error(exc)
