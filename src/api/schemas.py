from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiError(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    service: str


class TableResponse(BaseModel):
    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int


class DashboardSummary(BaseModel):
    drivers: int
    teams: int
    calibrated_rows: int
    strategy_rows: int
    recommended_strategies: int


class PipelineRunResponse(BaseModel):
    output_path: str
    row_count: int
    columns: list[str]


class StorageStatusResponse(BaseModel):
    backend: str
    database_configured: bool
    database_active: bool


class SyncOutputsResponse(BaseModel):
    calibrated_rows: int
    strategy_rows: int


class GenerateCalibratedRequest(BaseModel):
    practice_csv_path: Path | None = None
    historical_predictions_path: Path | None = None
    best_model_path: Path | None = None
    output_path: Path | None = None


class GenerateStrategiesRequest(BaseModel):
    calibrated_predictions_path: Path | None = None
    race_setup_path: Path | None = None
    output_path: Path | None = None


class LiveEvent(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    lap: int = Field(ge=1)
    event_type: str
    driver: str = ""
    target_driver: str = ""
    compound: str = ""
    notes: str = ""


class RunLiveSimulationRequest(BaseModel):
    race: str | None = None
    setup_path: Path | None = None
    output_path: Path | None = None
    events: list[LiveEvent] = Field(default_factory=list)
