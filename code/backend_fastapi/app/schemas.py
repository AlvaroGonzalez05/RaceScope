from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class DriverOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    driver_code: str
    team_name: str | None = None
    driver_id: int | None = None  # kept for reference, driver_code is the primary identifier


class TeamOut(BaseModel):
    team_name: str


class CircuitOut(BaseModel):
    circuit_id: str


class StintCurve(BaseModel):
    model_config = ConfigDict(extra="allow")
    compound: str
    start_lap: int
    end_lap: int
    lap_time_data: list[float]
    tyre_life_data: list[float]


class PitWindow(BaseModel):
    model_config = ConfigDict(extra="allow")
    lap_min: int | None = None
    lap_max: int | None = None


class StrategyOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    strategy_id: str
    type: str
    compounds: list[str]
    stints: list[int]
    stint_curves: list[StintCurve]
    pit_windows: list[PitWindow]
    stop_laps: list[int]
    expected_time: float
    variance: float
    risk_score: float
    stop_profitability: list[float] = []


class ContextOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_laps: int
    track_temp: float
    air_temp: float
    pit_loss: float
    sc_probability: float


class StrategyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    context: ContextOut
    strategies: list[StrategyOut]
    degradation: dict
    compute_meta: dict | None = None


class CompareResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    driver_a: StrategyResponse
    driver_b: StrategyResponse
