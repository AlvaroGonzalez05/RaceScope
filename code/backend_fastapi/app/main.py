from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_RISK_LAMBDA, DEFAULT_STRATEGY_COUNT
from .data_store import load_features, metadata_for_year, seasons_available
from .strategy_engine import StrategyEngine

app = FastAPI(title="Race Strategy MVP", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, Dict] = {}


def _cache_get(key: str) -> Optional[Dict]:
    item = _cache.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > 3600:
        _cache.pop(key, None)
        return None
    return item["value"]


def _cache_set(key: str, value: Dict) -> None:
    _cache[key] = {"ts": time.time(), "value": value}


class StrategyRequest(BaseModel):
    year: int
    circuit_id: str = Field(..., description="Circuit short name")
    driver_id: int
    risk_bias: float = DEFAULT_RISK_LAMBDA
    n_strategies: int = DEFAULT_STRATEGY_COUNT
    debug_profile: bool = False


class CompareRequest(BaseModel):
    year: int
    circuit_id: str
    driver_id: int
    teammate_id: int
    risk_bias: float = DEFAULT_RISK_LAMBDA
    n_strategies: int = DEFAULT_STRATEGY_COUNT
    debug_profile: bool = False


# Shared handlers

def _get_seasons() -> List[int]:
    return seasons_available()


def _get_circuits(season: int) -> List[str]:
    meta = metadata_for_year(season)
    circuits = meta["circuits"]
    return circuits["circuit_id"].dropna().unique().tolist() if not circuits.empty else []


def _get_drivers(season: int) -> List[Dict]:
    meta = metadata_for_year(season)
    drivers = meta["drivers"]
    return drivers.to_dict(orient="records") if not drivers.empty else []


def _get_teams(season: int) -> List[str]:
    meta = metadata_for_year(season)
    teams = meta["teams"]
    return teams["team_name"].dropna().unique().tolist() if not teams.empty else []


def _post_strategy(req: StrategyRequest) -> Dict:
    cache_key = f"strategy:{req.year}:{req.circuit_id}:{req.driver_id}:{req.risk_bias}:{req.n_strategies}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    df = load_features()
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available. Run ingestion + preprocessing.")

    engine = StrategyEngine(df)
    payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_id=req.driver_id,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        debug_profile=req.debug_profile,
    )

    response = {
        "year": req.year,
        "circuit_id": req.circuit_id,
        "driver_id": req.driver_id,
        **payload,
    }
    _cache_set(cache_key, response)
    return response


def _post_compare(req: CompareRequest) -> Dict:
    cache_key = f"compare:{req.year}:{req.circuit_id}:{req.driver_id}:{req.teammate_id}:{req.risk_bias}:{req.n_strategies}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    df = load_features()
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available. Run ingestion + preprocessing.")

    engine = StrategyEngine(df)
    driver_payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_id=req.driver_id,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        opponent_id=req.teammate_id,
        debug_profile=req.debug_profile,
    )
    teammate_payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_id=req.teammate_id,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        opponent_id=req.driver_id,
        debug_profile=req.debug_profile,
    )

    response = {
        "year": req.year,
        "circuit_id": req.circuit_id,
        "driver": {"driver_id": req.driver_id, **driver_payload},
        "teammate": {"driver_id": req.teammate_id, **teammate_payload},
    }
    _cache_set(cache_key, response)
    return response


# Legacy routes (temporary compatibility)
@app.get("/metadata/seasons")
def get_seasons_legacy() -> List[int]:
    return _get_seasons()


@app.get("/metadata/circuits")
def get_circuits_legacy(season: int) -> List[str]:
    return _get_circuits(season)


@app.get("/metadata/drivers")
def get_drivers_legacy(season: int) -> List[Dict]:
    return _get_drivers(season)


@app.get("/metadata/teams")
def get_teams_legacy(season: int) -> List[str]:
    return _get_teams(season)


@app.post("/strategy")
def post_strategy_legacy(req: StrategyRequest) -> Dict:
    return _post_strategy(req)


@app.post("/compare")
def post_compare_legacy(req: CompareRequest) -> Dict:
    return _post_compare(req)


# Stable /api routes
@app.get("/api/metadata/seasons")
def get_seasons() -> List[int]:
    return _get_seasons()


@app.get("/api/metadata/circuits")
def get_circuits(season: int) -> List[str]:
    return _get_circuits(season)


@app.get("/api/metadata/drivers")
def get_drivers(season: int) -> List[Dict]:
    return _get_drivers(season)


@app.get("/api/metadata/teams")
def get_teams(season: int) -> List[str]:
    return _get_teams(season)


@app.post("/api/strategy")
def post_strategy(req: StrategyRequest) -> Dict:
    return _post_strategy(req)


@app.post("/api/compare")
def post_compare(req: CompareRequest) -> Dict:
    return _post_compare(req)


# Serve frontend build from same origin when available
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(_FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        target = _FRONTEND_DIST / full_path
        if target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(_FRONTEND_DIST / "index.html")
