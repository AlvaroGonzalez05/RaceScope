from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from cachetools import TTLCache
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from . import config
from .config import DEFAULT_RISK_LAMBDA, DEFAULT_STRATEGY_COUNT
from .config import CACHE_TTL_SECONDS, OPENF1_AUTH_ENABLED, ADMIN_API_KEY
from .data_store import load_features, metadata_for_year, seasons_available, load_snapshot_state, invalidate_features_cache
from .ingest import ingest_season
from .preprocess import build_features_for_year
from .schemas import StrategyResponse, CompareResponse, DriverOut

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security — admin API key
# ---------------------------------------------------------------------------

_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def verify_admin_key(key: str | None = Security(_admin_key_header)) -> None:
    """FastAPI dependency: validates X-Admin-Key header for admin routes.

    Uses secrets.compare_digest to prevent timing attacks.
    Raises 403 if ADMIN_API_KEY is not configured or the key does not match.
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Admin endpoints are disabled: ADMIN_API_KEY not configured.",
        )
    if not key or not secrets.compare_digest(key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key.")


# ---------------------------------------------------------------------------
# Rate limiting — in-memory, per-IP sliding window
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 30       # requests
_RATE_LIMIT_WINDOW = 60.0  # seconds


def rate_limit(request: Request) -> None:
    """FastAPI dependency: 30 req/min per IP for compute-heavy routes."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW
    hits = _rate_limit_store[client_ip]
    # Evict stale timestamps
    _rate_limit_store[client_ip] = [t for t in hits if t > window_start]
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT_MAX} requests per {int(_RATE_LIMIT_WINDOW)}s.",
        )
    _rate_limit_store[client_ip].append(now)


def _preload_models_sync() -> None:
    """Pre-warm the LRU model cache at startup to eliminate cold-start latency."""
    from .strategy_engine import _load_model_cached  # noqa: PLC0415

    try:
        _load_model_cached("__fallback__")  # triggers global.joblib load
        logger.info("startup: global model pre-loaded")
    except Exception as exc:
        logger.warning("startup: model pre-load skipped (%s)", exc)

    try:
        from .data_store import load_features  # noqa: PLC0415
        df = load_features()
        if not df.empty:
            driver_codes = df["driver_code"].dropna().unique().tolist()
            for code in driver_codes[:16]:
                try:
                    _load_model_cached(str(code))
                except Exception:
                    pass
            logger.info("startup: pre-loaded models for %d drivers", min(len(driver_codes), 16))
    except Exception as exc:
        logger.warning("startup: per-driver pre-load skipped (%s)", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _preload_models_sync)
    yield


app = FastAPI(title="Race Strategy MVP", version="0.2.0", lifespan=lifespan)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        request_id = str(uuid.uuid4())[:8]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: TTLCache = TTLCache(maxsize=128, ttl=CACHE_TTL_SECONDS)


def _cache_get(key: str) -> Optional[Dict]:
    return _cache.get(key)


def _cache_set(key: str, value: Dict) -> None:
    _cache[key] = value


def _resolve_data_mode() -> Dict[str, object]:
    snapshot_state = load_snapshot_state()
    stale_data = bool(snapshot_state.get("stale_mode", False))
    if stale_data:
        return {"data_mode": "snapshot", "stale_data": True}
    if not OPENF1_AUTH_ENABLED:
        # Intentional local/demo mode: snapshot is current source of truth.
        return {"data_mode": "snapshot", "stale_data": False}
    return {"data_mode": "live", "stale_data": False}


class StrategyRequest(BaseModel):
    year: int
    circuit_id: str = Field(..., description="Circuit short name")
    driver_code: str = Field(..., description="3-letter driver code (e.g. VER, HAM, ALO)")
    risk_bias: float = DEFAULT_RISK_LAMBDA
    n_strategies: int = DEFAULT_STRATEGY_COUNT
    debug_profile: bool = False
    force_recompute: bool = False


class CompareRequest(BaseModel):
    year: int
    circuit_id: str
    driver_code: str = Field(..., description="3-letter driver code")
    teammate_code: str = Field(..., description="3-letter teammate code")
    risk_bias: float = DEFAULT_RISK_LAMBDA
    n_strategies: int = DEFAULT_STRATEGY_COUNT
    debug_profile: bool = False
    force_recompute: bool = False


class IngestRequest(BaseModel):
    year: int

    @field_validator("year")
    @classmethod
    def year_must_be_valid(cls, v: int) -> int:
        if v < 2018 or v > 2030:
            raise ValueError("year must be between 2018 and 2030")
        return v


def _run_ingest(year: int) -> None:
    try:
        logger.info("background ingest starting year=%s", year)
        ingest_season(year)
        build_features_for_year(year)
        invalidate_features_cache()
        logger.info("background ingest complete year=%s", year)
    except Exception as exc:
        logger.error("background ingest failed year=%s error=%s", year, exc)


# Shared handlers

def _get_seasons() -> List[int]:
    return seasons_available()


def _get_circuits(season: int) -> List[str]:
    available = seasons_available()
    if season not in available:
        raise HTTPException(status_code=404, detail=f"Season {season} not found. Available: {available}")
    meta = metadata_for_year(season)
    circuits = meta["circuits"]
    return circuits["circuit_id"].dropna().unique().tolist() if not circuits.empty else []


def _get_drivers(season: int) -> List[Dict]:
    available = seasons_available()
    if season not in available:
        raise HTTPException(status_code=404, detail=f"Season {season} not found. Available: {available}")
    meta = metadata_for_year(season)
    drivers = meta["drivers"]
    return drivers.to_dict(orient="records") if not drivers.empty else []


def _get_teams(season: int) -> List[str]:
    available = seasons_available()
    if season not in available:
        raise HTTPException(status_code=404, detail=f"Season {season} not found. Available: {available}")
    meta = metadata_for_year(season)
    teams = meta["teams"]
    return teams["team_name"].dropna().unique().tolist() if not teams.empty else []


def _post_strategy(req: StrategyRequest) -> Dict:
    cache_key = f"strategy:{req.year}:{req.circuit_id}:{req.driver_code}:{req.risk_bias}:{req.n_strategies}"
    if not req.force_recompute:
        cached = _cache_get(cache_key)
        if cached:
            cached_with_meta = dict(cached)
            mode = _resolve_data_mode()
            cached_with_meta["compute_meta"] = {
                "cache_hit": True,
                "mc_executed": False,
                "elapsed_ms": 0,
                "data_mode": mode["data_mode"],
                "stale_data": mode["stale_data"],
            }
            return cached_with_meta

    df = load_features()
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available. Run ingestion + preprocessing.")

    started_at = time.perf_counter()
    # Lazy import avoids loading the full scientific stack at API startup.
    from .strategy_engine import StrategyEngine

    engine = StrategyEngine(df)
    payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_code=req.driver_code,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        debug_profile=req.debug_profile,
    )

    response = {
        "year": req.year,
        "circuit_id": req.circuit_id,
        "driver_code": req.driver_code,
        **payload,
        "compute_meta": {
            "cache_hit": False,
            "mc_executed": bool(payload.get("strategies")),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            **_resolve_data_mode(),
        },
    }
    if not req.force_recompute:
        _cache_set(cache_key, response)
    return response


def _post_compare(req: CompareRequest) -> Dict:
    cache_key = f"compare:{req.year}:{req.circuit_id}:{req.driver_code}:{req.teammate_code}:{req.risk_bias}:{req.n_strategies}"
    if not req.force_recompute:
        cached = _cache_get(cache_key)
        if cached:
            cached_with_meta = dict(cached)
            mode = _resolve_data_mode()
            cached_with_meta["compute_meta"] = {
                "cache_hit": True,
                "mc_executed": False,
                "elapsed_ms": 0,
                "data_mode": mode["data_mode"],
                "stale_data": mode["stale_data"],
            }
            return cached_with_meta

    df = load_features()
    if df.empty:
        raise HTTPException(status_code=400, detail="No features available. Run ingestion + preprocessing.")

    started_at = time.perf_counter()
    # Lazy import avoids loading the full scientific stack at API startup.
    from .strategy_engine import StrategyEngine

    engine = StrategyEngine(df)
    driver_payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_code=req.driver_code,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        opponent_code=req.teammate_code,
        debug_profile=req.debug_profile,
    )
    teammate_payload = engine.generate_strategies(
        year=req.year,
        circuit_id=req.circuit_id,
        driver_code=req.teammate_code,
        risk_bias=req.risk_bias,
        n_strategies=req.n_strategies,
        opponent_code=req.driver_code,
        debug_profile=req.debug_profile,
    )

    response = {
        "year": req.year,
        "circuit_id": req.circuit_id,
        "driver": {"driver_code": req.driver_code, **driver_payload},
        "teammate": {"driver_code": req.teammate_code, **teammate_payload},
        "compute_meta": {
            "cache_hit": False,
            "mc_executed": bool(driver_payload.get("strategies") or teammate_payload.get("strategies")),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            **_resolve_data_mode(),
        },
    }
    if not req.force_recompute:
        _cache_set(cache_key, response)
    return response


# Legacy routes (temporary compatibility)
@app.get("/metadata/seasons")
def get_seasons_legacy(request: Request) -> List[int]:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _get_seasons()


@app.get("/metadata/circuits")
def get_circuits_legacy(request: Request, season: int) -> List[str]:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _get_circuits(season)


@app.get("/metadata/drivers")
def get_drivers_legacy(request: Request, season: int) -> List[Dict]:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _get_drivers(season)


@app.get("/metadata/teams")
def get_teams_legacy(request: Request, season: int) -> List[str]:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _get_teams(season)


@app.post("/strategy")
def post_strategy_legacy(request: Request, req: StrategyRequest) -> Dict:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _post_strategy(req)


@app.post("/compare")
def post_compare_legacy(request: Request, req: CompareRequest) -> Dict:
    logger.warning("Deprecated route called: %s — use /api/ prefix", request.url.path)
    return _post_compare(req)


# Stable /api routes
@app.get("/api/metadata/seasons")
def get_seasons() -> List[int]:
    return _get_seasons()


@app.get("/api/metadata/circuits")
def get_circuits(season: int) -> List[str]:
    return _get_circuits(season)


@app.get("/api/metadata/drivers", response_model=list[DriverOut], response_model_exclude_none=True)
def get_drivers(season: int) -> List[Dict]:
    return _get_drivers(season)


@app.get("/api/metadata/teams")
def get_teams(season: int) -> List[str]:
    return _get_teams(season)


@app.post("/api/strategy", response_model=StrategyResponse, response_model_exclude_none=True)
def post_strategy(req: StrategyRequest, _rl: None = Depends(rate_limit)) -> Dict:
    return _post_strategy(req)


@app.post("/api/compare", response_model=CompareResponse, response_model_exclude_none=True)
def post_compare(req: CompareRequest, _rl: None = Depends(rate_limit)) -> Dict:
    return _post_compare(req)


@app.post("/api/admin/ingest", dependencies=[Depends(verify_admin_key)])
async def admin_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    logger.info("admin ingest triggered for year=%s", req.year)
    background_tasks.add_task(_run_ingest, req.year)
    return {"status": "ingestion_started", "year": req.year}


@app.get("/api/admin/ingest/status", dependencies=[Depends(verify_admin_key)])
async def admin_ingest_status():
    return load_snapshot_state()


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
