# Backend PLAN.md — RaceScope FastAPI

This is the local backend copy of the master execution plan. For full context on agent roles, QA protocol, and phase ordering, read the root `PLAN.md` first.

**Backend agents:** `BE-CORE`, `BE-DATA`, `BE-API`
**QA agent** reviews this area after each phase.

---

## Quick Reference — Backend TO_DO Items

All items are from `TO_DO.md`. Ownership assigned here.

| TO_DO Item | Agent | Phase |
|---|---|---|
| Replace `requests` with `httpx` async client | BE-CORE | 1 |
| Build live-data ingestion trigger (`POST /api/admin/ingest`) | BE-CORE | 1 |
| Add cache invalidation for `load_features()` | BE-CORE | 1 |
| Derive `pit_loss` from real OpenF1 data | BE-DATA | 1 |
| Derive `sc_probability` from session data | BE-DATA | 1 |
| Fix `circuit_id` normalization in `preprocess.py` | BE-DATA | 1 |
| Add Pydantic response models to all routes | BE-API | 2 |
| Add query param validation (404 for missing seasons) | BE-API | 2 |
| Make CORS origins env-driven | BE-API | 2 |
| Add structured logging (replace print() calls) | BE-API | 2 |
| Add request-ID middleware | BE-API | 2 |
| Deprecate legacy routes (add headers/warnings) | BE-API | 2 |
| Improve ingestion error handling (skipped sessions) | BE-API | 2 |
| Swap MD5 for SHA-256 in cache keying | BE-API | 2 |
| Bound in-memory `_cache` in `main.py` | BE-API | 2 |

---

## Setup Checklist (all backend agents run this first)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
python -c "import pandas, numpy, scipy; print('scientific stack ok')"
uvicorn app.main:app --reload --port 8000 &
curl -s http://localhost:8000/api/metadata/seasons
# Should return a JSON array of years. If empty, run the pipeline first.
pkill -f "uvicorn app.main:app"
```

If the scientific stack import fails, recreate the venv:
```bash
python3.11 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
```

---

## BE-CORE Detailed Instructions

See root `PLAN.md` Phase 1 → BE-CORE section for full task list.

### Credentials & environment

Credentials are already written to `code/backend_fastapi/.env` (gitignored — never commit).
The `.env` file is read automatically by `config.py` at import time via `_load_dotenv()`.
Do not hardcode credentials anywhere in source files.

Confirm the env is loaded correctly before starting work:
```bash
python -c "from app.config import OPENF1_USERNAME, OPENF1_RATE_PER_SECOND; print(OPENF1_USERNAME, OPENF1_RATE_PER_SECOND)"
# Expected: agonzaleztabernero@alu.icai.comillas.edu 6
```

### Rate limit constraints (premium tier)

| Limit | Value | Enforcement mechanism |
|---|---|---|
| Burst | 6 req/s | `MIN_INTERVAL = 0.167 s` floor in `_acquire_token()` |
| Sustained | 60 req/min | Token-bucket in `_acquire_token()` — see Task 2 in root PLAN.md |

Both `OPENF1_RATE_PER_SECOND` and `OPENF1_RATE_PER_MINUTE` are already defined in `config.py` and set in `.env`. The token-bucket implementation in BE-CORE Task 2 must read these values from config, not hardcode them.

### Other key constraints

- `httpx` must be added to `requirements.txt`. Version pin: `httpx>=0.27,<1.0`.
- The `get_sync` wrapper in `openf1_client.py` is only for CLI scripts. Do not call it from FastAPI route handlers.
- `POST /api/admin/ingest` must use `BackgroundTasks`, not `asyncio.create_task`, because the FastAPI request context must remain valid for proper background task lifecycle management.
- After adding the admin routes, verify the route appears in `/docs` with the correct request schema.

### Verification commands

```bash
# After implementing:
uvicorn app.main:app --reload --port 8000 &
sleep 2

# Test async client import
python -c "from app.openf1_client import OpenF1Client; print('ok')"

# Test admin ingest endpoint
curl -s -X POST http://localhost:8000/api/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"year": 2023}'
# Expected: {"status": "ingestion_started", "year": 2023}

# Test ingest status
curl -s http://localhost:8000/api/admin/ingest/status
# Expected: snapshot_state JSON

# Smoke test — must still work
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}' | python3 -m json.tool | head -20

pkill -f "uvicorn app.main:app"
```

---

## BE-DATA Detailed Instructions

See root `PLAN.md` Phase 1 → BE-DATA section for full task list.

### Key constraints

- `_context()` must remain a synchronous method (no async). The pit_loss and sc_probability computation reads from the already-loaded DataFrame — no I/O.
- The SC detection proxy (median_lap_time * 1.35) is a heuristic. If zero eligible laps are found, fall back immediately; do not raise exceptions.
- The `CIRCUIT_ID_CANONICAL` dict must use lowercase keys and canonical-cased values. All lookups must `.lower()` the raw input before the dict lookup.
- Do not modify the parquet schema — only the preprocessing logic that populates `circuit_id`.

### Verification commands

```bash
# Test that preprocess runs cleanly after circuit_id normalization
python -c "
from app.preprocess import build_features_for_year
df = build_features_for_year(2023)
print('circuits found:', df['circuit_id'].unique().tolist())
print('rows:', len(df))
"

# Test that strategy engine computes non-hardcoded values
python -c "
from app.data_store import load_features
from app.strategy_engine import StrategyEngine
df = load_features(2023)
engine = StrategyEngine(df)
ctx = engine._context(2023, 'Sakhir')
print('pit_loss:', ctx.pit_loss)
print('sc_probability:', ctx.sc_probability)
# pit_loss should NOT be exactly 22.5 if Sakhir has race data
# sc_probability should NOT be exactly 0.20 if Sakhir has ≥2 race sessions
"
```

---

## BE-API Detailed Instructions

See root `PLAN.md` Phase 2 → BE-API section for full task list.

### Key constraints

- `schemas.py` must live in `app/schemas.py` — do not put models in `main.py`.
- Use `response_model_exclude_none=True` on all routes to keep responses clean.
- `model_config = ConfigDict(extra="allow")` on `StrategyResponse` and `CompareResponse` only — not on simple metadata models.
- The SHA-256 change in `openf1_client.py` is a one-line edit. Existing cached `.json` files will have old MD5-named keys; they will be ignored (cache miss) on first access, which is the correct behaviour.
- For the `_cache` in `main.py`, replace the raw `dict` with:
  ```python
  from cachetools import TTLCache
  _cache: TTLCache = TTLCache(maxsize=128, ttl=CACHE_TTL_SECONDS)
  ```
  Add `cachetools` to `requirements.txt`.

### Verification commands

```bash
uvicorn app.main:app --reload --port 8000 &
sleep 2

# Check OpenAPI schema is coherent
curl -s http://localhost:8000/openapi.json | python3 -m json.tool | grep -c '"$ref"'
# Should return a positive number (schema is populated)

# Test 404 for missing season
curl -s "http://localhost:8000/api/metadata/circuits?season=1900"
# Expected: {"detail": "Season 1900 not found"}

# Test CORS header
curl -s -I -H "Origin: http://localhost:5173" http://localhost:8000/api/metadata/seasons | grep -i "access-control"
# Expected: Access-Control-Allow-Origin: http://localhost:5173

# Test X-Request-ID header
curl -s -I http://localhost:8000/api/metadata/seasons | grep -i "x-request-id"
# Expected: x-request-id: <8-char uuid fragment>

# Test deprecated route warning appears in logs
curl -s http://localhost:8000/strategy 2>&1
# Check uvicorn log output for deprecation warning

pkill -f "uvicorn app.main:app"
```

---

## QA Checklist — Backend

After Phase 1 (BE-CORE + BE-DATA):
- [ ] Server starts without import errors
- [ ] Smoke-test returns valid strategy JSON (same fields as baseline)
- [ ] `POST /api/admin/ingest` returns 200
- [ ] `circuit_id` values in features are canonical strings (no integers)
- [ ] `pit_loss` and `sc_probability` differ from 22.5/0.20 for circuits with data

After Phase 2 (BE-API):
- [ ] `/docs` renders all routes with typed schemas
- [ ] `/api/metadata/circuits?season=1900` returns 404
- [ ] `curl -H "Origin: http://localhost:5173"` on any route returns CORS header
- [ ] `X-Request-ID` header present on all responses
- [ ] No `print()` calls remain in `ingest.py` or `preprocess.py`
- [ ] Legacy routes still work but log deprecation warnings
- [ ] Smoke-test still returns valid strategy JSON
