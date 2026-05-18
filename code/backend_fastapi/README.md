# FastAPI Backend (Race Strategy MVP)

## Setup

```bash
python3 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Data Pipeline

```bash
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2023
python -m scripts.train_models --min-laps 260 --epochs 12
python -m scripts.train_profiles --min-laps 160
```

## Run API

```bash
uvicorn app.main:app --reload --port 8000
```

## Demo MVP (snapshot local recomendado)

Para una demo estable (sin depender de OpenF1 en directo):

1. En `.env`, usar `OPENF1_AUTH_ENABLED=false`.
2. Confirmar artefactos locales:
```bash
ls data/features/year=2023/features.parquet
ls models/global.joblib
ls models/driver_14.joblib
```
3. Arrancar API:
```bash
source .venv_demo/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Smoke checks:
```bash
curl -s http://127.0.0.1:8000/api/metadata/seasons
curl -s "http://127.0.0.1:8000/api/metadata/circuits?season=2023"
curl -s -X POST http://127.0.0.1:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14,"force_recompute":true}'
```

## Endpoints

- `GET /metadata/seasons`
- `GET /metadata/circuits?season=YYYY`
- `GET /metadata/drivers?season=YYYY`
- `POST /strategy`
- `POST /compare`

### Request options

`POST /strategy` y `POST /compare` aceptan:

- `force_recompute` (opcional, bool): cuando es `true`, salta la cache y fuerza una simulacion fresca (incluye Monte Carlo top-K).

Autenticacion OpenF1:

- `OPENF1_AUTH_ENABLED=true`
- `OPENF1_USERNAME=<usuario>`
- `OPENF1_PASSWORD=<password>`

Si OpenF1 falla y hay datos locales previos, se activa modo snapshot (`data/features/*`) con indicador `stale_data=true`.

### Compute meta

Las respuestas incluyen `compute_meta` para trazabilidad:

- `cache_hit`
- `mc_executed`
- `elapsed_ms`
- `data_mode` (`live|snapshot`)
- `stale_data`
