# FastAPI Backend (Race Strategy MVP)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline

```bash
python -m scripts.ingest_season --start 2018 --end 2025
python -m scripts.preprocess --start 2018 --end 2025
python -m scripts.train_models --min-laps 200 --epochs 8
python -m scripts.train_profiles --min-laps 120
```

## Run API

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /metadata/seasons`
- `GET /metadata/circuits?season=YYYY`
- `GET /metadata/drivers?season=YYYY`
- `POST /strategy`
- `POST /compare`
