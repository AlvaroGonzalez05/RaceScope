# RaceScope — FastAPI Backend

Motor de estrategia F1. Expone metadatos, estrategias y comparaciones vía REST.
Para comandos completos de setup, pipeline y tests ver `COMMANDS.md` en la raíz del repo.

---

## Setup rápido

```bash
python3.11 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Pipeline (orden obligatorio)

```bash
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2023
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5
python -m scripts.train_profiles --min-laps 160
```

## Arrancar API

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints estables (`/api/` prefix)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/metadata/seasons` | Temporadas con datos disponibles |
| GET | `/api/metadata/circuits?season=YYYY` | Circuitos de la temporada |
| GET | `/api/metadata/drivers?season=YYYY` | Pilotos de la temporada |
| GET | `/api/metadata/teams?season=YYYY` | Equipos de la temporada |
| POST | `/api/strategy` | Estrategias rankeadas para un piloto |
| POST | `/api/compare` | Comparativa head-to-head dos pilotos |
| POST | `/api/admin/ingest` | Ingesta en background (requiere X-Admin-Key) |
| GET | `/api/admin/ingest/status` | Estado de la ingesta (requiere X-Admin-Key) |

## Request `/api/strategy`

```json
{
  "year": 2023,
  "circuit_id": "Sakhir",
  "driver_code": "VER",
  "risk_bias": 0.15,
  "n_strategies": 5,
  "force_recompute": false
}
```

`driver_code` es el identificador principal (3 letras: VER, HAM, LEC…).

## Campos de respuesta destacados

```json
{
  "context": { "total_laps": 57, "pit_loss": 22.5, "sc_probability": 0.20, ... },
  "strategies": [
    {
      "strategy_id": "...",
      "type": "1-stop",
      "compounds": ["SOFT", "MEDIUM"],
      "stints": [28, 29],
      "stop_laps": [28],
      "expected_time": 5832.4,
      "variance": 12.1,
      "risk_score": 5834.2,
      "stop_profitability": [18.3],
      "stint_curves": [...]
    }
  ],
  "compute_meta": { "cache_hit": false, "elapsed_ms": 4200, "data_mode": "snapshot" }
}
```

`stop_profitability`: ganancia neta en segundos por cada parada (positivo = parar vale la pena).

## Modelos

Arquitectura activa: **Transformer v3 medium** (d_model=384, 8 capas, 14.3M parámetros).
Ver `ARCHITECTURE.md` para especificación técnica completa.

Artefactos en `models/`:
- `driver_<code>.joblib` — modelo por piloto (≥200 vueltas limpias)
- `global.joblib` — fallback global
- `driver_profile_<code>.joblib` — perfil paramétrico por piloto

## Modos de datos

| Variable `.env` | Comportamiento |
|---|---|
| `OPENF1_AUTH_ENABLED=false` (default) | Modo snapshot: sirve parquets locales, `stale_data=false` |
| `OPENF1_AUTH_ENABLED=true` | Modo live: llama a OpenF1; si falla, cae a snapshot con `stale_data=true` |
