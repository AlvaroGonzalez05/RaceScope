# COMMANDS.md — RaceScope Strategy Lab

Referencia rápida de comandos. Todos los paths son relativos a la raíz del repo.

---

## Generar credenciales (una sola vez)

```bash
# Admin API key — pegar en .env como ADMIN_API_KEY=<valor>
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Backend — setup inicial (una sola vez)

```bash
cd code/backend_fastapi
python3.11 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
cp .env.example .env          # no sobreescribir si ya existe
```

---

## Estructura de datos (Medallion)

```
data/
  bronze/   ← JSON crudo de OpenF1 (nunca transformado)
             sessions.json, laps.json, stints.json, weather.json,
             drivers.json, intervals.json, race_control.json, pit.json
             car_data.json.gz, location.json.gz   ← gzip por volumen (~3.7 Hz)
  silver/   ← Parquet limpio y tipado  (antes: data/raw/)
  gold/     ← Feature store para ML   (antes: data/features/)
```

---

## Pipeline de datos (en orden)

Desde `code/backend_fastapi` con el venv activo:

```bash
# 1. Ingesta OpenF1
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2

# 2. Preprocesado y feature store
python -m scripts.preprocess --year 2023

# 3. Entrenamiento del modelo Transformer v2 (todos los drivers + global fallback)
python -m scripts.train_models --model-version v2 --epochs 50 --patience 12 --min-laps 200 --val-frac 0.15

# 4. Perfiles paramétricos de piloto
python -m scripts.train_profiles --min-laps 160
```

Reentrenar todos los drivers (puede tardar ~1h en CPU con 50 epochs):
```bash
python -m scripts.train_models --model-version v2 --epochs 50 --patience 12 --min-laps 200 --val-frac 0.15
```

Entrenamiento con hiperparámetros optimizados (tras HPO):
```bash
python -m scripts.hparam_search --n-trials 50 --timeout-hours 4 --output hparam_results/
python -m scripts.train_models --model-version v2 --hparam-json hparam_results/best_hparams.json
```

---

## API — arranque

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
uvicorn app.main:app --reload --port 8000
```

Smoke test (caso estable: 2023 / Sakhir / driver 14) — ruta pública, sin key:
```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}' | python -m json.tool | head -40
```

Rutas de admin — requieren `X-Admin-Key`:
```bash
# Disparar ingesta (admin — requiere X-Admin-Key)
curl -s -X POST http://localhost:8000/api/admin/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: <tu_ADMIN_API_KEY>" \
  -d '{"year":2023}'

# Estado de ingesta (admin)
curl -s http://localhost:8000/api/admin/ingest/status \
  -H "X-Admin-Key: <tu_ADMIN_API_KEY>"
```

---

## Frontend

```bash
cd code/frontend
npm install

npm run dev          # Vite dev server en :5173
npm run build        # genera dist/ (servido por FastAPI en single-origin)
```

---

## Tests

Desde `code/backend_fastapi` con el venv activo:

```bash
# Suite completa (sin tests de torch — evita cuelgue de Metal en subprocess)
PYTEST_NO_TORCH=1 pytest

# Tests de torch — solo desde terminal interactiva
pytest tests/test_transformer_model.py -v

# Archivo o test concreto
pytest tests/test_strategy_engine.py -v
pytest tests/test_strategy_engine.py::test_pit_loss_fallback -v

# Stop en el primer fallo
pytest -x
```

---

## Demo / presentación (single-origin, sin CORS)

```bash
# 1. Build del frontend
cd code/frontend
npm run build

# 2. Arrancar solo el backend (sirve dist/ en /)
cd ../backend_fastapi
source .venv_demo/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abrir: `http://127.0.0.1:8000`

Preflight rápido antes de presentar:
```bash
pkill -f "uvicorn app.main:app" || true
lsof -i :8000 -n -P
ls code/backend_fastapi/data/gold/year=2023/features.parquet
ls code/backend_fastapi/models/global.joblib
```

---

## Benchmark — estrategia (latencias de API)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
python scripts/benchmark_strategy.py
# resultado en benchmark_report.json  (cold / warm / hot latencies)
```

## Benchmark — modelos Transformer (v1 vs v2)

```bash
cd code/backend_fastapi
# usa system python3.11 (matplotlib disponible fuera del venv)
python3.11 -m scripts.benchmark_models
# imprime tabla comparativa y escribe benchmark_models_report.json
```

## Visualización de entrenamiento

```bash
cd code/backend_fastapi
# genera 5 PNGs en plots/ a partir de los CSV en models/logs/
python3.11 -m scripts.plot_training --logs-dir models/logs --out-dir plots --max-epochs 50
# plots/global_curves.png, driver_grid.png, final_val_mae_bar.png,
#       convergence_epochs.png, loss_vs_valmae.png
```

## Evaluación 2025 (held-out)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
# requiere ingest + preprocess 2025 primero:
python -m scripts.ingest_season --year 2025 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2025
# evaluación (nunca usada en entrenamiento):
python -m scripts.evaluate_2025 --push-sensitivity-test
```

---

## Diagnóstico rápido

```bash
# Verificar imports del stack científico (detecta freeze de Metal)
.venv_demo/bin/python -c "import pandas, numpy, scipy, torch; print('ok')"

# Puerto ocupado
lsof -i :8000 -n -P
pkill -f "uvicorn app.main:app"

# Cache inconsistente tras re-pipeline
rm -rf code/backend_fastapi/cache/pace_curves/*
```

Flags útiles de la API:
```json
{ "force_recompute": true }   // ignora cache de curvas en disco
{ "debug_profile": true }     // incluye info del perfil paramétrico en la respuesta
```
