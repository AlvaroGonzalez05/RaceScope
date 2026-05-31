# COMMANDS.md — RaceScope Strategy Lab

Referencia rápida de comandos. Todos los paths son relativos a la raíz del repo.

---

## Arrancar la app

Un solo proceso. Compilar el frontend si hay cambios en `src/`, luego arrancar el backend.

```bash
# 1. Compilar frontend (solo si hay cambios en code/frontend/src/)
cd code/frontend
npm run build          # genera frontend/dist/

# 2. Arrancar backend (sirve también el frontend compilado)
cd ../backend_fastapi
source .venv_demo/bin/activate
uvicorn app.main:app --port 8000
```

Abrir: **`http://localhost:8000`**

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
  silver/   ← Parquet limpio y tipado
  gold/     ← Feature store para ML
```

---

## Pipeline de datos (en orden)

Desde `code/backend_fastapi` con el venv activo:

```bash
# 1. Ingesta OpenF1
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2

# 2. Preprocesado y feature store
python -m scripts.preprocess --year 2023

# 3. Entrenamiento Transformer v3 (todos los drivers + global fallback)
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5

# 4. Perfiles paramétricos de piloto
python -m scripts.train_profiles --min-laps 160
```

Entrenamiento con versión específica:
```bash
python -m scripts.train_models --model-version v2 --epochs 15 --patience 5
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5
```

Búsqueda de hiperparámetros (solo v2):
```bash
python -m scripts.hparam_search --n-trials 50 --timeout-hours 4 --output hparam_results/
python -m scripts.train_models --model-version v2 --hparam-json hparam_results/best_hparams.json
```

---

## API — smoke test

```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_code":"VER"}' | python -m json.tool | head -40
```

Rutas de admin — requieren `X-Admin-Key`:
```bash
curl -s -X POST http://localhost:8000/api/admin/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: <tu_ADMIN_API_KEY>" \
  -d '{"year":2023}'

curl -s http://localhost:8000/api/admin/ingest/status \
  -H "X-Admin-Key: <tu_ADMIN_API_KEY>"
```

---

## Tests

Desde `code/backend_fastapi` con el venv activo:

```bash
pytest                                          # suite completa
pytest tests/test_transformer_model.py -v      # solo modelo transformer
pytest tests/test_transformer_model.py -k v3   # solo tests v3
pytest tests/test_strategy_engine.py -v        # solo strategy engine
pytest -x                                      # parar en el primer fallo
```

---

## Benchmark — rendimiento de arquitecturas (v1 vs v2 vs v3)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
python -m scripts.benchmark_architectures --n-train 512 --epochs 5 --n-sim 100 --laps 60
# resultado en reports/benchmark_architectures.csv
```

## Benchmark — latencias de API (cold / warm / hot)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
python scripts/benchmark_strategy.py
# resultado en benchmark_report.json
```

---

## Evaluación 2025 (held-out)

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
python -m scripts.ingest_season --year 2025 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2025
python -m scripts.evaluate_2025 --push-sensitivity-test
```

---

## Preflight antes de presentar

```bash
pkill -f "uvicorn app.main:app" || true   # matar instancia anterior
lsof -i :8000 -n -P                       # verificar puerto libre
ls code/backend_fastapi/data/gold/year=2023/features.parquet
ls code/backend_fastapi/models/global.joblib
```

---

## Diagnóstico rápido

```bash
# Verificar imports del stack científico
cd code/backend_fastapi
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
