# RaceScope Strategy Lab

## 1) Vision del producto
RaceScope es una aplicacion para explorar y comparar estrategias de Formula 1 antes de carrera (`Pre-race`).

Objetivos actuales:
- Abrir en `Home` con una landing interactiva de producto.
- Mostrar estrategias realistas por piloto con degradacion y ventanas de pit.
- Permitir comparacion multi-piloto en filas independientes.
- Mantener tiempos de respuesta bajos con cache y evaluacion escalable.

Objetivos siguientes fases:
- `Live`: adaptacion durante carrera.
- `Rewatch`: replay estrategico.
- `Explore`: analisis historico avanzado.

---

## 2) Arquitectura completa
La arquitectura actual se divide en tres bloques.

- Backend API y simulacion: `code/backend_fastapi`
- Frontend web: `code/frontend`
- Datos/modelos/cache: `code/backend_fastapi/data`, `code/backend_fastapi/models`, `code/backend_fastapi/cache`

### Flujo de alto nivel
1. Ingesta OpenF1 y guardado en parquet (capa bronze).
2. Preprocesado y feature store (capas silver → gold).
3. Entrenamiento de modelos (Transformer v3 por piloto + perfil parametrico).
4. Simulacion de estrategias (analitico + refino MC top-K).
5. API FastAPI expone metadata y estrategias.
6. Frontend consume API y renderiza comparaciones.

---

## 3) Pipeline de datos (OpenF1 -> features -> modelos -> simulacion)

### 3.1 Ingesta
Script:
- `code/backend_fastapi/scripts/ingest_season.py`

Entrada:
- OpenF1 (`sessions`, `laps`, `stints`, `weather`, `drivers`).

Salida:
- `code/backend_fastapi/data/bronze/year=<YYYY>/.../*.parquet`

### 3.2 Preprocesado
Script:
- `code/backend_fastapi/scripts/preprocess.py`

Salida principal:
- `code/backend_fastapi/data/gold/year=<YYYY>/features.parquet`
- `code/backend_fastapi/data/gold/metadata/*.parquet`

### 3.3 Entrenamiento Transformer v3
Script:
- `code/backend_fastapi/scripts/train_models.py`

Arquitectura activa: **Transformer v3 medium** (d_model=384, 8 capas, 15 features, 4 heads, ~14.3M params).

Salida:
- `code/backend_fastapi/models/driver_<code>.joblib`
- `code/backend_fastapi/models/global.joblib`

### 3.4 Entrenamiento perfil de piloto
Script:
- `code/backend_fastapi/scripts/train_profiles.py`

Salida:
- `code/backend_fastapi/models/driver_profile_<id>.joblib`
- `code/backend_fastapi/models/driver_profile_global.joblib`

---

## 4) Modelo de perfil de piloto
Archivo:
- `code/backend_fastapi/app/driver_profile.py`

El perfil representa estilo de conduccion de forma interpretable con parametros por compuesto y circuito:
- `base`: pace base
- `slope`: degradacion por vuelta
- `track_coef`: sensibilidad a temperatura de pista
- `air_coef`: sensibilidad a temperatura de aire
- `track_ref`, `air_ref`: referencias termicas

Fallbacks:
1. Perfil piloto+circuito+compuesto
2. Perfil piloto+compuesto
3. Perfil global por compuesto
4. Default numerico seguro

---

## 5) Motor de estrategia (tres motores colaborando)
Archivo:
- `code/backend_fastapi/app/strategy_engine.py`
- Documento detallado: `code/backend_fastapi/STRATEGY_CONSTRUCTION.md`

### 5.1 Fase 1 — Generacion analitica de candidatas
`_pace_table` consulta `DriverProfile` y devuelve `(pace_base, deg_rate)`
por compuesto, corregido por temperatura del contexto. Con eso,
`_candidate_strategies` resuelve la vuelta de parada optima por forma
cerrada (1-stop) o sistema lineal 2x2 (2-stop):

- `s* = ((pace_B - pace_A) + (deg_A - deg_B)/2 + deg_B * L) / (deg_A + deg_B)`

Cada candidata pasa un filtro de rentabilidad: si todas sus paradas
tienen `profit_i = (t_old - t_fresh) - pit_loss < 0`, se descarta antes
del ranking. Reglas F1: ≥1 parada, ≥2 compuestos distintos.

### 5.2 Precompute de curvas (Transformer)
Por request `(year, circuit_id, driver_id)` se generan o reutilizan
curvas de ritmo por compuesto. Estas curvas vienen del **Transformer v3**
y capturan no-linealidades (caida del neumatico al final del stint,
sensibilidad Circuit x Compound, embeddings de stint y vuelta).

Cache en disco:
- `code/backend_fastapi/cache/pace_curves/*.parquet`

### 5.3 Fase 2 — Ranking analitico (mean + lambda * var)
Para todas las candidatas filtradas:
- estimacion de esperanza de tiempo total sobre curvas del Transformer
- estimacion de varianza
- ajuste por SC y perdida de pit

### 5.4 Fase 3 — Refino Monte Carlo top-K
Solo las K mejores (`MC_TOP_K = 3`) se refinan con simulacion
estocastica del Transformer v3 (100 simulaciones batched por estrategia).

Ventaja:
- la vuelta de parada se calcula, no se busca por fuerza bruta
- el MC se ejecuta solo sobre el subconjunto ya prometedor
- distribucion realista del tiempo total, no un unico numero

### 5.5 Fase 4 — Filtro de salida
HARD-start (`compounds[0] == "HARD"`) se excluye duro del payload:
en F1 moderna es competitivamente irreal y no debe mostrarse aunque la
matematica lo permita.

---

## 6) API detallada
Archivo principal:
- `code/backend_fastapi/app/main.py`

### 6.1 Rutas estables (`/api`)
- `GET /api/metadata/seasons`
- `GET /api/metadata/circuits?season=YYYY`
- `GET /api/metadata/drivers?season=YYYY`
- `GET /api/metadata/teams?season=YYYY`
- `POST /api/strategy`
- `POST /api/compare`

### 6.2 Compatibilidad legacy
Se mantienen temporalmente:
- `/metadata/*`
- `/strategy`
- `/compare`

### 6.3 Request /strategy
```json
{
  "year": 2023,
  "circuit_id": "Sakhir",
  "driver_code": "VER",
  "risk_bias": 0.15,
  "n_strategies": 5,
  "debug_profile": false,
  "force_recompute": true
}
```

`driver_code` es el identificador principal (3 letras: VER, HAM, LEC…).

### 6.4 Response /strategy (resumen)
```json
{
  "year": 2023,
  "circuit_id": "Sakhir",
  "compute_meta": {"cache_hit": false, "elapsed_ms": 4200, "data_mode": "snapshot"},
  "context": {"total_laps": 57, "track_temp": 29.6, "air_temp": 25.9, "pit_loss": 22.5, "sc_probability": 0.2},
  "strategies": [
    {
      "type": "1-stop",
      "compounds": ["SOFT", "MEDIUM"],
      "stop_profitability": [18.3],
      "..."
    }
  ]
}
```

`stop_profitability`: ganancia neta en segundos por cada parada (positivo = parar vale la pena).

### 6.5 Errores esperables
- `400`: no hay features cargadas.
- `422`: payload invalido.
- `500`: fallo interno de simulacion/modelo.

---

## 7) Frontend
Directorio:
- `code/frontend`

### 7.1 Estructura principal
- `src/App.jsx`: estado global, tabs, home, burbuja contextual pre-race y ejecucion.
- `src/components/TopTabs.jsx`: barra superior `Home/Pre-race/Live/Rewatch/Explore`.
- `src/components/HomeLanding.jsx`: landing interactiva (entrada por defecto).
- `src/components/PreRaceContextBubble.jsx`: contexto de `Pre-race` (temporada, circuito, calcular).
- `src/components/DriverRow.jsx`: fila de piloto + visualizacion.
- `src/components/StrategyStrip.jsx`: carrusel horizontal de estrategias.
- `src/components/StrategyCurveChart.jsx`: grafico principal por stints.
- `src/components/ThemeToggle.jsx`: claro/oscuro.
- `src/styles.css`: sistema visual y responsive.

### 7.2 Estados de carga/error
Se implementan estados explicitos para metadata:
- `loading`
- `ready`
- `error`
- `empty`

La UI nunca deja selects vacios sin mensaje contextual.

### 7.3 Sistema de diseño
- Materiales sobrios, texturas sutiles.
- Bordes redondeados y sombras bajas.
- Tema claro/oscuro persistente.
- Comparacion por filas (segun sketch).
- Landing `Home` con motion moderno y estilo high-tech.

### 7.4 Navegacion y UX contextual
- `Home` es la pantalla de entrada por defecto.
- En `Pre-race` aparece una burbuja contextual junto a las tabs con:
  - temporada
  - circuito
  - boton `Calcular`
- En `Home/Live/Rewatch/Explore` esa burbuja no se renderiza.
- La zona de estrategias en `Pre-race` ocupa todo el ancho util (sin panel lateral de configuracion).

### 7.5 Responsive
- Desktop: sin scroll vertical global; scroll local en panel de filas/carruseles.
- Tablet: burbuja contextual debajo de tabs si no hay ancho suficiente.
- Mobile: flujo vertical legible.

---

## 8) Ejecucion local paso a paso

Nota de repositorio limpio:
- Este repo no versiona datasets, caches ni modelos entrenados.
- Tras clonar, hay que ejecutar ingesta/preprocesado/entrenamiento para regenerar artefactos locales.
- OpenF1 puede requerir autenticacion. Configura `code/backend_fastapi/.env` desde `code/backend_fastapi/.env.example`.

## 8.1 Backend
```bash
cd "code/backend_fastapi"
python3.11 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 8.1.1 Pipeline
```bash
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2023
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5
python -m scripts.train_profiles --min-laps 160
```

### 8.1.2 API
```bash
uvicorn app.main:app --reload --port 8000
```

## 8.2 Frontend
```bash
cd "code/frontend"
npm install
npm run build
```

Con build presente, FastAPI sirve la UI en el mismo origen (`http://localhost:8000`).

Modo alternativo dev con Vite:
```bash
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## 8.3 Demo MVP (presentacion)
Modo recomendado para demo: **snapshot local** (sin dependencia de red/OpenF1 durante la presentacion).

### Checklist preflight (2-3 min)
```bash
# 1) cortar procesos viejos
pkill -f "uvicorn app.main:app" || true
pkill -f "vite" || true

# 2) validar puertos libres
lsof -i :8000 -n -P
lsof -i :5173 -n -P

# 3) validar artefactos minimos
ls code/backend_fastapi/data/gold/year=2023/features.parquet
ls code/backend_fastapi/models/global.joblib
```

### Arranque de demo (single-origin)
```bash
cd "code/frontend"
npm run build

cd "../backend_fastapi"
source .venv_demo/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abrir: `http://127.0.0.1:8000`

### Guion rapido
1. `Home`: mostrar landing interactiva.
2. `Pre-race`: seleccionar temporada/circuito/pilotos.
3. Pulsar `Calcular`.
4. Mostrar top estrategias con curvas, stints y pit windows.
5. Caso estable de contingencia: `2023 / Sakhir / VER`.

---

## 9) Troubleshooting

### 9.1 Pantalla sin datos
Comprobar:
1. `GET /api/metadata/seasons` devuelve datos.
2. Existe `data/gold/year=.../features.parquet`.
3. No hay backend viejo en otro puerto/origen.
4. Para demo estable, usar snapshot local con `OPENF1_AUTH_ENABLED=false` en `.env`.
5. Si OpenF1 devuelve 401 en ingesta, revisar `OPENF1_USERNAME` y `OPENF1_PASSWORD`.

### 9.1.b Import freeze de stack cientifico
Si `uvicorn` no arranca o se queda colgado en imports:
1. Crear entorno nuevo (`.venv_demo`) y reinstalar dependencias.
2. Verificar import rapido:
```bash
cd "code/backend_fastapi"
.venv_demo/bin/python -c "import pandas, numpy, scipy, torch; print('ok')"
```
3. Relanzar API con ese entorno.

### 9.5 Checklist visual rapido (UI actual)
1. La app abre en `Home`.
2. La burbuja contextual solo aparece en `Pre-race`.
3. En `Pre-race`, las estrategias usan todo el ancho (sin configuracion lateral).
4. `Calcular` sigue siendo el unico trigger de simulacion.

### 9.2 Puerto ocupado
```bash
lsof -i :8000 -n -P
pkill -f "uvicorn app.main:app"
```

### 9.3 CORS/origen
Preferir single-origin (`frontend/dist` servido por FastAPI). Evita discrepancias `localhost` vs `localhost:5173`.

### 9.4 Falta de modelos/perfiles
Reejecutar:
```bash
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5
python -m scripts.train_profiles --min-laps 160
```

### 9.5 Cache inconsistente
Borrar cache local:
- `code/backend_fastapi/cache/pace_curves/*`
- Si hay snapshot local y fallo online, `compute_meta.stale_data=true` indicara que la API sirve ultimo estado persistido.

---

## 10) Benchmark y rendimiento
Reporte actual:
- `code/backend_fastapi/benchmark_report.json`

Interpretacion:
- `cold`: primer request
- `warm`: requests iniciales con caches en proceso
- `hot`: estado estabilizado con caches activas

Para benchmark local:
```bash
cd "code/backend_fastapi"
.venv_demo/bin/python scripts/benchmark_strategy.py
```

---

## 11) Decisiones tecnicas y tradeoffs
1. **Analitico + MC top-K**
   - reduce coste sin perder utilidad de ranking.
2. **Transformer v3 medium + perfil parametrico**
   - el Transformer memoriza la curva real de degradacion por piloto/circuito/compuesto; el perfil parametrico actua como fallback interpretable.
3. **stop_profitability explicito**
   - calculo marginal por parada evita sesgo hacia 2 paradas cuando el desgaste real es bajo.
4. **Single-origin / Vite proxy**
   - elimina clase de errores frecuentes de CORS: en dev el proxy de Vite reenvía `/api` a `:8000`; en demo FastAPI sirve `dist/` directamente.
5. **Compatibilidad legacy temporal**
   - evita romper clientes existentes durante migracion a `/api`.

---

## 12) Roadmap
1. `Live`: ingest incremental y replan en carrera.
2. `Rewatch`: timeline completo de eventos y decisiones.
3. `Explore`: comparativas historicas avanzadas.
4. `Queue/worker` para simulaciones concurrentes mas pesadas.
5. versionado de modelos y A/B de ranking.

---

## Smoke test recomendado
Caso base estable:
- `year=2023`
- `circuit_id=Sakhir`
- `driver_code=VER`

```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_code":"VER"}' | python -m json.tool | head -40
```
