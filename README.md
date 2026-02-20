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
1. Ingesta OpenF1 y guardado en parquet.
2. Preprocesado y feature store.
3. Entrenamiento de modelos (LSTM por piloto + perfil parametrico).
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
- `code/backend_fastapi/data/raw/year=<YYYY>/.../*.parquet`

### 3.2 Preprocesado
Script:
- `code/backend_fastapi/scripts/preprocess.py`

Salida principal:
- `code/backend_fastapi/data/features/year=<YYYY>/features.parquet`
- `code/backend_fastapi/data/features/metadata/*.parquet`

### 3.3 Entrenamiento LSTM
Script:
- `code/backend_fastapi/scripts/train_models.py`

Salida:
- `code/backend_fastapi/models/driver_<id>.joblib`
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

## 5) Motor de estrategia (analitico + MC top-K)
Archivo:
- `code/backend_fastapi/app/strategy_engine.py`

### 5.1 Precompute de curvas
Por request `(year, circuit_id, driver_id)` se generan o reutilizan curvas de ritmo por compuesto.

Cache en disco:
- `code/backend_fastapi/cache/pace_curves/*.parquet`

### 5.2 Fase A: evaluacion analitica
Para todas las estrategias candidatas:
- estimacion de esperanza de tiempo total
- estimacion de varianza
- ajuste por SC y perdida de pit

### 5.3 Fase B: refino Monte Carlo top-K
Solo las K mejores (`MC_TOP_K`) se refinan con simulacion estocastica.

Ventaja:
- casi la misma calidad de ranking
- coste computacional muy inferior a full-MC para todas

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
  "driver_id": 14,
  "risk_bias": 0.15,
  "n_strategies": 5,
  "debug_profile": false
}
```

### 6.4 Response /strategy (resumen)
```json
{
  "year": 2023,
  "circuit_id": "Sakhir",
  "driver_id": 14,
  "context": {"total_laps": 57, "track_temp": 29.6, "air_temp": 25.9, "pit_loss": 22.5, "sc_probability": 0.2},
  "strategies": [...],
  "degradation": {...}
}
```

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

## 8.1 Backend
```bash
cd "code/backend_fastapi"
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 8.1.1 Pipeline
```bash
python -m scripts.ingest_season --year 2023
python -m scripts.preprocess --year 2023
python -m scripts.train_models --min-laps 200 --epochs 8
python -m scripts.train_profiles --min-laps 120
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

## 9) Troubleshooting

### 9.1 Pantalla sin datos
Comprobar:
1. `GET /api/metadata/seasons` devuelve datos.
2. Existe `data/features/year=.../features.parquet`.
3. No hay backend viejo en otro puerto/origen.

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
python -m scripts.train_models --min-laps 200 --epochs 8
python -m scripts.train_profiles --min-laps 120
```

### 9.5 Cache inconsistente
Borrar cache local:
- `code/backend_fastapi/cache/pace_curves/*`

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
.venv311/bin/python scripts/benchmark_strategy.py
```

---

## 11) Decisiones tecnicas y tradeoffs
1. **Analitico + MC top-K**
   - reduce coste sin perder utilidad de ranking.
2. **Perfil parametrico + LSTM**
   - equilibrio entre interpretabilidad y capacidad predictiva.
3. **Single-origin**
   - elimina clase de errores frecuentes de CORS y origen cruzado.
4. **Compatibilidad legacy temporal**
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
- `driver_id=14`

```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}'
```
