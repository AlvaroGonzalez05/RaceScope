# Data Engineering para F1 y Analítica Deportiva

## Relevancia para RaceScope Strategy Lab

RaceScope implementa un pipeline completo de data engineering: ingestión desde OpenF1 API (con token bucket rate limiting, exponential backoff, y caché HTTP), almacenamiento en Parquet, feature engineering en `preprocess.py`, y una feature store simplificada en `data_store.py` con LRU cache. El backend FastAPI sirve la API de estrategia en tiempo real.

---

## 1. FUENTES DE DATOS DE F1

### 1.1 OpenF1 API — Fuente Principal de RaceScope

**OpenF1 Project (2023-presente). OpenF1: The Open Source API for Formula 1 Data.**
- **GitHub:** https://github.com/br-g/openf1
- **Documentación:** https://openf1.org/
- **Rate limits:** 3 req/s (free), 6 req/s burst / 60 req/min (premium/supporters)

Proyecto open-source impulsado por la comunidad que proporciona acceso completo a datos de Fórmula 1 desde la temporada 2023 en adelante. Ofrece más de 18 endpoints que cubren timing de vueltas, telemetría, información de pilotos, mensajes del control de carrera, meteorología y datos de circuito. Implementa limitación de tasa token-bucket para garantizar equidad en el acceso a la API. Powers muchos proyectos de análisis F1.

**Relevancia:** Fuente primaria de datos de RaceScope. El cliente en `app/ingest.py` implementa exactamente el patrón de token bucket documentado en la API para los límites de la cuenta premium. Los endpoints usados incluyen laps, stints, sessions, drivers y weather.

---

### 1.2 Ergast F1 API (Histórica, 1950-2024)

**Ergast Motor Racing Database (2006-2024). Ergast F1 API.**
- **URL:** https://ergast.com/mrd/
- **Nota:** Servicio deprecado a finales de 2024, supersedido por Jolpica-f1

La base de datos Ergast proporcionó datos históricos de F1 desde 1950 a través de una API REST con formatos XML/JSON/CSV. Fue la fuente de referencia estándar para investigación académica de F1 durante casi dos décadas. Demuestra el diseño de APIs REST para datos deportivos.

**Relevancia:** Referencia histórica importante para investigación académica de F1. Aunque RaceScope usa OpenF1 (datos desde 2023), Ergast es la fuente histórica que muchos papers de referencia usaron para entrenar modelos. La migración de Ergast a OpenF1/Jolpica refleja la evolución del ecosistema de datos de F1.

---

### 1.3 FastF1 Python Library

**Oehrly, T. (2021-presente). FastF1: Python Package for F1 Data Analysis.**
- **Documentación:** https://docs.fastf1.dev/
- **GitHub:** https://github.com/theOehrly/Fast-F1
- **PyPI:** fastf1

Wrapper Python de alta calidad alrededor de las APIs de telemetría F1 (usadas por las transmisiones de TV de F1) con caché automático para reducir las peticiones repetidas. Extrae datos de timing, telemetría (velocidad, DRS, RPM, throttle) y posición desde 2018 a través de DataFrames Pandas extendidos. Implementa caché automático y funciones personalizadas para análisis F1 académico y hobbyista.

**Relevancia:** Librería complementaria ampliamente usada en investigación académica de F1. Muchos de los papers de referencia en este TFG usan FastF1 como fuente de datos. Proporciona telemetría más detallada que OpenF1 para ciertos casos de uso.

---

## 2. ALMACENAMIENTO COLUMNAR — APACHE PARQUET

### 2.1 Dremel: Interactive Analysis of Web-Scale Datasets

**Melnik, S., et al. (2010). Dremel: Interactive Analysis of Web-Scale Datasets.**
- **Venue:** Proceedings of the VLDB Endowment, Vol. 3, Issue 1-2
- **URL:** https://research.google/pubs/dremel-interactive-analysis-of-web-scale-datasets/

Paper fundacional de Google que introduce el modelo de datos columnar anidado que inspiró Apache Parquet. Dremel usa niveles de definición y repetición para codificar datos anidados en formato columnar, logrando 100x mejora sobre MapReduce para consultas analíticas sobre terabytes de datos. El formato columnar permite "saltar" columnas no consultadas, reduciendo dramáticamente la I/O.

**Relevancia:** Base teórica del formato Parquet usado en RaceScope para `data/raw/` y `data/features/`. La partición por año (`year=<YYYY>`) en RaceScope implementa exactamente el patrón de partición columnar de Dremel para consultas eficientes.

---

### 2.2 Apache Parquet — Formato de Almacenamiento Columnar

**Apache Software Foundation (2013-presente). Apache Parquet Columnar Storage Format.**
- **URL:** https://parquet.apache.org/
- **Especificación:** https://parquet.apache.org/docs/file-format/

Apache Parquet es un formato de almacenamiento columnar open-source inspirado en Dremel. Usa codificación de tipo por columna (dictionary encoding, run-length encoding, bit packing) y compresión (Snappy, GZIP, LZ4). Proporciona compresión 10-100x frente a formatos de fila (CSV) y es el formato estándar de facto para data lakes y plataformas de analítica (Spark, Pandas, DuckDB).

**Relevancia:** Formato de almacenamiento principal de RaceScope. Los datos de OpenF1 se ingestan y almacenan en Parquet para lectura eficiente durante el entrenamiento de modelos y la generación de features. El uso de `pyarrow`/`pandas` para leer Parquet en `data_store.py` es el patrón estándar.

---

## 3. ARQUITECTURAS DE FEATURE STORES Y DATA PIPELINES

### 3.1 Feature Stores in Production Machine Learning

**Feast Documentation (2021-presente). Feast: Open Source Feature Store.**
- **URL:** https://feast.dev/
- **Whitepaper:** https://feast.dev/blog/what-is-a-feature-store/

Un feature store es la capa de infraestructura ML que centraliza la definición, almacenamiento y servido de features. Separa capas offline (entrenamiento en batch) y online (servido de baja latencia). Previene el "training/serving skew" — cuando las features en producción difieren de las usadas durante el entrenamiento. Feast es la solución open-source más popular.

**Relevancia:** RaceScope implementa un feature store simplificado en `data_store.py` (LRU cache sobre lecturas de Parquet) y una caché de curvas de pace en `cache/pace_curves/`. La arquitectura offline/online de los feature stores formales es el patrón que RaceScope implementa pragmáticamente.

---

### 3.2 The Data Lakehouse: A New Generation of Data Platforms

**Armbrust, M., et al. (2021). Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics.**
- **Venue:** CIDR 2021
- **URL:** https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf

Introduce el concepto de "data lakehouse" que combina la flexibilidad de los data lakes (almacenamiento de objetos barato, formatos abiertos como Parquet) con las capacidades de gestión de datos de los data warehouses (transacciones ACID, control de versiones, esquema). Delta Lake, Iceberg y Hudi son las implementaciones principales.

**Relevancia:** La arquitectura de datos de RaceScope (`data/raw/year=<YYYY>/` + `data/features/year=<YYYY>/`) implementa un patrón de data lakehouse simplificado usando Parquet con partición por año. El `snapshot_state.json` actúa como un registro de transacciones rudimentario.

---

## 4. RATE LIMITING Y ARQUITECTURAS DE APIs

### 4.1 Token Bucket Algorithm para Rate Limiting

**Varios autores (1986-presente). Token Bucket Algorithm.**
- **Wikipedia:** https://en.wikipedia.org/wiki/Token_bucket
- **RFC 2698** (Two Rate Three Color Marker): https://tools.ietf.org/html/rfc2698

El Token Bucket es el algoritmo estándar para limitación de tasa en APIs y sistemas de red. Define un "cubo" de capacidad fija que se rellena a tasa constante con tokens; las peticiones se sirven si hay tokens disponibles y se bloquean/descartan si el cubo está vacío. Soporta tráfico en ráfaga (burst) mientras se mantiene el ancho de banda promedio.

**Relevancia:** Implementado en `app/openf1_client.py` de RaceScope para respetar los límites de la API de OpenF1 (6 req/s burst, 60 req/min sostenido). El cubo de tokens garantiza que las peticiones respeten los límites del plan premium sin sobrecargar la API.

---

### 4.2 Exponential Backoff for Retrying Failed Requests

**Amazon Web Services (2016). Error Retries and Exponential Backoff in AWS.**
- **URL:** https://docs.aws.amazon.com/general/latest/gr/api-retries.html
- **Implementación de referencia:** Google API Design Guide: https://cloud.google.com/apis/design/errors

El backoff exponencial es el patrón estándar para reintentos de peticiones fallidas en sistemas distribuidos. El tiempo de espera entre reintentos se dobla exponencialmente (1s, 2s, 4s, 8s...) con jitter aleatorio para evitar "thunder herds". Evita sobrecargar un servicio en recuperación y es el patrón recomendado por todos los grandes proveedores de cloud.

**Relevancia:** Implementado en `app/openf1_client.py` de RaceScope para manejar errores transitorios de la API OpenF1 durante la ingestión de datos. Garantiza resiliencia del pipeline de datos ante fallos temporales de la API.

---

## 5. APIS WEB Y ARQUITECTURA BACKEND

### 5.1 FastAPI — Framework Web Moderno para Python

**Ramírez, S. (2018-presente). FastAPI: High-Performance ASGI Web Framework.**
- **Documentación oficial:** https://fastapi.tiangolo.com/
- **GitHub:** https://github.com/tiangolo/fastapi
- **Stars:** >75k (uno de los proyectos Python más populares en GitHub)

FastAPI es un framework web ASGI moderno (2018+) que usa anotaciones de tipos Python para validación automática de requests, generación de documentación OpenAPI/Swagger, y serialización/deserialización de datos con Pydantic. Construido sobre Starlette (routing/middleware) y Pydantic (validación de datos). Soporta WebSockets, concurrencia nativa con async/await.

**Relevancia:** Backend de RaceScope construido enteramente sobre FastAPI con Uvicorn como servidor ASGI. Las rutas `/api/strategy`, `/api/compare` y `/api/metadata/*` están implementadas como endpoints FastAPI con modelos Pydantic para validación de requests y responses.

---

### 5.2 ASGI: Asynchronous Server Gateway Interface

**Django/Python Community (2018-presente). ASGI Specification.**
- **Especificación:** https://asgi.readthedocs.io/
- **Uvicorn:** https://www.uvicorn.org/

ASGI es el estándar Python para manejo asíncrono de requests/responses. Contrasta con el WSGI síncrono (Flask, Django WSGI) al habilitar I/O no bloqueante, WebSockets y utilización eficiente de recursos para APIs de alta concurrencia. Uvicorn es el servidor ASGI de referencia para FastAPI.

**Relevancia:** RaceScope corre sobre Uvicorn (servidor ASGI), permitiendo el manejo de múltiples peticiones de estrategia concurrentes sin bloquear el event loop. Crítico para el rendimiento cuando múltiples usuarios calculan estrategias simultáneamente.

---

### 5.3 REST API Design Best Practices

**Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures (Chapter 5: REST).**
- **URL:** https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm

La disertación doctoral que define REST (Representational State Transfer). Establece las restricciones fundamentales: sin estado del cliente en el servidor, interfaz uniforme, caché, sistema en capas. El estilo arquitectónico REST es la base de las APIs web modernas.

**Relevancia:** Las rutas `/api/strategy`, `/api/compare` y `/api/metadata/*` de RaceScope siguen los principios REST. La presencia de rutas legacy (sin prefijo `/api/`) en el código es exactamente el tipo de violación de principios REST que Fielding documenta como problemática.

---

## 6. CACHING PARA ML EN PRODUCCIÓN

### 6.1 LRU Cache (Least Recently Used)

**Sleator, D., & Tarjan, R. (1985). Amortized Efficiency of List Update and Paging Rules.**
- **Venue:** Communications of the ACM, Vol. 28, Issue 2

El caché LRU es la política de reemplazo de caché más común: descarta el elemento accedido menos recientemente cuando el caché está lleno. Provably óptimo offline. Python implementa `functools.lru_cache` para memoización de funciones.

**Relevancia:** RaceScope usa `@lru_cache` extensivamente: en `_load_model_cached()`, `_load_pace_curves_cached()`, `_predict_stint_cached()` en `strategy_engine.py`, y en `load_features()` y `load_metadata()` en `data_store.py`. Previene la recarga de modelos LSTM (~50MB cada uno) en cada petición.

---

### 6.2 Disk-Based Caching con TTL para Curvas de Pace

La caché de curvas de pace en `cache/pace_curves/` de RaceScope implementa el patrón estándar de caché de dos niveles:
- **L1 (memoria):** LRU cache sobre modelos cargados
- **L2 (disco):** Parquet files en `cache/pace_curves/` con TTL de 24h

Este patrón es estándar en sistemas de ML en producción donde la generación de features es costosa (llamadas al modelo LSTM) pero los inputs cambian lentamente (las condiciones de carrera no cambian durante el día de preparación).

---

## 7. INGENIERÍA DE FEATURES DEPORTIVAS

### 7.1 Feature Engineering para Series Temporales Deportivas

**Literatura estándar de ML/Sports Analytics (2020-2024).**

Las features de ingeniería más importantes para datos de carrera F1 incluyen:

| Feature | Tipo | Técnica | Uso en RaceScope |
|---|---|---|---|
| `lap_number` | Numérico | Normalización ordinal | Input LSTM directo |
| `stint_age` | Numérico | Conteo desde último pit | Parámetro de degradación |
| `compound` | Categórico ordinal | Codificación ordinal (1/2/3) | Input LSTM + perfil |
| `track_temp` | Continuo | Centrado en referencia (30°C) | Coeficiente paramétrico |
| `air_temp` | Continuo | Centrado en referencia (22°C) | Coeficiente paramétrico |
| `lap_time` | Serie temporal | Z-score normalización | Target + feature LSTM |
| `session_type` | Categórico | Label encoding | Contexto de entrenamiento |
| `circuit_id` | Categórico | Normalización canónica | Segmentación de modelo |

**Relevancia:** La ingeniería de features de RaceScope sigue prácticas estándar documentadas en la literatura de ML aplicado a datos deportivos.

---

### 7.2 Normalización Canónica de IDs de Circuito

El mapa `CIRCUIT_ID_CANONICAL` en `preprocess.py` de RaceScope normaliza los múltiples aliases de nombres de circuito que aparecen en los datos de OpenF1. Este es un problema estándar en data engineering deportivo: las fuentes de datos externas usan nombres inconsistentes para las mismas entidades.

Ejemplo de literatura que documenta este problema:
- "Bahrain" vs "Sakhir" vs "Bahrain International Circuit"
- "Monaco" vs "Monte Carlo" vs "Circuit de Monaco"

La normalización canónica es un paso crítico de data quality previo a cualquier análisis.

---

## 8. ARQUITECTURA FRONTEND

### 8.1 React: Librería UI Basada en Componentes

**Facebook/Meta (2013-presente). React: A JavaScript Library for Building User Interfaces.**
- **URL:** https://react.dev/
- **Versión actual:** React 19 (2024)

React introduce el concepto de componentes reutilizables con estado local y el Virtual DOM para actualizaciones eficientes de la UI. El modelo de programación declarativo (describe qué mostrar, no cómo actualizar) simplifica la gestión de estado complejo en SPAs.

**Relevancia:** El frontend de RaceScope (`code/frontend/`) está construido sobre React. Los componentes `StrategyStrip`, `DriverRow` y `StrategyCurveChart` siguen el paradigma de componentes de React.

---

### 8.2 Vite: Build Tool de Nueva Generación

**Evans, E. (2020-presente). Vite: Next Generation Frontend Tooling.**
- **URL:** https://vitejs.dev/
- **GitHub:** https://github.com/vitejs/vite

Vite usa módulos ES nativos del navegador durante el desarrollo (no bundling), eliminando el tiempo de compilación de Webpack. El Hot Module Replacement (HMR) es instantáneo porque Vite actualiza solo los módulos modificados. En producción, usa Rollup para bundle optimizado.

**Relevancia:** Build tool del frontend de RaceScope (`npm run dev` usa el servidor de desarrollo de Vite; `npm run build` genera `frontend/dist/` servido por FastAPI). La elección de Vite sobre Create React App refleja las mejores prácticas actuales del ecosistema React.

---

### 8.3 Recharts: Librería de Visualización para React

**Recharts (2015-presente). Recharts: A Composable Charting Library Built on React Components.**
- **URL:** https://recharts.org/
- **GitHub:** https://github.com/recharts/recharts

Recharts es una librería de visualización basada en D3.js y React para gráficos composables. Usa componentes React para definir gráficos declarativamente (LineChart, AreaChart, etc.). Integración nativa con el ciclo de vida de React y gestión de estado.

**Relevancia:** El componente `StrategyCurveChart.jsx` de RaceScope usa Recharts para renderizar curvas de tiempo de vuelta por stint. La integración nativa con React permite actualizar los gráficos reactivamente cuando cambia la estrategia seleccionada.

---

## 9. SÍNTESIS: STACK TECNOLÓGICO DE RACESCOPE

| Capa | Tecnología | Justificación en Literatura |
|---|---|---|
| **API de datos** | OpenF1 API | Única API F1 open-source gratuita con datos 2023+ |
| **Almacenamiento** | Apache Parquet | Formato columnar estándar para analítica (Dremel paper) |
| **Feature Engineering** | Pandas + PyArrow | Librería estándar de Python para análisis de datos tabulares |
| **Rate Limiting** | Token Bucket | Algoritmo estándar de RFC 2698 |
| **Retry Logic** | Exponential Backoff | Patrón estándar de AWS/Google Cloud |
| **Backend** | FastAPI + Uvicorn | Framework Python ASGI más popular para ML APIs |
| **Caché L1** | LRU Cache (Python) | Política óptima de reemplazo de caché |
| **Caché L2** | Parquet + TTL 24h | Persistencia eficiente de features precalculadas |
| **ML** | PyTorch + LSTM | Framework DL estándar, arquitectura proven para series temporales |
| **Frontend** | React + Vite | Stack web moderno, componentes declarativos |
| **Visualización** | Recharts | Librería de gráficos React con soporte nativo |
