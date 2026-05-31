# Construcción de estrategias en RaceScope

Documento de análisis del módulo `app/strategy_engine.py`. Describe
cómo RaceScope genera, evalúa y ordena las estrategias de parada que
devuelve el endpoint `POST /api/strategy`.

El motor sigue una arquitectura de **tres motores colaborando**:

1. **DriverProfile lineal** — decide *cuándo parar* por cálculo
   analítico de primer principio.
2. **Transformer v3** — predice ritmo con no-linealidades y refina el
   Top-K con Monte Carlo.
3. **Heurística mean-variance** — ordena las candidatas filtradas con
   `mean + λ·var`.

El flujo completo se ejecuta dentro de
`StrategyEngine.generate_strategies` (`strategy_engine.py`). Por encima
hay un bloque de **derivación de contexto** que fija los parámetros
físicos de la carrera; por debajo, un bloque de **post-procesado y
ranking** que aplica sesgos suaves, deduplica, filtra HARD-start y
construye la respuesta.

---

## 1. Contexto de carrera

Antes de generar nada, el motor reconstruye el contexto
físico-deportivo de la combinación `(year, circuit_id)` a partir del
feature store (`data/gold/year=<YYYY>/features.parquet`). El resultado
es un `RaceContext` con seis campos:

| Campo | Origen | Fallback |
|---|---|---|
| `total_laps` | `lap_number.max()` sobre sesión `RACE` | 55 |
| `track_temp` | media de `track_temp` sobre la combinación | 30.0 |
| `air_temp` | media de `air_temp` sobre la combinación | 22.0 |
| `pit_loss` | `_derive_pit_loss` (mediana histórica de outlaps) | `PIT_LOSS_FALLBACK = 22.5` |
| `sc_probability` | `_derive_sc_probability` (anomalías de ritmo) | `SC_PROBABILITY_FALLBACK = 0.20` |
| `year` | argumento de entrada | — |

### 1.1 `_derive_pit_loss`

Estima el tiempo perdido en una parada agrupando los stints históricos
de cada piloto y midiendo el delta entre el outlap (`stint_age == 1`) y
la mediana del stint. Aplica un criterio de cordura `5 ≤ delta ≤ 60 s`
y exige al menos 5 muestras válidas; si no las hay, recurre al fallback.
El resultado se recorta a **`[PIT_LOSS_MIN = 15.0, PIT_LOSS_MAX = 45.0]`**
— límites realistas: una parada en F1 rara vez supera los 45 s y nunca
baja de 15 s.

### 1.2 `_derive_sc_probability`

Detecta sesiones con coche de seguridad mirando rachas de ≥3 vueltas
consecutivas en las que la mediana del paddock excede `1.35 × mediana
de carrera`. La probabilidad final es `sc_sessions / total_sessions`,
recortada al rango `[SC_PROBABILITY_MIN, SC_PROBABILITY_MAX]`. Exige al
menos dos sesiones históricas.

---

## 2. Tabla de pace por compuesto (`_pace_table`)

Aquí entra el **motor analítico de primer principio**. Para cada
compuesto `{SOFT, MEDIUM, HARD}` el motor obtiene un par
`(pace_base, deg_rate)`:

```python
pace_base = params.base
          + params.track_coef * (context.track_temp - params.track_ref)
          + params.air_coef   * (context.air_temp   - params.air_ref)
deg_rate  = max(params.slope, 0.0)
```

Donde `params` viene de `resolve_profile_params(profile, circuit_id,
compound)` (`app/driver_profile.py`), con **4 niveles de fallback**:

1. `(driver, circuit, compound)` específico
2. `driver_defaults[compound]`
3. `global_defaults[compound]`
4. `ProfileParams(base=90.0, slope=0.05, …)`

El profile es lineal: `lap_time(stint_age) = base + slope · stint_age`.
Lineal es suficiente para decidir la **vuelta de parada óptima**, que
es la decisión clave de esta fase.

---

## 3. Cota de vida de neumático

`_tyre_life_bounds` define, por compuesto y circuito, el rango
`(min_stint_len, max_stint_len)` físicamente viable. La construcción:

1. Calcula `stint_age.max()` por `(driver_id, session_key,
   stint_number, compound)`.
2. Toma los cuantiles 0.2 y 0.8 → `(q1, q9)`.
3. Aplica un **suelo de carrera** (`_RACE_MAX_FLOOR`): SOFT 25,
   MEDIUM 35, HARD 45 vueltas.

El suelo es deliberado: los datos de prácticas tienden a ser stints
cortos, así que el cuantil 0.8 observado puede ser muy inferior al
máximo real de carrera. Sin él, **no se generarían estrategias SOFT-start
largas** porque la matemática no encontraría hueco.

---

## 4. Generación analítica de candidatas

`_candidate_strategies` ya no enumera por combinatoria ciega — **resuelve
analíticamente la vuelta de parada óptima** para cada combinación de
compuestos.

### 4.1 Modelo del tiempo total — 1-stop

Para una estrategia A→B con `stop_lap = s` sobre `L` vueltas:

```
T(s) = Σ_{i=0..s-1} (pace_A + deg_A · i)
     + pit_loss
     + Σ_{j=0..L-s-1} (pace_B + deg_B · j)
```

Derivando `dT/ds = 0` (forma cerrada):

```
s* = ((pace_B − pace_A) + (deg_A − deg_B)/2 + deg_B · L) / (deg_A + deg_B)
```

`s*` se redondea, se clampa a las cotas físicas del compuesto y al
mínimo de 5 vueltas por stint. Si `deg_A + deg_B ≈ 0` (sin
degradación), la función es lineal y se usa el centro de la ventana
viable.

### 4.2 Modelo 2-stop

Para A→B→C con stops `(s1, s2)`, las ecuaciones `∂T/∂s1 = 0` y
`∂T/∂s2 = 0` dan un sistema lineal 2×2:

```
[deg_A + deg_B,    -deg_B    ] [s1]   [(pace_B − pace_A) + (deg_A − deg_B)/2          ]
[   -deg_B,     deg_B + deg_C] [s2] = [(pace_C − pace_B) + (deg_B − deg_C)/2 + deg_C·L]
```

Resuelto con `np.linalg.solve`. Si el determinante es ~0, se usa
reparto uniforme `(L/3, 2L/3)` como fallback.

### 4.3 Filtros físicos

Antes de aceptar una candidata:

- `stint_length ≥ 5` vueltas (descartar stints triviales).
- `stint_length ≤` cota superior del compuesto.
- Al menos 1 parada (siempre garantizado: no se generan 0-stops).
- Al menos 2 compuestos distintos en la estrategia (regla F1; descarta
  A-A, A-A-A).

### 4.4 Filtro de rentabilidad

Cada parada se evalúa con el principio del usuario:

```
profit_i = (tiempo_con_viejo − tiempo_con_fresco) − pit_loss
```

donde "tiempo" es la suma de las `remaining` vueltas del stint
siguiente, asumiendo que el neumático viejo arrastra ya
`stint_so_far` vueltas de degradación acumulada. Si **todos** los
`profit_i` son negativos, la candidata se descarta antes del ranking.
Esto promueve `_stop_profitability` de decorador del payload a filtro
de generación.

---

## 5. Curvas de ritmo del Transformer

Paralelamente a la generación de candidatas, el motor precomputa las
**curvas de tiempos por vuelta** que el piloto haría rodando con cada
compuesto desde la vuelta 1. Aquí entra el **Transformer v3** (el motor
de pace de alta fidelidad):

1. Carga el `DriverProfile` y genera una semilla paramétrica (lineal).
2. Pasa esa semilla a `TransformerPaceModel.predict_stint`, que
   devuelve la curva con no-linealidades capturadas (caída al final del
   stint, sensibilidad cruzada Circuit × Compound, embeddings de stint
   y vuelta de carrera).

El resultado se cachea en disco como Parquet en
`cache/pace_curves/<year>_<circuit>_<driver>_<Ttrack>_<Tair>.parquet`,
TTL 24 h. Invalidación: `force_recompute: true` en el endpoint.

Estas curvas son la **única fuente de tiempos por vuelta** que ve el
frontend (`stint_curves` en el payload) y la entrada del scoring
analítico de la sección 6.

---

## 6. Pre-scoring analítico

`_analytical_eval` recorre los stints del candidato y construye un par
`(mean, var)` del tiempo total esperado. Los términos son aditivos:

| Término | Forma | Significado |
|---|---|---|
| Ritmo puro | `Σ curves[c][:stint_len]` | suma de vueltas previstas (Transformer) |
| Tráfico | `μ · stint_len` con `σ²·stint_len` | 0.15 s/vuelta media, 0.05 s/vuelta σ |
| Pit loss | mezcla `(1−p_sc)·normal + p_sc·reducido` | reducido = `max(12, pit_loss − 8)` |
| SC global | `p_sc · 15 s` | coste medio adicional por SC con varianza |

Función objetivo:

```
score = mean + risk_bias · var
```

con `risk_bias = DEFAULT_RISK_LAMBDA = 0.15`. Trade-off mean-variance:
a mayor `risk_bias`, más penaliza las estrategias volátiles.

---

## 7. Refinamiento Monte Carlo (Top-K)

Sólo las `MC_TOP_K = 3` mejores pasan por `_simulate_strategy`. Aquí el
Transformer despliega su capacidad estocástica completa.

### 7.1 Sucesos de carrera (vectorizados)

- `sc_events ∼ Bernoulli(sc_probability)` por simulación.
- `sc_laps ∼ Uniform[5, total_laps − 5]`.
- `pit_loss` reducido a 12 s mínimo (−8 s) cuando el SC cae a ≤2
  vueltas de un stop programado.

### 7.2 Rollout por modelo

Dispatch por tipo de modelo cargado:

- **Transformer v3** (camino activo). Rollout **batched** de 100
  simulaciones por stint. Para cada simulación se muestrea de la
  `PracticeDistribution` del piloto: intención de ritmo (`phase_a`,
  `phase_b` normalizadas), temperaturas. Contexto `(T, 15)` con
  `stint_number_norm`. El embedding Circuit × Compound actúa como gate
  multiplicativo en la rama de degradación. Salida: `(n_sim, stint_len)`
  tiempos por vuelta. Se suma y se añade ruido de tráfico
  N(`μ·L`, `σ²·L`).
- **Transformer v2.** Igual flujo, contexto `(T, 14)`.
- **Transformer v1 / LSTM.** Caminos legacy mantenidos para
  compatibilidad.

Acumulación final:

```
totals += pit_loss · n_stops
totals += np.where(sc_events, 15, 0)
```

Devuelve `(mean_MC, var_MC, totals_list)`. El score refinado vuelve a
aplicarse con la misma fórmula `mean + λ·var`.

---

## 8. Post-procesado y ranking final

Sobre la lista completa (todas las candidatas; las Top-K llevan
`mean`/`var` refinados), `generate_strategies` aplica filtros en orden:

### 8.1 Sesgo suave sobre primer compuesto

```python
if first_compound == "MEDIUM":
    score += 1.0   # nudge suave (SOFT preferida por defecto)
```

HARD-start **no** se penaliza en el score: se filtra duro a la salida
(8.3).

### 8.2 Sesgo vs rival (modo compare)

Si se pasa `opponent_code`, se calcula la mejor `mean` analítica del
rival y se penaliza al principal por la diferencia:

```python
if mean > opponent_best:
    score += (mean − opponent_best) · 0.25
```

### 8.3 Filtro duro de HARD-start

```python
if candidate.compounds[0].upper() == "HARD":
    continue  # nunca llega al payload
```

Decisión de producto: en F1 moderna salir en duro es competitivamente
irreal. Aunque la matemática lo permita, no se muestra al usuario.

### 8.4 Deduplicación por ventana de parada

`_cluster_key` discretiza los `stop_laps` en bins de
`PIT_WINDOW_BIN = 5` vueltas. Si una estrategia comparte clave con otra
ya aceptada, se descarta. Evita devolver cinco variantes de "parar en
la 18-20".

### 8.5 Salida por estrategia

Cada estrategia final lleva:

- `strategy_id`: hash SHA-1 de `(year, circuit, driver, type, compounds,
  stints, stop_laps, pit_windows)` truncado a 16 caracteres.
- `expected_time`, `variance`, `risk_score`.
- `stint_curves`: por stint, `lap_time_data` (curva Transformer) y
  `tyre_life_data` (vida normalizada 0-100% sobre la curva monotónica).
- `stop_profitability`: lista de Δ por stop, calculada con las curvas
  del Transformer (no con la tabla lineal; las dos coinciden en el
  signo pero la del Transformer es más fiel).

---

## 9. Reparto de motores

| Tarea | Motor | Por qué |
|---|---|---|
| Decidir vuelta de parada | `DriverProfile` lineal | Forma cerrada, mil candidatas en ms |
| Filtrar candidatas no rentables | `DriverProfile` lineal | Decisión de primer principio |
| Curvas mostradas al usuario | Transformer v3 | Pace de alta fidelidad con no-linealidades |
| Ranking inicial (todas) | Heurística `mean + λ·var` sobre curvas Transformer | Ordenar barato |
| Refinamiento Top-3 | Monte Carlo con Transformer v3 | Incertidumbre realista, distribución no puntual |
| Filtro HARD-start | Regla de salida | Producto: no mostrar lo irreal |

---

## 10. Flujo completo

```
generate_strategies(year, circuit, driver)
├── _context                      → RaceContext (laps, T_track, T_air, pit_loss, p_sc)
├── _compound_stats               → base/slope por compuesto (datos crudos, fallback)
├── _tyre_life_bounds             → (min, max) por compuesto + suelos de carrera
├── _pace_table                   → {compound: (pace_base, deg_rate)} corregido por T
├── _load_model                   → Transformer v3 (con fallback a v2/v1/LSTM)
├── _precompute_pace_curves       → curvas por compuesto (Transformer) + cache
├── _candidate_strategies         → resolver s* analítico, filtrar por profit_i ≥ 0
├── opponent (opcional)           → opp_pace_table + opp_curves → opponent_best
├── for each candidate:
│     _analytical_eval            → mean_an, var_an
│     score = mean_an + λ·var_an + sesgo_medium + sesgo_rival
├── ordenar por score
├── Top-3 → _simulate_strategy → mean_MC, var_MC → re-score
├── ordenar candidato completo con scores refinados
├── filtro: cluster_key + HARD-start excluido
└── empaquetar respuesta (fingerprint, stint_curves, stop_profitability,
                          contexto, degradación)
```

---

## 11. Puntos sensibles para la memoria

- El profile asume **degradación lineal**. El Transformer corrige eso
  en las curvas y en el MC; documentarlo como "lineal donde basta,
  no-lineal donde importa".
- El **suelo de carrera** (`_RACE_MAX_FLOOR`) sigue siendo una decisión
  de ingeniería, no aprendida.
- El **filtro HARD-start** es una regla de producto, no del modelo.
- Las cotas de `pit_loss` (`[15, 45]`) son priors físicos basados en
  observación real de F1 moderna.
- El término de SC añade `+15 s` plano cuando se activa: aproximación
  gruesa que no modela la ganancia neta de "pit gratis bajo SC" más
  allá de la reducción del `pit_loss` ya descrita.
