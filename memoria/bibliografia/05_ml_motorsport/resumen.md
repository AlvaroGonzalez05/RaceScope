# Machine Learning Aplicado a Motorsport y Predicción de Rendimiento

## Relevancia para RaceScope Strategy Lab

RaceScope aplica redes LSTM para predicción de tiempos de vuelta en F1, con modelos específicos por piloto (`models/driver_<id>.joblib`) entrenados sobre secuencias de 10 vueltas con 8 features: lap_number, stint_age, compound (codificado 1/2/3), session_type, circuit_id, track_temp, air_temp, lap_time (normalizado z-score). Además implementa perfiles paramétricos de piloto por circuito/compuesto como sistema complementario de dos niveles.

---

## 1. PREDICCIÓN DE TIEMPOS DE VUELTA EN F1 CON DEEP LEARNING

### 1.1 Data-Driven Pit Stop Decision Support for Formula 1 Using Deep Learning Models

**Rapp, E., et al. (2025). Data-driven Pit Stop Decision Support for Formula 1 Using Deep Learning Models.**
- **Venue:** Frontiers in Artificial Intelligence, Vol. 8, Article 1673148
- **DOI:** 10.3389/frai.2025.1673148
- **URL:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full

Evalúa cinco arquitecturas de deep learning (Bi-LSTM, TCN-GRU, GRU, InceptionTime, CNN-BiLSTM) para predecir ventanas óptimas de pit stop y degradación de neumáticos usando datos de telemetría F1 reales. El modelo Bi-LSTM demostró rendimiento superior en la captura de dependencias temporales de largo rango en datos de carrera. Precisión máxima del 97.19% en predicción de tiempos de vuelta. El framework proporciona soporte de decisiones para estrategas prediciendo ventanas de pit stop con alta precisión.

**Relevancia:** Directamente aplicable — proporciona métodos de deep learning de vanguardia para predicción de timing de pit stop que podría mejorar la fase de refinamiento Monte Carlo del motor de estrategia de RaceScope. La comparación de múltiples arquitecturas ofrece insights sobre modelado temporal de dinámica de carrera.

---

### 1.2 Explainable Reinforcement Learning for Formula One Race Strategy

**Thomas, D. (2025). Explainable Reinforcement Learning for Formula One Race Strategy.**
- **arXiv:** 2501.04068
- **URL:** https://arxiv.org/abs/2501.04068

Introduce RSRL (Race Strategy Reinforcement Learning), un modelo de deep recurrent Q-network (DRQN) entrenado mediante simuladores de carrera Monte Carlo. Formula la estrategia de carrera como un Markov Decision Process donde la selección de compuestos y el timing de pit stops se optimizan. Logra una posición promedio de finalización P5.33 en el Gran Premio de Bahréin 2023, superando las estrategias baseline Monte Carlo. Enfatiza la explicabilidad en la toma de decisiones IA.

**Relevancia:** Altamente relevante — el enfoque RL + MC es complementario al motor de dos fases de RaceScope. Los aspectos de explicabilidad abordan directamente la necesidad de justificar recomendaciones estratégicas a los ingenieros de carrera.

---

### 1.3 Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy

**Anonymous Authors (2025). Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy.**
- **arXiv:** 2501.04067
- **URL:** https://arxiv.org/abs/2501.04067

Aborda la predicción de series temporales de estados de energía de neumáticos usando arquitecturas LSTM y Transformer con características de explicabilidad. Analiza datos de telemetría histórica de Mercedes-AMG PETRONAS F1 para predecir la degradación de neumáticos y niveles de energía a través de los stints de carrera. Los modelos XGBoost y deep learning logran predicciones dentro de 1.7 vueltas de media. Enfatiza la interpretabilidad de las predicciones de redes neuronales para la toma de decisiones estratégicas.

**Relevancia:** Importante para las features relacionadas con neumáticos en la fase analítica — los enfoques LSTM para predicción de energía de neumáticos soportan directamente la generación de curvas de pace por piloto de RaceScope. Los aspectos de explicabilidad se alinean con la necesidad de recomendaciones de estrategia interpretables.

---

### 1.4 Learning-Based Multi-Agent Race Strategies in Formula 1

**Anonymous Authors (2026). Learning-Based Multi-Agent Race Strategies in Formula 1.**
- **arXiv:** 2602.23056
- **URL:** https://arxiv.org/abs/2602.23056

Propone agentes de aprendizaje por refuerzo que optimizan conjuntamente la asignación de energía, el modelado de degradación de neumáticos, la interacción aerodinámica entre coches y las decisiones de pit stop mientras consideran las estrategias de los rivales. La formulación multi-agente captura interacciones competitivas y comparación cabeza a cabeza de estrategias, directamente aplicable al diseño del endpoint de comparación de RaceScope.

**Relevancia:** Altamente relevante — extiende RaceScope de la optimización de piloto único al análisis competitivo multi-piloto. Proporciona métodos RL para modelar interacciones estratégicas entre pilotos.

---

### 1.5 PREDICTING LAP TIMES IN A FORMULA 1 RACE USING DEEP LEARNING ALGORITHMS

**Tilburg University (2023). Predicting Lap Times in a Formula 1 Race Using Deep Learning Algorithms.**
- **URL:** https://arno.uvt.nl/show.cgi?fid=180319

Tesis exhaustiva sobre la aplicación de algoritmos de deep learning a la predicción de tiempos de vuelta F1. Evalúa múltiples arquitecturas incluyendo LSTM apilado, GRU y combinaciones CNN-LSTM para modelar dependencias temporales en datos de carrera vuelta a vuelta. Demuestra que las arquitecturas recurrentes con formatos de ventana móvil logran rendimiento competitivo. Proporciona orientación práctica para el despliegue en producción de redes neuronales de series temporales.

**Relevancia:** Referencia metodológica directa — comparación detallada de arquitecturas RNN para predicción de series temporales F1. El enfoque de ventana móvil y la selección de arquitectura informan las elecciones del modelo LSTM de RaceScope para el componente de perfil de piloto.

---

### 1.6 Deep Neural Network-Based Lap Time Forecasting of Formula 1 Racing

**ACE Open (2023). Deep Neural Network-Based Lap Time Forecasting of Formula 1 Racing.**
- **DOI:** 10.13140/RG.2.2.17644.13122
- **URL:** https://www.researchgate.net/publication/379012640

Propone redes neuronales profundas para reemplazar la simulación tradicional de tiempos de vuelta como herramientas de predicción de rendimiento más rápidas. Demuestra la capacidad de las DNNs para capturar relaciones no lineales entre tiempos de vuelta y variables de carrera incluyendo compuesto, edad del stint y características del circuito.

**Relevancia:** Soporta el modelado de perfiles de piloto — la predicción de rendimiento específica por circuito y piloto es fundamental para el componente de perfil de piloto. El enfoque de categorización (piloto × circuito × compuesto) se aplica directamente a la ingeniería de features de RaceScope.

---

## 2. HERRAMIENTAS Y ARQUITECTURAS ESPECÍFICAS DE F1

### 2.1 Virtual Strategy Engineer con ANNs

**Heilmeier, A., et al. (2020). Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit Motorsport.**
- **Venue:** Applied Sciences, Vol. 10, Issue 21, Article 7805
- **DOI:** 10.3390/app10217805
- **URL:** https://www.mdpi.com/2076-3417/10/21/7805

Presenta un ingeniero de estrategia virtual basado en dos redes neuronales artificiales. La primera decide si hacer pit stop; la segunda selecciona el compuesto. Procesa el estado de carrera (posición, combustible, condición de neumáticos, vuelta) para producir recomendaciones. Demuestra el despliegue práctico de redes neuronales en sistemas de soporte de decisiones en tiempo real para motorsport.

**Relevancia:** Arquitectura de referencia para sistemas de decisión de estrategia basados en NN. El diseño de dos redes refleja la descomposición de estrategia de RaceScope (timing de pit + selección de compuesto).

---

### 2.2 Formula-E Race Strategy Development with Neural Networks and MCTS

**Harman, M., Li, Y., & Langdon, W. B. (2020). Formula-E Race Strategy Development using Artificial Neural Networks and Monte Carlo Tree Search.**
- **Venue:** Neural Computing and Applications, Vol. 32, pp. 10567–10581
- **DOI:** 10.1007/s00521-020-04871-1
- **URL:** https://link.springer.com/article/10.1007/s00521-020-04871-1

Investiga modelos de predicción ANN para reemplazar la simulación tradicional de tiempos de vuelta, habilitando una predicción de rendimiento más rápida. Aplica búsqueda Monte Carlo Tree Search para exploración de estrategias. El enfoque híbrido refleja directamente el diseño analítico + refinamiento MC de RaceScope.

**Relevancia:** Paralelo arquitectónico directo al enfoque de dos fases de RaceScope (ANN para scoring analítico + MCTS para refinamiento de estrategia).

---

### 2.3 Deep-Racing: Embedded DNN para Predicción de Carrera F1

**Fatima, S., & Johrendt, J. (2023). Deep-Racing: An Embedded Deep Neural Network Model for F1 Race Prediction.**
- **Venue:** International Journal of Machine Learning, Vol. 13, No. 3
- **URL:** https://www.ijml.org/vol13/IJML-V13N3-1135-MT23-337.pdf

Presenta la arquitectura EDNN llamada Deep-Racing que predice timing óptimo de pit stop y posiciones finales de carrera. Entrenada en 169,000 vueltas de temporadas F1 2015-2022. Logra precision de predicción de pit stop de 0.56, recall de 0.83 y F1-score de 0.67. Demuestra el despliegue exitoso de DNNs para predicción integrada de pit stop y resultado de carrera en datos F1 reales.

**Relevancia:** Baseline de referencia directamente comparable — valida los enfoques de redes neuronales profundas para predicción de timing de pit stop a escala. El aprendizaje multi-tarea (timing de pit stop + resultado de carrera) demuestra cómo modelos únicos pueden soportar múltiples objetivos estratégicos.

---

## 3. ML EN MOTORSPORT: TELEMETRÍA Y ANÁLISIS DE RENDIMIENTO

### 3.1 AI-Enabled Prediction of Sim Racing Performance Using Telemetry Data

**Hojaji, F., Toth, A. J., Joyce, J. M., & Campbell, M. J. (2024). AI-Enabled Prediction of Sim Racing Performance Using Telemetry Data.**
- **Venue:** Computers & Education: X Reality, Vol. 4, Article 100047
- **DOI:** 10.1016/j.cexr.2024.100047
- **URL:** https://www.sciencedirect.com/science/article/pii/S2451958824000472

Aplica deep learning a datos de telemetría (velocidad, RPM, ángulo de dirección, freno, acelerador, aceleración) para predicción de rendimiento en carreras de simulación. Identifica factores críticos que impactan el rendimiento de conducción. Valida el enfoque de ingeniería de features usando señales de telemetría de motorsport.

**Relevancia:** Valida el uso de datos de telemetría para predicción de rendimiento. Proporciona insights sobre qué features de telemetría son más predictivas del rendimiento, lo que informa la selección de features de input del LSTM de RaceScope.

---

### 3.2 Advanced Machine Learning for Formula 1 Race Performance Prediction

**ResearchGate (2025). Advanced Machine Learning Approaches for Formula 1 Race Performance Prediction: A Comprehensive Analysis of Championship Point Forecasting.**
- **DOI:** 10.13140/RG.2.2.29445.87047
- **URL:** https://www.researchgate.net/publication/394015807

Analiza 589,081 tiempos de vuelta individuales en 1,125 carreras (1950-2024) usando métodos de ensemble, gradient boosting y modelos de regresión. El modelo Gradient Boosting óptimo logra R²=0.999, RMSE=0.197, MAE=0.125. Demuestra ingeniería de features efectiva para datos históricos de F1.

**Relevancia:** Benchmark de precisión para predicción de rendimiento F1. Los resultados de R²=0.999 demuestran que los modelos basados en datos pueden predecir con alta fidelidad cuando se dispone de features bien ingenieriadas.

---

### 3.3 A Data-Driven Analysis of Formula 1 Car Races Outcome

**Springer Nature (2023). A Data-Driven Analysis of Formula 1 Car Races Outcome.**
- **Venue:** Computer and Multimedia Technology, Springer
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-26438-2_11

Framework basado en datos que reduce 21 features de carrera a 4 componentes ortogonales principales explicando ~70% de la varianza. Encuentra que el rendimiento del piloto en las fases tempranas de la carrera predice los resultados finales. Identifica las features clave que influencian los resultados de carrera mediante reducción de dimensionalidad PCA.

**Relevancia:** Insights de reducción de features — el análisis PCA sugiere qué parámetros de carrera contribuyen más a los resultados. El hallazgo de que el rendimiento en las primeras fases predice los resultados finales soporta la función de scoring analítico de la fase de ranking de estrategias.

---

## 4. DEEP LEARNING PARA PREDICCIÓN DE RENDIMIENTO DEPORTIVO

### 4.1 Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions

**Hewamalage, H., Bergmeir, C., & Bandara, K. (2020). Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions.**
- **Venue:** International Journal of Forecasting, Vol. 37, Issue 1, pp. 388-427
- **DOI:** 10.1016/j.ijforecast.2020.06.008
- **URL:** https://arxiv.org/abs/1909.00590

Survey exhaustivo de arquitecturas RNN (LSTM, GRU) para forecasting de series temporales. Demuestra rendimiento competitivo frente a modelos ARIMA y ETS. Valida la selección de RNN para predicción de secuencias de tiempos de vuelta con ventanas de contexto de 10 vueltas.

**Relevancia:** Review que posiciona LSTM en el contexto más amplio de forecasting de series temporales. Proporciona comparativa sistemática que ayuda a justificar la elección de LSTM frente a alternativas estadísticas clásicas para la predicción de tiempos de vuelta de F1.

---

### 4.2 Hybrid Transformer-LSTM Model for Athlete Performance Prediction

**Ajgel, R., Ajgel, N., Osman, M., & Elhoseny, M. (2023). Hybrid Transformer-LSTM Model for Athlete Performance Prediction in Sports Training Management.**
- **Venue:** Informatica, Vol. 47, No. 3
- **URL:** https://www.informatica.si/index.php/informatica/article/view/8013

Propone un framework híbrido Transformer-LSTM que combina mecanismos de atención con capas recurrentes para capturar interacciones de features globales y dependencias temporales localizadas. Logra F1-Score 92.1%, AUC-ROC 96.3%. Arquitectura aplicable al modelado de secuencias de rendimiento de piloto en motorsport.

**Relevancia:** La arquitectura híbrida Transformer-LSTM es una mejora natural del LSTM puro de RaceScope. Proporciona framework para combinar la captura de patrones globales de Transformer con el procesamiento temporal local de LSTM.

---

### 4.3 Sports Match Prediction Using Attention-Based LSTM Networks

**ScienceDirect (2021). Sports Match Prediction Model for Training and Exercise Using Attention-Based LSTM Network.**
- **Venue:** Computers & Education: X Reality
- **URL:** https://www.sciencedirect.com/science/article/pii/S2352864821000602

Desarrolla un LSTM basado en atención (AS-LSTM) para predecir resultados de partidos usando datos históricos de equipos. Demuestra la efectividad de los mecanismos de atención con LSTM para datos deportivos secuenciales.

**Relevancia:** El mecanismo de atención con LSTM es relevante para el modelo de perfil de piloto de RaceScope — la atención podría aprender a enfocarse en fases críticas de carrera (vueltas de estrategia, degradación de neumáticos clave) al predecir tiempos de vuelta.

---

## 5. COMPARATIVA DE ARQUITECTURAS ML PARA MOTORSPORT

### Resumen de Arquitecturas Evaluadas en la Literatura para Predicción F1

| Arquitectura | Paper | F1 Accuracy / RMSE | Características | Relevancia RaceScope |
|---|---|---|---|---|
| **Bi-LSTM** | Rapp et al. 2025 | 97.19% | Mejor para dependencias temporales largas | Alternativa a LSTM actual |
| **DRQN (RL)** | Thomas 2025 | P5.33 avg position | Aprendizaje por refuerzo + MC | Enfoque complementario |
| **EDNN** | Fatima & Johrendt 2023 | F1-score 0.67 | Multi-tarea (pit + posición) | Baseline comparable |
| **XGBoost** | Miao et al. 2025 | Dentro 1.7 vueltas | Explicable; rápido | Comparativa no-DL |
| **Gradient Boosting** | ResearchGate 2025 | R²=0.999 | Alta precisión global | Benchmark general |
| **Transformer-LSTM** | Ajgel et al. 2023 | AUC-ROC 96.3% | Híbrido; atención global | Mejora futura |
| **TCN-GRU** | Rapp et al. 2025 | Competitivo | Convolucional + recurrente | Alternativa arquitectónica |

---

## 6. INGENIERÍA DE FEATURES PARA DATOS DEPORTIVOS SECUENCIALES

### 6.1 Normalización en Series Temporales para Redes Neuronales

**Deep Adaptive Input Normalization for Time Series Forecasting (2019)**
- **arXiv:** 1902.07892
- **URL:** https://arxiv.org/abs/1902.07892

Aborda los desafíos de normalización en series temporales no estacionarias. Propone normalización adaptativa del input para manejar cambios de distribución. Crítico para la estrategia de normalización z-score de RaceScope y el manejo de variabilidad de tiempos de vuelta entre diferentes sesiones de carrera.

**Relevancia:** La normalización z-score de RaceScope (lap_mean, lap_std por sesión) sigue este principio de adaptación local para manejar la no-estacionariedad de los datos de F1.

---

### 6.2 Codificación de Compuestos de Neumáticos

La codificación ordinal de compuestos (SOFT=1, MEDIUM=2, HARD=3) en RaceScope refleja la jerarquía física de suavidad/dureza del compuesto y corresponde a los principios de codificación para variables categóricas ordinales documentados en la literatura de preprocessing de ML. La codificación ordinal captura la relación de orden entre compuestos mejor que la codificación one-hot para features que tienen un orden intrínseco meaningful.

---

## CONCLUSIÓN: POSICIONAMIENTO DE RACESCOPE EN ML PARA MOTORSPORT

RaceScope contribuye al campo del ML aplicado a motorsport mediante:

1. **Sistema E2E completo**: Desde ingestión de datos OpenF1 hasta recomendaciones de estrategia accionables, cubriendo la brecha entre modelos académicos y herramientas de decisión prácticas.

2. **Modelos específicos por piloto y circuito**: A diferencia de muchos papers que usan modelos globales, RaceScope entrena 21 modelos LSTM específicos por piloto con fallback jerárquico.

3. **Motor de dos fases innovador**: La combinación de scoring analítico rápido (todos los candidatos) con refinamiento Monte Carlo (top-K) optimiza el balance entre exhaustividad y precisión.

4. **Datos OpenF1 en producción**: Uso de la API OpenF1 open-source como fuente de datos real para entrenamiento y serving, demostrando viabilidad con datos públicos accesibles.
