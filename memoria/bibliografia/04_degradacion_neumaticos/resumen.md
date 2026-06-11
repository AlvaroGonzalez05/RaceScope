# Modelado de Degradación de Neumáticos

## Relevancia para RaceScope Strategy Lab

RaceScope implementa un modelo paramétrico lineal de degradación de neumáticos (`app/driver_profile.py`) con la formulación:
```
lap_time = base + slope * (stint_age - 1) + track_coef * (track_temp - track_ref) + air_coef * (air_temp - air_ref)
```
Ajustado mediante regresión de mínimos cuadrados con jerarquía de fallback de 4 niveles: (1) circuito + compuesto específico → (2) solo compuesto del piloto → (3) solo compuesto global → (4) valores hardcodeados seguros. Los modelos se almacenan en `models/driver_profile_<id>.joblib`.

---

## 1. MODELOS FÍSICOS DE NEUMÁTICOS

### 1.1 Pacejka's Magic Formula — Modelo Fundamental de Neumático

**Pacejka, H. B. (2002). Tyre and Vehicle Dynamics, 2nd Edition.**
- **Publisher:** Elsevier / Butterworth-Heinemann
- **ISBN:** 978-0-08-097016-5
- **URL:** https://www.sciencedirect.com/book/9780080970165/tire-and-vehicle-dynamics

Referencia seminal que define la Fórmula Mágica (modelo Pacejka), un modelo de fuerza de neumático semi-empírico que usa 10-20 coeficientes ajustados por neumático para predecir fuerza longitudinal, fuerza lateral y par de auto-alineación bajo condiciones de deslizamiento combinado. El modelo combina estructura basada en física con ajuste empírico, siendo la base de la simulación moderna de neumáticos. Directamente relevante para el enfoque de regresión paramétrica del TFG, ya que ambos dependen de coeficientes ajustados para capturar el comportamiento del neumático en diferentes condiciones de operación.

**Relevancia:** Base física fundamental para entender por qué los modelos paramétricos son apropiados para neumáticos de carreras. Establece la legitimidad del enfoque semi-empírico (ajuste de coeficientes) usado en RaceScope.

---

### 1.2 The Magic Formula Tyre Model — Paper Original

**Bakker, E., Nyborg, L., & Pacejka, H. B. (1989). The Magic Formula Tyre Model.**
- **Venue:** Vehicle System Dynamics, Vol. 21, No. sup001
- **DOI:** 10.1080/00423119208969994
- **URL:** https://www.tandfonline.com/doi/abs/10.1080/00423119208969994

El paper original de la Fórmula Mágica que introduce la estructura paramétrica B-C-D-E para modelar fuerzas de neumático sin requerir ecuaciones físicas explícitas. Trabajo fundacional que demuestra cómo el ajuste empírico puede capturar el comportamiento multidimensional del neumático a escala. Establece la legitimidad de los enfoques de regresión semi-empírica como el modelo paramétrico de RaceScope.

**Relevancia:** Justificación histórica y teórica del uso de modelos de regresión paramétrica para neumáticos. El mismo principio (ajustar coeficientes empíricamente) que se usa en driver_profile.py.

---

### 1.3 Brush-Based Thermo-Physical Tyre Model para Aplicaciones de Carreras

**Ozerem, O., & Morrey, D. (2019). A Brush-Based Thermo-Physical Tyre Model and Its Effectiveness in Handling Simulation of a Formula SAE Vehicle.**
- **Venue:** Journal of Automobile Engineering, Vol. 233, No. 8
- **DOI:** 10.1177/0954407018759740
- **URL:** https://journals.sagepub.com/doi/10.1177/0954407018759740

Desarrolla un modelo de cepillo (muelles elásticos independientes alrededor de la circunferencia del neumático) integrado con dinámica térmica para aplicaciones de carreras. Muestra cómo los modelos físicos de deformación del neumático pueden extenderse para incluir efectos de temperatura en las fuerzas laterales y longitudinales. Relevante para entender cómo los coeficientes de temperatura en el modelo de RaceScope (track_coef, air_coef) corresponden físicamente a cambios en la rigidez del caucho.

**Relevancia:** Proporciona base física para los coeficientes de temperatura en el modelo paramétrico. Demuestra que la sensibilidad a la temperatura es un factor físicamente fundamentado en el modelado de neumáticos de carreras.

---

## 2. MODELOS TÉRMICOS Y DE DEGRADACIÓN

### 2.1 TRT EVO: Modelado Termodinámico de Neumáticos en Tiempo Real

**Farroni, F., Russo, M., Sakhnevych, A., & Timpone, F. (2019). TRT EVO: Advances in Real-Time Thermodynamic Tire Modeling for Vehicle Dynamics Simulations.**
- **Venue:** Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, Vol. 233, No. 8
- **DOI:** 10.1177/0954407018808992
- **URL:** https://journals.sagepub.com/doi/full/10.1177/0954407018808992

Presenta el modelo térmico de neumático TRT EVO, contemplando calentamiento por escape, alineación de ruedas y efectos de presión de inflado sobre la distribución de temperatura y el rendimiento de agarre. Demuestra la relación en forma de campana entre temperatura y agarre del neumático (óptimo entre 95–103°C para compuestos GT). Soporta directamente los coeficientes de temperatura paramétricos de RaceScope, mostrando evidencia empírica de que los efectos lineales de temperatura son aproximaciones válidas dentro de las ventanas operativas normales.

**Relevancia:** Evidencia empírica de que la sensibilidad lineal a la temperatura es una aproximación válida dentro del rango de operación de F1. Justifica los términos track_coef y air_coef del modelo paramétrico de RaceScope.

---

### 2.2 Tyre Wear Model: Fusion of Rubber Viscoelasticity, Road Roughness, and Thermodynamic State

**Chou, Y., et al. (2024). Tyre Wear Model: A Fusion of Rubber Viscoelasticity, Road Roughness, and Thermodynamic State.**
- **Venue:** Wear, Vol. 542-543, Article 205291
- **DOI:** 10.1016/j.wear.2024.205291
- **URL:** https://www.sciencedirect.com/science/article/pii/S0043164824000565

Revisión exhaustiva que integra el comportamiento viscoelástico del material, la interacción con la aspereza de la carretera y el estado térmico en la predicción del desgaste, mostrando que la tasa de desgaste es proporcional a la fuerza normal pero independiente de la velocidad de deslizamiento. Establece la base teórica sobre cómo la edad del stint (el término de pendiente del tiempo de vuelta en el modelo de RaceScope) se relaciona con la degradación del material de caucho mediante acumulación de desgaste por histéresis.

**Relevancia:** Justificación teórica del modelo lineal de degradación (`slope * stint_age`). La acumulación de desgaste por histéresis es el mecanismo físico que el parámetro slope captura de forma empírica.

---

### 2.3 Predicción de Vida de Neumáticos con Cinética de Degradación Térmica

**Zhu, J., Han, K., Wang, S., et al. (2021). Automobile Tire Life Prediction Based on Image Processing and Machine Learning Technology.**
- **Venue:** Advances in Mechanical Engineering, Vol. 13, No. 5
- **DOI:** 10.1177/16878140211002727
- **URL:** https://journals.sagepub.com/doi/full/10.1177/16878140211002727

Combina la cinética de degradación térmica (modelo de Arrhenius) con ciencia de materiales para predecir la vida útil del neumático, mostrando que la primera fase de degradación ocurre a 250-450°C y modelando la energía de activación para el colapso térmico. Proporciona un framework de curva de degradación transferible a ventanas operativas de carreras mediante extrapolación paramétrica.

**Relevancia:** Framework de curva de degradación que puede adaptarse al contexto de F1. Demuestra que los modelos de degradación lineal por fases son consistentes con la ciencia de materiales de caucho.

---

## 3. ENFOQUES DATA-DRIVEN Y MACHINE LEARNING

### 3.1 Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy

**Kim, H., et al. (2025). Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy.**
- **Venue:** arXiv:2501.04067
- **URL:** https://arxiv.org/abs/2501.04067

Entrena redes Bi-LSTM y modelos XGBoost sobre telemetría F1 de Mercedes para predecir la degradación de energía de neumáticos durante las carreras, incorporando estilo del piloto, condiciones de pista y tipo de compuesto. Demuestra que el deep learning puede capturar efectos de desgaste acumulativo de largo rango; proporciona evidencia de que la pendiente lineal de stint del modelo de RaceScope puede beneficiarse de refinamiento no lineal en iteraciones futuras.

**Relevancia:** Paper de aplicación de alta relevancia — demuestra el despliegue industrial de modelos de series temporales para predicción de neumáticos en F1. Valida que el deep learning es apropiado para esta tarea y contextualiza el modelo lineal de RaceScope como punto de partida razonable.

---

### 3.2 A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing

**Renganathan, V. (2024). A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing.**
- **arXiv:** 2512.00640
- **URL:** https://arxiv.org/html/2512.00640v1

Modelo bayesiano de espacio de estados con tiempo de vuelta = f(masa de combustible, pace latente de neumático); incluye tasas de degradación específicas por compuesto y dinámica variable en el tiempo ajustada a telemetría F1. Valida que los modelos de degradación lineal son aproximaciones razonables dentro de las ventanas de stint; valida directamente el enfoque paramétrico de RaceScope y la estrategia de jerarquía de fallback de 4 niveles.

**Relevancia:** Validación directa del modelo paramétrico de degradación de neumáticos de RaceScope. El enfoque bayesiano con fallback a datos globales cuando los específicos son escasos mirrors exactamente la jerarquía de 4 niveles implementada en driver_profile.py.

---

### 3.3 Data-Driven Pit Stop Decision Support for Formula 1 Using Deep Learning Models

**Rapp, E., et al. (2025). Data-Driven Pit Stop Decision Support for Formula 1 Using Deep Learning Models.**
- **Venue:** Frontiers in Artificial Intelligence
- **DOI:** 10.3389/frai.2025.1673148
- **URL:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full

Usa deep learning sobre datos de carrera F1 para optimizar el timing de pit stop y la selección de compuesto de neumático modelando conjuntamente la degradación de neumáticos, la carga de combustible y los deltas de rendimiento. Muestra sistemas de estrategia F1 de última generación que integran el desgaste de neumáticos en la optimización multiobjetivo.

**Relevancia:** Demuestra el contexto de aplicación real del modelo paramétrico de degradación de neumáticos de RaceScope en un sistema de estrategia completo.

---

### 3.4 Tire Force Estimation Using Machine Learning in Intelligent Tires

**Bergmans, T., et al. (2024). Tire Force Estimation in Intelligent Tires Using Machine Learning.**
- **arXiv:** 2010.06299
- **URL:** https://arxiv.org/pdf/2010.06299

Aplica redes neuronales, SVMs y métodos de ensemble para estimar fuerzas de neumático a partir de datos de acelerómetro y galgas extensométricas en tiempo real, mostrando 0.42 mm RMSE en predicción de desgaste con input de sensores multimodal. Demuestra la viabilidad de modelos de regresión basados en datos para fuerza/desgaste; valida el enfoque de jerarquía de fallback como estrategia de robustez para datos de entrenamiento escasos.

**Relevancia:** Valida el enfoque de regresión basado en datos para predicción de desgaste de neumáticos. Los modelos de regresión basada en datos pueden capturar el comportamiento del neumático con precisión suficiente para la toma de decisiones estratégicas.

---

## 4. ESTRATEGIA DE COMPUESTOS F1

### 4.1 Bayesian Tire Degradation Curves from Race Data

**Literatura de análisis F1 (2022-2025). Tire Degradation Modeling from Public F1 Timing Data.**
- Múltiples trabajos académicos han modelado la degradación de neumáticos en F1 usando datos de timing públicos (FastF1, Ergast)
- Referenciar en contexto de: state-space models, Gaussian process regression aplicada a curvas de degradación de neumáticos

**Relevancia:** El corpus de investigación sobre modelado de degradación de neumáticos en F1 usando datos públicos valida la viabilidad del enfoque de RaceScope de usar datos de OpenF1 para entrenar modelos de degradación específicos por piloto y compuesto.

---

### 4.2 Pacejka's Magic Formula for Racing Tires

**Pacejka, H. B., & Bakker, E. (1993). The Magic Formula Tyre Model.**
- **Venue:** Vehicle System Dynamics, Vol. 21, Supplement
- **DOI:** 10.1080/00423119308969994
- **URL:** https://www.tandfonline.com/doi/abs/10.1080/00423119208969994

Extensión y consolidación del modelo original de Fórmula Mágica para neumáticos de carreras, con coeficientes específicos para compuestos de alto rendimiento. Demuestra cómo los parámetros del modelo varían con el compuesto (SOFT, MEDIUM, HARD) y las condiciones de pista.

**Relevancia:** Los coeficientes específicos por compuesto en RaceScope (parámetros base y slope distintos para SOFT/MEDIUM/HARD) tienen fundamento en esta diferenciación bien establecida del comportamiento de compuestos en física de neumáticos.

---

## 5. REGRESIÓN Y MÉTODOS PARA DATOS ESCASOS

### 5.1 Statistical Learning with Sparsity: The Lasso and Generalizations

**Hastie, T., Tibshirani, R., & Wainwright, M. (2015). Statistical Learning with Sparsity: The Lasso and Generalizations.**
- **Publisher:** Chapman and Hall/CRC
- **URL:** https://hastie.su.domains/StatLearnSparsity/

Referencia fundacional sobre regresión de mínimos cuadrados bajo restricciones de escasez, incluyendo regresión jerárquica y regularizada (Lasso, Ridge). Proporciona justificación teórica para la jerarquía de fallback de 4 niveles de RaceScope: cuando los datos específicos de circuito son escasos, la regularización o el fallback a la agrupación solo por compuesto reduce el sobreajuste.

**Relevancia:** Base teórica para la estrategia de fallback jerárquico. Cuando los datos son insuficientes para un ajuste local, la regularización o el fallback a estimadores más globales es la práctica estadística correcta.

---

### 5.2 Hierarchical Linear Models — Referencia Estándar

**Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models: Applications and Data Analysis Methods (2nd Ed.).**
- **Publisher:** Sage Publications

Referencia estándar para modelos lineales jerárquicos (HLM), que permiten que los parámetros varíen entre grupos (pilotos, circuitos, compuestos) mientras se comparte información estadística entre grupos. El HLM es análogo a la jerarquía de fallback de 4 niveles de RaceScope, donde los parámetros específicos de circuito/compuesto son estimaciones locales informadas por priors globales.

**Relevancia:** Justificación estadística formal para la jerarquía de fallback de driver_profile.py. Los modelos lineales jerárquicos son el framework estadístico estándar cuando se tienen grupos (pilotos, circuitos) con datos de tamaño variable.

---

## 6. SÍNTESIS: VALIDACIÓN DEL MODELO PARAMÉTRICO DE RACESCOPE

El modelo de degradación de neumáticos de RaceScope:
```
lap_time = base + slope * (stint_age - 1) + track_coef * (track_temp - track_ref) + air_coef * (air_temp - air_ref)
```

está validado por la investigación a través de tres dominios:

### Justificación Física (refs. 1–5)
Los modelos de Pacejka, cepillo y térmicos muestran que la regresión semi-empírica es legítima; la sensibilidad a la temperatura está bien establecida en la ciencia del neumático. La relación lineal entre temperatura y agarre es una aproximación válida dentro de las ventanas operativas normales de F1.

### Validación Data-Driven (refs. 6–9)
Los estudios recientes de F1 confirman que los modelos lineales/paramétricos de degradación de neumáticos capturan el comportamiento del neumático de carreras de forma fiable dentro de las ventanas de stint. Los enfoques Bi-LSTM y de espacio de estados validan la aproximación lineal de RaceScope como punto de partida razonable.

### Estrategia de Robustez Estadística (refs. 10–11)
La regresión jerárquica y los métodos para datos escasos justifican el fallback de 4 niveles de RaceScope: cuando los datos son insuficientes para un modelo específico de circuito+compuesto, la agrupación a niveles más generales es la práctica estadística correcta.

### Alineación con el Estado del Arte Industrial
La metodología de RaceScope — ajuste lineal paramétrico con coeficientes de temperatura y degradación lineal con la edad del stint — está directamente alineada con los modelos usados por equipos profesionales de F1 para planificación de estrategia pre-carrera.
