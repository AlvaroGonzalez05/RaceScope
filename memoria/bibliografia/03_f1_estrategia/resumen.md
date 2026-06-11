# Optimización de Estrategia en Fórmula 1

## Relevancia para RaceScope Strategy Lab

RaceScope implementa un motor de dos fases para optimización de estrategias de pit stop en F1: scoring analítico de todos los candidatos (1-stop y 2-stop con diferentes compuestos y ventanas de pit) seguido de refinamiento Monte Carlo de los top-K. El motor considera probabilidad de Safety Car (P=0.20), pérdida de pit stop (22.5s mediana), y enumera estrategias válidas mediante bounds de vida de neumáticos (percentiles 20/80 empíricos).

---

## 1. PAPERS DE OPTIMIZACIÓN ACADÉMICA DE PIT STOPS

### 1.1 Optimizing Pit Stop Strategies in Formula 1 with Dynamic Programming and Game Theory

**Aguad, F., & Thraves, C. (2024). Optimizing Pit Stop Strategies in Formula 1 with Dynamic Programming and Game Theory.**
- **Venue:** European Journal of Operational Research, Vol. 319, Issue 3, pp. 908-919
- **DOI:** 10.1016/j.ejor.2024.05.031
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0377221724005484

Formula el problema de optimización de pit stop como un juego Stackelberg de suma cero usando programación dinámica. El framework modela la competición entre dos pilotos donde el líder elige la estrategia primero y el seguidor responde óptimamente. Los resultados muestran una mejora promedio de 2.3 segundos en el tiempo de carrera y una reducción del 17.8% en la probabilidad de undercut. Extiende el DP básico a DP estocástico para manejar eventos aleatorios como banderas amarillas y cambios meteorológicos.

**Relevancia:** Referencia fundamental — aborda directamente el problema competitivo de pit stop con elementos tanto deterministas como estocásticos. La formulación del juego Stackelberg proporciona base teórica para el modelado de estrategia competitiva.

---

### 1.2 On the Optimization of Pit Stop Strategies via Dynamic Programming

**Carrasco Heine, O. F., & Thraves, C. (2023). On the Optimization of Pit Stop Strategies via Dynamic Programming.**
- **Venue:** Central European Journal of Operations Research, Vol. 31, pp. 1-25
- **DOI:** 10.1007/s10100-022-00806-4
- **URL:** https://link.springer.com/article/10.1007/s10100-022-00806-4

Presenta enfoques de programación dinámica fundacionales para optimización de pit stops, determinando las vueltas óptimas de parada y selección de compuesto de neumáticos entre tres compuestos. Extiende el DP clásico a DP estocástico (SDP) incorporando eventos aleatorios (banderas amarillas, lluvia). Las soluciones SDP tienden a retrasar pit stops para beneficiarse de posibles oportunidades de bandera amarilla.

**Relevancia:** Base teórica esencial — la formulación DP proporciona la base algorítmica para la fase analítica del ranking de estrategias. Las extensiones estocásticas soportan directamente la fase de refinamiento Monte Carlo.

---

### 1.3 Optimización de Estrategias de Pit Stop en Carreras F1 — Universidad de Chile

**Repositorio Universidad de Chile (2020). Optimization of Pit Stop Strategies in Formula 1 Racing.**
- **URL:** https://repositorio.uchile.cl/bitstream/handle/2250/199664/Optimization-of-pit-stop-strategies-in-Formula-1-racing.pdf

Tesis/informe sobre optimización de pit stop en F1 usando modelos deterministas y estocásticos. Compara estrategias bajo diferentes condiciones de carrera y calcula intervalos de confianza para el tiempo final de carrera. Proporicona precedente académico para investigación de optimización de estrategia de pit stop F1.

**Relevancia:** Precedente académico para el dominio de investigación de RaceScope. Valida el dominio de investigación y proporciona referencias metodológicas para manejar la dinámica de carrera y la incertidumbre.

---

## 2. SIMULACIÓN Y MACHINE LEARNING PARA ESTRATEGIA F1

### 2.1 Data-Driven Pit Stop Decision Support for Formula 1 Using Deep Learning Models

**Rapp, E., et al. (2025). Data-driven pit stop decision support for Formula 1 using deep learning models.**
- **Venue:** Frontiers in Artificial Intelligence, Vol. 8, Article 1673148
- **DOI:** 10.3389/frai.2025.1673148
- **URL:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full

Evalúa cinco arquitecturas de deep learning (Bi-LSTM, TCN-GRU, GRU, InceptionTime, CNN-BiLSTM) para predecir ventanas óptimas de pit stop y degradación de neumáticos. El Bi-LSTM logra el mejor rendimiento modelando dependencias temporales de largo rango. Precisión máxima del 97.19% en predicción de tiempos de vuelta. Demuestra la validez de la elección de arquitectura LSTM en RaceScope.

**Relevancia:** Referencia moderna de ML directamente aplicable — proporciona métodos de deep learning para predicción de temporización de pit stop que podrían mejorar la fase de refinamiento Monte Carlo del motor de estrategia.

---

### 2.2 Explainable Reinforcement Learning for Formula One Race Strategy

**Thomas, D. (2025). Explainable Reinforcement Learning for Formula One Race Strategy.**
- **arXiv:** 2501.04068
- **URL:** https://arxiv.org/html/2501.04068v1

Introduce RSRL (Race Strategy Reinforcement Learning), un modelo de RL diseñado para controlar estrategias de carrera en entornos simulados, ofreciendo una alternativa más rápida a las estrategias hardcodeadas y basadas en Monte Carlo. El modelo logra una posición de llegada promedio de P5.33 en el Grand Prix de Bahréin 2023, superando a los enfoques Monte Carlo baseline. Enfatiza la explicabilidad en la toma de decisiones de IA.

**Relevancia:** Altamente relevante — presenta un enfoque RL complementario al motor de dos fases. Los aspectos de explicabilidad abordan directamente la necesidad de justificar las recomendaciones estratégicas.

---

### 2.3 Towards Learning-Based Formula 1 Race Strategies

**Arxiv (2024). Towards Learning-Based Formula 1 Race Strategies.**
- **arXiv:** 2512.21570
- **URL:** https://arxiv.org/abs/2512.21570

Presenta frameworks complementarios para optimizar estrategias de carrera F1, considerando conjuntamente la asignación de energía, el desgaste de neumáticos y el timing de pit stops. La investigación combina programación no lineal entera mixta (MINLP) para decisiones de pit stop con entornos de aprendizaje por refuerzo para optimización de estrategia.

**Relevancia:** Referencia crítica — la integración de MINLP con RL refleja directamente el enfoque de dos fases de RaceScope. La optimización conjunta de energía, desgaste de neumáticos y pit stops aborda los inputs clave para la fase de scoring analítico.

---

### 2.4 Application of Monte Carlo Methods to Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport

**Heilmeier, A., et al. (2020). Application of Monte Carlo Methods to Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport.**
- **Venue:** Applied Sciences, Vol. 10, Issue 12, Article 4229
- **DOI:** 10.3390/app10124229
- **URL:** https://www.mdpi.com/2076-3417/10/12/4229

Estudio exhaustivo sobre la aplicación de simulación Monte Carlo para contabilizar eventos probabilísticos de carrera (coches de seguridad, accidentes, variabilidad en tiempos de vuelta y duraciones de pit stop). El framework modela miles de simulaciones de carrera para estimar la probabilidad de éxito de diferentes estrategias. Los equipos F1 simulan ~300 millones de permutaciones de carrera usando esta metodología.

**Relevancia:** Referencia metodológica central — valida el enfoque Monte Carlo para la fase de refinamiento. Proporciona distribuciones de probabilidad específicas y modelos para eventos aleatorios que podrían incorporarse al motor de simulación de RaceScope.

---

### 2.5 Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions

**Heilmeier, A., et al. (2020). Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit Motorsport.**
- **Venue:** Applied Sciences, Vol. 10, Issue 21, Article 7805
- **DOI:** 10.3390/app10217805
- **URL:** https://www.mdpi.com/2076-3417/10/21/7805

Presenta un ingeniero de estrategia virtual (VSE) basado en dos redes neuronales artificiales que decide si un piloto debe hacer una parada en boxes y qué compuesto de neumático montar. El sistema procesa el estado de la carrera (posición, combustible, condición de neumáticos, número de vuelta) para producir recomendaciones de pit stop.

**Relevancia:** Directamente aplicable — proporciona arquitectura de decisión de estrategia basada en ANN que podría complementar o reemplazar las heurísticas hardcodeadas. El diseño de dos redes (clasificador pit/no-pit + selector de compuesto) refleja el enfoque de descomposición de estrategia de RaceScope.

---

### 2.6 Planning Formula One Race Strategies Using Discrete-Event Simulation

**Revista JORS (2009). Planning Formula One Race Strategies Using Discrete-Event Simulation.**
- **Venue:** Journal of the Operational Research Society, Vol. 60, Issue 7, pp. 926-938
- **DOI:** 10.1057/palgrave.jors.2602626
- **URL:** https://ideas.repec.org/a/pal/jorsoc/v60y2009i7d10.1057_palgrave.jors.2602626.html

Trabajo fundacional temprano sobre simulación de eventos discretos para estrategia F1. Desarrolla modelos de simulación describiendo efectos del consumo de combustible y la degradación de neumáticos sobre tiempos de vuelta basados en datos observados. Las simulaciones imitan eventos en pista incluyendo mezcla de coches en la salida, pit stops, adelantamientos, situaciones de Safety Car y abandonos.

**Relevancia:** Fundamento histórico — establece la simulación de eventos discretos como metodología probada para análisis de estrategia F1. Trabajo temprano que motivó los enfoques de dos fases actuales en investigación moderna.

---

### 2.7 Simulating Formula One Race Strategies

**Sulsters, C. (2017). Simulating Formula One Race Strategies.**
- **Venue:** VU Business Analytics Thesis
- **URL:** https://vu-business-analytics.github.io/internship-office/papers/paper-sulsters.pdf

Tesis de máster presentando un modelo de simulación para estrategias de carrera F1 que determina el timing óptimo de pit stop modelando efectos del consumo de combustible y la degradación de neumáticos sobre tiempos de vuelta. El simulador contempla eventos en pista (mezcla, pit stops, adelantamientos, Safety Cars, abandonos).

**Relevancia:** Referencia práctica — proporciona ejemplo funcional de implementación de simulación de carrera de extremo a extremo. Los enfoques de modelado de combustible y neumáticos son directamente aplicables al componente de perfil de piloto de RaceScope.

---

## 3. ANÁLISIS ESTADÍSTICO Y BAYESIANO DE RESULTADOS F1

### 3.1 Bayesian Analysis of Formula One Race Results

**Fórmula One Journal of Quantitative Analysis in Sports (2023). Bayesian Analysis of Formula One Race Results: Disentangling Driver Skill and Constructor Advantage.**
- **DOI:** 10.1515/jqas-2022-0021
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10660124/

Desarrolla un novedoso método de regresión logística de rank-ordered multinivel bayesiano para modelar posiciones finales individuales de carrera de 2014-2021 (era híbrida). El análisis muestra que aproximadamente el 88% de la varianza en los resultados de carrera se explica por la ventaja del constructor, mientras que la habilidad del piloto representa ~12%. Los parámetros son directamente interpretables como log-odds ratios de superar a competidores.

**Relevancia:** Proporciona base estadística para clasificación de habilidad de piloto y modelado de rendimiento del constructor. El enfoque bayesiano multinivel ofrece métodos de cuantificación de incertidumbre aplicables a los modelos de pace específicos de piloto de RaceScope.

---

### 3.2 A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing

**Renganathan, V. (2024). A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing.**
- **arXiv:** 2512.00640
- **URL:** https://arxiv.org/abs/2512.00640

Introduce un framework de modelado bayesiano de espacio de estados para estimar dinámicas latentes de degradación de neumáticos usando datos de timing disponibles públicamente de la API Python FastF1. Los tiempos de vuelta se modelan como función de la masa de combustible y el pace latente de neumáticos, con las paradas en boxes representadas como reinicios de estado. Demuestra rendimiento predictivo superior comparado con baselines ARIMA.

**Relevancia:** Componente crítico — el modelado de degradación de neumáticos es fundamental para las fases de perfil de piloto y scoring analítico. El enfoque bayesiano proporciona cuantificación de incertidumbre esencial para la simulación Monte Carlo.

---

## 4. GAME THEORY Y ESTRATEGIA COMPETITIVA EN F1

### 4.1 Game Theory in Formula 1: From Physical to Strategic Interactions

**arxiv (2025). Game Theory in Formula 1: From Physical to Strategic Interactions.**
- **arXiv:** 2503.05421
- **URL:** https://arxiv.org/html/2503.05421v1

Proporciona un framework de teoría de juegos para la optimización de estrategia de carrera F1 multi-agente, integrando efectos aerodinámicos de estela física, optimización de trayectoria y decisiones estratégicas de pit stop. El paper modela las interacciones competitivas entre pilotos como escenarios de juego donde ambos jugadores optimizan sus estrategias considerando las respuestas de los oponentes. Aborda el escenario realista donde las decisiones de pit stop deben contemplar las decisiones del equipo rival y las dinámicas de undercut/overcut.

**Relevancia:** Altamente relevante para análisis competitivo — extiende la estrategia de piloto único a escenarios competitivos multi-piloto. Esencial para modelar dinámicas de carrera realistas donde las ventanas de pit stop dependen de las acciones de los competidores.

---

### 4.2 Learning-Based Multi-Agent Race Strategies in Formula 1

**arxiv (2026). Learning-based Multi-agent Race Strategies in Formula 1.**
- **arXiv:** 2602.23056
- **URL:** https://arxiv.org/html/2602.23056v1

Aborda el aprendizaje por refuerzo multi-agente para estrategias de carrera F1 competitivas donde los agentes aprenden a equilibrar la gestión de energía, la degradación de neumáticos, la interacción aerodinámica entre coches y las decisiones de pit stop mientras consideran las estrategias rivales.

**Relevancia:** Referencia avanzada — extiende RaceScope de la optimización de piloto único al análisis competitivo multi-piloto. Proporciona métodos RL para modelar interacciones estratégicas entre pilotos.

---

## 5. PREDICCIÓN DE RESULTADOS F1 CON ML

### 5.1 Predicting Formula 1 Race Outcomes: A Machine Learning Approach

**Jafri, A. (2024). Predicting Formula 1 Race Outcomes: A Machine Learning Approach.**
- **URL:** https://aliabdullahjafri.com/static/media/Ali_Jafri_CapstoneProject1_Fall2024.c7244022875d46bec5d9.pdf

Enfoque exhaustivo de ML para predicción de resultados de carrera F1 utilizando múltiples algoritmos e ingeniería de features. Demuestra que la posición de carrera contribuye el 75.8% a la precisión de predicción, con variaciones estacionales al 23.8%. Usa análisis exhaustivo de importancia de features para identificar drivers clave de resultados de carrera. Incorpora XGBoost, Random Forests y enfoques de redes neuronales.

**Relevancia:** Proporciona insights de ingeniería de features y enfoque de aprendizaje multi-tarea. El análisis de importancia de features informa qué variables de piloto/circuito/compuesto son más predictivas para la fase de scoring analítico.

---

### 5.2 Deep-Racing: An Embedded Deep Neural Network Model for F1 Race Prediction

**Fatima, S., & Johrendt, J. (2023). Deep-Racing: An Embedded Deep Neural Network Model for F1 Race Prediction.**
- **Venue:** International Journal of Machine Learning, Vol. 13, No. 3
- **URL:** https://www.ijml.org/vol13/IJML-V13N3-1135-MT23-337.pdf

Presenta la arquitectura EDNN (Embedded Deep Neural Network) llamada Deep-Racing que predice los timing óptimos de pit stop y las posiciones finales de carrera. Entrenada en 169,000 vueltas de las temporadas F1 2015-2022, el modelo logra precisión de predicción de pit stop de 0.56, recall de 0.83 y F1-score de 0.67.

**Relevancia:** Directamente aplicable — valida enfoques de redes neuronales profundas para predicción de timing de pit stop a escala. El aprendizaje multi-tarea (timing de pit stop + resultado de carrera) demuestra cómo modelos únicos pueden soportar múltiples objetivos estratégicos.

---

### 5.3 PREDICTING LAP TIMES IN A FORMULA 1 RACE USING DEEP LEARNING ALGORITHMS

**Tilburg University (2023). Predicting Lap Times in a Formula 1 Race Using Deep Learning Algorithms.**
- **URL:** https://arno.uvt.nl/show.cgi?fid=180319

Tesis exhaustiva sobre la aplicación de algoritmos de deep learning a la predicción de tiempos de vuelta F1. Evalúa múltiples arquitecturas incluyendo LSTM apilado, GRU y combinaciones CNN-LSTM para modelar dependencias temporales en datos de carrera vuelta a vuelta. Demuestra que las arquitecturas recurrentes con formatos de ventana móvil logran rendimiento competitivo en datasets diversos.

**Relevancia:** Referencia metodológica — comparación detallada de arquitecturas RNN para predicción de series temporales F1. El enfoque de ventana móvil y la selección de arquitectura informan directamente las elecciones del modelo LSTM de RaceScope para el componente de perfil de piloto.

---

### 5.4 Real-Time Decision Making in Motorsports: Analytics for Improving Professional Car Race Strategy

**MIT Thesis (2015). Real-Time Decision Making in Motorsports: Analytics for Improving Professional Car Race Strategy.**
- **URL:** https://dspace.mit.edu/handle/1721.1/100310

Investigación del MIT Sloan sobre métodos analíticos para predicción dentro de carrera y toma de decisiones en tiempo real en carreras de coches profesionales. Propone un Lap Time Prediction Model para F1 basado en redes neuronales LSTM entrenadas en datos históricos de carrera. Establece la predicción de tiempos de vuelta basada en LSTM como metodología probada.

**Relevancia:** Referencia seminal del MIT — establece la predicción de tiempos de vuelta basada en LSTM como metodología probada. Proporciona base académica para sistemas de soporte de decisiones en tiempo real en optimización de estrategia F1.

---

## 6. DATOS Y APIs DE F1

### 6.1 OpenF1 API

**OpenF1 Project (2023-presente). OpenF1 API — The open source API for Formula 1 data.**
- **GitHub:** https://github.com/br-g/openf1
- **Docs:** https://openf1.org/

Proyecto open-source impulsado por la comunidad que proporciona acceso exhaustivo a datos F1 a través de formatos JSON/CSV con más de 18 endpoints. Implementa limitación de tasa token-bucket para la equidad en el acceso a la API. Infraestructura esencial para pipelines de ingestión de datos deportivos. Usado como fuente de datos principal en RaceScope.

**Relevancia:** Fuente primaria de datos de RaceScope — proporciona tiempos de vuelta, telemetría, información de pilotos y datos de eventos de carrera que alimentan todo el pipeline de entrenamiento.

---

### 6.2 FastF1 Python Library

**Oehrly, T. (2021-presente). FastF1 — Python Package for F1 Data Analysis.**
- **Docs:** https://docs.fastf1.dev/
- **GitHub:** https://github.com/theOehrly/Fast-F1

Wrapper Python alrededor de APIs de telemetría F1 con caché automático. Extrae datos de timing, telemetría y posición (desde 2018+) a través de DataFrames Pandas extendidos. Implementa caché automático y funciones personalizadas para análisis F1.

**Relevancia:** Librería complementaria a OpenF1 para análisis de datos F1 en Python. Usada por la comunidad de investigación académica para acceder a datos F1 de forma estructurada.

---

## SÍNTESIS: POSICIONAMIENTO DE RACESCOPE EN LA LITERATURA

La literatura académica revela varios enfoques dominantes para la optimización de estrategia F1:

1. **Simulación Monte Carlo**: Metodología fundacional usada por equipos profesionales para evaluar millones de permutaciones de carrera considerando desgaste de neumáticos, coches de seguridad e interacciones competitivas.

2. **Aprendizaje por Refuerzo**: Enfoque emergente combinando redes neuronales con simulación de carrera para optimización de estrategia adaptativa y con aprendizaje.

3. **Programación Dinámica y Teoría de Juegos**: Base teórica para optimización de pit stop bajo competencia. Las formulaciones de juego Stackelberg capturan dinámicas líder-seguidor.

4. **Redes Neuronales Profundas**: Implementación práctica para predicción de tiempos de vuelta, degradación de neumáticos y timing de pit stop.

5. **Métodos Bayesianos y Estadísticos**: Proporcionan cuantificación de incertidumbre, clasificación de habilidad de piloto e inferencia posterior para decisiones estratégicas.

**RaceScope se alinea con esta literatura** combinando scoring analítico (fase DP/teoría de juegos) con refinamiento Monte Carlo (fase de simulación estocástica), integrando perfiles de piloto basados en LSTM, y aprovechando datos de OpenF1 para ingeniería de features.
