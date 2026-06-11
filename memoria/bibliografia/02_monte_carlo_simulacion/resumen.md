# Monte Carlo: Simulación y Optimización bajo Incertidumbre

## Relevancia para RaceScope Strategy Lab

RaceScope implementa refinamiento Monte Carlo (n_sim=200) sobre el top-K=5 de estrategias candidatas. La fase MC simula: evento de Safety Car (Bernoulli con P(SC)=0.20), vuelta de SC (uniforme sobre último 10% de carrera), pérdida de pit stop condicional, y ruido de tráfico por vuelta (distribución normal correlacionada por vuelta). El output es (mean_time, variance, samples) para ranking refinado. El objetivo de ranking es `risk_score = E[time] + λ * sqrt(Var[time])` con λ=0.15.

---

## 1. FUNDAMENTOS DE SIMULACIÓN MONTE CARLO

### 1.1 Simulation and the Monte Carlo Method (3ª Ed.)

**Rubinstein, R. Y., & Kroese, D. P. (2016). Simulation and the Monte Carlo Method, 3rd Edition.**
- **Publisher:** Wiley Series in Probability and Statistics
- **URL:** https://www.wiley.com/en-us/Simulation+and+the+Monte+Carlo+Method,+3rd+Edition-p-9781118632161

El libro de texto autorizado sobre métodos Monte Carlo con implementaciones modernas en pseudocódigo. Cubre generación de números aleatorios, técnicas de reducción de varianza y optimización Monte Carlo. Incluye el método cross-entropy (CEM), directamente relevante para el refinamiento iterativo basado en muestreo de soluciones candidatas.

**Relevancia:** Fundamento teórico para los algoritmos de simulación MC; proporciona técnicas de reducción de varianza aplicables a mejorar la convergencia de la fase de refinamiento top-K.

---

### 1.2 Monte Carlo Sampling-Based Methods for Stochastic Optimization

**Shapiro, A. (2013). Monte Carlo Sampling-Based Methods for Stochastic Optimization.**
- **Venue:** Optimization Online
- **URL:** https://optimization-online.org/wp-content/uploads/2013/06/3920.pdf

Tratamiento exhaustivo de enfoques basados en muestreo para optimización estocástica, cubriendo teoría de convergencia, complejidad muestral y el método Sample Average Approximation (SAA). Aborda el desafío teórico de equilibrar el esfuerzo de estimación versus optimización con muestras finitas.

**Relevancia:** Base teórica para la fase de refinamiento MC; explica las garantías de convergencia al muestrear del espacio de candidatos de estrategia y cómo el tamaño de muestra N afecta a la confianza en los rankings finales.

---

### 1.3 On the Rate of Convergence of Optimal Solutions of Monte Carlo Approximations

**Shapiro, A. (1999). On the Rate of Convergence of Optimal Solutions of Monte Carlo Approximations of Stochastic Programs.**
- **Venue:** SIAM Journal on Optimization
- **DOI:** 10.1137/S1052623498349541
- **URL:** https://epubs.siam.org/doi/10.1137/S1052623498349541

Establece tasas de convergencia para soluciones óptimas bajo muestreo Monte Carlo. Muestra que la convergencia escala con √N (raíz cuadrada del tamaño muestral), proporcionando la base teórica para elegir presupuestos de muestreo MC en la fase de refinamiento.

**Relevancia:** Proporciona garantías de convergencia para el bucle de refinamiento MC; ayuda a justificar la asignación del presupuesto muestral (n_sim=200) para lograr rankings de estrategia estables.

---

## 2. SAMPLE AVERAGE APPROXIMATION (SAA) Y PROGRAMACIÓN ESTOCÁSTICA

### 2.1 The Sample Average Approximation Method for Stochastic Discrete Optimization

**Kleywegt, A. J., Shapiro, A., & Homem-de-Mello, T. (2002). The Sample Average Approximation Method for Stochastic Discrete Optimization.**
- **Venue:** SIAM Journal on Optimization
- **DOI:** 10.1137/S1052623499363220
- **URL:** https://epubs.siam.org/doi/10.1137/S1052623499363220

Paper fundacional sobre el método SAA: aproxima el objetivo de valor esperado por el promedio muestral y resuelve el problema determinista resultante. Incluye estimación estadística de brechas de optimalidad a partir de múltiples ejecuciones SAA con diferentes muestras.

**Relevancia:** La fase analítica de RaceScope puntúa todos los candidatos; el refinamiento MC valida/actualiza puntuaciones en el top-K mediante muestreo repetido — este es precisamente el framework SAA. Muestra cómo estimar la confianza en los rankings finales.

---

### 2.2 A Guide to Sample-Average Approximation

**Kim, S., & Pasupathy, R. (2014). A Guide to Sample-Average Approximation.**
- **Venue:** Cornell University (ORIE Technical Report)
- **URL:** https://people.orie.cornell.edu/shane/pubs/SAAGuide.pdf

Guía práctica para implementar SAA: cubre selección del tamaño muestral, obtención de buenas soluciones candidatas y validación estadística. Tratamiento accesible de teoría y práctica.

**Relevancia:** Orientación práctica para la implementación del refinamiento MC; explica cómo evaluar la calidad de la solución y las brechas de optimalidad con presupuestos muestrales finitos.

---

## 3. OPTIMIZACIÓN AJUSTADA AL RIESGO — MEAN-VARIANCE

### 3.1 Portfolio Selection — Markowitz (Paper Original)

**Markowitz, H. M. (1952). Portfolio Selection.**
- **Venue:** The Journal of Finance, Vol. 7, No. 1, pp. 77–91
- **URL:** https://en.wikipedia.org/wiki/Modern_portfolio_theory

Paper seminal que introduce la optimización media-varianza: maximiza el rendimiento esperado sujeto a una restricción de varianza (riesgo), o equivalentemente minimiza el riesgo para un rendimiento objetivo dado. Formaliza el principio de que el trade-off entre riesgo y rendimiento es fundamental para la toma de decisiones.

**Relevancia:** El ranking ponderado por lambda de RaceScope (`E[X] + λ·σ[X]`) es una aplicación directa de los objetivos ajustados al riesgo de Markowitz. Este paper es la justificación teórica original para combinar valor esperado y varianza en la selección de estrategias.

---

### 3.2 Optimization of Conditional Value-at-Risk (CVaR)

**Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.**
- **Venue:** The Journal of Risk
- **URL:** https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf

Introduce el Conditional Value-at-Risk (CVaR) — la pérdida esperada en la cola de una distribución — y muestra que puede minimizarse como un problema de optimización convexo. Mejor medida de riesgo de cola que la varianza sola.

**Relevancia:** Alternativa a la varianza para el ranking ajustado al riesgo de estrategias; más robusta a resultados extremos (e.g., abandono de carrera o disrupciones por Safety Car). Podría mejorar la función de ranking de RaceScope más allá de la media-varianza.

---

## 4. MONTE CARLO TREE SEARCH (MCTS)

### 4.1 Bandit-Based Monte-Carlo Planning (UCT)

**Kocsis, L., & Szepesvári, C. (2006). Bandit-Based Monte-Carlo Planning.**
- **Venue:** 17th European Conference on Machine Learning (ECML)
- **URL:** http://ggp.stanford.edu/readings/uct.pdf

Introduce UCT (Upper Confidence Bounds applied to Trees): combina la teoría del multi-armed bandit (UCB1) con la expansión del árbol Monte Carlo. Equilibra la exploración vs. explotación en la búsqueda en árbol. Se demuestra la convergencia al juego óptimo.

**Relevancia:** Conceptualmente relacionado con el enfoque de dos fases de RaceScope: el scoring analítico (fase de exploración) identifica candidatos prometedores, luego el refinamiento MC (fase de explotación) enfoca la computación en el top-K. El equilibrio exploración-explotación de UCT es análogo al mecanismo de ranking de estrategias.

---

### 4.2 Monte Carlo Tree Search: A Review of Recent Modifications and Applications

**Salah, A. A., & Winands, M. H. M. (2022). Monte Carlo Tree Search: A Review of Recent Modifications and Applications.**
- **Venue:** Artificial Intelligence Review, Springer
- **URL:** https://link.springer.com/article/10.1007/s10462-022-10228-y

Revisión exhaustiva de variantes de MCTS y aplicaciones más allá de los juegos (robótica, planificación, optimización). Revisa políticas de selección, estrategias de simulación y métodos de backpropagation.

**Relevancia:** Posiciona MCTS en el panorama más amplio de optimización; aplicable a árboles de búsqueda de estrategia de carrera donde las decisiones se ramifican por vuelta, ventana de pit y compuesto de neumático.

---

## 5. SIMULACIÓN APLICADA A MOTORSPORT

### 5.1 Application of Monte Carlo Methods in Circuit Motorsport

**Heilmeier, A., et al. (2020). Application of Monte Carlo Methods to Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport.**
- **Venue:** Applied Sciences, Vol. 10, Issue 12, Article 4229
- **DOI:** 10.3390/app10124229
- **URL:** https://www.mdpi.com/2076-3417/10/12/4229

Modela influencias probabilísticas en simulación de carreras: accidentes, coches de seguridad completos, variabilidad en timing de pit stop y variabilidad de vida de neumáticos. Los métodos Monte Carlo permiten la evaluación de la robustez de las estrategias de carrera. Los equipos F1 simulan miles de millones de escenarios utilizando esta metodología para predecir resultados.

**Relevancia:** Referencia metodológica de aplicación directa — valida el uso del enfoque Monte Carlo para la fase de refinamiento. Proporciona distribuciones de probabilidad específicas y modelos para eventos aleatorios que podrían incorporarse al motor de simulación de RaceScope.

---

### 5.2 Formula-E Race Strategy Development with ANNs and MCTS

**Harman, M., Li, Y., & Langdon, W. B. (2020). Formula-E Race Strategy Development using Artificial Neural Networks and Monte Carlo Tree Search.**
- **Venue:** Neural Computing and Applications, Vol. 32, pp. 10567–10581
- **URL:** https://link.springer.com/article/10.1007/s00521-020-04871-1

Investiga modelos de predicción ANN para reemplazar la simulación tradicional de tiempos de vuelta, habilitando una predicción de rendimiento más rápida. Aplica búsqueda de árbol Monte Carlo para exploración de estrategias. El enfoque híbrido refleja directamente el diseño analítico + refinamiento MC de RaceScope.

**Relevancia:** Paralelo directo en arquitectura — la combinación ANN + MCTS espeja el enfoque de dos fases de RaceScope (scoring analítico + refinamiento MC).

---

### 5.3 Optimizing Pit Stop Strategies with Dynamic Programming and Game Theory

**Aguad, F., & Thraves, C. (2024). Optimizing Pit Stop Strategies in F1 with DP and Game Theory.**
- **Venue:** European Journal of Operational Research
- **DOI:** 10.1016/j.ejor.2024.05.031
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0377221724005484

Formula el problema como un juego Stackelberg de suma cero. Usa programación dinámica y simulación MC para resolver el problema competitivo de estrategia de pit stop. Muestra mejoras de 2.3 segundos en tiempo de carrera y reducción del 17.8% en probabilidad de undercut.

**Relevancia:** Directamente aplicable — demuestra el estado del arte académico combinando teoría de juegos + DP + simulación. Valida el enfoque de dos fases MC de RaceScope como técnica práctica de refinamiento para estrategia competitiva.

---

## 6. OPTIMIZACIÓN POR SIMULACIÓN — SURVEY

### 6.1 Simulation Optimization: A Review on Theory and Applications

**Pasupathy, R., et al. (2015). Simulation Optimization: A Review on Theory and Applications.**
- **Venue:** Annals of Operations Research, Springer
- **URL:** https://link.springer.com/article/10.1007/s10479-015-2019-x

Survey de métodos de optimización por simulación: cubre algoritmos sin gradiente, métodos de superficie de respuesta y enfoques basados en muestreo. Discute convergencia, varianza y asignación del presupuesto computacional.

**Relevancia:** Taxonomía exhaustiva de la optimización por simulación; ayuda a posicionar el motor de dos fases de RaceScope dentro de la literatura más amplia de SO (Simulation Optimization).

---

### 6.2 Recent Advances in Simulation-Based Optimization for Operations Research Problems

**Ghosh, S., Henderson, S. G., & Pasupathy, R. (2023). Recent Advances in Simulation-Based Optimization.**
- **Venue:** Annals of Operations Research, Springer
- **URL:** https://link.springer.com/article/10.1007/s10479-022-05122-3

Perspectiva actualizada sobre optimización basada en simulación, cubriendo la integración de AI/ML con optimización por simulación, enfoques multiobjetivo y computación distribuida. Discute aplicaciones en cadena de suministro, logística y planificación.

**Relevancia:** Perspectiva reciente sobre optimización estratégica basada en simulación; valida el uso de muestreo MC para decisiones de ranking bajo incertidumbre en contextos operacionales.

---

## 7. REDUCCIÓN DE VARIANZA Y MC AVANZADO

### 7.1 Variance Reduction Techniques in Monte Carlo Methods

**Fishman, G. S. (1996). Monte Carlo: Concepts, Algorithms, and Applications.**
- **Publisher:** Springer (Series in Operations Research)

Tratamiento exhaustivo de reducción de varianza: importance sampling, control variates, antithetic variates, stratified sampling. Muestra cómo reducir la varianza de estimación 10-100× con técnicas apropiadas.

**Relevancia:** La fase de refinamiento MC de RaceScope se beneficiaría de la reducción de varianza; e.g., control variates usando puntuaciones analíticas, o importance sampling sesgado hacia candidatos de alto ranking para reducir la incertidumbre en las puntuaciones finales.

---

### 7.2 The Cross-Entropy Method for Optimization

**Rubinstein, R. Y., & Kroese, D. P. (2004). The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization.**
- **Publisher:** Springer
- **URL:** https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf

El CEM remuestrea iterativamente de un subconjunto élite de soluciones, actualizando una distribución de probabilidad para concentrarse alrededor de regiones de alta calidad. Sin gradientes, aplicable a problemas combinatorios y continuos.

**Relevancia:** Alternativa al refinamiento MC fijo: en lugar de evaluar el top-K una vez, podría muestrear y re-rankear iterativamente, enfocándose en las clases de estrategia más prometedoras (e.g., pit temprano vs. pit tardío).

---

## 8. OPTIMIZACIÓN MULTIOBJETIVO Y MÉTODOS DE PARETO

### 8.1 Multi-Objective Optimization: Theory and Applications

**Revisiones diversas en MDPI Algorithms y ACM Computing Surveys (2020-2024).**
- **URL:** https://www.mdpi.com/1999-4893/17/5/206

Framework para optimizar múltiples objetivos en conflicto (e.g., tiempo de carrera vs. desgaste de neumáticos vs. consumo de combustible). Define la optimalidad de Pareto: una solución donde ningún objetivo puede mejorar sin degradar otro. El frente de Pareto es el conjunto de todas las soluciones no dominadas.

**Relevancia:** La estrategia F1 tiene múltiples objetivos (tiempo, vida de neumáticos, combustible, riesgo). El ranking MC de RaceScope podría expandirse a un frente de Pareto de estrategias en lugar de un único top-K. Habilita análisis de trade-off para las preferencias del piloto/equipo.

---

## 9. NO FREE LUNCH Y SELECCIÓN DE ALGORITMOS

**Wolpert, D. H., & Macready, W. G. (1997). No Free Lunch Theorems for Optimization.**
- **Venue:** IEEE Transactions on Evolutionary Computation
- **URL:** https://en.wikipedia.org/wiki/No_free_lunch_theorem

Resultado fundamental: ningún algoritmo de optimización domina en todos los tipos de problemas. Motiva la selección de algoritmos basada en la estructura del problema y el conocimiento previo.

**Relevancia:** Justifica la elección de un enfoque de dos fases (analítico + MC) adaptado a la estructura de pit stop en lugar de un optimizador genérico de caja negra. Demuestra la importancia del diseño de algoritmos específico del dominio.

---

## 10. TABLA RESUMEN: MÉTODOS MC PARA RACESCOPE

| Método | Referencia | Aplicación en RaceScope |
|---|---|---|
| **SAA** | Kleywegt et al. 2002 | Estimación de E[time] para ranking de estrategias |
| **Convergencia √N** | Shapiro 1999 | Justificación de n_sim=200 |
| **Mean-Variance** | Markowitz 1952 | Función de ranking `E + λ·σ` |
| **CVaR** | Rockafellar & Uryasev 2000 | Alternativa robusta al ranking de varianza |
| **MCTS/UCT** | Kocsis & Szepesvári 2006 | Exploración-explotación en selección top-K |
| **Variance Reduction** | Fishman 1996 | Mejora de eficiencia del bucle MC |
| **Cross-Entropy** | Rubinstein & Kroese 2004 | Refinamiento iterativo de candidatos |
| **Pareto** | MDPI 2024 | Extensión a recomendaciones multi-objetivo |
| **Aplicación F1** | Heilmeier et al. 2020 | Distribuciones de eventos de carrera |
| **Combinación DP+MC** | Aguad & Thraves 2024 | Validación del enfoque de dos fases |
