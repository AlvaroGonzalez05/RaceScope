# Estado del Arte en Forecasting de Series Temporales

## Relevancia para RaceScope Strategy Lab

RaceScope utiliza LSTM para predicción de series temporales de tiempos de vuelta en F1. Esta sección posiciona esa elección dentro del panorama actual (2020-2025) de arquitecturas para forecasting de series temporales, incluyendo las competiciones de referencia (M4, M5), modelos fundacionales, y el debate sobre si los Transformers realmente superan a los métodos más simples en este dominio.

---

## 1. COMPETICIONES DE FORECASTING Y BENCHMARKS

### 1.1 M4 Forecasting Competition

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 Time Series and 61 Forecasting Methods.**
- **Venue:** International Journal of Forecasting, Vol. 36, Issue 1, pp. 54-74
- **DOI:** 10.1016/j.ijforecast.2019.04.014
- **URL:** https://www.sciencedirect.com/science/article/pii/S0169207019301128

La competición M4 con 100,000 series temporales reales demostró que en 2018 los métodos de deep learning puro (incluyendo LSTM) NO superaban a los métodos estadísticos clásicos. El método ganador (de Uber/Smyl) fue un híbrido que combinaba Exponential Smoothing con LSTM. Esta conclusión motivó una revisión crítica de cuándo los métodos de DL son apropiados.

**Relevancia:** Pone en perspectiva el uso de LSTM en RaceScope — en datasets pequeños o con estructura estadística clara, el LSTM puro puede no ser la mejor opción. El enfoque híbrido del motor de dos fases de RaceScope (modelos paramétricos + LSTM) refleja la lección principal de M4: combinar métodos estadísticos con DL funciona mejor.

---

### 1.2 M5 Forecasting Competition: El Auge del Deep Learning

**Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 Accuracy Competition: Results, Findings and Conclusions.**
- **URL:** https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf

La competición M5 (2020) con datos de ventas de Walmart mostró que los métodos de machine learning (LightGBM, XGBoost, redes neuronales) eran muy competitivos y en muchos casos superaban a los métodos estadísticos. Esto marcó un punto de inflexión en el que DL demostró valor en forecasting de series temporales a gran escala.

**Relevancia:** La evolución M4→M5 ilustra que con suficientes datos y complejidad del problema, el DL (incluyendo LSTM) sí aporta valor. Los datos de F1 de múltiples temporadas y pilotos justifican el uso de LSTM en RaceScope.

---

### 1.3 Monash Time Series Forecasting Archive

**Godahewa, R., et al. (2021). Monash Time Series Forecasting Archive.**
- **Venue:** NeurIPS 2021 (Datasets and Benchmarks Track)
- **arXiv:** 2105.06643
- **URL:** https://arxiv.org/abs/2105.06643

Archivo estandarizado de 58 datasets de series temporales de dominio público para benchmarking de métodos de forecasting. Incluye datos de deportes, tráfico, energía, y otros dominios. Proporciona el framework de evaluación estándar de la comunidad.

**Relevancia:** Establece el protocolo de evaluación estándar para comparar métodos de forecasting. Relevante para la evaluación rigurosa del modelo LSTM de RaceScope frente a baselines.

---

## 2. EL DEBATE LSTM vs. TRANSFORMERS vs. MLP EN SERIES TEMPORALES

### 2.1 Are Transformers Effective for Time Series Forecasting?

**Zeng, A., et al. (2023). Are Transformers Effective for Time Series Forecasting?**
- **Venue:** AAAI 2023
- **arXiv:** 2205.13504
- **URL:** https://arxiv.org/abs/2205.13504

Paper provocativo que cuestiona el uso de Transformers para forecasting de series temporales. Demuestra que un modelo lineal simple (DLinear) supera a Autoformer, FEDformer y otros Transformers en múltiples benchmarks. La conclusión es que muchos Transformers para series temporales sobreajustan a los benchmarks evaluados.

**Relevancia:** Advertencia importante sobre el uso acrítico de arquitecturas complejas para series temporales. Sugiere que para RaceScope, el modelo LSTM + perfil paramétrico puede ser suficiente y que las mejoras deberían validarse cuidadosamente contra baselines simples.

---

### 2.2 Revisiting Long-Term Time Series Forecasting: An Investigation on Linear Mapping

**Li, Z., et al. (2023). Revisiting Long-Term Time Series Forecasting: An Investigation on Linear Mapping.**
- **arXiv:** 2305.14535
- **URL:** https://arxiv.org/abs/2305.14535

Investiga por qué las proyecciones lineales simples son tan competitivas con Transformers en forecasting de largo plazo. Argumenta que muchos benchmarks tienen alta estacionariedad que favorece a los modelos lineales. Proporciona análisis de cuándo los modelos no lineales (LSTM, Transformers) aportan valor real.

**Relevancia:** Relevante para entender cuándo el LSTM de RaceScope aporta valor frente a modelos lineales simples. La degradación de neumáticos tiene componentes no lineales (curva de degradación en forma de "S") donde el LSTM debería superar a la regresión lineal.

---

## 3. MODELOS FUNDACIONALES PARA SERIES TEMPORALES

### 3.1 TimesFM: Modelo Fundacional de Google para Series Temporales

**Das, A., et al. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting.**
- **Venue:** ICML 2024
- **arXiv:** 2310.10688
- **URL:** https://arxiv.org/abs/2310.10688
- **Blog:** https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- **GitHub:** https://github.com/google-research/timesfm

TimesFM (200M parámetros) es un Transformer solo-decodificador pre-entrenado en más de 100,000 millones de puntos temporales reales. Procesa patches de 32 puntos temporales consecutivos como tokens. Rendimiento en zero-shot competitivo con modelos supervisados. TimesFM 2.0 extiende el contexto a 2048 puntos, mejorando un 25%.

**Relevancia:** Representa el estado del arte actual (2024) en modelos fundacionales para series temporales. Para RaceScope, TimesFM podría proporcionar una capacidad de predicción de zero-shot para nuevos circuitos o pilotos sin datos históricos suficientes, complementando el sistema de fallback actual.

---

### 3.2 MOIRAI: Modelo Fundacional de Salesforce para Forecasting Universal

**Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers.**
- **Venue:** ICML 2024
- **arXiv:** 2402.02592
- **URL:** https://arxiv.org/abs/2402.02592
- **Blog:** https://www.salesforce.com/blog/moirai/
- **GitHub:** https://github.com/SalesforceAIResearch/uni2ts

MOIRAI es entrenado en 27,000 millones de observaciones de 9 dominios diferentes. Soporta datos de cualquier variante (longitud variable), frecuencias arbitrarias y atención any-variate para datos heterogéneos. MOIRAI 2.0 logra el ranking #1 en GIFT-Eval leaderboard. MOIRAI-MoE es una variante de Mixture-of-Experts con 65x menos parámetros activos que los competidores.

**Relevancia:** MOIRAI puede manejar los datos heterogéneos de F1 (temperaturas de pista, combustible, tipo de compuesto como features adicionales). Podría proporcionar predicciones de zero-shot para el sistema de fallback de RaceScope.

---

### 3.3 Lag-Llama: Primer Modelo Fundacional Open-Source para Forecasting Probabilístico

**Rasul, K., et al. (2023). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.**
- **arXiv:** 2310.08278
- **URL:** https://arxiv.org/abs/2310.08278
- **GitHub:** https://github.com/time-series-foundation-models/lag-llama

Lag-Llama es el primer modelo fundacional open-source para forecasting probabilístico de series temporales. Usa un Transformer solo-decodificador con features de lag (similar a ARIMA en concepto) como covariables. Pre-entrenado en un corpus diverso; generalización zero-shot fuerte. Produce distribuciones de probabilidad (no solo predicciones puntuales), habilitando cuantificación de incertidumbre.

**Relevancia:** El forecasting probabilístico de Lag-Llama se alinea con el scoring ajustado al riesgo de RaceScope (`E[time] + λ*σ`). Las distribuciones de probabilidad output podrían alimentar directamente el cálculo de varianza en la fase de scoring analítico.

---

## 4. ARQUITECTURAS AVANZADAS: REVISIÓN CRONOLÓGICA

### 4.1 N-BEATS (2020)

**Oreshkin, B. N., et al. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.**
- **Venue:** ICLR 2020
- **arXiv:** 1905.10437
- **URL:** https://arxiv.org/abs/1905.10437

Primer modelo de DL puro que supera de forma consistente a los métodos estadísticos estándar en la competición M4. Usa bloques fully-connected apilados con enlaces residuales (backcast + forecast). Versión interpretable: aprende componentes de tendencia (polinomios) + estacionalidad (armónicos de Fourier). Agnóstico al dominio.

---

### 4.2 N-HiTS (2023)

**Challu, C., et al. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.**
- **Venue:** AAAI 2023
- **arXiv:** 2201.12886
- **URL:** https://arxiv.org/abs/2201.12886

Extiende N-BEATS con muestreo multi-tasa e interpolación jerárquica para descomponer series temporales en diferentes componentes de frecuencia. Logra mejora del 20% en precisión frente a Transformers con computación 50x más rápida. Relevante para la modelización multi-escala de la degradación de neumáticos (variaciones de vuelta a vuelta + tendencia de stint).

---

### 4.3 Temporal Fusion Transformer (2021)

**Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.**
- **Venue:** International Journal of Forecasting, Vol. 37, Issue 4
- **DOI:** 10.1016/j.ijforecast.2021.03.012
- **URL:** https://arxiv.org/abs/1912.09363

TFT combina redes de selección de variables (VSN), LSTM para procesamiento local, y self-attention para dependencias de largo rango. Soporta inputs estáticos (habilidad del piloto), observados (tiempo de vuelta actual) y conocidos futuros (predicción meteorológica). Mejora del 36-69% frente a DeepAR. Interpretabilidad nativa de features y temporalidad.

**Relevancia directa para RaceScope:** La VSN de TFT podría seleccionar automáticamente qué features (temperatura, stint_age, compuesto) son más relevantes para cada predicción. La distinción entre inputs estáticos (perfil de piloto) y dinámicos (datos de carrera en tiempo real) se alinea perfectamente con la arquitectura de RaceScope.

---

### 4.4 PatchTST (2023)

**Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.**
- **Venue:** ICLR 2023
- **arXiv:** 2211.14730
- **URL:** https://arxiv.org/abs/2211.14730

PatchTST segmenta series temporales en patches (análogos a palabras en NLP) y aplica Transformers a nivel de patch. Logra mejora del 21% en MSE con reducción cuadrática de la computación. El procesamiento independiente por canal permite modelado separado de cada feature.

**Relevancia para RaceScope:** El concepto de patch podría segmentar las carreras en fases estratégicas distintas. Procesamiento por canal permitiría modelos independientes para cada feature (temperatura, stint_age, etc.) que luego se combinan.

---

## 5. TRANSFERENCIA DE APRENDIZAJE EN SERIES TEMPORALES

### 5.1 Transfer Learning with Foundational Models for Time Series

**Arxiv (2024). Transfer Learning with Foundational Models for Time Series Forecasting using Low-Rank Adaptations.**
- **arXiv:** 2410.11539
- **URL:** https://arxiv.org/abs/2410.11539

Los modelos fundacionales pre-entrenados en corpus diverso de series temporales pueden fine-tunearse en el dominio objetivo con muestras mínimas etiquetadas. Las adaptaciones de bajo rango (LoRA) reducen los parámetros entrenables. Habilita despliegue rápido a nuevos contextos de motorsport sin reentrenamiento desde cero.

**Relevancia:** Solución al problema de cold-start en RaceScope para nuevos pilotos o circuitos con datos insuficientes. En lugar del fallback actual al modelo global, se podría usar un modelo fundacional fine-tuneado.

---

### 5.2 Transfer Learning Across Datasets for Time Series Forecasting

**Fawaz, H. I., et al. (2020). Transfer Learning for Time Series Classification.**
- **Venue:** IEEE International Conference on Big Data 2018
- **URL:** https://arxiv.org/abs/1811.01533

Pre-entrenar modelos en datasets fuente de gran tamaño y fine-tunear en dominios objetivo más pequeños. Los modelos de DL entrenados en datasets grandes generalizan a dominios más pequeños. La jerarquía de representaciones es más difícil de aprender en series temporales que en imágenes.

**Relevancia:** El comportamiento de los pilotos entre circuitos exhibe patrones compartidos (degradación de neumáticos, adelantamientos). La transferencia entre circuitos podría arrancar las predicciones de Singapore usando datos de Monaco.

---

## 6. NORMALIZACIÓN E INGENIERÍA DE FEATURES

### 6.1 Reversible Instance Normalization (RevIN)

**Kim, T., et al. (2021). Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift.**
- **Venue:** ICLR 2022
- **arXiv:** 2105.11199
- **URL:** https://arxiv.org/abs/2105.11199

RevIN normaliza instancias individuales (no el dataset completo) y desnormaliza las predicciones al final. Maneja cambios de distribución (distribution shift) entre entrenamiento y test. Mejora del 18% en MSE en benchmarks de forecasting.

**Relevancia:** La normalización z-score global de RaceScope puede verse como una versión simplificada de RevIN. Implementar RevIN podría mejorar la robustez a cambios de condiciones entre temporadas o circuitos muy diferentes.

---

### 6.2 Cyclical Encoding para Features Temporales

**NVIDIA Blog (2021). Three Approaches to Encoding Time Information as Features for ML Models.**
- **URL:** https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/

Las features cíclicas (hora del día, vuelta de la carrera, posición en el campeonato) deben codificarse como pares (sin, cos) para preservar la distancia circular. La codificación ordinal crea discontinuidades artificiales (vuelta 57→1 en una carrera de 57 vueltas). La codificación cíclica mejora el rendimiento de los modelos de ML en estas features.

**Relevancia:** El número de vuelta en RaceScope (feature lap_number) podría beneficiarse de codificación cíclica cuando se modelan patrones de carrera que dependen de la posición relativa en la carrera (últimas vueltas vs. primeras).

---

## 7. DEBATE ACTUAL: ¿NECESITAMOS ARQUITECTURAS COMPLEJAS?

### El Argumento a Favor de la Simplicidad

Varios papers recientes (2023-2024) cuestionan la complejidad de las arquitecturas de forecasting modernas:

1. **DLinear** (Zeng et al. 2023): Un modelo lineal supera a muchos Transformers
2. **TimesNet** (Wu et al. 2023): Las CNNs 2D para series temporales 1D son competitivas
3. **FITS** (Xu et al. 2023): 10k parámetros superan a Transformers de millones

### La Respuesta del Dominio

Para el caso de RaceScope, las series temporales de F1 tienen características específicas que justifican el LSTM:

- **No estacionariedad**: La distribución de tiempos de vuelta cambia con degradación de neumáticos, combustible y condiciones de pista → necesita capacidad no lineal
- **Dependencias a largo plazo**: El stint_age a vuelta 35 depende de las condiciones desde vuelta 1 → necesita memoria a largo plazo
- **Heterogeneidad**: Diferentes pilotos, circuitos y compuestos → necesita representaciones aprendidas
- **Dataset pequeño**: ~100-300 vueltas por piloto/circuito → regularización es crítica

### Conclusión

El LSTM es una elección defensible y bien justificada para RaceScope, especialmente considerando:
1. El tamaño del dataset (pequeño, específico por piloto/circuito)
2. La naturaleza no estacionaria de los datos
3. Las dependencias temporales dentro del stint
4. La necesidad de interpretabilidad para estrategas de F1

Los modelos fundacionales (TimesFM, MOIRAI) representan el siguiente paso natural cuando se disponga de más datos históricos o se necesite capacidad de zero-shot.
