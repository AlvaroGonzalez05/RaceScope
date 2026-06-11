# Long Short-Term Memory (LSTM) y Deep Learning para Series Temporales

## Relevancia para RaceScope Strategy Lab

RaceScope utiliza redes LSTM (`app/models_lstm.py`) para predecir tiempos por vuelta y degradación de neumáticos. El modelo `LSTMPaceNet` tiene arquitectura `nn.LSTM(input_dim, hidden_dim=64) → nn.Linear(64, 1)`, entrenado con loss MSE y optimizador Adam (lr=1e-3). Procesa ventanas de 10 vueltas (context window) con 8 features de entrada: lap_number, stint_age, compound, session_type, circuit_id, track_temp, air_temp, lap_time (normalizado con z-score).

---

## 1. PAPERS FUNDACIONALES

### 1.1 Long Short-Term Memory (LSTM) — Paper Original

**Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.**
- **Venue:** Neural Computation, Vol. 9, No. 8, pp. 1735–1780
- **DOI:** 10.1162/neco.1997.9.8.1735
- **URL:** https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory

El paper seminal que introduce Long Short-Term Memory, resolviendo el problema del gradiente desvaneciente en RNNs estándar mediante carruseles de error constante y unidades de puerta multiplicativas. Las LSTMs usan puertas forget, input y output para controlar el flujo de información a través de celdas de memoria, permitiendo aprender dependencias que abarcan más de 1000 pasos temporales. La arquitectura demuestra que redes recurrentes pueden capturar dependencias a largo plazo en secuencias complejas.

**Relevancia para el TFG:** Arquitectura central del motor de simulación de RaceScope para modelar la dinámica temporal de los tiempos por vuelta y la degradación de neumáticos. Los mecanismos de puerta son esenciales para capturar variaciones a corto plazo (vuelta a vuelta) y tendencias de largo plazo (evolución del stint).

---

### 1.2 Finding Structure in Time — RNN Original

**Elman, J. L. (1990). Finding Structure in Time.**
- **Venue:** Cognitive Science, Vol. 14, No. 2, pp. 179–211
- **URL:** https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1

La Simple Recurrent Network (SRN) de Elman es el trabajo fundacional sobre procesamiento de datos secuenciales mediante conexiones recurrentes. Demuestra cómo los patrones de unidades ocultas retroalimentados a sí mismos permiten a las redes desarrollar representaciones internas que reflejan la estructura temporal. Trabajo pionero en usar redes neuronales para aprendizaje de secuencias sin unidades lingüísticas explícitas.

**Relevancia:** Proporciona la base teórica para entender cómo las conexiones recurrentes capturan la dinámica temporal en escenarios de carrera. Los principios de la SRN subyacen a las arquitecturas LSTM modernas usadas en el motor de estrategia.

---

### 1.3 Vanishing Gradient — La Motivación Teórica de LSTM

**Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent Is Difficult.**
- **Venue:** IEEE Transactions on Neural Networks, Vol. 5, pp. 157–166
- **DOI:** 10.1109/72.279181
- **URL:** https://ieeexplore.ieee.org/document/279181/

Analiza formalmente por qué el descenso de gradiente estándar falla al aprender dependencias a largo plazo en RNNs. Los autores demuestran que los gradientes se desvanecen o explotan exponencialmente durante backpropagation through time (BPTT), haciendo progresivamente difícil capturar dependencias más allá de unos pocos pasos temporales. Este análisis motivó directamente el desarrollo de LSTM.

**Relevancia:** Crítico para entender por qué se necesitó LSTM para modelar tiempos de vuelta a lo largo de distancias de carrera completas. Explica los desafíos computacionales en el entrenamiento de perfiles de piloto.

---

### 1.4 On the Difficulty of Training Recurrent Neural Networks

**Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks.**
- **Venue:** Proceedings of ICML 2013
- **arXiv:** 1211.5063
- **URL:** https://arxiv.org/abs/1211.5063

Analiza los problemas de gradiente desvaneciente y explosivo desde perspectivas analítica, geométrica y de sistemas dinámicos. Propone el gradient norm clipping como solución efectiva para gradientes explosivos, convirtiéndose en práctica estándar en el entrenamiento de RNNs. Proporciona análisis formal sobre cómo los gradientes de error decaen o se amplifican durante BPTT.

**Relevancia:** Justifica la implementación de gradient clipping en el bucle de entrenamiento PyTorch. Explica por qué LSTM fue elegido frente a RNN vainilla para modelar stints de 300+ vueltas.

---

### 1.5 Generating Sequences With Recurrent Neural Networks

**Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.**
- **arXiv:** 1308.0850
- **URL:** https://arxiv.org/abs/1308.0850

Demuestra cómo las redes LSTM pueden generar secuencias complejas con estructura de largo rango prediciendo un punto de datos a la vez. Extiende la generación de secuencias a la transducción de secuencias, habilitando transformaciones input-output sin alineaciones predefinidas. Este framework demostró ser fundamental para muchas aplicaciones sequence-to-sequence.

**Relevancia:** Relevante para entender cómo LSTM puede adaptarse para predicción de secuencias de tiempos de vuelta. El framework de generación de secuencias informa la fase de scoring analítico que rankea estrategias de pit stop.

---

## 2. VARIANTES Y MEJORAS DE LSTM

### 2.1 Gated Recurrent Units (GRU)

**Cho, K., et al. (2014). Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation.**
- **Venue:** EMNLP 2014
- **arXiv:** 1406.1078
- **URL:** https://arxiv.org/abs/1406.1078

Introduce el Gated Recurrent Unit (GRU), una variante simplificada de LSTM con menos parámetros. Las GRUs combinan las puertas forget e input en una sola "reset gate" y fusionan el estado de celda con el estado oculto. Con solo dos puertas frente a las tres de LSTM, las GRUs son computacionalmente más ligeras manteniendo rendimiento comparable en tareas de modelado de secuencias.

**Relevancia:** GRU ofrece una alternativa más ligera a LSTM para el modelado de degradación de neumáticos y tiempos de vuelta. Potencialmente útil para la eficiencia de despliegue en el backend FastAPI manteniendo la capacidad de modelado temporal.

---

**Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.**
- **arXiv:** 1412.3555
- **URL:** https://arxiv.org/abs/1412.3555

Proporciona una comparación empírica exhaustiva entre GRU y LSTM en múltiples datasets y tareas. El estudio revela que ninguna arquitectura supera consistentemente a la otra, con el rendimiento variando por aplicación y selección de hiperparámetros.

**Relevancia:** Valida que la elección de LSTM para RaceScope es justificable, aunque GRU podría explorarse como alternativa más ligera.

---

### 2.2 Bidirectional LSTM

**Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks.**
- **Venue:** IEEE Transactions on Signal Processing, Vol. 45, pp. 2673–2681
- **DOI:** 10.1109/78.650093
- **URL:** https://dl.acm.org/doi/10.1109/78.650093

Los RNNs bidireccionales procesan secuencias en dirección temporal tanto hacia adelante como hacia atrás, permitiendo a las redes acceder al contexto futuro al hacer predicciones. Esta arquitectura demostró ser particularmente efectiva para tareas donde la secuencia completa está disponible antes del procesamiento.

**Relevancia:** Podría mejorar el modelado de degradación de neumáticos al permitir que la red considere condiciones de carrera pasadas y futuras. Relevante en análisis offline donde el historial completo de carrera está disponible.

---

### 2.3 Mecanismos de Atención con LSTM

**Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.**
- **Venue:** ICLR 2015
- **arXiv:** 1409.0473
- **URL:** https://arxiv.org/abs/1409.0473

Introduce el mecanismo de atención (atención aditiva de Bahdanau) para modelos sequence-to-sequence. En lugar de comprimir secuencias de entrada completas en vectores de contexto de tamaño fijo, la atención permite que los decodificadores se enfoquen dinámicamente en diferentes elementos de entrada al generar cada salida.

**Relevancia:** Los mecanismos de atención podrían mejorar el modelo de perfil de piloto al enfocarse en fases críticas de carrera (qualificación, configuración de neumáticos) para predecir tiempos de vuelta.

---

## 3. LSTM vs. ALTERNATIVAS MODERNAS (Estado del Arte)

### 3.1 Attention Is All You Need — Transformers

**Vaswani, A., et al. (2017). Attention Is All You Need.**
- **Venue:** NeurIPS 2017
- **arXiv:** 1706.03762
- **URL:** https://arxiv.org/abs/1706.03762

Paper revolucionario que introduce la arquitectura Transformer, reemplazando la recurrencia completamente con mecanismos de multi-head self-attention. Los Transformers logran mayor paralelizabilidad comparados con RNNs, reduciendo dramáticamente el tiempo de entrenamiento mientras alcanzan rendimiento state-of-the-art en tareas de secuencia.

**Relevancia:** Mientras RaceScope usa LSTM, los Transformers representan el estándar actual para modelado de secuencias. Podrían considerarse como alternativa para predicción de tiempos de vuelta si la velocidad de inferencia en tiempo real es menos crítica que la precisión.

---

### 3.2 Temporal Convolutional Networks (TCN)

**Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.**
- **arXiv:** 1803.01271
- **URL:** https://arxiv.org/abs/1803.01271

Compara sistemáticamente TCNs con LSTMs en benchmarks diversos de modelado de secuencias. Las TCNs usan convoluciones causales y dilatadas para capturar dependencias temporales manteniendo eficiencia computacional. El estudio demuestra que las TCNs frecuentemente superan a LSTMs en benchmarks estándar con mejor paralelización y mayor memoria efectiva.

**Relevancia:** Las TCNs ofrecen una alternativa computacionalmente eficiente con potencialmente mayor receptividad temporal. Podrían explorarse para predicciones de estrategia multi-vuelta donde se necesitan horizontes temporales más largos.

---

### 3.3 N-BEATS

**Oreshkin, B. N., et al. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.**
- **Venue:** ICLR 2020
- **arXiv:** 1905.10437
- **URL:** https://arxiv.org/abs/1905.10437

Introduce un enfoque de deep learning puro para forecasting de series temporales univariadas usando capas fully-connected con enlaces residuales hacia adelante y atrás. La arquitectura es interpretable, agnóstica al dominio, y logra rendimiento state-of-the-art en competiciones de predicción importantes (M3, M4).

**Relevancia:** Altamente relevante para predicción univariada de tiempos de vuelta. El framework de expansión de base interpretable podría explicar las elecciones de estrategia a los usuarios.

---

### 3.4 Informer: Efficient Transformer for Long-Sequence Time-Series Forecasting

**Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.**
- **Venue:** AAAI 2021 (Best Paper Award)
- **arXiv:** 2012.07436
- **URL:** https://arxiv.org/abs/2012.07436

Aborda las limitaciones computacionales de los Transformers para secuencias largas introduciendo ProbSparse self-attention (complejidad O(L log L)) y attention distilling. La arquitectura descompone series temporales en componentes de tendencia y estacionalidad.

**Relevancia:** Podría manejar eficientemente escenarios de forecasting multi-carrera. La descomposición tendencia-estacional se alinea naturalmente con los patrones de degradación de neumáticos de F1.

---

### 3.5 Autoformer

**Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.**
- **Venue:** NeurIPS 2021
- **arXiv:** 2106.13008
- **URL:** https://arxiv.org/abs/2106.13008

Mejora Informer mediante descomposición estacional-tendencia usando media móvil y mecanismos de auto-correlación que reemplazan el self-attention. La arquitectura descompone progresivamente series en tendencias y componentes estacionales durante el forecasting. Implementado en los Juegos Olímpicos de Invierno 2022 para predicción meteorológica.

**Relevancia:** La descomposición estacional-tendencia captura naturalmente el comportamiento de neumáticos F1: tendencia de degradación inicial más variaciones periódicas entre stints.

---

### 3.6 S4: Structured State Space Models

**Gu, A., Goel, K., & Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces.**
- **Venue:** ICLR 2022
- **arXiv:** 2111.00396
- **URL:** https://arxiv.org/abs/2111.00396

S4 aprovecha modelos de espacio de estados en tiempo continuo parametrizados mediante descomposición diagonal más de bajo rango para modelar eficientemente secuencias largas. Logra state-of-the-art en Long Range Arena, manejando secuencias de longitud 16K donde todos los métodos anteriores fallan.

**Relevancia:** Ofrece complejidad O(N) para secuencias largas. Podría modelar dependencias multi-carrera o estrategias de temporada completa con superior eficiencia comparado con la complejidad O(N²) de LSTM.

---

### 3.7 Mamba: Selective State Space Models

**Gu, A., & Dao, C. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.**
- **arXiv:** 2312.00752
- **URL:** https://arxiv.org/abs/2312.00752

Extiende S4 con transiciones de estado dependientes del input mediante un mecanismo de selección, permitiendo al modelo enfocarse o suprimir selectivamente información. Logra 5x mayor throughput que Transformers con complejidad lineal en tiempo.

**Relevancia:** Representa la arquitectura post-Transformer de vanguardia con complejidad lineal. Ideal para procesar datasets completos de carrera o múltiples temporadas con overhead computacional mínimo.

---

### 3.8 PatchTST

**Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.**
- **Venue:** ICLR 2023
- **arXiv:** 2211.14730
- **URL:** https://arxiv.org/abs/2211.14730

PatchTST segmenta series temporales en patches (palabras) y aplica Transformers a nivel de patch en lugar de punto. Logra 21% de mejora en MSE y 16.7% en MAE respecto a enfoques Transformer anteriores reduciendo la computación cuadráticamente.

**Relevancia:** El concepto de patching podría segmentar carreras en fases estratégicas (vuelta de apertura, gestión de neumáticos, clímax). El procesamiento independiente por canal permite modelado independiente de piloto y compuesto.

---

## 4. ENTRENAMIENTO DE LSTM — MEJORES PRÁCTICAS

### 4.1 Adam Optimizer

**Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.**
- **arXiv:** 1412.6980
- **URL:** https://arxiv.org/abs/1412.6980

Introduce el optimizador Adam (Adaptive Moment Estimation), combinando ventajas de AdaGrad y RMSProp. Adam adapta learning rates por parámetro basándose en estimaciones de primer y segundo momento, habilitando convergencia rápida con poca sintonización de hiperparámetros. Se ha convertido en el optimizador por defecto para deep learning.

**Relevancia:** RaceScope usa Adam (lr=1e-3) en el pipeline de entrenamiento para perfiles de piloto y modelos globales. Proporciona scheduling de learning rate adaptativo sin sintonización explícita.

---

### 4.2 Dropout en RNNs

**Gal, Y., & Ghahramani, Z. (2016). A Theoretically Grounded Application of Dropout in Recurrent Neural Networks.**
- **Venue:** NeurIPS 2016
- **arXiv:** 1512.05287
- **URL:** https://arxiv.org/abs/1512.05287

Proporciona justificación teórica para aplicar dropout a RNNs, demostrando que aplicar la misma máscara de dropout a todos los pasos temporales equivale a inferencia variacional aproximada. Propone schedules de dropout concretos para LSTMs.

**Relevancia:** Crítico para RaceScope ya que los modelos específicos de piloto entrenan en datasets relativamente pequeños (un piloto, un circuito, ~100-200 vueltas). Proporciona un enfoque principiado a tasas de dropout que mantienen la coherencia temporal.

---

### 4.3 Layer Normalization

**Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.**
- **arXiv:** 1607.06450
- **URL:** https://arxiv.org/abs/1607.06450

Propone la normalización de capa como alternativa a la normalización de lote que normaliza activaciones dentro de cada muestra en lugar de a través del lote. Layer norm funciona efectivamente con RNNs y tamaños de lote pequeños, eliminando la dependencia de estadísticas de lote.

**Relevancia:** Preferible a la normalización de lote para el entrenamiento de LSTM con datasets pequeños específicos de circuito. Proporciona entrenamiento estable sin dependencia de lote.

---

### 4.4 Early Stopping

**Prechelt, L. (1998). Early Stopping - But When?**
- **Venue:** Neural Networks: Tricks of the Trade (Springer)
- **URL:** https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks.pdf

Analiza el early stopping como método de regularización, comparando criterios de parada (plateau del error de validación, estancamiento del gradiente). Demuestra que el early stopping es teóricamente equivalente a la regularización L2.

**Relevancia:** El early stopping debe implementarse al entrenar perfiles de piloto en datasets pequeños (--min-laps 160) para prevenir sobreajuste. Ayuda a equilibrar precisión versus generalización a nuevas condiciones de carrera.

---

## 5. LIBROS Y RECURSOS DE REFERENCIA

### 5.1 Deep Learning — Goodfellow et al.

**Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.**
- **Publisher:** MIT Press
- **URL:** https://www.deeplearningbook.org/

El libro de texto definitivo sobre deep learning proporciona cobertura exhaustiva de RNNs, LSTMs y modelado de secuencias para regresión. Incluye orientación práctica sobre diseño de arquitecturas, flujo de gradientes y procedimientos de entrenamiento para modelos temporales.

**Relevancia:** Referencia esencial para los fundamentos de LSTM en contextos multivariados. Cubre el framework matemático subyacente al modelado de tiempos de vuelta y degradación de neumáticos con múltiples canales de input.

---

### 5.2 The Elements of Statistical Learning

**Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd Ed.).**
- **Publisher:** Springer
- **URL:** https://hastie.su.domains/ElemStatLearn/

Libro de texto definitivo sobre aprendizaje estadístico. El capítulo 7 cubre regresión y funciones de pérdida. El MSE se presenta como la pérdida estándar para regresión por su tractabilidad matemática (convexidad, diferenciabilidad) y alineación con la estimación de mínimos cuadrados.

**Relevancia:** MSE como objetivo de optimización para regresión de tiempos de vuelta en RaceScope. Entender las propiedades del MSE ayuda a explicar por qué los tiempos de vuelta atípicos (por accidentes o pit stops) necesitan filtrado antes del entrenamiento LSTM.

---

## 6. TABLA COMPARATIVA DE ARQUITECTURAS

| Arquitectura | Año | Complejidad | Ventaja Principal | Desventaja | Relevancia para TFG |
|---|---|---|---|---|---|
| **LSTM** | 1997 | O(N·H²) | Resuelve gradiente desvaneciente | Inferencia secuencial lenta | Implementación actual ✓ |
| **GRU** | 2014 | O(N·H²) | Menos parámetros que LSTM | Ligeramente menos expresivo | Alternativa ligera |
| **Bi-LSTM** | 1997 | O(N·H²) | Contexto futuro disponible | Requiere secuencia completa | Mejora análisis offline |
| **Transformer** | 2017 | O(N²·H) | Paralelizable; patrones globales | Memoria intensiva | Alternativa moderna |
| **TCN** | 2018 | O(N·H²) | Paralelizable; mayor memoria efectiva | Campo receptivo fijo | Alternativa eficiente |
| **N-BEATS** | 2020 | O(N·H²) | Expansión de base interpretable | Justificación teórica limitada | Baseline fuerte |
| **Informer** | 2021 | O(N log N) | Eficiente; descomposición tendencia-estacional | Arquitectura compleja | Forecasting largo rango |
| **Autoformer** | 2021 | O(N log N) | Auto-correlación + descomposición | Asunción estacional | Natural para degradación |
| **S4** | 2021 | O(N) | Complejidad lineal; secuencias muy largas | Parametrización compleja | Escalado multi-carrera |
| **Mamba** | 2023 | O(N) | Atención selectiva; SOTA | Reciente (validación pendiente) | Elección a futuro |
| **PatchTST** | 2023 | O(N log N) | Patching reduce complejidad; pre-entrenamiento | Asunción de independencia de canal | Estrategias multi-fase |

---

## 7. PAPER DIRECTAMENTE APLICADO A F1 + LSTM

**Miao, Y., et al. (2025). Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy.**
- **arXiv:** 2501.04067
- **URL:** https://arxiv.org/abs/2501.04067

Aborda directamente la predicción de energía de neumáticos en F1 usando deep learning de series temporales con características de explicabilidad. Propone modelos que predicen la degradación del rendimiento de neumáticos a lo largo de stints de carrera. Integra la física del modelo de neumáticos con predicciones de redes neuronales, habilitando recomendaciones de estrategia interpretables.

**Relevancia:** Paper de aplicación de alta relevancia que demuestra el despliegue industrial de modelos de series temporales para estrategia F1. Valida que el deep learning (incluyendo variantes LSTM/RNN) es apropiado para predicción de neumáticos, el diferenciador central de RaceScope.

---

## RESUMEN: JUSTIFICACIÓN DE ELECCIÓN DE LSTM PARA RACESCOPE

La elección de LSTM en RaceScope para predicción de tiempos de vuelta y degradación de neumáticos está sólidamente fundamentada en la investigación de deep learning:

1. **Fiabilidad demostrada** en modelado de secuencias (27+ años de investigación)
2. **Interpretabilidad** a través de mecanismos de puerta que modelan naturalmente la degradación de neumáticos
3. **Eficiencia práctica** para cálculos de estrategia en tiempo real en peticiones `/api/strategy`
4. **Adaptación a datasets pequeños** mediante dropout y regularización, crítico para entrenamiento específico por circuito

Para mejoras futuras, considerar:
- **Arquitecturas híbridas**: Combinar encoder LSTM con decoder Transformer para generación de estrategias
- **Cuantificación de incertidumbre**: Implementar regresión por cuantiles para decisiones sensibles al riesgo
- **Evolución arquitectónica**: Test A/B de N-BEATS o PatchTST como comparaciones baseline
