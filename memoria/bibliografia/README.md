# Bibliografía — RaceScope Strategy Lab

**TFG:** Business Analytics + GITT
**Título:** RaceScope Strategy Lab — F1 Pre-Race Strategy Explorer
**Autor:** Álvaro González Tabernero
**Institución:** ICAI, Universidad Pontificia Comillas
**Fecha de compilación:** Marzo 2026

---

## Descripción

Esta carpeta contiene la bibliografía completa del TFG, organizada por bloques temáticos. Cada subcarpeta incluye un `resumen.md` con:
- Citas completas (autores, año, título, venue, DOI/URL)
- Resumen extendido de 4-5 frases por paper
- Relevancia específica para el proyecto RaceScope

El archivo `referencias.bib` contiene todas las referencias en formato BibTeX para uso directo con LaTeX.

---

## Estructura de Carpetas

```
bibliografia/
├── README.md                          ← Este archivo (índice maestro)
├── referencias.bib                    ← Todas las referencias en BibTeX
├── papers/                            ← PDFs descargados (si aplica)
│
├── 01_lstm_deep_learning/
│   └── resumen.md                     ← LSTM, RNN, DL fundacional + variantes
│
├── 02_monte_carlo_simulacion/
│   └── resumen.md                     ← MC theory, SAA, risk-adjusted ranking
│
├── 03_f1_estrategia/
│   └── resumen.md                     ← F1 pit stop optimization, race simulation
│
├── 04_degradacion_neumaticos/
│   └── resumen.md                     ← Tire physics, ML-based degradation models
│
├── 05_ml_motorsport/
│   └── resumen.md                     ← DL applied to motorsport performance
│
├── 06_series_temporales_estado_arte/
│   └── resumen.md                     ← SOTA: N-BEATS, TFT, TimesFM, Mamba
│
└── 07_data_engineering_f1/
    └── resumen.md                     ← OpenF1, Parquet, FastAPI, React/Vite
```

---

## Referencias por Bloque Temático

### 01 — LSTM y Deep Learning (Sección relevante de la memoria: Arquitectura del Modelo ML)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Hochreiter & Schmidhuber — Long Short-Term Memory | 1997 | Neural Computation | **Fundacional** |
| Elman — Finding Structure in Time | 1990 | Cognitive Science | RNN base |
| Bengio et al. — Learning Long-Term Dependencies | 1994 | IEEE TNN | Vanishing gradient |
| Pascanu et al. — On the Difficulty of Training RNNs | 2013 | ICML | Gradient clipping |
| Graves — Generating Sequences With RNNs | 2013 | arXiv | Sequence generation |
| Cho et al. — GRU / Encoder-Decoder | 2014 | EMNLP | GRU alternative |
| Jozefowicz et al. — Empirical Evaluation of GRU vs LSTM | 2015 | arXiv | Architecture comparison |
| Schuster & Paliwal — Bidirectional RNNs | 1997 | IEEE Signal Processing | Bi-LSTM |
| Bahdanau et al. — Neural MT with Alignment (Attention) | 2015 | ICLR | Attention mechanisms |
| Vaswani et al. — Attention Is All You Need | 2017 | NeurIPS | Transformer |
| Bai et al. — TCN vs LSTM | 2018 | arXiv | TCN alternative |
| Oreshkin et al. — N-BEATS | 2020 | ICLR | Strong baseline |
| Zhou et al. — Informer | 2021 | AAAI | Long-range efficient |
| Wu et al. — Autoformer | 2021 | NeurIPS | Decomposition |
| Gu et al. — S4 State Space | 2021 | ICLR | Linear complexity |
| Gu & Dao — Mamba | 2023 | arXiv | Selective SSM SOTA |
| Nie et al. — PatchTST | 2023 | ICLR | Patch Transformer |
| Challu et al. — N-HiTS | 2023 | AAAI | Hierarchical interp. |
| Kingma & Ba — Adam Optimizer | 2014 | arXiv | Training practice |
| Gal & Ghahramani — Dropout in RNNs | 2016 | NeurIPS | Regularization |
| Ba et al. — Layer Normalization | 2016 | arXiv | Normalization |
| Goodfellow et al. — Deep Learning (textbook) | 2016 | MIT Press | Reference textbook |
| Hastie et al. — Elements of Statistical Learning | 2009 | Springer | Stats foundation |
| Miao et al. — Tyre Energy Prediction F1 (LSTM) | 2025 | arXiv | **F1-specific DL** |

---

### 02 — Monte Carlo y Optimización bajo Incertidumbre (Sección: Motor de Simulación)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Rubinstein & Kroese — Simulation and the Monte Carlo Method | 2016 | Wiley | **Manual de referencia** |
| Shapiro — MC Sampling-Based Stochastic Optimization | 2013 | Optimization Online | Convergencia teórica |
| Shapiro — Rate of Convergence of MC Approximations | 1999 | SIAM | Convergencia √N |
| Kleywegt et al. — SAA Method | 2002 | SIAM | SAA framework |
| Kim & Pasupathy — Guide to SAA | 2014 | Cornell ORIE | SAA práctica |
| Markowitz — Portfolio Selection | 1952 | J. Finance | Mean-variance |
| Rockafellar & Uryasev — CVaR Optimization | 2000 | J. Risk | Tail risk |
| Kocsis & Szepesvári — UCT / MCTS | 2006 | ECML | Exploración-explotación |
| Salah & Winands — MCTS Review | 2022 | AI Review | MCTS survey |
| Pasupathy et al. — Simulation Optimization Review | 2015 | Ann. OR | Survey SO |
| Rubinstein & Kroese — Cross-Entropy Method | 2004 | Springer | Varianza reducida |
| Heilmeier et al. — MC in Circuit Motorsport | 2020 | Applied Sciences | **MC en F1** |
| Harman et al. — ANN + MCTS Formula-E | 2020 | Neural Computing | MC en motorsport |
| Aguad & Thraves — DP + Game Theory F1 | 2024 | EJOR | **Validación two-phase** |

---

### 03 — Estrategia F1 y Pit Stops (Sección: Contexto de Dominio)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Aguad & Thraves — DP + Game Theory F1 | 2024 | EJOR | **Top reference** |
| Carrasco & Thraves — DP Pit Stop Optimization | 2023 | CEJOR | DP fundacional |
| Rapp et al. — DL for F1 Pit Stop Decision Support | 2025 | Frontiers AI | **SOTA aplicado** |
| Thomas — Explainable RL for F1 Strategy | 2025 | arXiv | RL approach |
| Arxiv — Towards Learning-Based F1 Race Strategies | 2024 | arXiv | MINLP + RL |
| Heilmeier et al. — MC in Circuit Motorsport | 2020 | Applied Sciences | MC F1 |
| Heilmeier et al. — Virtual Strategy Engineer ANN | 2020 | Applied Sciences | VSE system |
| Revista JORS — F1 Discrete-Event Simulation | 2009 | JORS | Historical foundation |
| Sulsters — Simulating F1 Race Strategies | 2017 | VU BA Thesis | Academic baseline |
| Renganathan — State-Space Tire Degradation F1 | 2024 | arXiv | Bayesian tire model |
| Arxiv — Game Theory in F1 | 2025 | arXiv | Competitive dynamics |
| Arxiv — Multi-Agent Race Strategies F1 | 2026 | arXiv | Multi-agent RL |
| MIT Thesis — Real-Time Decision Making Motorsports | 2015 | MIT | LSTM for F1 |
| Jafri — ML for F1 Race Outcomes | 2024 | Capstone | Feature engineering |
| Deep-Racing — DNN for F1 Prediction | 2023 | IJML | DNN F1 baseline |
| OpenF1 Project | 2023- | Open Source | **Fuente de datos** |
| FastF1 Library | 2021- | Python | Data access library |
| JQAS — Bayesian Analysis F1 Results | 2023 | JQAS | Statistical modeling |

---

### 04 — Degradación de Neumáticos (Sección: Modelo de Perfil de Piloto)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Pacejka — Tyre and Vehicle Dynamics (textbook) | 2002 | Elsevier | **Manual de referencia** |
| Bakker, Nyborg & Pacejka — Magic Formula | 1989 | Vehicle System Dynamics | Magic formula original |
| Ozerem & Morrey — Brush Thermo-Physical Tire | 2019 | J. Auto. Engineering | Thermal brush model |
| Farroni et al. — TRT EVO Thermodynamic Tire | 2019 | Proc. Inst. Mech. Eng. | Temperature sensitivity |
| Chou et al. — Tire Wear (Viscoelasticity + Thermo) | 2024 | Wear | Wear physics |
| Miao et al. — Tire Energy Prediction F1 | 2025 | arXiv | **LSTM para neumáticos F1** |
| Renganathan — State-Space Tire Degradation | 2024 | arXiv | Bayesian degradation |
| Rapp et al. — DL Pit Stop Decision Support | 2025 | Frontiers AI | DL + tire degradation |
| Hastie et al. — Statistical Learning with Sparsity | 2015 | CRC | Sparse regression |
| Raudenbush & Bryk — Hierarchical Linear Models | 2002 | Sage | Hierarchical models |

---

### 05 — ML en Motorsport (Sección: Trabajos Relacionados)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Rapp et al. — DL F1 Pit Stop | 2025 | Frontiers AI | **Most relevant** |
| Thomas — Explainable RL F1 | 2025 | arXiv | RL approach |
| Miao et al. — Tire Energy F1 | 2025 | arXiv | Tire + DL |
| Multi-Agent F1 RL | 2026 | arXiv | Multi-agent |
| Tilburg U. — DL Lap Time Prediction | 2023 | Thesis | Architecture comparison |
| Heilmeier — VSE with ANN | 2020 | Applied Sciences | ANN strategy system |
| Harman et al. — ANN+MCTS Formula-E | 2020 | Neural Computing | Hybrid approach |
| Deep-Racing DNN | 2023 | IJML | DNN baseline |
| Hojaji et al. — AI Sim Racing Telemetry | 2024 | C&E: XR | Telemetry ML |
| ResearchGate — Advanced ML F1 Performance | 2025 | ResearchGate | R²=0.999 benchmark |
| Springer — Data-Driven F1 Races Analysis | 2023 | Springer | PCA analysis |
| Hewamalage et al. — RNNs for Time Series (Survey) | 2020 | IJF | RNN survey |
| Ajgel et al. — Hybrid Transformer-LSTM Sports | 2023 | Informatica | Hybrid architecture |
| MIT Thesis — Real-Time Motorsport Analytics | 2015 | MIT | LSTM F1 foundation |

---

### 06 — Estado del Arte Series Temporales (Sección: Comparativa Arquitectural)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| Makridakis et al. — M4 Competition | 2020 | IJF | **Benchmark de referencia** |
| Makridakis et al. — M5 Competition | 2022 | Statistical Modeling | DL evolution |
| Godahewa et al. — Monash TS Archive | 2021 | NeurIPS | Benchmark datasets |
| Zeng et al. — Are Transformers Effective for TS? | 2023 | AAAI | Critical analysis |
| Das et al. — TimesFM (Google) | 2024 | ICML | Foundation model |
| Woo et al. — MOIRAI (Salesforce) | 2024 | ICML | Foundation model |
| Rasul et al. — Lag-Llama | 2023 | arXiv | Open-source FM |
| Oreshkin et al. — N-BEATS | 2020 | ICLR | Strong DL baseline |
| Challu et al. — N-HiTS | 2023 | AAAI | Hierarchical forecasting |
| Lim et al. — TFT | 2021 | IJF | Interpretable multi-horizon |
| Nie et al. — PatchTST | 2023 | ICLR | Patch Transformer |
| Zhou et al. — Informer | 2021 | AAAI | Efficient Transformer |
| Wu et al. — Autoformer | 2021 | NeurIPS | Auto-correlation |
| Kim et al. — RevIN | 2021 | ICLR | Normalization |
| Transfer Learning TS (LoRA) | 2024 | arXiv | Foundation fine-tuning |

---

### 07 — Data Engineering y Arquitectura (Sección: Implementación Técnica)

| Referencia | Año | Venue | Clave |
|---|---|---|---|
| OpenF1 API | 2023- | Open Source | **Fuente de datos** |
| Ergast F1 API | 2006-2024 | Open Source | Datos históricos |
| FastF1 Python Library | 2021- | Python | Data access |
| Melnik et al. — Dremel (Google) | 2010 | VLDB | **Parquet fundacional** |
| Apache Parquet | 2013- | Apache | Columnar storage |
| Feast Feature Store | 2019- | Open Source | Feature store pattern |
| Armbrust et al. — Lakehouse | 2021 | CIDR | Data lakehouse |
| Token Bucket Algorithm | RFC 2698 | IETF | Rate limiting |
| AWS — Exponential Backoff | 2016 | AWS Docs | Retry pattern |
| Ramírez — FastAPI | 2018- | Open Source | Backend framework |
| ASGI Specification | 2018- | Python | Async web standard |
| Fielding — REST Architecture | 2000 | PhD Dissertation | REST principles |
| React | 2013- | Meta/Open Source | Frontend framework |
| Vite | 2020- | Open Source | Build tool |
| Recharts | 2015- | Open Source | Visualization |
| Sleator & Tarjan — LRU Cache | 1985 | CACM | Cache policy |

---

## Estadísticas de la Bibliografía

| Bloque | Nº de Referencias | Tipo Principal |
|---|---|---|
| 01 LSTM y DL | 24 | Papers académicos peer-reviewed |
| 02 Monte Carlo | 14 | Papers académicos + textbooks |
| 03 Estrategia F1 | 18 | Papers académicos + tesis |
| 04 Degradación neumáticos | 10 | Papers académicos + textbooks |
| 05 ML en Motorsport | 14 | Papers académicos + tesis |
| 06 Estado del arte TS | 15 | Papers académicos |
| 07 Data Engineering | 16 | Docs técnicas + papers |
| **TOTAL** | **~111** | (con solapamiento intencional) |

**Referencias únicas estimadas:** ~80-85 (algunos papers aparecen en varios bloques por relevancia cruzada)

---

## Guía de Uso en la Memoria

### Para la sección de Introducción/Motivación
→ `03_f1_estrategia/` (contexto de dominio F1) + `07_data_engineering_f1/` (OpenF1, herramientas)

### Para la sección de Estado del Arte / Trabajos Relacionados
→ `05_ml_motorsport/` (trabajos similares en F1) + `06_series_temporales_estado_arte/` (comparativa arquitectural)

### Para la sección de Metodología — Modelos ML
→ `01_lstm_deep_learning/` (justificación de LSTM) + `04_degradacion_neumaticos/` (justificación del modelo paramétrico)

### Para la sección de Metodología — Motor de Estrategia
→ `02_monte_carlo_simulacion/` (justificación MC + mean-variance) + `03_f1_estrategia/` (DP + game theory)

### Para la sección de Implementación
→ `07_data_engineering_f1/` (stack técnico completo)

### Para la sección de Discusión / Trabajo Futuro
→ `06_series_temporales_estado_arte/` (Transformers, TimesFM, Mamba como mejoras futuras)

---

*Bibliografía compilada en marzo 2026 mediante investigación sistemática de arXiv, Semantic Scholar, Google Scholar, Frontiers AI, IEEE Xplore, ScienceDirect y documentación oficial de proyectos open-source.*
