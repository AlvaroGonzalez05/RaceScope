# Declaración de uso de herramientas de Inteligencia Artificial Generativa

Este documento recoge, en términos prácticos, el uso que se ha hecho de herramientas de Inteligencia Artificial Generativa (IAG) durante el desarrollo del Trabajo Fin de Grado **«RaceScope: Optimización de la estrategia de carrera en Fórmula 1 mediante Inteligencia Artificial»**, presentado por **Álvaro González Tabernero** en el marco del Doble Grado en Ingeniería en Tecnologías de Telecomunicación y Análisis de Negocios/Business Analytics (GITT-BA) de la Universidad Pontificia Comillas (ICAI), curso 2025/26.

Las categorías que figuran aquí coinciden con las declaradas en el anexo de la memoria del TFG. Se acompañan de ejemplos concretos del repositorio para que cualquier revisor pueda contrastar el alcance real del uso.

---

## Herramienta utilizada

- **Asistente**: Claude Code (Anthropic), modelo Opus 4.x.
- **Modo de uso**: terminal local sobre el repositorio del proyecto, con acceso de lectura y escritura a los archivos.
- **Papel**: asistencia a tareas concretas planteadas por el autor. Cada cambio sugerido por la herramienta ha sido revisado y aceptado por el autor antes de incorporarse al repositorio.

---

## Tareas en las que ha intervenido

1. **Crítico** — contra-argumentos sobre decisiones de modelado (Transformer v3 frente a LSTM, cota de validez física, criterio media-varianza) para reforzar la defensa de las decisiones tomadas.
2. **Referencias** — búsqueda de candidatas bibliográficas (arXiv, DOIs, documentación oficial), que el autor ha contrastado contra la fuente primaria antes de añadirlas a `memoria/src/main.bib`.
3. **Metodólogo** — sugerencias de técnicas aplicables (normalización térmica, envoltura empírica por circuito, etc.) que el autor ha implementado, evaluado y validado.
4. **Interpretador de código** — análisis preliminar de los CSV de evaluación 2025 (`code/backend_fastapi/reports/eval_2025_4drivers.csv`) para extraer métricas agregadas (MAPE, sesgo, MAE/RMSE) que después se han incorporado a la memoria. Además, se ha prestado ayuda para la depuración de errores y debugging.
5. **Constructor de plantillas** — andamiaje de estructuras LaTeX (tablas con `tabularx`, sección de resumen ejecutivo y abstract, listas con `enumitem`) que el autor ha adaptado al estilo del documento.
6. **Corrector de estilo literario y de lenguaje** — pasadas de revisión sobre la prosa de los capítulos 1–6 según las indicaciones del autor y bajo una guía de estilo personal.
7. **Generador previo de diagramas de flujo y contenido** — esbozo inicial de las figuras TikZ (arquitectura del sistema, pipeline de datos) que el autor ha refinado y validado.
8. **Sintetizador y divulgador de libros complicados** — resúmenes de referencias técnicas extensas (artículos sobre Transformers, documentación de PyTorch, reglamento técnico/deportivo FIA) para acelerar la lectura.
9. **Revisor** — comprobaciones cruzadas de citas.
10. **Traductor** — traducción al inglés del Project Abstract a partir del texto en español redactado por el autor.
11. **Gestión del control de versiones** — redacción de mensajes de commit, organización de los archivos a versionar y ejecución asistida de operaciones rutinarias de Git y GitHub (`git status`, `git add`, `git diff`, `git commit`).

---

## Tareas que no se han delegado

- Diseño científico y metodológico del proyecto (definición del problema, arquitectura del motor de estrategia, criterios de validación).
- Implementación de los componentes nucleares del backend (pipeline de datos, modelos Transformer y paramétricos, motores 1–3 del *strategy engine*, API FastAPI, frontend React).
- Recolección, limpieza y validación final de los datos de OpenF1.
- Toma de decisiones sobre alcance, hipótesis, limitaciones y líneas de trabajo futuro.
- Verificación de las salidas del modelo y de las cifras reportadas.
- Redacción original de los capítulos de la memoria.

---

## Trazabilidad

Los *commits* del repositorio que incluyen contribución directa de la herramienta llevan el campo `Co-Authored-By: Claude Opus 4.X` en su mensaje. La traza completa puede recuperarse con:

```bash
git log --grep="Co-Authored-By: Claude"
```

---

Álvaro González Tabernero — junio de 2026.
