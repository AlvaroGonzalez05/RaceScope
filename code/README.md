# RaceScope
**Análisis interactivo de telemetría y estrategia en Fórmula 1**  
Proyecto para la asignatura *Desarrollo de Aplicaciones de Visualización de Datos* (ICAI).

## Nueva arquitectura (MVP estrategia)

- Backend FastAPI: `code/backend_fastapi`
- Frontend React + Vite: `code/frontend`

Consulta los README de cada módulo para ejecución.

## Descripción

RaceScope es una aplicación web que obtiene datos de la API pública de **OpenF1**, los filtra y procesa para generar visualizaciones interactivas sobre el rendimiento de los pilotos en un Gran Premio.  

La aplicación incluye módulos de telemetría básica, comparaciones de ritmos, estimación sencilla de tiempos y degradación de neumáticos mediante un modelo de regresión lineal regularizada, y un simulador de estrategias que se ejecuta únicamente tras la selección del equipo, piloto y circuito por parte del usuario.  

El objetivo es ofrecer una herramienta clara, rápida y con la granularidad suficiente para que un usuario pueda construir comparaciones útiles y comprender la estrategia de forma sencilla.

## Objetivos

- Desarrollar una interfaz interactiva para explorar y comparar telemetría y ritmos.
- Implementar un modelo predictivo sencillo para estimar tiempos base y degradación.
- Simular estrategias de carrera con combinaciones simples de stints.
- Mantener un diseño accesible y fácil de interpretar.

## Funcionalidades principales

- Telemetría básica: velocidad, RPM, marchas, aceleración y frenada por vuelta.
- Comparador de ritmos: distribución de tiempos por stint y compuesto.
- Degradación estimada: curvas de pérdida de ritmo por neumático.
- Estrategia simple: cálculo de las tres combinaciones de paradas más prometedoras.

## Pipeline de datos

1. Obtención de datos desde la API de OpenF1.  
2. Procesado: limpieza de nombres, detección de vueltas anómalas, etiquetado de stints y compuestos.  
3. Estructura tabular: vueltas como unidad central, enlazadas a sesiones, eventos y pilotos.  
4. Modelo predictivo sencillo: regresión lineal regularizada (L2) para estimar tiempo base y degradación.  
5. Selección del usuario: equipo, piloto y circuito.  
6. Simulación de estrategia: cálculo de tiempos totales aproximados y combinaciones de paradas.  
7. Visualización interactiva: comparaciones y vistas dinámicas en la interfaz.  

## Plan inicial de trabajo

1. Integración con la API de OpenF1 y caché local.  
2. Desarrollo de visualizaciones iniciales (ritmos y telemetría básica).  
3. Implementación del modelo predictivo sencillo.  
4. Módulo de simulación de estrategias.  
5. Optimización de rendimiento y refinamiento de la interfaz.  

## Stack tecnológico

- Python + Dash / Plotly para la interfaz.  
- Pandas para procesamiento de datos.  
- scikit-learn para el modelo predictivo.  
- Archivos locales (Parquet/CSV) como almacenamiento ligero.  

## Instrucciones de ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/AlvaroGonzalez05/RaceScope_DAVD.git
   cd <repo>
