# Ficha CRISP-ML(Q) · Churn Predictor

CRISP-ML(Q) organiza un proyecto de ML en seis fases. La Q (quality assurance) pide que
cada fase declare sus requisitos, los riesgos que pueden hacerla fallar y el control que
los detecta. Esta ficha es el registro de esas decisiones: se versiona junto al código.

## Fase 1 · Negocio y datos

| Pregunta | Respuesta |
|---|---|
| Problema de negocio | La fintech pierde clientes y hoy contacta a todos por igual en las campañas de retención |
| Decisión que apoya el modelo | A qué clientes ofrecer retención este mes |
| Criterio de éxito de negocio | Gastar menos por cliente que la política actual (contactar a todos) |
| Criterio de éxito de ML | F1 ≥ 0.60 y ROC-AUC ≥ 0.70 en test |
| Criterio de éxito económico | Costo esperado por cliente menor que contactar a todos y que no contactar a nadie |
| Costo de un falso negativo | S/ 200 (cliente que se va sin oferta) |
| Costo de un falso positivo | S/ 100 (oferta a quien no se iba) |
| Factibilidad | 300 clientes, 7 features, clase balanceada (49.7 % churn) |

## Riesgos y controles por fase

| Fase | Riesgo | Control (quality gate) | Dónde vive |
|---|---|---|---|
| 1. Negocio y datos | El dataset cambia de esquema o trae valores imposibles | Contrato de datos | `src/validate_data.py`, job `data-validation` |
| 1. Negocio y datos | `"None"` leído como nulo y luego imputado | `keep_default_na=False` + test | `src/common.py`, `tests/test_data_contract.py` |
| 2. Preparación | Training-serving skew: la API preprocesa distinto que el entrenamiento | Preprocesamiento dentro del Pipeline | `src/features.py`, `tests/test_features.py` |
| 2. Preparación | Data leakage: el escalador ve datos de test | El Pipeline se ajusta solo con train, fold por fold | `src/train.py` |
| 3. Modelado | Elegir modelo mirando test (optimismo) | Selección por validación cruzada | `src/train.py` |
| 3. Modelado | Un modelo que no aporta sobre lo trivial | Baseline registrado en MLflow y cuatro candidatos comparados por CV | `src/train.py` |
| 4. Evaluación | Buena métrica promedio, mala en un segmento | Recall por segmento, brecha máxima | `src/evaluate_and_register.py` |
| 4. Evaluación | Promover un modelo peor que el vigente | Champion / challenger | `src/evaluate_and_register.py` |
| 5. Despliegue | La API sirve algo distinto a lo evaluado | Test de skew contra la API en ejecución | `tests/test_skew.py`, job `smoke-test` |
| 5. Despliegue | Entradas inválidas llegan al modelo | Contrato en el esquema de la API (422) | `app/main.py` |
| 6. Monitoreo | Los datos de producción cambian | PSI + KS por feature, PSI de predicciones | `src/monitor.py` |
| 6. Monitoreo | El modelo se degrada sin que cambien los datos | F1 con etiquetas tardías | `src/monitor.py`, `model-monitoring.yml` |

## Decisiones registradas

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-11-16 | Umbral de decisión elegido por costo, no 0.5 | Un FN cuesta el doble que un FP |
| 2026-11-16 | El modelo se selecciona por F1 de validación cruzada | Test queda reservado para la evaluación final |
| 2026-11-16 | Cuarto candidato `GradientBoostingClassifier` agregado al modelado | Comparar una alternativa de boosting por árboles contra logistic, random forest y xgboost |
| 2026-11-16 | Se mantienen los costos S/ 200 (FN) y S/ 100 (FP) | Reflejan la campaña de retención del caso y permiten un umbral distinto de 0.5 |
| 2026-11-16 | `@champion` solo se mueve si el modelo pasa el quality gate y no cuesta más que el vigente | Evitar promover un modelo peor que el que ya está en producción |

## Estado de controles · fases 5 y 6 (2026-11-16)

| Fase | Control | Estado | Dónde se implementó |
|---|---|---|---|
| 5. Despliegue | La API sirve exactamente lo evaluado (sin skew) | Implementado | `app/main.py` + `tests/test_skew.py` (local, contenedor, staging y prod) |
| 5. Despliegue | Entradas inválidas rechazadas antes del modelo | Implementado | `CustomerFeatures` con `Literal` y rangos → HTTP 422 |
| 5. Despliegue | Imagen mínima y sin root | Implementado | `requirements-serve.txt` + `Dockerfile` |
| 5. Despliegue | Publicación trazable por commit | Implementado | `build-and-publish` en `mlops.yml` (tag SHA y `latest` desde `main`) |
| 5. Despliegue | Entorno de prueba antes de producción | Implementado | ECS `staging` desde `develop` + aprobación para `prod` |
| 5. Despliegue | Rollback si la versión nueva no levanta | Implementado | Circuit breaker de ECS + verificación en `scripts/deploy_ecs.sh` |
| 6. Monitoreo | Data drift y prediction drift sin etiquetas | Implementado | `src/monitor.py` (PSI/KS) y `/metrics` con Prometheus |
| 6. Monitoreo | Concept drift con etiquetas tardías | Implementado (cuando llegan etiquetas) | F1 por ventana en `monitor.py` |
| 6. Monitoreo | Reentrenamiento automático | Implementado | `model-monitoring.yml` (exit 42 → `mlops.yml`) |
| 6. Monitoreo | Alertas y dashboard | Parcial (stack opcional) | `monitoring/` con Prometheus, Pushgateway y Grafana |

## Riesgos que la solución todavía no cubre

| Riesgo | Por qué importa | Mitigación propuesta |
|---|---|---|
| El log `logs/predictions.jsonl` vive dentro del contenedor ECS y se pierde al reemplazar la tarea | Sin ese log, el monitoreo de drift de un entorno desplegado se queda sin datos | Exportar las predicciones a S3 o CloudWatch Logs y leer el monitor desde ahí |
| El concept drift depende de etiquetas que llegan semanas después | En `competitor` el F1 cae a 0.38 sin que se mueva ningún PSI; el sistema reacciona recién cuando llegan las etiquetas | Etiquetas proxy (cancelaciones, tickets, caída de uso) y ventanas de performance parciales |
| El champion se promueve por métricas offline; no hay monitoreo por segmento en producción | Un modelo puede ser bueno en promedio y fallar en un grupo | Métricas de recall por segmento en `/metrics` y alertas por grupo |

## Decisiones registradas · despliegue y monitoreo

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-11-16 | La API carga el Pipeline completo y el umbral desde `production_model_info.json` | Elimina el preprocesamiento duplicado y usa la decisión de negocio de la fase 4 |
| 2026-11-16 | `develop` despliega en staging y `main` en producción, con aprobación manual | Probar en infraestructura real antes de llegar a los clientes |
| 2026-11-16 | El monitor corre semanalmente y dispara `mlops.yml` con exit 42 | Cierra el ciclo de Continuous Training |
| 2026-11-16 | Se mantiene el stack de Prometheus/Grafana como opcional | El laboratorio funciona solo con el reporte de `monitor.py`; el stack agrega dashboard y alerta |
