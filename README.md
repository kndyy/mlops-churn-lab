# MLOps Churn Lab

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

- Validar los datos antes de entrenar.
- Garantizar que la API prediga lo mismo que el modelo evaluado.
- Elegir el modelo con evidencia y decidir con criterios de negocio.
- Promover a producción solo lo que pasa un quality gate.
- Detectar cuándo el modelo deja de funcionar y reentrenarlo.
- Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 18 passed
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
uvicorn app.main:app --port 8000      # http://localhost:8000/docs
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── src/
│   ├── data_prep.py                # limpieza, encoding, escalado y split
│   ├── train.py                    # 3 modelos con MLflow tracking
│   └── evaluate_and_register.py    # registro en MLflow Model Registry
├── tests/                          # tests de la API con el modelo mockeado
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-prep → train → register → smoke-test
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Amazon ECR, drift, reentrenamiento automático |

## Resultados del laboratorio 14.1

Pipeline CRISP-ML(Q) con cuatro candidatos comparados por validación cruzada estratificada
(5 folds, 240 clientes de train). El set de test se reservó para la evaluación final.

| Modelo | F1 CV | ROC-AUC CV |
|---|---:|---:|
| baseline (siempre churn) | 0.6629 | — |
| logistic | 0.6666 ± 0.0454 | 0.7315 |
| random_forest | **0.6900** ± 0.0426 | **0.7326** |
| xgboost | 0.6829 ± 0.0656 | 0.7100 |
| gradient_boosting (cuarto modelo) | 0.6476 ± 0.0761 | 0.6858 |

El cuarto modelo candidato es `GradientBoostingClassifier(n_estimators=100, max_depth=3,
learning_rate=0.1)`. Quedó último por F1 y por ROC-AUC de validación cruzada, así que **no
cambia el modelo elegido**: `random_forest` sigue siendo el mejor y es el que se registra
como `@champion`.

Evaluación en test con el umbral elegido por costo (`0.3`):

| Métrica | Valor |
|---|---:|
| F1 | 0.7013 |
| Precision | 0.5745 |
| Recall | 0.9000 |
| ROC-AUC | 0.7989 |
| Costo por cliente | S/ 43.33 |

El quality gate pasa sus 4 checks: F1 ≥ 0.60, ROC-AUC ≥ 0.70, brecha de recall ≤ 0.25 y
costo menor que contactar a todos (S/ 50.00). La model card completa se genera en
`reports/model_card.md` y el detalle en `reports/evaluation_report.json`.

```bash
python src/validate_data.py
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
pytest tests/ -q
```

## Resultados del laboratorio 14.2

La fase 5 sirve el Pipeline dentro de una API FastAPI que ya no preprocesa: recibe las
7 columnas crudas y aplica el umbral guardado en `production_model_info.json`. La imagen
de inferencia (`requirements-serve.txt`) se publica en Amazon ECR y se despliega en dos
entornos ECS Fargate con Terraform: `staging` desde `develop` y `prod` desde `main` con
aprobación manual. La fase 6 mide data drift, prediction drift y F1 con `src/monitor.py`,
y `model-monitoring.yml` dispara `mlops.yml` cuando la decisión es REENTRENAR.

### Validación local

| Comando | Resultado |
|---|---|
| `pytest tests/ -q` | 30 passed, 2 skipped (los de skew necesitan la API) |
| `API_URL=http://localhost:8000 pytest tests/test_skew.py -v` | 2 passed (uvicorn y contenedor) |
| `curl localhost:8000/metrics \| grep churn_` | `churn_predictions_total` y `churn_probability_bucket` |
| `simulate_traffic.py --n 300` | 300 requests: 56 BAJO, 125 MEDIO, 119 ALTO |
| `simulate_traffic.py --n 300 --scenario price_hike` | 300 requests: 28 BAJO, 94 MEDIO, 178 ALTO |

El bug de la guía 11 queda cerrado: el mismo cliente con `Month-to-Month` devuelve
`0.7709` (ALTO) y con `Two Year`, `0.4691` (MEDIO). Una categoría fuera de contrato
devuelve HTTP 422 sin llegar al modelo.

### Monitoreo: price_hike contra competitor

`price_hike` sube la tarifa y dispara reclamos. El PSI de `monthly_charges` y
`support_tickets_last_6m` se vuelve crítico y el monitor decide **REENTRENAR** sin
necesidad de etiquetas. `competitor` capta clientes de contrato largo: las features no
cambian (PSI entre 0.02 y 0.06), pero la relación entre features y churn sí, y el F1 cae
de 0.70 a 0.38. Eso es **concept drift** y solo se ve con etiquetas.

| Escenario | Semana del primer síntoma | Señal | Decisión |
|---|---|---|---|
| `price_hike` | semana 5 | PSI 0.584 (ALERTA) → 2.418 (REENTRENAR) | REENTRENAR en semana 7 |
| `competitor` | semana 6 | F1 0.507 → 0.384, PSI ≤ 0.055 | REENTRENAR en semana 6 si hay etiquetas |

En `price_hike` el F1 no baja: hay data drift sin pérdida de performance y se reentrena
por precaución. En `competitor` el F1 se desploma con los datos "en verde": ningún
monitor de datos lo detecta.

**Reto: latencia de etiquetas.** En churn la etiqueta real llega semanas después. Si llega
con 4 semanas de retraso, la caída de F1 de la semana 6 se conoce en la semana 10. El
sistema tardaría **4 semanas** en reaccionar a `competitor`; las señales de data drift y
prediction drift no lo detectan porque en ese escenario no se mueven. Para acortar la
reacción habría que usar etiquetas proxy (cancelaciones, tickets, caída de uso) o ventanas
de performance con etiquetas parciales.
