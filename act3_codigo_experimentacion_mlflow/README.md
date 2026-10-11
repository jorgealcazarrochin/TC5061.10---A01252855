# TC5061 - Actividad 3: Código de experimentación base con MLflow

Script de entrenamiento parametrizado que predice el **overall rating** de un jugador del
dataset *EA SPORTS FC 27 Player Ratings* y registra cada corrida en **MLflow Tracking**.

Es la continuación de la Actividad 2: el notebook se conserva únicamente para la exploración
de datos y toda la lógica de carga, limpieza, entrenamiento y evaluación vive ahora en
`train.py`.

**Autor:** Jorge David Alcázar Rochín - A01252855

---

## Contenido del repositorio

```
.
├── train.py              # script de entrenamiento parametrizado
├── run_experiments.sh    # corre las 5 configuraciones de la actividad
├── requirements.txt      # dependencias con versiones fijas
├── data/
│   └── players.csv       # dataset (19,789 jugadores, snapshot 2026-09-12)
└── README.md
```

---

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Ejecución

El script corre con valores por defecto sin necesidad de pasar ningún argumento:

```bash
python train.py
```

Para cambiar la configuración se pasan los argumentos por línea de comandos:

```bash
python train.py --model random_forest --n-estimators 300 --max-depth 12 --min-samples-leaf 5 --seed 42
```

Para ver todos los argumentos disponibles con su descripción:

```bash
python train.py --help
```

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--data` | `data/players.csv` | Ruta del CSV con los datos |
| `--experiment` | `fc27_experimentacion` | Nombre del experimento en MLflow |
| `--tracking-uri` | `sqlite:///mlflow.db` | Backend de MLflow Tracking |
| `--model` | `random_forest` | Algoritmo: `linear`, `ridge` o `random_forest` |
| `--n-estimators` | `100` | Número de árboles (random forest) |
| `--max-depth` | `None` | Profundidad máxima de los árboles (random forest) |
| `--min-samples-leaf` | `1` | Mínimo de muestras por hoja (random forest) |
| `--alpha` | `1.0` | Regularización (ridge) |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--seed` | `42` | Semilla aleatoria |

---

## Las cinco corridas de la actividad

```bash
bash run_experiments.sh
```

Equivale a ejecutar:

```bash
python train.py --model linear --seed 42
python train.py --model random_forest --n-estimators 100 --seed 42
python train.py --model random_forest --n-estimators 300 --seed 42
python train.py --model random_forest --n-estimators 300 --max-depth 8 --seed 42
python train.py --model random_forest --n-estimators 300 --min-samples-leaf 10 --seed 42
```

Las cinco quedan agrupadas en el experimento `fc27_experimentacion`.

---

## Ver los resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Y abrir `http://localhost:5000`. En la vista **Model training → Runs** se pueden seleccionar
las corridas y usar el botón **Compare** para contrastarlas.

---

## Qué se registra en cada corrida

- **Parámetros:** modelo, `n_estimators`, `max_depth`, `min_samples_leaf`, `alpha`,
  `test_size`, `seed`, número y lista de variables, tamaño de los conjuntos.
- **Métricas:** `mae`, `rmse`, `r2` y `mae_dummy` (un modelo que siempre predice la media,
  como referencia).
- **Artefactos:** `metricas.json` y la gráfica `real_vs_predicho.png`.
- **Modelo entrenado**, con `mlflow.sklearn.log_model`.

---

## Reproducibilidad

La semilla se recibe con `--seed` (default 42) y se aplica en tres puntos:

1. `random.seed()` y `np.random.seed()` al inicio de la ejecución.
2. El `random_state` del `train_test_split`, para que la división sea siempre la misma.
3. El `random_state` del modelo, para que el random forest construya los mismos árboles.

Dos ejecuciones con los mismos argumentos producen métricas idénticas. Se puede comprobar
corriendo dos veces el mismo comando y comparando los valores de `mae`, `rmse` y `r2`.

Las versiones de las librerías están fijadas en `requirements.txt`.
