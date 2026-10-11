import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Variables que usa el modelo (las mismas que se eligieron en el notebook)
FEATURES = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physicality",
    "movement_reactions",
    "mentality_composure",
    "skill_moves",
    "weak_foot",
    "edad",
]
TARGET = "overall_rating"


def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena un modelo de regresion sobre el dataset EA FC 27 "
        "y registra la corrida en MLflow."
    )
    p.add_argument(
        "--data",
        default="data/players.csv",
        help="Ruta del CSV con los datos (default: data/players.csv)",
    )
    p.add_argument(
        "--experiment",
        default="fc27_experimentacion",
        help="Nombre del experimento en MLflow (default: fc27_experimentacion)",
    )
    p.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="Backend de MLflow Tracking (default: sqlite:///mlflow.db)",
    )

    # Hiperparametros
    p.add_argument(
        "--model",
        default="random_forest",
        choices=["linear", "ridge", "random_forest"],
        help="Algoritmo a entrenar (default: random_forest)",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Numero de arboles, solo para random_forest (default: 100)",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidad maxima de los arboles, solo para random_forest "
        "(default: None, sin limite)",
    )
    p.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimo de muestras por hoja, solo para random_forest (default: 1)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularizacion, solo para ridge (default: 1.0)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporcion del conjunto de prueba (default: 0.2)",
    )

    # Semilla
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria, controla el split y el modelo (default: 42)",
    )

    return p.parse_args()


def fijar_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)


def cargar_datos(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {ruta}")
    df = pd.read_csv(ruta)
    print(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


def limpiar_datos(df):
    df = df.copy()
    df = df.drop(columns=["height_cm", "weight_kg"], errors="ignore")

    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
    fecha_snapshot = pd.to_datetime(df["snapshot_date"].iloc[0])
    df["edad"] = ((fecha_snapshot - df["birthdate"]).dt.days / 365.25).round(1)

    antes = len(df)
    df = df[df["position"] != "GK"]
    print(
        f"Limpieza: se quitaron {antes - len(df)} porteros, quedan {len(df)} jugadores"
    )

    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def construir_modelo(args):
    if args.model == "linear":
        return LinearRegression()
    if args.model == "ridge":
        return Ridge(alpha=args.alpha, random_state=args.seed)
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )


def evaluar(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    metricas = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }
    return metricas, y_pred


def guardar_artefactos(y_test, y_pred, metricas, params, carpeta="artefactos"):
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)

    ruta_json = carpeta / "metricas.json"
    ruta_json.write_text(
        json.dumps({"parametros": params, "metricas": metricas}, indent=2),
        encoding="utf-8",
    )

    ruta_png = carpeta / "real_vs_predicho.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.2, color="steelblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2,
        label="prediccion perfecta",
    )
    plt.xlabel("overall_rating real")
    plt.ylabel("overall_rating predicho")
    plt.title("Real vs Predicho (conjunto de prueba)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=100)
    plt.close()

    return [ruta_json, ruta_png]


def registrar_modelo(modelo, nombre="modelo"):

    try:
        mlflow.sklearn.log_model(
            modelo,
            name=nombre,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    except TypeError:
        mlflow.sklearn.log_model(modelo, name=nombre)


def main():
    args = parse_args()
    fijar_semilla(args.seed)

    # Datos
    df = cargar_datos(args.data)
    df = limpiar_datos(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    # Entrenamiento
    modelo = construir_modelo(args)
    modelo.fit(X_train, y_train)

    # Evaluacion
    metricas, y_pred = evaluar(modelo, X_test, y_test)

    # Referencia: un modelo que siempre predice la media
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    metricas["mae_dummy"] = mean_absolute_error(y_test, dummy.predict(X_test))

    print("\nResultados:")
    for nombre, valor in metricas.items():
        print(f"  {nombre:10s}: {valor:.4f}")

    # Parametros que se van a registrar
    params = {
        "modelo": args.model,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "alpha": args.alpha,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_features": len(FEATURES),
        "features": ", ".join(FEATURES),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Registro en MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    nombre_run = (
        f"{args.model}_n{args.n_estimators}_d{args.max_depth}"
        f"_leaf{args.min_samples_leaf}_s{args.seed}"
    )
    with mlflow.start_run(run_name=nombre_run):
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)

        for ruta in guardar_artefactos(y_test, y_pred, metricas, params):
            mlflow.log_artifact(str(ruta))

        registrar_modelo(modelo)

        run_id = mlflow.active_run().info.run_id

    print(f"\nCorrida registrada en el experimento '{args.experiment}'")
    print(f"run_name: {nombre_run}")
    print(f"run_id  : {run_id}")


if __name__ == "__main__":
    main()
