import random
from pathlib import Path
import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("hello_mlflow")

with mlflow.start_run(run_name="primer_run"):

    
    learning_rate = 0.01
    epochs = 5

    mlflow.log_param("modelo", "regresion_lineal_dummy")
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)

    
    random.seed(42)
    loss = 1.0
    accuracy = 0.0

    for epoch in range(epochs):
        loss = loss * 0.7 + random.uniform(0.0, 0.02)
        accuracy = 1.0 - loss
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        print(f"epoca {epoch}: loss={loss:.4f}  accuracy={accuracy:.4f}")

    mlflow.log_metric("loss_final", loss)
    mlflow.log_metric("accuracy_final", accuracy)

    
    reporte = Path("resultados.txt")
    reporte.write_text(
        "Resumen del experimento hello_mlflow\n"
        "------------------------------------\n"
        "modelo........: regresion_lineal_dummy\n"
        f"learning_rate.: {learning_rate}\n"
        f"epochs........: {epochs}\n"
        f"loss final....: {loss:.4f}\n"
        f"accuracy final: {accuracy:.4f}\n",
        encoding="utf-8",
    )
    mlflow.log_artifact(str(reporte))

    run_id = mlflow.active_run().info.run_id
    print("\nRun registrado correctamente.")
