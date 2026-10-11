set -e

echo '=== Corrida 1: baseline lineal (referencia de la actividad anterior) ==='
python train.py --model linear --seed 42

echo '=== Corrida 2: random forest por defecto ==='
python train.py --model random_forest --n-estimators 100 --seed 42

echo '=== Corrida 3: random forest mas arboles ==='
python train.py --model random_forest --n-estimators 300 --seed 42

echo '=== Corrida 4: random forest con profundidad limitada ==='
python train.py --model random_forest --n-estimators 300 --max-depth 8 --seed 42

echo '=== Corrida 5: random forest con hojas mas grandes ==='
python train.py --model random_forest --n-estimators 300 --min-samples-leaf 10 --seed 42

echo
echo 'Listo. Revisa las corridas con:'
echo '  mlflow ui --backend-store-uri sqlite:///mlflow.db'
