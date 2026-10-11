"""Ejecuta cinco configuraciones bajo un único experimento MLflow.

Uso: python run_five_runs.py --data-path IoTProcessed_Data.csv
Si se omite --data-path, train.py descarga el dataset mediante KaggleHub.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT = "agricultura_iot_fan_baseline_6_sensores"
CONFIGS = [
    (100, 8, 1),
    (200, 8, 1),
    (300, 12, 1),
    (200, 16, 2),
    (400, 0, 3),  # 0 representa árboles sin límite de profundidad
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="Ruta del CSV local; opcional con KaggleHub.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--compare-only", action="store_true", help="Solo crea el CSV a partir de corridas ya registradas.")
    args = parser.parse_args()

    # Un URI SQLite relativo debe resolverse en el mismo directorio en padre e hijos.
    data_path = str(Path(args.data_path).resolve()) if args.data_path else None
    os.chdir(HERE)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    script = HERE / "train.py"
    if not args.compare_only:
        client = MlflowClient(tracking_uri=args.tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        prior_runs = client.search_runs([experiment.experiment_id], max_results=1000)
        successful_names = {r.info.run_name for r in prior_runs if r.info.status == "FINISHED"}
        for trees, depth, leaf in CONFIGS:
            depth_name = "none" if depth == 0 else str(depth)
            run_name = f"rf_trees{trees}_depth{depth_name}_leaf{leaf}_seed{args.seed}"
            if run_name in successful_names:
                print(f"Se conserva corrida ya terminada: {run_name}")
                continue
            command = [sys.executable, str(script), "--n-estimators", str(trees),
                       "--max-depth", str(depth), "--min-samples-leaf", str(leaf),
                       "--seed", str(args.seed), "--experiment-name", EXPERIMENT,
                       "--tracking-uri", args.tracking_uri]
            if data_path:
                command.extend(["--data-path", data_path])
            print("\nEjecutando:", " ".join(command), flush=True)
            subprocess.run(command, cwd=HERE, check=True)

    experiment = MlflowClient(tracking_uri=args.tracking_uri).get_experiment_by_name(EXPERIMENT)
    runs = MlflowClient(tracking_uri=args.tracking_uri).search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"]
    )
    expected_names = {f"rf_trees{t}_depth{'none' if d == 0 else d}_leaf{leaf}_seed{args.seed}"
                      for t, d, leaf in CONFIGS}
    latest_by_name = {}
    for run in runs:
        run_name = run.info.run_name
        if run_name not in expected_names or run.info.status != "FINISHED":
            continue
        latest_by_name.setdefault(run_name, {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "n_estimators": run.data.params.get("n_estimators"),
            "max_depth": run.data.params.get("max_depth"),
            "min_samples_leaf": run.data.params.get("min_samples_leaf"),
            "seed": run.data.params.get("seed"),
            "accuracy": run.data.metrics.get("accuracy"),
            "f1_macro": run.data.metrics.get("f1_macro"),
            "precision_macro": run.data.metrics.get("precision_macro"),
            "recall_macro": run.data.metrics.get("recall_macro"),
        })
    rows = list(latest_by_name.values())
    if len(rows) < 5:
        raise RuntimeError(f"Se esperaban cinco corridas terminadas; se encontraron {len(rows)}.")
    comparison = pd.DataFrame(rows).sort_values("run_name")
    output = HERE / "comparison_results.csv"
    comparison.to_csv(output, index=False)
    print(f"\nCorridas encontradas: {len(comparison)}")
    print(comparison.to_string(index=False))
    print(f"\nComparación exportada: {output}")
    print(f"Experimento MLflow: {EXPERIMENT}")


if __name__ == "__main__":
    main()
