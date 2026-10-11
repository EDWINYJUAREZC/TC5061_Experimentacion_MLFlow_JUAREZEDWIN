"""Baseline reproducible de clasificación para el conjunto IoT agrícola.

Ejemplo:
  python train.py --data-path IoTProcessed_Data.csv --n-estimators 200 \
    --max-depth 12 --min-samples-leaf 1 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

# MLflow y matplotlib generan directorios temporales. Guardarlos junto al script
# permite ejecutar el flujo en entornos Windows con %TEMP% restringido.
_LOCAL_TEMP = Path(__file__).resolve().parent / ".run_tmp"
_LOCAL_TEMP.mkdir(exist_ok=True)
os.environ["TEMP"] = str(_LOCAL_TEMP)
os.environ["TMP"] = str(_LOCAL_TEMP)
tempfile.tempdir = str(_LOCAL_TEMP)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_ID = "wisam1985/iot-agriculture-2024"
TARGET = "Fan_actuator_ON"
# El nombre con errata se mantiene porque así aparece en el archivo fuente.
FEATURES = ["tempreature", "humidity", "water_level", "N", "P", "K"]
EXPERIMENT = "agricultura_iot_fan_baseline_6_sensores"


def parse_args() -> argparse.Namespace:
    """Define argumentos CLI con valores por defecto reproducibles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="CSV local; si se omite, descarga con kagglehub.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Número de árboles del Random Forest.")
    parser.add_argument("--max-depth", type=int, default=12, help="Profundidad máxima; 0 significa sin límite.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Mínimo de observaciones por hoja.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla usada en partición y modelo.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fracción reservada para prueba.")
    parser.add_argument("--experiment-name", default=EXPERIMENT, help="Experimento MLflow compartido.")
    parser.add_argument("--tracking-uri", default=None, help="URI de Tracking; por defecto ./mlflow.db.")
    args = parser.parse_args()
    if args.n_estimators < 1 or args.min_samples_leaf < 1:
        parser.error("n-estimators y min-samples-leaf deben ser >= 1")
    if args.max_depth < 0:
        parser.error("max-depth debe ser >= 0")
    if not 0 < args.test_size < 1:
        parser.error("test-size debe estar entre 0 y 1")
    return args


def load_data(data_path: str | None) -> tuple[pd.DataFrame, str]:
    """Lee un CSV local o descarga el dataset público con KaggleHub."""
    if data_path:
        path = Path(data_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No existe el CSV especificado: {path}")
        return pd.read_csv(path), str(path)
    import kagglehub
    root = Path(kagglehub.dataset_download(DATASET_ID))
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontraron CSV en la descarga: {root}")
    # El conjunto esperado contiene un CSV. Si hay varios, se concatenan con el mismo esquema.
    frames = [pd.read_csv(p) for p in csvs]
    return pd.concat(frames, ignore_index=True, sort=False), DATASET_ID


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Limpia encabezados, duplicados, tipos y etiquetas; conserva los seis sensores."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    absent = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if absent:
        raise KeyError(f"Faltan columnas requeridas {absent}; recibidas: {df.columns.tolist()}")

    # Conversión explícita: valores no numéricos pasan a NaN y se imputan dentro del pipeline.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid_y = y.notna() & y.isin([0, 1])
    removed_target = int((~valid_y).sum())
    df = df.loc[valid_y].copy()
    y = y.loc[valid_y].astype("int8")
    X = df[FEATURES].copy()
    if X.isna().all(axis=0).any():
        all_missing = X.columns[X.isna().all()].tolist()
        raise ValueError(f"Sensores enteramente vacíos no se pueden imputar: {all_missing}")
    info = {
        "raw_rows": int(len(raw)), "rows_after_cleaning": int(len(df)),
        "duplicate_rows_removed": duplicates, "rows_removed_missing_or_invalid_target": removed_target,
        "feature_count": len(FEATURES), "feature_columns": ",".join(FEATURES),
        "missing_sensor_values": int(X.isna().sum().sum()),
    }
    # Detectar extremos con IQR sin eliminarlos: sensores agrícolas pueden registrar
    # episodios reales fuera del rango habitual y se requiere validación de dominio.
    for col in FEATURES:
        observed = X[col].dropna()
        q1, q3 = observed.quantile([0.25, 0.75])
        iqr = q3 - q1
        info[f"iqr_outliers_{col}"] = int(((observed < q1 - 1.5 * iqr) | (observed > q3 + 1.5 * iqr)).sum())
    return X, y, info


def build_model(args: argparse.Namespace) -> Pipeline:
    """Crea imputación mediana seguida de un clasificador Random Forest."""
    preprocessing = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), FEATURES)],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None if args.max_depth == 0 else args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1,  # evita variaciones por paralelismo y mantiene resultados reproducibles
        )),
    ])


def train_evaluate(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace):
    """Particiona estratificadamente, entrena y calcula métricas de clasificación."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    model = build_model(args)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
    }
    return model, X_test, y_test, prediction, metrics


def log_run(args, source: str, X, y, info, model, y_test, prediction, metrics) -> str:
    """Registra hiperparámetros, métricas, matriz de confusión y modelo en MLflow."""
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)
    depth = "none" if args.max_depth == 0 else str(args.max_depth)
    run_name = f"rf_trees{args.n_estimators}_depth{depth}_leaf{args.min_samples_leaf}_seed{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model": "RandomForestClassifier", "n_estimators": args.n_estimators,
            "max_depth": depth, "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced", "test_size": args.test_size,
            "seed": args.seed, "dataset": source,
            "target": TARGET, "feature_columns": ",".join(FEATURES), **info,
        })
        mlflow.log_metrics(metrics)
        # Huella para identificar la versión exacta del archivo usado en la corrida.
        if Path(source).is_file():
            digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            mlflow.set_tag("dataset_sha256", digest)
        mlflow.set_tag("seed_policy", "train_test_split y RandomForest; n_jobs=1")
        # Mantener el temporal dentro del proyecto evita depender de permisos en %TEMP%.
        local_temp = Path(__file__).resolve().parent / ".run_tmp"
        local_temp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agri_mlflow_", dir=local_temp) as temp_dir:
            temp = Path(temp_dir)
            matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Matriz de confusión - prueba")
            fig.tight_layout()
            figure_path = temp / "confusion_matrix.png"
            fig.savefig(figure_path, dpi=150)
            plt.close(fig)
            metrics_path = temp / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            mlflow.log_artifact(str(figure_path), artifact_path="evaluation")
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(model, name="baseline_model")
        print(f"Run ID: {run.info.run_id}")
        print(f"Experimento: {args.experiment_name}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Filas: {info['rows_after_cleaning']:,} | Predictores: {len(FEATURES)} ({', '.join(FEATURES)})")
        print(json.dumps(metrics, indent=2))
        return run.info.run_id


def main() -> None:
    args = parse_args()
    raw, source = load_data(args.data_path)
    X, y, info = prepare_data(raw)
    model, _, y_test, prediction, metrics = train_evaluate(X, y, args)
    log_run(args, source, X, y, info, model, y_test, prediction, metrics)


if __name__ == "__main__":
    main()
