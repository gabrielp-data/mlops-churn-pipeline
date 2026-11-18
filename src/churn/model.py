"""Treino com pipeline completo e três conjuntos estratificados."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import CategoricalEncoder, FeatureBuilder, prepare_features
from churn.lineage import git_commit


def configure_tracking(cfg: Settings = settings) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def split_data(cfg: Settings):
    frame = load_clean(cfg.data_path, cfg.id_column, cfg.target)
    X = prepare_features(frame)
    y = (frame[cfg.target] == cfg.positive_label).astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=y_dev,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(cfg: Settings) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("encoder", CategoricalEncoder()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    random_state=cfg.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def train_candidate(cfg: Settings, X_train, y_train, X_val, y_val):
    pipeline = build_model(cfg).fit(X_train, y_train)
    metrics = evaluate(pipeline, X_val, y_val)
    with mlflow.start_run(run_name=f"rf-n{cfg.n_estimators}-d{cfg.max_depth}") as run:
        mlflow.log_params(
            {
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "random_state": cfg.random_state,
                "test_size": cfg.test_size,
                "validation_size": cfg.validation_size,
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "source_sha256": hashlib.sha256(
                    b"".join(
                        path.name.encode() + path.read_bytes()
                        for path in sorted(Path(__file__).parent.glob("*.py"))
                    )
                ).hexdigest(),
                "data_file": cfg.data_path.name,
                "data_md5": hashlib.md5(cfg.data_path.read_bytes()).hexdigest(),
                "input_contract": "raw-customer-v1",
            }
        )
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
        encoder = pipeline.named_steps["encoder"]
        contract = {
            "feature_columns": list(X_train.columns),
            "category_mappings": {
                col: list(categories)
                for col, categories in zip(
                    encoder._columns, encoder._encoder.categories_
                )
            },
            "positive_label": cfg.positive_label,
            "threshold": 0.5,
        }
        mlflow.log_dict(contract, "feature_contract.json")
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.iloc[:3],
            code_paths=[str(Path(__file__).parent)],
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in (
                    "mlflow",
                    "cloudpickle",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "pandera",
                    "pydantic",
                    "pydantic-settings",
                    "pyyaml",
                )
            ],
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        candidate = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "val_roc_auc": metrics["roc_auc"],
        }
    return pipeline, candidate, metrics


def evaluate_holdout(pipeline, candidate, X_test, y_test):
    metrics = evaluate(pipeline, X_test, y_test)
    with mlflow.start_run(run_id=candidate["run_id"]):
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
    return metrics


def train(cfg: Settings = settings):
    configure_tracking(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cfg)
    pipeline, candidate, _ = train_candidate(cfg, X_train, y_train, X_val, y_val)
    metrics = evaluate_holdout(pipeline, candidate, X_test, y_test)
    write_json(
        cfg.candidates_path,
        {"candidates": [candidate], "best_run_id": candidate["run_id"]},
    )
    return pipeline, metrics


def main():
    _, metrics = train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
