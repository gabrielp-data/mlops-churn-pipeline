"""Treino do modelo de churn — orquestra o pipeline de ponta a ponta.

Entrada: ``python -m churn.model`` (ou ``uv run python -m churn.model``).

O artefato é um ``sklearn.Pipeline`` completo:

    features derivadas → encoding das categóricas → RandomForest

Assim o modelo registrado no MLflow recebe o **dado cru** (as 19 colunas do
cliente) e quem consome — API, lote, painel — não precisa reimplementar o
pré-processamento. Treino e inferência usam exatamente as mesmas transformações.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn.config import Settings, settings
from churn.data import load_clean
from churn.evaluate import evaluate
from churn.features import (
    CategoricalEncoder,
    add_derived_features,
    split_features_target,
)
from churn.lineage import data_md5, git_commit

EXPERIMENT = "churn"

# O MLflow 3 serializa sklearn com skops (formato seguro, sem pickle arbitrário):
# tipos do próprio projeto precisam ser declarados como confiáveis.
TRUSTED_TYPES = [
    "churn.features.CategoricalEncoder",
    "churn.features.add_derived_features",
]


def build_pipeline(cfg: Settings, **rf_params) -> Pipeline:
    """Pipeline dado cru → probabilidade de churn, com semente fixa (reprodutível)."""
    params = {"n_estimators": cfg.n_estimators, "random_state": cfg.random_state, **rf_params}
    return Pipeline([
        ("features", FunctionTransformer(add_derived_features)),
        ("encoder", CategoricalEncoder()),
        ("model", RandomForestClassifier(**params)),
    ])


def load_split(cfg: Settings = settings):
    """Dados validados → (x_train, x_test, y_train, y_test), estratificado."""
    frame = load_clean(cfg.data_path, cfg.id_column, cfg.target)
    features, labels = split_features_target(frame, cfg.target, cfg.positive_label)
    return train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )


def log_pipeline(pipe: Pipeline, x_train: pd.DataFrame, cfg: Settings) -> None:
    """Loga linhagem + modelo (com assinatura e exemplo) na run ativa do MLflow."""
    mlflow.set_tags({"git_commit": git_commit(), "data_md5": _safe_data_md5(cfg)})
    signature = infer_signature(x_train, pipe.predict(x_train))
    mlflow.sklearn.log_model(
        pipe,
        name="model",
        signature=signature,
        input_example=x_train.iloc[:5],
        skops_trusted_types=TRUSTED_TYPES,
    )


def save_artifact(pipe: Pipeline, path: Path) -> None:
    """Persiste o pipeline treinado (saída do estágio DVC)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(pipe, handle)


def train(cfg: Settings = settings) -> tuple[Pipeline, dict[str, float]]:
    """Executa o pipeline: dados → split → treino → avaliação → MLflow."""
    x_train, x_test, y_train, y_test = load_split(cfg)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="rf"):
        mlflow.log_params(cfg.model_dump(mode="json"))
        pipe = build_pipeline(cfg).fit(x_train, y_train)
        metrics = evaluate(pipe, x_test, y_test)
        mlflow.log_metrics(metrics)
        log_pipeline(pipe, x_train, cfg)

    save_artifact(pipe, cfg.artifact_path)
    return pipe, metrics


def _safe_data_md5(cfg: Settings) -> str:
    pointer = cfg.data_path.parent / (cfg.data_path.name + ".dvc")
    try:
        return data_md5(pointer)
    except Exception:
        return "no-dvc"


def main() -> None:
    _, metrics = train()
    print("Modelo treinado. Métricas no teste:")
    for name, value in metrics.items():
        print(f"  {name:>10}: {value:.4f}")


if __name__ == "__main__":
    main()
