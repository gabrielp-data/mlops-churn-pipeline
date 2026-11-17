"""Tarefas do pipeline de treino — Encontros 9–11 (orquestração).

Cada função aqui é uma *tarefa* do DAG do Airflow (``dags/churn_training.py``),
mas é Python puro: não importa nada do Airflow. Isso permite

* testar as tarefas com pytest, sem subir um scheduler;
* rodar o pipeline inteiro localmente com ``uv run python -m churn.pipeline``.

Regra de ouro do Airflow: o que trafega entre tarefas (XCom) deve ser pequeno e
serializável em JSON — ids de run, métricas, versões. Nunca o DataFrame ou o
modelo em si; esses ficam no disco / no MLflow.

Fluxo:

    validate_data → train_candidate → quality_gate → decide_promotion
                                                     ├─ promote_champion
                                                     └─ register_challenger
"""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn.config import Settings, settings
from churn.data import load_clean
from churn.model import _safe_data_md5, train
from churn.registry import MODEL_NAME, promote_challenger, register_model
from churn.registry import promote_champion as _set_champion_alias

PROMOTE = "promote_champion"
KEEP = "register_challenger"


class QualityGateError(RuntimeError):
    """Modelo candidato abaixo do mínimo aceitável — o DAG deve falhar aqui."""


# ---------------------------------------------------------------- tarefas
def validate_data(cfg: Settings = settings) -> dict:
    """Carrega e valida o dataset (pandera). Falha cedo se o dado estiver ruim."""
    df = load_clean(cfg.data_path, cfg.id_column, cfg.target)
    churn_rate = float((df[cfg.target] == cfg.positive_label).mean())
    return {"rows": int(len(df)), "churn_rate": churn_rate, "data_md5": _safe_data_md5(cfg)}


def train_candidate(cfg: Settings = settings) -> dict:
    """Treina um modelo candidato (run rastreada no MLflow) e devolve id + métricas."""
    _, metrics = train(cfg)
    run = mlflow.last_active_run()
    if run is None:  # pragma: no cover — train() sempre abre uma run
        raise RuntimeError("Nenhuma run do MLflow encontrada após o treino.")
    return {"run_id": run.info.run_id, "metrics": metrics}


def quality_gate(candidate: dict, cfg: Settings = settings) -> dict:
    """Barra candidatos com roc_auc abaixo de ``cfg.min_roc_auc``."""
    roc_auc = candidate["metrics"]["roc_auc"]
    if roc_auc < cfg.min_roc_auc:
        raise QualityGateError(
            f"roc_auc={roc_auc:.4f} < mínimo {cfg.min_roc_auc:.4f}; candidato descartado."
        )
    return candidate


def champion_roc_auc(model_name: str = MODEL_NAME) -> float | None:
    """roc_auc da run por trás do ``@champion`` atual (``None`` se não houver campeão)."""
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get("roc_auc")


def decide_promotion(candidate: dict, model_name: str = MODEL_NAME) -> str:
    """Branch: candidato vira campeão se superar o roc_auc do campeão atual."""
    current = champion_roc_auc(model_name)
    if current is None or candidate["metrics"]["roc_auc"] > current:
        return PROMOTE
    return KEEP


def promote_champion(candidate: dict, model_name: str = MODEL_NAME) -> int:
    """Registra o candidato e move o alias ``@champion`` para ele."""
    mv = register_model(candidate["run_id"])
    _set_champion_alias(int(mv.version), model_name)
    return int(mv.version)


def register_challenger(candidate: dict, model_name: str = MODEL_NAME) -> int:
    """Registra o candidato como ``@challenger`` (o campeão permanece)."""
    mv = register_model(candidate["run_id"])
    promote_challenger(int(mv.version), model_name)
    return int(mv.version)


# ------------------------------------------------------ execução local
def run(cfg: Settings = settings) -> dict:
    """Executa o DAG em sequência, sem Airflow — útil em dev e em testes."""
    report = validate_data(cfg)
    print(f"[validate] {report['rows']} linhas, churn={report['churn_rate']:.1%}")

    candidate = quality_gate(train_candidate(cfg), cfg)
    print(f"[train]    run={candidate['run_id'][:8]} roc_auc={candidate['metrics']['roc_auc']:.4f}")

    branch = decide_promotion(candidate)
    version = (promote_champion if branch == PROMOTE else register_challenger)(candidate)
    alias = "champion" if branch == PROMOTE else "challenger"
    print(f"[registry] v{version} → @{alias}")
    return {"branch": branch, "version": version, **candidate}


if __name__ == "__main__":
    run()
