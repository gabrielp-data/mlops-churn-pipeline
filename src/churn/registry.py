"""Promove apenas modelos completos da execução descrita em candidates.json."""

import json

import mlflow
from mlflow import MlflowClient

from churn.config import Settings, settings
from churn.model import write_json

MODEL_NAME = settings.model_name


def register_model(run_id: str, cfg: Settings = settings):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    run = MlflowClient().get_run(run_id)
    uri = run.data.tags.get("model_uri")
    if run.info.status != "FINISHED" or not uri:
        raise ValueError(
            "Run incompleto ou sem pipeline registrado. Execute o treino novamente."
        )
    return mlflow.register_model(uri, cfg.model_name)


def promote_champion(version: int, model_name: str = MODEL_NAME):
    MlflowClient().set_registered_model_alias(model_name, "champion", str(version))


def promote_challenger(version: int, model_name: str = MODEL_NAME):
    MlflowClient().set_registered_model_alias(model_name, "challenger", str(version))


def load_champion(model_name: str = MODEL_NAME, cfg: Settings = settings):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")


def promote_best(cfg: Settings = settings):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    summary = json.loads(cfg.candidates_path.read_text(encoding="utf-8"))
    client = MlflowClient()
    candidates = [client.get_run(item["run_id"]) for item in summary["candidates"]]
    eligible = [
        run
        for run in candidates
        if run.info.status == "FINISHED"
        and run.data.tags.get("model_uri")
        and "val_roc_auc" in run.data.metrics
    ]
    if not eligible:
        raise ValueError("Nenhum candidato válido com modelo e métrica de validação.")
    best = max(eligible, key=lambda run: run.data.metrics["val_roc_auc"])
    mv = register_model(best.info.run_id, cfg)
    promote_champion(int(mv.version), cfg.model_name)
    info = {
        "model_name": cfg.model_name,
        "version": str(mv.version),
        "run_id": mv.run_id,
    }
    write_json(cfg.promotion_path, info)
    return info


if __name__ == "__main__":
    print(json.dumps(promote_best(), indent=2))
