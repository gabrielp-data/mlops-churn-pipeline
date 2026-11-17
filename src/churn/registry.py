"""Model Registry — Encontro 7.

Registra a melhor run no MLflow Model Registry, promove aliases e lista versões.

Uso:
    uv run python -m churn.registry           # registra o melhor run e aplica @champion
    uv run python -m churn.registry --export  # copia o @champion para artifacts/champion/

Depois: ``uv run mlflow ui``
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from churn.config import settings

MODEL_NAME = "churn"
INFO_FILE = "champion_info.json"


def register_model(run_id: str, artifact_path: str = "model") -> mlflow.entities.model_registry.ModelVersion:
    """Registra o artefato de uma run no Model Registry e retorna a versão."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.register_model(model_uri, MODEL_NAME)


def promote_champion(version: int, model_name: str = MODEL_NAME) -> None:
    """Marca `@champion` na versão informada."""
    MlflowClient().set_registered_model_alias(model_name, "champion", version)


def promote_challenger(version: int, model_name: str = MODEL_NAME) -> None:
    """Marca `@challenger` na versão informada."""
    MlflowClient().set_registered_model_alias(model_name, "challenger", version)


def load_champion(model_name: str = MODEL_NAME):
    """Carrega o modelo marcado como @champion via pyfunc."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")


def champion_info(model_name: str = MODEL_NAME) -> dict:
    """Metadados do @champion: versão, run, métricas, hiperparâmetros e linhagem."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "champion")
    run = client.get_run(mv.run_id)
    return {
        "model_name": model_name,
        "alias": "@champion",
        "version": int(mv.version),
        "run_id": mv.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "metrics": run.data.metrics,
        "hyperparameters": run.data.params,
        "data_md5": run.data.tags.get("data_md5"),
        "git_commit": run.data.tags.get("git_commit"),
    }


def export_champion(dest: Path = settings.champion_dir, model_name: str = MODEL_NAME) -> dict:
    """Copia o pacote MLflow do @champion + ``champion_info.json`` para ``dest``.

    O Registry vive no ``mlflow.db`` local; a imagem/containers da API leem
    essa cópia, sem depender do banco nem dos caminhos absolutos do ``mlruns/``.
    """
    info = champion_info(model_name)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(f"models:/{model_name}@champion", dst_path=str(dest))
    (dest / INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def list_versions(model_name: str = MODEL_NAME) -> None:
    """Imprime versões registradas com seus aliases."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"  Nenhuma versão registrada para '{model_name}'.")
        return
    for v in versions:
        # search_model_versions não preenche aliases; get_model_version sim
        aliases = client.get_model_version(model_name, v.version).aliases
        print(f"  v{v.version}  run={v.run_id[:8]}  status={v.status}  aliases={aliases}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Registry do churn")
    parser.add_argument("--export", action="store_true", help="exporta o @champion atual")
    if parser.parse_args().export:
        info = export_champion()
        print(f"@champion v{info['version']} exportado para {settings.champion_dir}/")
        return

    runs = mlflow.search_runs(
        experiment_names=["churn"],
        order_by=["metrics.roc_auc DESC"],
    )
    if runs.empty:
        print("Nenhuma run encontrada. Execute churn.model ou churn.experiments primeiro.")
        return

    best = runs.iloc[0]
    run_id = best["run_id"]
    roc_auc = float(best["metrics.roc_auc"])
    print(f"Melhor run: {run_id[:8]}  roc_auc={roc_auc:.4f}")

    mv = register_model(run_id)
    print(f"Modelo registrado → versão {mv.version}")

    promote_champion(int(mv.version))
    print(f"Alias @champion → v{mv.version}")

    print("\nVersões registradas:")
    list_versions()


if __name__ == "__main__":
    main()
