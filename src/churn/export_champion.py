"""Exporta uma versão imutável do campeão para servir sem acessar o registry."""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from churn.config import Settings, settings
from churn.model import write_json


def export_champion(cfg: Settings = settings, promotion_path: Path | None = None):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient()
    if promotion_path is not None:
        selected = json.loads(promotion_path.read_text(encoding="utf-8"))
        mv = client.get_model_version(selected["model_name"], selected["version"])
        if mv.run_id != selected["run_id"]:
            raise ValueError("Manifesto de promoção não corresponde ao registry.")
    else:
        mv = client.get_model_version_by_alias(cfg.model_name, "champion")
    run = client.get_run(mv.run_id)
    info = {
        "model_name": mv.name,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "alias_at_export": "champion",
        "metrics": run.data.metrics,
        "params": run.data.params,
        "git_commit": run.data.tags.get("git_commit"),
        "source_sha256": run.data.tags.get("source_sha256"),
        "data_file": run.data.tags.get("data_file"),
        "data_md5": run.data.tags.get("data_md5"),
        "exported_at": datetime.now(UTC).isoformat(),
    }
    target = cfg.champion_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".champion-") as temp:
        # A versão é resolvida uma vez; alteração do alias não mistura artefatos.
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{mv.name}/{mv.version}",
                dst_path=temp,
            )
        )
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path="feature_contract.json",
            dst_path=str(downloaded),
        )
        write_json(downloaded / "champion_info.json", info)
        mlflow.sklearn.load_model(str(downloaded))  # valida antes de substituir
        if target.exists():
            # Só a pasta de exportação configurada é substituída.
            if not (target / "champion_info.json").is_file():
                raise ValueError(
                    f"Destino existente não é uma exportação do projeto: {target}"
                )
            shutil.rmtree(target)
        shutil.copytree(downloaded, target)
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_champion(promotion_path=args.promotion), indent=2))
