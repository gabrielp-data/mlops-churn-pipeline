"""Treino → Registry (@champion) → export → predição em lote, com MLflow isolado."""
import json

import mlflow
import pytest

from churn import registry
from churn.model import train
from churn.predict import predict


@pytest.fixture()
def champion(mlflow_env):
    _, metrics = train(mlflow_env)
    run_id = mlflow.last_active_run().info.run_id
    mv = registry.register_model(run_id)
    registry.promote_champion(int(mv.version))
    return mlflow_env, run_id, metrics


def test_champion_info(champion):
    _, run_id, metrics = champion
    info = registry.champion_info()
    assert info["version"] == 1
    assert info["run_id"] == run_id
    assert info["metrics"]["roc_auc"] == pytest.approx(metrics["roc_auc"])
    assert info["hyperparameters"]["n_estimators"] == "20"
    assert info["data_md5"] == "no-dvc"


def test_export_champion(champion):
    cfg, run_id, _ = champion
    registry.export_champion(cfg.champion_dir)
    assert (cfg.champion_dir / "MLmodel").exists()
    exported = json.loads((cfg.champion_dir / registry.INFO_FILE).read_text())
    assert exported["run_id"] == run_id

    # o pacote exportado é um pipeline completo: recebe dado cru
    model = mlflow.sklearn.load_model(str(cfg.champion_dir))
    assert hasattr(model, "predict_proba")


def test_predict_em_lote_com_dado_cru(champion, tmp_path):
    cfg, _, _ = champion
    out = tmp_path / "preds.csv"
    result = predict(str(cfg.data_path), str(out))
    assert len(result) == 200
    assert result["churn_probability"].between(0, 1).all()
    assert out.exists()
