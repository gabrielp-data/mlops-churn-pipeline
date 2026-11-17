"""Testes das tarefas do DAG (churn.pipeline) — sem Airflow, com MLflow isolado."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from churn import pipeline
from churn.config import Settings


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch) -> Settings:
    """Dataset sintético + MLflow (tracking e registry) em SQLite temporário."""
    rng = np.random.default_rng(0)
    n = 200
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(20, 110, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    # sinal aprendível: contrato mensal + pouco tempo de casa → churn
    churn = np.where((contract == "Month-to-month") & (tenure < 24), "Yes", "No")
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": yes_no(),
        "PaymentMethod": "Electronic check", "MonthlyCharges": monthly,
        "TotalCharges": (monthly * tenure).round(2).astype(str), "Churn": churn,
    })
    data_path = tmp_path / "churn.csv"
    df.to_csv(data_path, index=False)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)  # artefatos do mlflow (./mlruns) ficam no tmp
    yield Settings(
        data_path=data_path,
        artifact_path=tmp_path / "artifacts" / "model.pkl",
        n_estimators=20,
    )
    mlflow.set_tracking_uri(None)


def test_validate_data(cfg):
    report = pipeline.validate_data(cfg)
    assert report["rows"] == 200
    assert 0.0 < report["churn_rate"] < 1.0


def test_quality_gate_reprova():
    with pytest.raises(pipeline.QualityGateError):
        pipeline.quality_gate({"run_id": "x", "metrics": {"roc_auc": 0.5}}, Settings(min_roc_auc=0.75))


def test_quality_gate_aprova():
    cand = {"run_id": "x", "metrics": {"roc_auc": 0.9}}
    assert pipeline.quality_gate(cand, Settings(min_roc_auc=0.75)) is cand


def test_primeiro_run_vira_champion_depois_challenger(cfg):
    first = pipeline.run(cfg)
    assert first["branch"] == pipeline.PROMOTE
    assert pipeline.champion_roc_auc() == pytest.approx(first["metrics"]["roc_auc"])

    # mesmo dado + mesma semente → mesmo roc_auc → não supera o campeão
    second = pipeline.run(cfg)
    assert second["branch"] == pipeline.KEEP

    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias("churn", "champion").version) == first["version"]
    assert int(client.get_model_version_by_alias("churn", "challenger").version) == second["version"]


def test_decide_promotion_supera_campeao(cfg):
    first = pipeline.run(cfg)
    better = {"run_id": first["run_id"], "metrics": {"roc_auc": first["metrics"]["roc_auc"] + 0.01}}
    assert pipeline.decide_promotion(better) == pipeline.PROMOTE
