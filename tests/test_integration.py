"""Treino → registry → exportação → lote/API com MLflow e modelos reais."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from churn.api_lookup import create_lookup_app
from churn.app import create_app
from churn.config import Settings
from churn.db import fetch_customer, import_customers
from churn.experiments import run_grid
from churn.export_champion import export_champion
from churn.inference import ChampionModel
from churn.predict import predict
from churn.registry import load_champion, promote_best


@pytest.fixture
def dataset(raw_frame, tmp_path):
    frame = pd.concat([raw_frame] * 20, ignore_index=True)
    frame["customerID"] = [f"TEST-{i}" for i in range(len(frame))]
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_full_lifecycle(dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, frame = dataset
    cfg = Settings(
        _env_file=None,
        data_path=path,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment_name="integration",
        n_estimators=3,
        candidates_path=tmp_path / "candidates.json",
        promotion_path=tmp_path / "champion.json",
        champion_dir=tmp_path / "champion",
    )
    summary = run_grid(cfg, n_estimators=(3, 5), depths=(2,))
    assert len(summary["candidates"]) == 2
    runs = [mlflow.get_run(c["run_id"]) for c in summary["candidates"]]
    assert all(run.data.tags.get("model_uri") for run in runs)
    assert sum("test_roc_auc" in run.data.metrics for run in runs) == 1

    promoted = promote_best(cfg)
    assert promoted["run_id"] == summary["best_run_id"]
    exported = export_champion(cfg, cfg.promotion_path)
    champion = ChampionModel(cfg.champion_dir)
    assert exported["run_id"] == champion.info["run_id"]
    raw = frame.drop(columns=["Churn"])
    expected = champion.predict_proba(raw)

    # O pyfunc registrado também recebe categorias brutas.
    registry_model = load_champion(cfg.model_name, cfg)
    from churn.features import prepare_features

    np.testing.assert_array_equal(
        registry_model.predict(prepare_features(raw)), expected >= 0.5
    )

    unlabeled = tmp_path / "new-customers.csv"
    raw.to_csv(unlabeled, index=False)
    batch = tmp_path / "predictions.parquet"
    result = predict(str(unlabeled), str(batch), cfg.champion_dir)
    np.testing.assert_allclose(result.churn_probability, expected, rtol=0, atol=0)

    # Sem identificador também funciona.
    no_id = tmp_path / "no-id.csv"
    raw.drop(columns=["customerID"]).to_csv(no_id, index=False)
    assert len(predict(str(no_id), model_dir=cfg.champion_dir)) == len(raw)

    database_url = f"sqlite:///{(tmp_path / 'customers.db').as_posix()}"
    import_customers(path, create_engine(database_url))
    cfg_db = cfg.model_copy(
        update={
            "database_url": Settings(
                _env_file=None,
                database_url=database_url,
            ).database_url
        }
    )
    with TestClient(create_app(cfg_db)) as client:
        response = client.get("/results/TEST-0")
        assert response.status_code == 200, response.text
        assert response.json()["churn_probability"] == expected[0]
        payload = json.loads(prepare_features(raw.iloc[:1]).to_json(orient="records"))[
            0
        ]
        online = client.post("/predict", json=payload)
        assert online.status_code == 200, online.text
        assert online.json()["churn_probability"] == expected[0]
        assert online.json()["run_id"] == promoted["run_id"]
        assert client.get("/model-info").json()["version"] == promoted["version"]

    with TestClient(create_lookup_app(batch)) as client:
        assert client.get("/results/TEST-0").json()["churn_probability"] == expected[0]
        assert client.get("/results/missing").status_code == 404
        assert client.post("/predict", json={}).status_code == 404

    # Reexportar a mesma versão não perde o pacote.
    assert export_champion(cfg)["run_id"] == promoted["run_id"]


def test_database_import_is_transactional(dataset, tmp_path):
    path, frame = dataset
    url = f"sqlite:///{(tmp_path / 'customers.db').as_posix()}"
    engine = create_engine(url)
    assert import_customers(path, engine) == len(frame)
    with pytest.raises(IntegrityError):
        import_customers(path, engine)
    cfg = Settings(_env_file=None, database_url=url)
    assert fetch_customer("TEST-0", cfg)["customerID"] == "TEST-0"
    assert fetch_customer("' OR 1=1 --", cfg) is None
