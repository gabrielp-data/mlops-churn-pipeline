"""Apps da AWS: scoring, lote (v1) e two-tier com banco (v2).

O PostgreSQL é substituído por um SQLite com a mesma tabela `customers` — a API
usa SQLAlchemy, então a query é a mesma.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from churn import api_lookup, api_v2, batch_scoring, db
from churn.scoring import ChampionModel, risk_level, to_model_input
from tests.conftest import make_synthetic


@pytest.fixture()
def champion(exported_champion):
    return ChampionModel(exported_champion.champion_dir)


# ------------------------------------------------------------ scoring
def test_risk_level():
    assert [risk_level(p) for p in (0.1, 0.3, 0.59, 0.6)] == ["Baixo", "Médio", "Médio", "Alto"]


def test_contrato_rejeita_categoria_desconhecida(raw_frame):
    with pytest.raises(ValueError, match="Vitalicio"):
        to_model_input(raw_frame.assign(Contract="Vitalicio"))


def test_score_one(champion, raw_frame):
    record = raw_frame.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    out = champion.score_one(record)
    assert out["churn_prediction"] in (0, 1)
    assert 0 <= out["churn_probability"] <= 1
    assert champion.model_summary()["version"] == 1


def test_champion_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="--export"):
        ChampionModel(tmp_path)


# ------------------------------------------------------------ v1 (lote)
def test_batch_scoring_e_api_lookup(exported_champion, champion, tmp_path, monkeypatch):
    out = tmp_path / "batch.parquet"
    result = batch_scoring.score_all(exported_champion.data_path, out, champion)
    assert len(result) == 200
    assert set(result["risk_level"]) <= {"Baixo", "Médio", "Alto"}

    monkeypatch.setattr(api_lookup, "OUTPUT_PATH", out)
    with TestClient(api_lookup.app) as c:
        assert c.get("/health").json()["records_available"] == 200
        body = c.get("/results/C0000").json()
        assert body["customer_id"] == "C0000"
        assert body["model_version"] == 1
        assert c.get("/results/NAO-EXISTE").status_code == 404


# ------------------------------------------------------------ v2 (two-tier)
@pytest.fixture()
def v2(exported_champion, tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'customers.db'}")
    df = make_synthetic()
    df.loc[1, "Contract"] = "Vitalicio"  # registro sujo no banco
    df.to_sql("customers", engine, index=False)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    monkeypatch.setattr(api_v2.settings, "champion_dir", exported_champion.champion_dir)
    with TestClient(api_v2.app) as c:
        yield c


def test_v2_health(v2):
    body = v2.get("/health").json()
    assert body["database"]["status"] == "connected"
    assert body["database"]["records"] == 200
    assert body["model"]["version"] == 1


def test_v2_customer_e_results(v2):
    assert v2.get("/customers/C0000").json()["customer"]["customerID"] == "C0000"
    body = v2.get("/results/C0000").json()
    assert body["data_source"] == "postgresql"
    assert body["actual_churn"] in {"Yes", "No"}
    assert 0 <= body["churn_probability"] <= 1


def test_v2_nao_encontrado_e_contrato(v2):
    assert v2.get("/results/NAO-EXISTE").status_code == 404
    resp = v2.get("/results/C0001")  # Contract = Vitalicio no banco
    assert resp.status_code == 422
    assert "Vitalicio" in resp.json()["detail"]


def test_v2_predict_online_estima_total(v2):
    body = v2.post("/predict", json={"tenure": 2, "MonthlyCharges": 90.0}).json()
    assert body["inference_type"] == "real_time_online"
    assert body["risk_level"] in {"Baixo", "Médio", "Alto"}
    assert v2.post("/predict", json={"Contract": "Vitalicio"}).status_code == 422


def test_v2_banco_fora_do_ar(exported_champion, monkeypatch):
    def sem_banco():
        raise RuntimeError("DB_PASSWORD não definida")
    monkeypatch.setattr(db, "get_engine", sem_banco)
    monkeypatch.setattr(api_v2.settings, "champion_dir", exported_champion.champion_dir)
    with TestClient(api_v2.app) as c:
        assert c.get("/health").json()["database"]["status"].startswith("error")
        assert c.get("/results/C0000").status_code == 503
        assert c.post("/predict", json={}).status_code == 200  # online não depende do banco
