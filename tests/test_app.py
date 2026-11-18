"""Contrato HTTP com pipeline real; mocks só para falhas externas."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from churn.app import create_app
from churn.config import Settings
from churn.inference import ChampionModel
from churn.model import build_model


@pytest.fixture
def client(raw_frame, tmp_path):
    cfg = Settings(_env_file=None, data_path=tmp_path / "customers.csv", n_estimators=5)
    raw_frame.to_csv(cfg.data_path, index=False)
    champion = ChampionModel.__new__(ChampionModel)
    champion.model = build_model(cfg).fit(
        raw_frame, (raw_frame.Churn == "Yes").astype(int)
    )
    champion.info = {"version": "7", "run_id": "test-run", "model_name": "churn"}
    with TestClient(create_app(cfg, model_factory=lambda path: champion)) as client:
        yield client


def payload(raw_frame):
    return raw_frame.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()


def test_health_and_prediction(client, raw_frame):
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post("/predict", json=payload(raw_frame))
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["model_version"] == "7"
    assert result["label"] == ("Yes" if result["churn"] else "No")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 73),
        ("SeniorCitizen", 2),
        ("MonthlyCharges", -1),
        ("TotalCharges", -10),
        ("Contract", "Vitalício"),
    ],
)
def test_invalid_payload(client, raw_frame, field, value):
    data = payload(raw_frame)
    data[field] = value
    assert client.post("/predict", json=data).status_code == 422


def test_missing_field(client, raw_frame):
    data = payload(raw_frame)
    data.pop("tenure")
    assert client.post("/predict", json=data).status_code == 422


def test_lookup_and_online_match(client, raw_frame):
    online = client.post("/predict", json=payload(raw_frame)).json()
    lookup = client.get("/results/A1")
    assert lookup.status_code == 200
    assert lookup.json()["churn_probability"] == online["churn_probability"]
    assert lookup.json()["actual_churn"] == "No"
    assert client.get("/customers/not-found").status_code == 404


def test_database_outage_does_not_stop_prediction(client, raw_frame, monkeypatch):
    def unavailable(*args):
        raise OperationalError("query", {}, Exception("private connection details"))

    monkeypatch.setattr("churn.db.check_database", unavailable)
    monkeypatch.setattr("churn.db.fetch_customer", unavailable)
    assert client.get("/health").status_code == 503
    response = client.get("/results/A1")
    assert response.status_code == 503
    assert "private" not in response.text
    assert client.post("/predict", json=payload(raw_frame)).status_code == 200
