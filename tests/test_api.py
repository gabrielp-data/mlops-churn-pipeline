import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError


def test_lookup_and_prediction_use_same_model(api, customer):
    with TestClient(api.app) as client:
        lookup = client.get("/results/7590-VHVEG")
        payload = {
            k: v for k, v in customer.items() if k not in ("customerID", "Churn")
        }
        payload["SeniorCitizen"] = str(payload["SeniorCitizen"])
        response = client.post("/predict", json=payload)
        assert lookup.status_code == response.status_code == 200
        assert (
            response.json()["churn_probability"]
            == lookup.json()["churn_probability"]
            == 0.5439
        )
        assert client.get("/model-info").json()["run_name"] == "rf-n500-d8"


@pytest.mark.parametrize("customer_id", ["missing", "' OR '1'='1"])
def test_missing_id_and_sql_metacharacters_do_not_return_records(api, customer_id):
    with TestClient(api.app) as client:
        assert client.get(f"/customers/{customer_id}").status_code == 404


@pytest.mark.parametrize("payload", [{"Contract": "Vitalicio"}, {"tenure": -1}])
def test_invalid_prediction_returns_422(api, payload):
    with TestClient(api.app) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_database_failure_preserves_online_inference(api, monkeypatch):
    def unavailable():
        raise OperationalError("test", {}, Exception("offline"))

    monkeypatch.setattr(api, "get_engine", unavailable)
    with TestClient(api.app) as client:
        assert client.get("/customers/7590-VHVEG").status_code == 503
        response = client.post("/predict", json={})
        assert response.status_code == 200
        assert 0 <= response.json()["churn_probability"] <= 1
