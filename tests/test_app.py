"""Testes da API FastAPI (churn.app) com um @champion real exportado."""
import mlflow
import pytest
from fastapi.testclient import TestClient

from churn import app as app_module
from churn import registry
from churn.model import train

VALID_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.5, "TotalCharges": 666.0,
}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Treina, registra, promove e exporta um @champion (uma vez por módulo)."""
    from churn.config import Settings
    from tests.conftest import make_synthetic

    tmp = tmp_path_factory.mktemp("api")
    data_path = tmp / "churn.csv"
    make_synthetic().to_csv(data_path, index=False)
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)
    mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
    cfg = Settings(data_path=data_path, artifact_path=tmp / "m.pkl",
                   champion_dir=tmp / "champion", n_estimators=20)
    train(cfg)
    mv = registry.register_model(mlflow.last_active_run().info.run_id)
    registry.promote_champion(int(mv.version))
    registry.export_champion(cfg.champion_dir)
    yield cfg
    mlflow.set_tracking_uri(None)
    mp.undo()


def _client(monkeypatch, **overrides):
    for key, value in overrides.items():
        monkeypatch.setattr(app_module.settings, key, value)
    return TestClient(app_module.app)


@pytest.fixture()
def client(exported, monkeypatch):
    with _client(monkeypatch, model_uri=str(exported.champion_dir), data_path=exported.data_path) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "customers_loaded": 200}


def test_model_info(client, exported):
    info = client.get("/model-info").json()
    assert info["version"] == 1
    assert info["alias"] == "@champion"
    assert "roc_auc" in info["metrics"]
    assert info["source"] == str(exported.champion_dir)


def test_model_info_do_registry(exported, monkeypatch):
    with _client(monkeypatch, model_uri="models:/churn@champion", data_path=exported.data_path) as c:
        assert c.get("/model-info").json()["version"] == 1


def test_predict_novo_cliente(client):
    body = client.post("/predict", json=VALID_CUSTOMER).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["label"] == ("Yes" if body["churn"] else "No")
    assert body["risk"] in {"Baixo", "Médio", "Alto"}
    assert body["model_version"] == 1


def test_predict_campo_faltando(client):
    incompleto = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_predict_categoria_fora_do_contrato(client):
    assert client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Vitalicio"}).status_code == 422


def test_predict_por_customer_id(client):
    body = client.get("/predict/C0000").json()
    assert body["customer_id"] == "C0000"
    assert body["actual_churn"] in {"Yes", "No"}
    assert body["customer_source"] == "base_historica"


def test_predict_cliente_inexistente(client):
    assert client.get("/predict/NAO-EXISTE").status_code == 404


def test_simulacao_e_se(client):
    base = client.get("/predict/C0000").json()
    sim = client.post("/predict/C0000", json={"tenure": 72, "TotalCharges": 5000.0}).json()
    assert sim["customer_source"] == "base_historica_com_simulacao"
    assert sim["simulated_values"] == {"tenure": 72, "TotalCharges": 5000.0}
    assert sim["actual_churn"] == base["actual_churn"]


def test_sem_modelo_responde_503(exported, monkeypatch, tmp_path):
    with _client(monkeypatch, model_uri=str(tmp_path / "nada"), data_path=exported.data_path) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=VALID_CUSTOMER).status_code == 503
        assert c.get("/model-info").status_code == 503
