"""API do app-churn-v2 — arquitetura em duas camadas (two-tier).

Camada de aplicação (1ª EC2): este FastAPI + o @champion em memória.
Camada de dados (2ª EC2): PostgreSQL em Docker com a tabela ``customers``
carregada a partir do churn.csv bruto (CSV → postgres).

O lookup por código faz um ``SELECT`` no banco da outra máquina e o @champion
pontua o registro retornado. A inferência online (``POST /predict``) não depende
do banco: se o Postgres cair, as consultas respondem 503 e o /predict segue no ar.

Uso: ``uv run uvicorn churn.api_v2:app --port 8000`` (com o .env do banco)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from churn import db
from churn.config import settings
from churn.scoring import ChampionModel

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    champion = ChampionModel(settings.champion_dir)
    _state["champion"] = champion
    print(f"Modelo carregado: {champion.info.get('model_name')} v{champion.info.get('version')} @champion")
    yield
    _state.clear()


app = FastAPI(
    title="Churn v2 — two-tier",
    description="Lookup de clientes no PostgreSQL remoto + inferência em tempo real com o @champion.",
    version="2.1.0",
    lifespan=lifespan,
)


def _champion() -> ChampionModel:
    return _state["champion"]


def _fetch_customer(customer_id: str) -> dict:
    """SELECT no PostgreSQL → registro do cliente como dicionário."""
    try:
        with db.get_engine().connect() as conn:
            row = conn.execute(db.QUERY_BY_ID, {"cid": customer_id}).mappings().first()
    except (SQLAlchemyError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Banco PostgreSQL indisponível: {exc}") from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' não encontrado no PostgreSQL.")
    record = dict(row)
    if record.get("MonthlyCharges") is not None:  # NUMERIC chega como Decimal
        record["MonthlyCharges"] = float(record["MonthlyCharges"])
    return record


def _score(record: dict) -> dict:
    try:
        return _champion().score_one(record)
    except ValueError as exc:  # categoria fora do contrato do modelo
        raise HTTPException(status_code=422, detail=str(exc)) from exc


YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class CustomerPayload(BaseModel):
    """Cliente novo. TotalCharges vazio/0 → estimado como tenure × MonthlyCharges."""

    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(default=12, ge=0, le=72)
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: NoInternet = "No"
    OnlineBackup: NoInternet = "No"
    DeviceProtection: NoInternet = "No"
    TechSupport: NoInternet = "No"
    StreamingTV: NoInternet = "No"
    StreamingMovies: NoInternet = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(default=75.0, ge=0.0)
    TotalCharges: float | None = Field(default=None, ge=0.0)


@app.get("/")
def root():
    return {
        "servico": "Churn API v2 (two-tier: app + PostgreSQL)",
        "endpoints": {
            "GET /health": "status do modelo e do banco",
            "GET /model-info": "versão, métricas e linhagem do @champion",
            "GET /customers/{customerID}": "registro do cliente no PostgreSQL",
            "GET /results/{customerID}": "registro do banco + score do modelo",
            "POST /predict": "inferência online para cliente novo",
            "GET /docs": "documentação interativa (Swagger)",
        },
    }


@app.get("/health")
def health():
    cfg = db.db_settings()
    database = {"host": cfg["host"], "port": cfg["port"], "database": cfg["database"]}
    try:
        with db.get_engine().connect() as conn:
            database["records"] = conn.execute(db.COUNT).scalar_one()
        database["status"] = "connected"
    except Exception as exc:  # noqa: BLE001 — /health sempre responde
        database["status"] = f"error: {exc.__class__.__name__}: {exc}".splitlines()[0]
    return {"status": "online", "model": _champion().model_summary(), "database": database}


@app.get("/model-info")
def model_info():
    """Metadados do @champion em uso: versão no Registry, run, métricas e linhagem."""
    return _champion().info


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    """Consulta pura: o registro do cliente exatamente como está no banco."""
    return {"data_source": "postgresql", "customer": _fetch_customer(customer_id)}


@app.get("/results/{customer_id}")
def get_customer_score(customer_id: str):
    """Busca o cliente no PostgreSQL e calcula o score do @champion sobre o registro."""
    record = _fetch_customer(customer_id)
    result = _score(record)
    return {
        "customer_id": record["customerID"],
        **result,
        "actual_churn": record["Churn"],
        "contract_type": record["Contract"],
        "monthly_charges": record["MonthlyCharges"],
        "tenure": record["tenure"],
        "data_source": "postgresql",
        "model": _champion().model_summary(),
        "customer": record,
    }


@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    """Inferência online para um cliente novo (não depende do banco)."""
    data = customer.model_dump()
    if not data["TotalCharges"]:
        data["TotalCharges"] = float(data["tenure"] * data["MonthlyCharges"])
    return {**_score(data), "inference_type": "real_time_online", "model": _champion().model_summary()}
