"""API REST de predição de churn — Encontro 12.

Serve o modelo ``@champion`` do MLflow Model Registry.

Endpoints:
  GET  /                       → status
  GET  /health                 → API no ar? modelo e base carregados?
  GET  /model-info             → versão, run, métricas, hiperparâmetros e linhagem do @champion
  GET  /predict/{customer_id}  → predição para um cliente da base histórica
  POST /predict/{customer_id}  → idem, simulando outros tenure/MonthlyCharges/TotalCharges ("e se")
  POST /predict                → predição para um cliente novo (todas as colunas)

De onde vem o modelo (``CHURN_MODEL_URI``):
  * ``models:/churn@champion`` (padrão) → direto do Registry, para dev local;
  * um diretório (ex.: ``artifacts/champion``) → cópia gerada por
    ``python -m churn.registry --export``, usada no container.

Uso:
    uv run uvicorn churn.app:app --reload --port 8000
    Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from churn.config import settings
from churn.data import coerce_total_charges, load_raw
from churn.registry import INFO_FILE, champion_info

EXEMPLO_ID = "7590-VHVEG"
THRESHOLD = 0.5

_state: dict = {}


# --------------------------------------------------------------- carga
def _load_model(uri: str):
    """Carrega o pipeline sklearn + metadados (do Registry ou de um export local)."""
    model = mlflow.sklearn.load_model(uri)
    if uri.startswith("models:/"):
        info = champion_info(uri.removeprefix("models:/").split("@")[0])
    else:
        info_path = Path(uri) / INFO_FILE
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return model, {**info, "source": uri, "status": "loaded"}


def _load_customers(path: Path) -> pd.DataFrame | None:
    """Base histórica de clientes (indexada por customerID) para consulta por código."""
    if not path.exists():
        return None
    df = coerce_total_charges(load_raw(path))
    return df.set_index(settings.id_column)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sem modelo a API ainda sobe: /health e /model-info dizem o motivo e /predict responde 503.
    try:
        _state["model"], _state["info"] = _load_model(settings.model_uri)
    except Exception as exc:  # noqa: BLE001
        _state["model"] = None
        _state["info"] = {"source": settings.model_uri, "status": "indisponível", "error": str(exc)}
    _state["customers"] = _load_customers(settings.data_path)
    yield
    _state.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Serve o modelo @champion do MLflow Model Registry.",
    version="0.2.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------- schemas
YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class CustomerIn(BaseModel):
    """Todas as colunas de um cliente — mesmo contrato do ChurnSchema (pandera)."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class Simulation(BaseModel):
    """Valores numéricos a simular para um cliente existente (todos opcionais)."""

    tenure: int | None = Field(None, ge=0, le=72, examples=[6])
    MonthlyCharges: float | None = Field(None, ge=0, examples=[85.0])
    TotalCharges: float | None = Field(None, ge=0, examples=[510.0])


class PredictionOut(BaseModel):
    churn: bool
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    label: Literal["Yes", "No"]
    risk: Literal["Baixo", "Médio", "Alto"]
    model_version: int | None = None


class CustomerPredictionOut(PredictionOut):
    customer_id: str
    actual_churn: str | None = None
    customer_source: Literal["base_historica", "base_historica_com_simulacao"]
    simulated_values: dict[str, int | float] = {}


# ------------------------------------------------------------ inferência
def _risk(proba: float) -> str:
    return "Baixo" if proba < 0.3 else "Médio" if proba < 0.6 else "Alto"


def _score(features: pd.DataFrame) -> dict:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modelo @champion indisponível: {_state['info'].get('error')}")
    proba = float(model.predict_proba(features)[0, 1])
    churn = proba >= THRESHOLD
    return {
        "churn": churn,
        "churn_probability": round(proba, 4),
        "label": "Yes" if churn else "No",
        "risk": _risk(proba),
        "model_version": _state["info"].get("version"),
    }


def _predict_customer(customer_id: str, sim: Simulation | None = None) -> dict:
    customers = _state.get("customers")
    if customers is None:
        raise HTTPException(status_code=503, detail=f"Base de clientes não encontrada em {settings.data_path}.")
    if customer_id not in customers.index:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' não encontrado na base.")

    row = customers.loc[[customer_id]]
    actual = str(row[settings.target].iloc[0])
    features = row.drop(columns=[settings.target]).reset_index(drop=True)
    simulated = sim.model_dump(exclude_none=True) if sim else {}
    for column, value in simulated.items():
        features[column] = value

    return {
        **_score(features),
        "customer_id": customer_id,
        "actual_churn": actual,
        "customer_source": "base_historica_com_simulacao" if simulated else "base_historica",
        "simulated_values": simulated,
    }


# --------------------------------------------------------------- rotas
@app.get("/", tags=["infra"])
def root():
    return {"status": "online", "message": "API de churn ativa. Acesse /docs"}


@app.get("/health", tags=["infra"])
def health():
    """Verifica se a API está no ar e o que foi carregado."""
    customers = _state.get("customers")
    return {
        "status": "ok",
        "model_loaded": _state.get("model") is not None,
        "customers_loaded": 0 if customers is None else len(customers),
    }


@app.get("/model-info", tags=["modelo"])
def model_info():
    """Versão, run, métricas, hiperparâmetros e linhagem (DVC + Git) do @champion."""
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail=_state["info"])
    return _state["info"]


@app.get("/predict/{customer_id}", response_model=CustomerPredictionOut, tags=["inferência"])
def predict_customer(customer_id: str = PathParam(..., examples=[EXEMPLO_ID])):
    """Predição do @champion para um cliente da base histórica."""
    return _predict_customer(customer_id)


@app.post("/predict/{customer_id}", response_model=CustomerPredictionOut, tags=["inferência"])
def simulate_customer(sim: Simulation, customer_id: str = PathParam(..., examples=[EXEMPLO_ID])):
    """Simulação "e se": troca tenure/MonthlyCharges/TotalCharges de um cliente existente."""
    return _predict_customer(customer_id, sim)


@app.post("/predict", response_model=PredictionOut, tags=["inferência"])
def predict(customer: CustomerIn):
    """Predição para um cliente novo, informando todas as colunas."""
    return _score(pd.DataFrame([customer.model_dump()]))
