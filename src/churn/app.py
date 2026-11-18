"""API online servindo exatamente o pacote exportado do campeão."""

from contextlib import asynccontextmanager

import pandera.errors
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from churn import db
from churn.config import Settings, settings
from churn.inference import ChampionModel


class CustomerIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionOut(BaseModel):
    churn: bool
    churn_prediction: int
    churn_probability: float = Field(ge=0, le=1)
    label: str
    risk_level: str
    model_version: str
    run_id: str


def create_app(cfg: Settings = settings, model_factory=ChampionModel):
    @asynccontextmanager
    async def lifespan(app):
        app.state.champion = model_factory(cfg.champion_dir)
        yield
        app.state.champion = None

    app = FastAPI(
        title="Churn · Clientes e previsões", version="0.2.0", lifespan=lifespan
    )

    def score(record):
        try:
            return app.state.champion.score_one(record)
        except (
            ValueError,
            pandera.errors.SchemaErrors,
            pandera.errors.SchemaError,
        ) as exc:
            raise HTTPException(
                422,
                detail="Dados fora do contrato do modelo. Confira categorias e valores.",
            ) from exc

    def customer(customer_id):
        try:
            record = db.fetch_customer(customer_id, cfg)
        except (SQLAlchemyError, OSError, RuntimeError) as exc:
            raise HTTPException(503, detail="Base de clientes indisponível.") from exc
        if record is None:
            raise HTTPException(404, detail="Cliente não encontrado.")
        return record

    @app.get("/health")
    def health():
        try:
            database = db.check_database(cfg)
        except (SQLAlchemyError, RuntimeError):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "model_loaded": True,
                    "database": {"status": "unavailable"},
                },
            )
        return {"status": "ok", "model_loaded": True, "database": database}

    @app.get("/model-info")
    def model_info():
        return app.state.champion.info

    @app.post("/predict", response_model=PredictionOut)
    def predict(payload: CustomerIn):
        return score(payload.model_dump())

    @app.get("/customers/{customer_id}")
    def get_customer(customer_id: str):
        return {
            "customer": customer(customer_id),
            "data_source": "postgresql" if cfg.database_url else "csv",
        }

    @app.get("/results/{customer_id}")
    def get_result(customer_id: str):
        record = customer(customer_id)
        return {
            "customer_id": customer_id,
            "customer": record,
            "actual_churn": record.get("Churn"),
            "data_source": "postgresql" if cfg.database_url else "csv",
            **score(record),
        }

    return app


app = create_app()
