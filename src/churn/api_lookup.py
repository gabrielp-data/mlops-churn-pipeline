"""API v1: consulta scores já calculados, sem carregar ou executar o modelo."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.config import settings


def create_lookup_app(batch_path: Path = settings.batch_path):
    @asynccontextmanager
    async def lifespan(app):
        frame = pd.read_parquet(batch_path)
        if "customerID" not in frame or frame["customerID"].duplicated().any():
            raise ValueError("Lote precisa ter customerID único para consulta.")
        app.state.results = frame.set_index("customerID")
        app.state.info = json.loads(
            batch_path.with_suffix(batch_path.suffix + ".json").read_text(
                encoding="utf-8"
            )
        )
        yield

    app = FastAPI(title="Churn · Consulta em lote", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": "batch_lookup",
            "records": len(app.state.results),
        }

    @app.get("/model-info")
    def model_info():
        return app.state.info

    @app.get("/results/{customer_id}")
    def result(customer_id: str):
        if customer_id not in app.state.results.index:
            raise HTTPException(404, detail="Cliente não encontrado no lote.")
        row = app.state.results.loc[customer_id]
        return {
            "customer_id": customer_id,
            "churn_prediction": int(row["churn_pred"]),
            "churn_probability": float(row["churn_probability"]),
            "risk_level": row["risk_level"],
            "model_version": str(row["model_version"]),
            "run_id": row["run_id"],
            "data_source": "batch",
        }

    return app


app = create_lookup_app()
