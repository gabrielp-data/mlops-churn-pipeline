"""API do app-churn-v1 — só consulta resultados pré-calculados (sem inferência).

Lê ``data/batch_predictions.parquet`` (gerado por ``churn.batch_scoring``).

Uso: ``uv run uvicorn churn.api_lookup:app --port 8000``
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.batch_scoring import OUTPUT_PATH

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(OUTPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode `python -m churn.batch_scoring` antes.")
    _state["batch"] = pd.read_parquet(path).set_index("customer_id")
    yield
    _state.clear()


app = FastAPI(
    title="Churn v1 — consulta em lote",
    description="Lookup de resultados pré-computados pelo @champion (sem inferência ativa).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "records_available": len(_state["batch"]), "source": str(OUTPUT_PATH)}


@app.get("/results/{customer_id}")
def get_batch_result(customer_id: str):
    batch = _state["batch"]
    if customer_id not in batch.index:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' não encontrado na base.")
    return {"customer_id": customer_id, **batch.loc[customer_id].to_dict()}
