"""Scoring dos apps da AWS com o @champion exportado (``artifacts/champion/``).

Usado pela pontuação em lote (v1), pela API de consulta (v1) e pela API two-tier
(v2). O modelo é o pipeline completo do treino (features + encoding + RF), então
um cliente vindo do CSV, do PostgreSQL ou do formulário é transformado
**exatamente** como no treino.

Antes de pontuar, o registro é checado contra o mesmo contrato do treino
(``ChurnSchema``): categoria desconhecida (ex.: ``Contract = "Vitalicio"``) vira
``ValueError`` — a API devolve 422 em vez de chutar.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import pandera.errors

from churn.config import settings
from churn.data import coerce_total_charges
from churn.registry import INFO_FILE
from churn.schema import ChurnSchema

THRESHOLD = 0.5
_INPUT_SCHEMA = ChurnSchema.to_schema().remove_columns([settings.target])


def risk_level(prob: float) -> str:
    return "Alto" if prob >= 0.6 else ("Médio" if prob >= 0.3 else "Baixo")


def to_model_input(records: pd.DataFrame) -> pd.DataFrame:
    """Registros crus → entrada do modelo: sem id/alvo, TotalCharges numérico, contrato checado."""
    df = records.drop(columns=[settings.id_column, settings.target], errors="ignore")
    df = coerce_total_charges(df)
    try:
        return _INPUT_SCHEMA.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as exc:
        fails = exc.failure_cases[["column", "failure_case"]].drop_duplicates()
        detail = "; ".join(f"{c}={v!r}" for c, v in fails.itertuples(index=False))
        raise ValueError(f"Dado fora do contrato do modelo: {detail}") from exc


class ChampionModel:
    """O @champion exportado + seus metadados (``champion_info.json``)."""

    def __init__(self, model_dir: Path = settings.champion_dir):
        model_dir = Path(model_dir)
        if not (model_dir / "MLmodel").exists():
            raise FileNotFoundError(
                f"Modelo @champion não encontrado em {model_dir}. "
                "Rode `uv run python -m churn.registry --export` primeiro."
            )
        # flavor sklearn: dá acesso ao predict_proba (o pyfunc só devolve a classe)
        self.model = mlflow.sklearn.load_model(str(model_dir))
        info_path = model_dir / INFO_FILE
        self.info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    def predict_proba(self, records: pd.DataFrame):
        return self.model.predict_proba(to_model_input(records))[:, 1]

    def score_one(self, record: dict) -> dict:
        prob = float(self.predict_proba(pd.DataFrame([record]))[0])
        return {
            "churn_prediction": int(prob >= THRESHOLD),
            "churn_probability": round(prob, 4),
            "risk_level": risk_level(prob),
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def model_summary(self) -> dict:
        metrics = self.info.get("metrics", {})
        return {
            **{k: self.info.get(k) for k in ("model_name", "alias", "version", "run_name")},
            "roc_auc": metrics.get("roc_auc"),
            "accuracy": metrics.get("accuracy"),
        }
