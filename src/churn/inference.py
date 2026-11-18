"""Uma única implementação de scoring para API, PostgreSQL e lote."""

import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from churn.config import settings


def risk_level(probability: float) -> str:
    return (
        "Alto" if probability >= 0.6 else ("Médio" if probability >= 0.3 else "Baixo")
    )


class ChampionModel:
    def __init__(self, model_dir: Path = settings.champion_dir):
        model_dir = Path(model_dir)
        if not (model_dir / "MLmodel").is_file():
            raise FileNotFoundError(
                f"Campeão ausente em {model_dir}. Execute treino, registry e export_champion."
            )
        self.info = json.loads(
            (model_dir / "champion_info.json").read_text(encoding="utf-8")
        )
        self.model = mlflow.sklearn.load_model(str(model_dir))

    def predict_proba(self, records: pd.DataFrame):
        return self.model.predict_proba(records)[:, 1]

    def score_one(self, record: dict):
        probability = float(self.predict_proba(pd.DataFrame([record]))[0])
        churn = probability >= 0.5
        return {
            "churn": churn,
            "churn_prediction": int(churn),
            "churn_probability": probability,
            "label": "Yes" if churn else "No",
            "risk_level": risk_level(probability),
            "model_version": self.info["version"],
            "run_id": self.info["run_id"],
        }
