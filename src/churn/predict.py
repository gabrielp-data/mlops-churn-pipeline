"""Inferência em lote usando o modelo @champion do Registry — Encontro 8.

O @champion é um pipeline completo (features + encoding + modelo), então recebe
o dado cru: aqui só tiramos o id, convertemos TotalCharges e removemos o alvo.

Uso:
    uv run python -m churn.predict --input data/churn.csv
    uv run python -m churn.predict --input data/churn.csv --output predictions.csv
"""
from __future__ import annotations

import argparse

import mlflow.sklearn
import pandas as pd

from churn.config import settings
from churn.data import coerce_total_charges, drop_identifier
from churn.registry import MODEL_NAME


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    """CSV cru → colunas de entrada do modelo (sem id e sem alvo)."""
    df = coerce_total_charges(drop_identifier(raw, settings.id_column))
    return df.drop(columns=[settings.target], errors="ignore")


def predict(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Carrega @champion e pontua o CSV de entrada."""
    raw = pd.read_csv(input_path)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    proba = model.predict_proba(to_features(raw))[:, 1]

    result = raw[[settings.id_column]].copy()
    result["churn_probability"] = proba.round(4)
    result["churn_pred"] = (proba >= 0.5).astype(int)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Predições salvas em {output_path}")
    else:
        print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score com o modelo @champion")
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", default=None, help="CSV de saída (opcional)")
    args = parser.parse_args()
    predict(args.input, args.output)


if __name__ == "__main__":
    main()
