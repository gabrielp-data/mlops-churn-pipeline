"""Consumo do modelo de produção — Encontro 7.

Carrega ``models:/churn@champion`` pelo flavor ``pyfunc`` (agnóstico de
framework) e pontua 3 clientes reais da base, como faria qualquer consumidor.

Uso:
    uv run python -m churn.test_champion
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import settings
from churn.predict import to_features
from churn.registry import load_champion


def run(path: Path = settings.data_path, n: int = 3) -> pd.Series:
    print("1. Carregando o modelo de produção (@champion)...")
    model = load_champion()
    print("2. Preparando clientes reais da base (dado cru, o pipeline faz o resto)...")
    raw = pd.read_csv(path).head(n)
    print("3. Executando inferência...")
    preds = pd.Series(model.predict(to_features(raw)), index=raw[settings.id_column])
    print("\nPrevisão de churn (0 = Não, 1 = Sim):")
    print(preds.to_string())
    return preds


if __name__ == "__main__":
    run()
