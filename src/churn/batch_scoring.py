"""Pontuação em lote (app-churn-v1): aplica o @champion a todos os clientes.

Gera ``data/batch_predictions.parquet``, consultado pela API da v1
(``churn.api_lookup``), que não faz inferência: só lê os resultados prontos.
No build da imagem da AWS, este passo roda dentro do ``docker build``.

Uso: ``uv run python -m churn.batch_scoring``
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import settings
from churn.data import coerce_total_charges
from churn.scoring import THRESHOLD, ChampionModel, risk_level

OUTPUT_PATH = Path("data/batch_predictions.parquet")


def score_all(data_path: Path = settings.data_path, output: Path = OUTPUT_PATH,
              champion: ChampionModel | None = None) -> pd.DataFrame:
    champion = champion or ChampionModel()
    df = coerce_total_charges(pd.read_csv(data_path))
    probs = champion.predict_proba(df)

    result = pd.DataFrame({
        "customer_id": df[settings.id_column],
        "churn_prediction": (probs >= THRESHOLD).astype(int),
        "churn_probability": np.round(probs, 4),
        "risk_level": [risk_level(p) for p in probs],
        "actual_churn": df[settings.target],
        "contract_type": df["Contract"],
        "monthly_charges": df["MonthlyCharges"],
        "total_charges": df["TotalCharges"],
        "tenure": df["tenure"].astype(int),
        "model_version": champion.info.get("version"),
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def main() -> None:
    champion = ChampionModel()
    print(f"--> Modelo: {champion.info.get('model_name')} v{champion.info.get('version')} @champion "
          f"({champion.info.get('run_name')})")
    result = score_all(champion=champion)
    print(f"--> {len(result)} predições salvas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
