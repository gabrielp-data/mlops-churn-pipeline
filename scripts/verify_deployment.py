"""Verifica um deploy real por HTTP e compara os resultados com o lote local."""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from churn.features import prepare_features


def verify(
    api_url, panel_url, database_down=False, batch_url=None, batch_panel_url=None
):
    data = pd.read_csv("data/telco-public.csv")
    expected = pd.read_parquet("data/batch_predictions.parquet").set_index("customerID")
    info = json.loads(Path("models/champion/champion_info.json").read_text())
    sample = data.iloc[[0, 1, 100]]
    sample = pd.concat([sample, data.loc[data.tenure == 0].head(1)])
    with requests.Session() as client:
        health = client.get(f"{api_url}/health", timeout=15)
        assert health.status_code == (503 if database_down else 200), health.text
        if not database_down:
            assert health.json()["database"] == {
                "status": "connected",
                "source": "postgresql",
            }
        panel = client.get(f"{panel_url}/_stcore/health", timeout=15)
        assert panel.status_code == 200, panel.text
        actual_info = client.get(f"{api_url}/model-info", timeout=15)
        actual_info.raise_for_status()
        assert actual_info.json()["run_id"] == info["run_id"]
        for _, row in sample.iterrows():
            customer_id = row.customerID
            payload = json.loads(
                prepare_features(pd.DataFrame([row])).to_json(orient="records")
            )[0]
            response = client.post(f"{api_url}/predict", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            assert result["run_id"] == info["run_id"]
            assert (
                abs(
                    result["churn_probability"]
                    - expected.loc[customer_id, "churn_probability"]
                )
                < 1e-12
            )
            lookup = client.get(f"{api_url}/results/{customer_id}", timeout=15)
            if database_down:
                assert lookup.status_code == 503, lookup.text
            else:
                lookup.raise_for_status()
                assert lookup.json()["data_source"] == "postgresql"
                assert lookup.json()["churn_probability"] == result["churn_probability"]
            if batch_url:
                batch = client.get(f"{batch_url}/results/{customer_id}", timeout=15)
                batch.raise_for_status()
                assert batch.json()["churn_probability"] == result["churn_probability"]
                assert batch.json()["run_id"] == info["run_id"]
        if not database_down:
            assert (
                client.get(f"{api_url}/results/missing", timeout=15).status_code == 404
            )
        payload["tenure"] = -1
        assert (
            client.post(f"{api_url}/predict", json=payload, timeout=15).status_code
            == 422
        )
        if batch_url:
            assert (
                client.post(
                    f"{batch_url}/predict", json=payload, timeout=15
                ).status_code
                == 404
            )
            assert (
                client.get(f"{batch_panel_url}/_stcore/health", timeout=15).status_code
                == 200
            )
    print("Verificação HTTP e equivalência de probabilidades: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--panel-url", default="http://127.0.0.1:8501")
    parser.add_argument("--database-down", action="store_true")
    parser.add_argument("--batch-url")
    parser.add_argument("--batch-panel-url")
    args = parser.parse_args()
    verify(
        args.api_url,
        args.panel_url,
        args.database_down,
        args.batch_url,
        args.batch_panel_url,
    )
