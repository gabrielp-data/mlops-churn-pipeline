"""Pontuação em lote; o CSV pode não ter Churn nem customerID."""

import argparse
import json
from pathlib import Path

import pandas as pd

from churn.config import settings
from churn.inference import ChampionModel, risk_level


def predict(
    input_path: str, output_path: str | None = None, model_dir: Path | None = None
):
    raw = pd.read_csv(input_path)
    champion = ChampionModel(model_dir or settings.champion_dir)
    probabilities = champion.predict_proba(raw)
    result = (
        raw[["customerID"]].copy()
        if "customerID" in raw
        else pd.DataFrame(index=raw.index)
    )
    result["churn_pred"] = (probabilities >= 0.5).astype(int)
    result["churn_probability"] = probabilities
    result["risk_level"] = [risk_level(p) for p in probabilities]
    result["model_version"] = champion.info["version"]
    result["run_id"] = champion.info["run_id"]
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".parquet":
            result.to_parquet(destination, index=False)
        else:
            result.to_csv(destination, index=False)
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(champion.info, indent=2),
            encoding="utf-8",
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.data_path))
    parser.add_argument("--output", default=str(settings.batch_path))
    parser.add_argument("--model-dir", type=Path, default=settings.champion_dir)
    args = parser.parse_args()
    result = predict(args.input, args.output, args.model_dir)
    print(f"{len(result)} registros gravados em {args.output}")


if __name__ == "__main__":
    main()
