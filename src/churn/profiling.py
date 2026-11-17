"""Profiling do dataset — Encontro 4.

Retrato do dado antes de qualquer feature: tipos, estatísticas, nulos e
distribuição do alvo. Se o ``ydata-profiling`` estiver instalado, também gera
um relatório HTML (``churn_profile_report.html``, ignorado pelo Git).

Uso:
    uv run python -m churn.profiling
    uv run python -m churn.profiling --input data/churn.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from churn.config import settings

REPORT_FILE = Path("churn_profile_report.html")


def profile(df: pd.DataFrame, target: str = settings.target) -> dict:
    """Resumo do dataset em um dicionário (usado pelo CLI e pelos testes)."""
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nulls = df.isnull().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicated_ids": int(df[settings.id_column].duplicated().sum()),
        "nulls": {k: int(v) for k, v in nulls[nulls > 0].items()},
        "total_charges_blank": int(total_charges.isna().sum()),
        "target_distribution": (df[target].value_counts(normalize=True) * 100).round(2).to_dict(),
    }


def run_profiling(path: Path = settings.data_path) -> dict:
    """Imprime o retrato do dataset e tenta gerar o relatório HTML."""
    df = pd.read_csv(path)
    print("=== PROFILING DO DATASET ===")
    print("\n1. Estrutura e tipos:")
    df.info()
    print("\n2. Resumo estatístico (numéricas):")
    print(df.describe().T)

    summary = profile(df)
    print("\n3. Nulos por coluna:", summary["nulls"] or "nenhum")
    print(f"   TotalCharges em branco (texto no CSV): {summary['total_charges_blank']}")
    print(f"   customerID duplicados: {summary['duplicated_ids']}")
    print(f"\n4. Distribuição do alvo ({settings.target}, %):", summary["target_distribution"])

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("\nPara o relatório HTML completo: uv pip install ydata-profiling")
    else:
        ProfileReport(df, title="Relatório de Profiling - Churn").to_file(REPORT_FILE)
        print(f"\nRelatório HTML gerado: {REPORT_FILE}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profiling do dataset de churn")
    parser.add_argument("--input", type=Path, default=settings.data_path)
    run_profiling(parser.parse_args().input)


if __name__ == "__main__":
    main()
