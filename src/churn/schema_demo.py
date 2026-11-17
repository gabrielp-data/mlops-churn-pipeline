"""Quebra proposital do contrato de dados — Encontro 4.

Adultera 3 linhas reais da base (id duplicado, ``tenure`` fora de 0–72,
``MonthlyCharges`` negativo e ``Contract`` desconhecido) e mostra o
``ChurnSchema`` (pandera) barrando todas de uma vez (``lazy=True``).

Uso:
    uv run python -m churn.schema_demo
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from churn.config import settings
from churn.data import coerce_total_charges, drop_identifier
from churn.schema import ChurnSchema


def break_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve as 3 primeiras linhas com violações de propósito."""
    bad = df.head(3).copy().reset_index(drop=True)
    bad.loc[1, settings.id_column] = bad.loc[0, settings.id_column]  # id duplicado
    bad.loc[0, "tenure"] = 999                                       # fora de 0–72
    bad.loc[1, "MonthlyCharges"] = -15.0                              # cobrança negativa
    bad.loc[2, "Contract"] = "Vitalicio"                              # categoria inexistente
    return bad


def failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Valida ``df`` e retorna a tabela de falhas do pandera (vazia se passar)."""
    try:
        ChurnSchema.validate(df, lazy=True)
    except pa_errors.SchemaErrors as err:
        return err.failure_cases[["column", "check", "failure_case"]]
    return pd.DataFrame(columns=["column", "check", "failure_case"])


def main(path: Path = settings.data_path) -> None:
    bad = break_contract(pd.read_csv(path))
    # o id é checado à parte (o schema valida o dado já sem o identificador)
    dup = bad[settings.id_column][bad[settings.id_column].duplicated(keep=False)]
    if not dup.empty:
        print(f"customerID duplicado: {sorted(set(dup))}")
    cases = failure_cases(coerce_total_charges(drop_identifier(bad, settings.id_column)))
    print(cases.to_string(index=False))


if __name__ == "__main__":
    main()
