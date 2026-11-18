"""Acesso parametrizado a clientes, com PostgreSQL e SQLite para testes."""

import argparse
import json
from functools import lru_cache

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from churn.config import Settings, settings
from churn.features import FEATURE_COLUMNS, prepare_features

metadata = MetaData()
customers = Table(
    "customers",
    metadata,
    Column("customerID", String, primary_key=True),
    *[
        Column(
            name,
            Integer
            if name in ("SeniorCitizen", "tenure")
            else Float
            if name in ("MonthlyCharges", "TotalCharges")
            else String,
            nullable=False,
        )
        for name in FEATURE_COLUMNS
    ],
    Column("Churn", String, nullable=True),
)


@lru_cache(maxsize=4)
def _engine(url: str) -> Engine:
    kwargs = {"connect_timeout": 5} if url.startswith("postgresql") else {}
    return create_engine(
        url, pool_pre_ping=True, connect_args=kwargs, hide_parameters=True
    )


def get_engine(cfg: Settings = settings) -> Engine:
    if cfg.database_url is None:
        raise RuntimeError("CHURN_DATABASE_URL não configurada.")
    return _engine(cfg.database_url.get_secret_value())


def import_customers(path, engine: Engine):
    raw = pd.read_csv(path)
    if (
        "customerID" not in raw
        or raw["customerID"].isna().any()
        or raw["customerID"].duplicated().any()
    ):
        raise ValueError("O CSV precisa de customerID preenchido e único.")
    clean = prepare_features(raw)
    clean.insert(0, "customerID", raw["customerID"].astype(str))
    if "Churn" in raw:
        if not raw["Churn"].isin(["Yes", "No"]).all():
            raise ValueError("Churn precisa ser Yes ou No.")
        clean["Churn"] = raw["Churn"]
    else:
        clean["Churn"] = None
    metadata.create_all(engine)
    records = json.loads(clean.to_json(orient="records"))
    # Transação única: IDs duplicados no banco cancelam toda a carga.
    with engine.begin() as conn:
        conn.execute(insert(customers), records)
    return len(records)


def fetch_customer(customer_id: str, cfg: Settings = settings):
    if cfg.database_url is not None:
        with get_engine(cfg).connect() as conn:
            row = (
                conn.execute(
                    select(customers).where(customers.c.customerID == customer_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None
    # Modo local de demonstração; em AWS use CHURN_DATABASE_URL.
    raw = pd.read_csv(cfg.data_path, dtype={"customerID": str})
    matches = raw.loc[raw["customerID"] == customer_id]
    if matches.empty:
        return None
    record = json.loads(matches.iloc[:1].to_json(orient="records"))[0]
    record["TotalCharges"] = float(prepare_features(matches).iloc[0]["TotalCharges"])
    return record


def check_database(cfg: Settings):
    if cfg.database_url is None:
        return {"status": "not_configured", "source": "csv"}
    with get_engine(cfg).connect() as conn:
        conn.execute(text("SELECT 1"))
        conn.execute(select(customers.c.customerID).limit(1))
    return {"status": "connected", "source": "postgresql"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Importa clientes em uma tabela vazia."
    )
    parser.add_argument("--input", default=str(settings.data_path))
    args = parser.parse_args()
    print(f"{import_customers(args.input, get_engine())} clientes importados.")
