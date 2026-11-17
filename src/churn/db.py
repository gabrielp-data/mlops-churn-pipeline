"""Conexão com o PostgreSQL da camada de dados (2ª EC2 — churn-db-host).

As credenciais NUNCA ficam no código. Vêm de variáveis de ambiente, que podem
estar num ``.env`` local (ignorado pelo Git) ou no ``docker run --env-file`` da EC2.

    DB_HOST      IP do banco (privado 172.31.x.x dentro da VPC; público fora dela)
    DB_PORT      porta (padrão 5432)
    DB_NAME      banco (padrão churn_db)
    DB_USER      usuário (padrão postgres)
    DB_PASSWORD  senha (obrigatória, sem padrão)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

CUSTOMER_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
# nomes entre aspas: o Postgres preserva maiúsculas (customerID, SeniorCitizen...)
QUERY_BY_ID = text(
    f'SELECT {", ".join(chr(34) + c + chr(34) for c in CUSTOMER_COLUMNS)} '
    'FROM customers WHERE "customerID" = :cid'
)
COUNT = text("SELECT count(*) FROM customers")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Carrega um .env simples (CHAVE=valor) sem sobrescrever o ambiente."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def db_settings() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "churn_db"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
    }


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = db_settings()
    if not cfg["password"]:
        raise RuntimeError("DB_PASSWORD não definida. Configure o .env (veja .env.example).")
    # pool_pre_ping descarta conexões mortas; connect_timeout evita travar a API
    # se o Security Group bloquear a porta 5432.
    return create_engine(
        URL.create("postgresql+psycopg2", **cfg),
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
