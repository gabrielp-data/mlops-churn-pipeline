"""Configuração central do projeto (Pydantic).

Todo caminho, hiperparâmetro e constante mágica do script original vive aqui —
nunca hardcoded no meio da lógica. Valores podem ser sobrescritos por variáveis
de ambiente com prefixo ``CHURN_`` ou por um arquivo ``.env``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros do pipeline de churn."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        protected_namespaces=(),
        extra="ignore",
        env_ignore_empty=True,
        populate_by_name=True,
    )

    # dados
    data_path: Path = Path("data/churn.csv")
    id_column: str = "customerID"
    target: str = "Churn"
    positive_label: str = "Yes"

    # split / modelo
    test_size: float = Field(default=0.2, gt=0, lt=0.5)
    validation_size: float = Field(default=0.25, gt=0, lt=0.5)
    random_state: int = 42
    n_estimators: int = Field(default=200, ge=1)
    max_depth: int | None = None

    # saída
    tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        validation_alias=AliasChoices("CHURN_TRACKING_URI", "MLFLOW_TRACKING_URI"),
    )
    experiment_name: str = "churn"
    model_name: str = "churn"
    candidates_path: Path = Path("artifacts/candidates.json")
    promotion_path: Path = Path("artifacts/champion.json")
    champion_dir: Path = Path("models/champion")
    batch_path: Path = Path("data/batch_predictions.parquet")
    database_url: SecretStr | None = None
    api_url: str = "http://127.0.0.1:8000"


settings = Settings()
