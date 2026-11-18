"""Engenharia de features.

Transformações *stateless* são funções puras. O encoding de categóricas guarda
estado (as categorias vistas) — por isso vive numa classe com fit/transform,
que aprende SOMENTE no treino e evita vazamento de dados.

Nota de projeto: não há normalização de escala aqui de propósito. Random Forest
não é sensível à escala das features; a normalização por números mágicos do
script original era desnecessária.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from churn.schema import FeatureSchema

FEATURE_COLUMNS = list(FeatureSchema.to_schema().columns)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida entradas brutas sem exigir Churn e fixa a ordem das colunas."""
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    clean = frame[FEATURE_COLUMNS].copy()
    clean["TotalCharges"] = (
        pd.to_numeric(
            clean["TotalCharges"].replace(r"^\s*$", 0, regex=True), errors="raise"
        )
        .fillna(0.0)
        .astype(float)
    )
    return FeatureSchema.validate(clean, lazy=True)[FEATURE_COLUMNS]


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Transformação sem estado empacotada junto com o modelo."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_derived_features(prepare_features(X))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas (pura, row-wise — segura antes do split)."""
    out = df.copy()
    out["gasto_por_mes"] = out["TotalCharges"] / (out["tenure"] + 1)
    return out


def split_features_target(
    df: pd.DataFrame, target: str, positive_label: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) de y (alvo binário 0/1)."""
    features = df.drop(columns=[target])
    labels = (df[target] == positive_label).astype(int)
    return features, labels


class CategoricalEncoder(TransformerMixin, BaseEstimator):
    """Codifica colunas categóricas em inteiros.

    Aprende as categorias no ``fit`` (apenas com o treino) e as aplica no
    ``transform`` de treino e teste — o encapsulamento é o que impede o
    vazamento que existia no script monolítico.
    """

    def __init__(self) -> None:
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._columns: list[str] = []

    def fit(self, features: pd.DataFrame, y=None) -> CategoricalEncoder:
        self._columns = features.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
        self._encoder.fit(features[self._columns])
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        out[self._columns] = self._encoder.transform(out[self._columns])
        return out

    def fit_transform(
        self, features: pd.DataFrame, y=None, **fit_params
    ) -> pd.DataFrame:
        return self.fit(features).transform(features)
