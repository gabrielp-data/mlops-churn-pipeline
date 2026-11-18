import importlib
import json
import os
from pathlib import Path
from uuid import uuid4

import cloudpickle
import pandas as pd
import pytest
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "streamlit_cloud" / "champion"


@pytest.fixture
def customer():
    # Public IBM example also displayed in the deployed app.
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No",
    }


@pytest.fixture(scope="session")
def model_bundle():
    with (CHAMPION / "model.pkl").open("rb") as handle:
        model = cloudpickle.load(handle)
    contract = json.loads((CHAMPION / "feature_contract.json").read_text())
    return model, contract


@pytest.fixture
def database(tmp_path, customer):
    url = os.environ.get("TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    engine = create_engine(url)
    # Each PostgreSQL test gets a new schema; never touch existing app tables.
    schema = None
    if engine.dialect.name == "postgresql":
        schema = "test_" + uuid4().hex
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine.dispose()
        url = engine.url.update_query_dict(
            {"options": f"-csearch_path={schema}"}
        ).render_as_string(hide_password=False)
        engine = create_engine(url)
    pd.DataFrame([customer]).to_sql("customers", engine, index=False)
    yield engine, url
    engine.dispose()
    if schema:
        admin = create_engine(os.environ["TEST_DATABASE_URL"])
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def dashboard(database):
    st.cache_resource.clear()
    app = AppTest.from_file(
        str(ROOT / "streamlit_cloud" / "app.py"), default_timeout=30
    )
    app.secrets["DB_URL"] = database[1]
    app.run()
    assert not app.exception
    yield app
    st.cache_resource.clear()


@pytest.fixture
def api(database, monkeypatch):
    from src import inference

    # Exercise the actual API with the exact artifact shipped to Streamlit.
    champion = inference.ChampionModel(CHAMPION)
    monkeypatch.setattr(inference, "ChampionModel", lambda: champion)
    module = importlib.import_module("src.api_v2")
    monkeypatch.setattr(module, "champion", champion)
    monkeypatch.setattr(module, "get_engine", lambda: database[0])
    return module
