import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Mini-amostra com o schema do Telco, incluindo um TotalCharges em branco."""
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4"],
            "gender": ["Male", "Female", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [12, 0, 34, 5],
            "PhoneService": ["Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"],
            "MonthlyCharges": [55.5, 89.1, 20.0, 45.3],
            "TotalCharges": ["666.0", " ", "680.0", "226.5"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


def make_synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético no formato do Telco, com sinal aprendível."""
    import numpy as np

    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(20, 110, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    churn = np.where((contract == "Month-to-month") & (tenure < 24), "Yes", "No")
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": yes_no(),
        "PaymentMethod": "Electronic check", "MonthlyCharges": monthly,
        "TotalCharges": (monthly * tenure).round(2).astype(str), "Churn": churn,
    })


@pytest.fixture()
def mlflow_env(tmp_path, monkeypatch):
    """CSV sintético + MLflow (tracking e registry) isolados em tmp_path."""
    import mlflow

    from churn.config import Settings

    data_path = tmp_path / "churn.csv"
    make_synthetic().to_csv(data_path, index=False)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield Settings(
        data_path=data_path,
        artifact_path=tmp_path / "artifacts" / "churn_model.pkl",
        champion_dir=tmp_path / "artifacts" / "champion",
        n_estimators=20,
    )
    mlflow.set_tracking_uri(None)
