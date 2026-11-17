"""Painel Streamlit da API de churn — consome a FastAPI (não carrega o modelo).

Uso:
    uv run streamlit run src/churn/dashboard.py      # http://localhost:8501
    API_URL=http://localhost:8000 (padrão)
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXEMPLO_ID = "7590-VHVEG"
RISK_COLOR = {"Baixo": "🟢", "Médio": "🟡", "Alto": "🔴"}

st.set_page_config(page_title="Churn · @champion", page_icon="📊", layout="wide")
st.title("📊 Predição de Churn — modelo @champion")
st.caption(f"API: `{API_URL}`")


def call(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=10, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(f"Não foi possível conectar à API em `{API_URL}`. Ela está rodando?")
        st.stop()
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        st.error(f"Erro {resp.status_code}: {detail}")
        st.stop()
    return resp.json()


def show_prediction(data: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidade de churn", f"{data['churn_probability']:.1%}")
    c2.metric("Classe prevista", "Churn" if data["churn"] else "Não churn")
    c3.metric("Risco", f"{RISK_COLOR[data['risk']]} {data['risk']}")
    if data.get("actual_churn") is not None:
        st.caption(
            f"Churn real na base: `{data['actual_churn']}` · origem: `{data['customer_source']}`"
            + (f" · simulado: `{data['simulated_values']}`" if data.get("simulated_values") else "")
        )


tab_cliente, tab_novo, tab_modelo = st.tabs(["Consulta por cliente", "Novo cliente", "Modelo @champion"])

with tab_cliente:
    customer_id = st.text_input("Código do cliente (customerID)", value=EXEMPLO_ID).strip()
    simular = st.checkbox("Simular outros valores numéricos (\"e se\")")
    sim = {}
    if simular:
        s1, s2, s3 = st.columns(3)
        sim["tenure"] = s1.number_input("tenure (meses)", 0, 72, 6)
        sim["MonthlyCharges"] = s2.number_input("MonthlyCharges", 0.0, 200.0, 85.0, step=5.0)
        sim["TotalCharges"] = s3.number_input("TotalCharges", 0.0, 10000.0, 510.0, step=10.0)
    if st.button("Consultar", type="primary", disabled=not customer_id):
        data = call("POST", f"/predict/{customer_id}", json=sim) if simular else call("GET", f"/predict/{customer_id}")
        show_prediction(data)

with tab_novo:
    yn = ["No", "Yes"]
    net = ["No", "Yes", "No internet service"]
    with st.form("novo"):
        a, b, c = st.columns(3)
        payload = {
            "gender": a.selectbox("gender", ["Female", "Male"]),
            "SeniorCitizen": int(b.selectbox("SeniorCitizen", [0, 1])),
            "Partner": c.selectbox("Partner", yn),
            "Dependents": a.selectbox("Dependents", yn),
            "tenure": b.number_input("tenure", 0, 72, 12),
            "PhoneService": c.selectbox("PhoneService", yn, index=1),
            "MultipleLines": a.selectbox("MultipleLines", ["No", "Yes", "No phone service"]),
            "InternetService": b.selectbox("InternetService", ["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": c.selectbox("OnlineSecurity", net),
            "OnlineBackup": a.selectbox("OnlineBackup", net),
            "DeviceProtection": b.selectbox("DeviceProtection", net),
            "TechSupport": c.selectbox("TechSupport", net),
            "StreamingTV": a.selectbox("StreamingTV", net),
            "StreamingMovies": b.selectbox("StreamingMovies", net),
            "Contract": c.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": a.selectbox("PaperlessBilling", yn, index=1),
            "PaymentMethod": b.selectbox(
                "PaymentMethod",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            ),
            "MonthlyCharges": c.number_input("MonthlyCharges", 0.0, 200.0, 55.5),
            "TotalCharges": a.number_input("TotalCharges", 0.0, 10000.0, 666.0),
        }
        if st.form_submit_button("Prever", type="primary"):
            show_prediction(call("POST", "/predict", json=payload))

with tab_modelo:
    info = call("GET", "/model-info")
    m = info.get("metrics", {})
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Versão no Registry", f"v{info.get('version', '?')}")
    k2.metric("ROC AUC (teste)", f"{m['roc_auc']:.3f}" if "roc_auc" in m else "—")
    k3.metric("Recall", f"{m['recall']:.3f}" if "recall" in m else "—")
    k4.metric("Acurácia", f"{m['accuracy']:.3f}" if "accuracy" in m else "—")
    st.subheader("Linhagem")
    st.markdown(
        f"- **Run:** `{info.get('run_name')}` (`{info.get('run_id')}`)\n"
        f"- **Dado (DVC md5):** `{info.get('data_md5')}`\n"
        f"- **Código (git):** `{info.get('git_commit')}`\n"
        f"- **Origem do modelo:** `{info.get('source')}`"
    )
    st.subheader("Hiperparâmetros")
    st.json(info.get("hyperparameters", {}))
