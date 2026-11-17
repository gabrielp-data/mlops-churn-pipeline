"""Tela do app-churn-v1 — consulta de scores pré-calculados em lote.

Uso: ``uv run streamlit run src/churn/app_churn_v1.py`` (com churn.api_lookup na :8000)
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="App Churn v1 — consulta em lote", page_icon="📋", layout="wide")
st.title("📋 Consulta de Scores de Churn (v1)")
st.caption("Container único na AWS (FastAPI + Streamlit via Supervisord) · resultados pré-calculados pelo @champion")

st.sidebar.header("Consulta de resultados")
customer_id = st.sidebar.text_input("Código do cliente (customerID)", value="7590-VHVEG").strip()

if st.sidebar.button("Buscar resultado", type="primary"):
    if not customer_id:
        st.error("Informe um código de cliente válido.")
        st.stop()
    try:
        res = requests.get(f"{API_URL}/results/{customer_id}", timeout=5)
    except requests.exceptions.ConnectionError:
        st.error(f"Falha ao comunicar com o backend em {API_URL}.")
        st.stop()
    if res.status_code == 404:
        st.warning(f"O cliente '{customer_id}' não consta na base consolidada.")
    elif res.status_code != 200:
        st.error(f"Erro {res.status_code}: {res.text}")
    else:
        data = res.json()
        st.success(f"Resultado localizado para o cliente **{customer_id}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classificação", "Churn (1)" if data["churn_prediction"] == 1 else "Não churn (0)")
        c2.metric("Probabilidade de cancelamento", f"{data['churn_probability'] * 100:.2f}%")
        c3.metric("Nível de risco", data["risk_level"])
        c4.metric("Churn real (base)", data.get("actual_churn", "—"))
        st.caption(f"Modelo v{data.get('model_version')} · processado em {data.get('processed_at')}")
        st.divider()
        st.subheader("Detalhes da consulta em lote")
        st.json(data)
