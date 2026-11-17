"""Tela do app-churn-v2 — consulta no PostgreSQL + nova inferência em tempo real.

Uso: ``uv run streamlit run src/churn/app_churn_v2.py`` (com churn.api_v2 na :8000)
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="App Churn v2 — two-tier", page_icon="📊", layout="wide")
st.title("📊 Painel de Retenção de Clientes (v2)")
st.caption("Duas camadas: aplicação + modelo numa EC2, PostgreSQL em Docker em outra EC2")


def legenda_modelo(resposta: dict) -> str:
    m = resposta.get("model") or {}
    auc = m.get("roc_auc")
    auc_txt = f" · ROC AUC de teste {auc:.3f}" if auc is not None else ""
    return f"Modelo: `{m.get('model_name')}` v{m.get('version')} {m.get('alias')} ({m.get('run_name')}){auc_txt}"


def erro(res: requests.Response) -> None:
    detail = res.json().get("detail") if res.headers.get("content-type", "").startswith("application/json") else res.text
    msgs = {404: "Cliente não encontrado no banco.", 422: "Dado fora do contrato do modelo.",
            503: "Banco de dados indisponível."}
    st.error(f"{msgs.get(res.status_code, f'Erro {res.status_code}')} {detail}")


# ---------------------------------------------------------------------------
# BLOCO 1 — consulta por código no PostgreSQL (camada de dados, 2ª EC2)
# ---------------------------------------------------------------------------
st.subheader("🔍 1. Consulta de cliente no banco (PostgreSQL)")
c_id, c_btn = st.columns([3, 1])
search_id = c_id.text_input("Código do cliente (customerID)", value="7590-VHVEG").strip()
c_btn.write("")
c_btn.write("")
if c_btn.button("Buscar cliente", use_container_width=True):
    try:
        res = requests.get(f"{API_URL}/results/{search_id}", timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Serviço backend indisponível em {API_URL}.")
        st.stop()
    st.session_state["consulta"] = (search_id, res.status_code, res.json() if res.status_code == 200 else res)

# o resultado fica em session_state para não sumir quando o Bloco 2 é enviado
if "consulta" in st.session_state:
    cid, status, data = st.session_state["consulta"]
    if status != 200:
        erro(data)
    else:
        st.success(f"Cliente **{cid}** encontrado no PostgreSQL e avaliado pelo modelo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Previsão do modelo", "Churn (1)" if data["churn_prediction"] == 1 else "Não churn (0)")
        m2.metric("Probabilidade", f"{data['churn_probability'] * 100:.2f}%")
        m3.metric("Risco", data["risk_level"])
        m4.metric("Churn real (base)", data["actual_churn"])
        st.caption(legenda_modelo(data))
        st.markdown("**Registro retornado pelo banco**")
        st.dataframe([data["customer"]], use_container_width=True)
        with st.expander("Resposta completa da API (JSON)"):
            st.json(data)

st.divider()

# ---------------------------------------------------------------------------
# BLOCO 2 — nova inferência em tempo real (online serving)
# ---------------------------------------------------------------------------
st.subheader("⚡ 2. Nova inferência em tempo real (novo cliente)")
net = ["No", "Yes", "No internet service"]
with st.form("nova_inferencia"):
    a, b, c = st.columns(3)
    a.markdown("**Contrato**")
    tenure = a.slider("Tempo de contrato (meses)", 0, 72, 6)
    contract = a.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    payment = a.selectbox("Método de pagamento", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = a.selectbox("Fatura digital (Paperless)", ["Yes", "No"])
    b.markdown("**Custos e serviços**")
    monthly = b.number_input("Valor mensal ($)", 0.0, 200.0, 85.0, step=1.0)
    total = b.number_input("Valor total acumulado ($ — 0 = automático)", 0.0, 10000.0, 0.0, step=10.0)
    internet = b.selectbox("Serviço de internet", ["Fiber optic", "DSL", "No"])
    tech = b.selectbox("Suporte técnico", net)
    c.markdown("**Segurança e perfil**")
    online_sec = c.selectbox("Segurança online", net)
    online_bak = c.selectbox("Backup online", net)
    device = c.selectbox("Proteção de dispositivo", net)
    senior = c.selectbox("Cliente idoso (SeniorCitizen)", [0, 1])
    enviar = st.form_submit_button("⚡ Fazer chamada no modelo", type="primary", use_container_width=True)

if enviar:
    no_net = internet == "No"
    payload = {
        "SeniorCitizen": senior, "tenure": int(tenure), "Contract": contract,
        "PaymentMethod": payment, "PaperlessBilling": paperless,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total) or None,
        "InternetService": internet,
        "TechSupport": "No internet service" if no_net else tech,
        "OnlineSecurity": "No internet service" if no_net else online_sec,
        "OnlineBackup": "No internet service" if no_net else online_bak,
        "DeviceProtection": "No internet service" if no_net else device,
        "StreamingTV": "No internet service" if no_net else "No",
        "StreamingMovies": "No internet service" if no_net else "No",
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Não foi possível conectar à API em {API_URL}.")
        st.stop()
    if resp.status_code != 200:
        erro(resp)
    else:
        result = resp.json()
        st.success("Inferência executada pelo @champion")
        r1, r2, r3 = st.columns(3)
        r1.metric("Classificação", "Risco de churn (1)" if result["churn_prediction"] == 1 else "Retenção estável (0)")
        r2.metric("Probabilidade", f"{result['churn_probability'] * 100:.2f}%")
        r3.metric("Nível de risco", result["risk_level"])
        st.caption(legenda_modelo(result))
        with st.expander("Resposta completa da API (JSON)"):
            st.json(result)
