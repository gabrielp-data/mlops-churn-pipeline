"""app-churn-v2 no Streamlit Community Cloud (duas camadas: app + PostgreSQL gerenciado).

O Streamlit Cloud roda um unico processo, entao a FastAPI do src/api_v2.py nao
sobe aqui: o painel aplica o @champion diretamente, com o MESMO contrato de
features do treino (src/features.py). Os dados dos clientes ficam num PostgreSQL
separado (ex.: Neon), cuja URL vem de st.secrets["DB_URL"].

Blocos:
  1. Consulta por codigo no PostgreSQL + score do modelo
  2. Nova inferencia em tempo real (gravada na tabela predictions)
  3. Historico das inferencias gravadas no banco
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import features_for_inference

CHAMPION_DIR = Path(__file__).resolve().parent / "champion"
DATA_URL = ("https://raw.githubusercontent.com/IBM/watsonx-ai-samples/"
            "e7345a897d3a440e0e8ee465150fe895bd5b29cb/cpd4.5/data/customer_churn/"
            "WA_FnUseC_TelcoCustomerChurn.csv")
CUSTOMER_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

st.set_page_config(page_title="App Churn v2 - Streamlit Cloud", layout="wide")


# --------------------------------------------------------------------------
# Modelo @champion (exportado do MLflow Model Registry para streamlit_cloud/champion)
# --------------------------------------------------------------------------
@st.cache_resource
def load_champion():
    with open(CHAMPION_DIR / "model.pkl", "rb") as f:
        model = cloudpickle.load(f)
    contract = json.loads((CHAMPION_DIR / "feature_contract.json").read_text(encoding="utf-8"))
    info = json.loads((CHAMPION_DIR / "champion_info.json").read_text(encoding="utf-8"))
    return model, contract, info


def risk_level(prob: float) -> str:
    return "Alto" if prob >= 0.6 else ("Medio" if prob >= 0.3 else "Baixo")


def score(record: dict) -> dict:
    model, contract, _ = load_champion()
    entrada = {k: v for k, v in record.items() if k not in ("customerID", "Churn")}
    prob = float(model.predict_proba(features_for_inference(pd.DataFrame([entrada]), contract))[0, 1])
    return {"churn_prediction": int(prob >= 0.5), "churn_probability": round(prob, 4),
            "risk_level": risk_level(prob)}


# --------------------------------------------------------------------------
# Camada de dados: PostgreSQL separado
# --------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    url = st.secrets["DB_URL"]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return create_engine(url, pool_pre_ping=True)


@st.cache_resource
def ensure_tables() -> int:
    """Na primeira execucao carrega o churn.csv bruto na tabela customers (CSV -> postgres)."""
    engine = get_engine()
    tables = inspect(engine).get_table_names()
    if "customers" not in tables:
        df = pd.read_csv(DATA_URL, dtype={"TotalCharges": str})[CUSTOMER_COLUMNS]
        df.to_sql("customers", engine, index=False, chunksize=1000, method="multi")
    if "predictions" not in tables:
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE predictions (created_at VARCHAR(30), payload TEXT, "
                "churn_prediction INTEGER, churn_probability DOUBLE PRECISION, "
                "risk_level VARCHAR(10), model_version VARCHAR(40))"))
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM customers")).scalar_one()


def fetch_customer(customer_id: str):
    cols = ", ".join(f'"{c}"' for c in CUSTOMER_COLUMNS)
    with get_engine().connect() as conn:
        row = conn.execute(text(f'SELECT {cols} FROM customers WHERE "customerID" = :cid'),
                           {"cid": customer_id}).mappings().first()
    if row is None:
        return None
    record = dict(row)
    record["MonthlyCharges"] = float(record["MonthlyCharges"])
    return record


def save_prediction(payload: dict, result: dict, version: str):
    with get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO predictions VALUES (:t, :p, :c, :pr, :r, :v)"),
            {"t": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
             "p": json.dumps(payload, ensure_ascii=False), "c": result["churn_prediction"],
             "pr": result["churn_probability"], "r": result["risk_level"], "v": version})


# --------------------------------------------------------------------------
# Tela
# --------------------------------------------------------------------------
_, _, info = load_champion()
modelo_txt = (f"Modelo: `{info['model_name']}` v{info['version']} {info['alias']} "
              f"({info['run_name']}) · acurácia de teste {info['test_accuracy'] * 100:.2f}%")

st.title("📊 Painel de Retenção de Clientes (v2)")
st.caption("Duas camadas: aplicação + modelo no Streamlit Community Cloud, PostgreSQL gerenciado em outro servidor")
st.caption(modelo_txt)

try:
    total = ensure_tables()
    st.success(f"PostgreSQL conectado · {total} clientes na tabela `customers`")
    db_ok = True
except Exception as exc:  # noqa: BLE001 - a inferencia online continua sem banco
    st.error(f"Banco PostgreSQL indisponível: {exc.__class__.__name__}. Configure DB_URL em Secrets.")
    db_ok = False

# BLOCO 1 -------------------------------------------------------------------
st.subheader("🔍 1. Consulta de Cliente no Banco (PostgreSQL)")
c1, c2 = st.columns([3, 1])
search_id = c1.text_input("Código do Cliente (customerID)", value="7590-VHVEG", help="Ex: 7590-VHVEG, 5575-GNVDE")
c2.write("")
c2.write("")
if c2.button("Buscar Cliente", width="stretch", disabled=not db_ok):
    record = fetch_customer(search_id.strip())
    if record is None:
        st.warning(f"Cliente '{search_id}' não encontrado no banco.")
    else:
        try:
            r = score(record)
            st.success(f"Cliente **{search_id}** encontrado no PostgreSQL e avaliado pelo modelo")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Previsão do Modelo", "Churn (1)" if r["churn_prediction"] else "Não Churn (0)")
            m2.metric("Probabilidade", f"{r['churn_probability'] * 100:.2f}%")
            m3.metric("Risco", r["risk_level"])
            m4.metric("Churn real (base)", record["Churn"])
            st.markdown("**Registro retornado pelo banco**")
            st.dataframe(pd.DataFrame([record]), width="stretch")
        except ValueError as exc:
            st.error(f"Dado fora do contrato do modelo: {exc}")

st.divider()

# BLOCO 2 -------------------------------------------------------------------
st.subheader("⚡ 2. Nova Inferência em Tempo Real (Simulação de Novo Cliente)")
with st.form("form_nova_inferencia"):
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Parâmetros Contratuais**")
        tenure = st.slider("Tempo de Contrato (Meses)", 1, 72, 6)
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Método de Pagamento", ["Electronic check", "Mailed check",
                                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Fatura Digital (Paperless)", ["Yes", "No"])
    with b:
        st.markdown("**Custos e Serviços Digitais**")
        monthly = st.number_input("Valor Mensal ($)", 15.0, 150.0, 85.0, 1.0)
        total_ch = st.number_input("Valor Total Acumulado ($ - deixe 0 para auto)", 0.0, value=0.0, step=10.0)
        internet = st.selectbox("Serviço de Internet", ["Fiber optic", "DSL", "No"])
        tech = st.selectbox("Suporte Técnico", ["No", "Yes", "No internet service"])
    with c:
        st.markdown("**Segurança e Perfil**")
        sec = st.selectbox("Segurança Online", ["No", "Yes", "No internet service"])
        bak = st.selectbox("Backup Online", ["No", "Yes", "No internet service"])
        dev = st.selectbox("Proteção de Dispositivo", ["No", "Yes", "No internet service"])
        senior = st.selectbox("Cliente Idoso (SeniorCitizen)", ["0", "1"])
    enviar = st.form_submit_button("⚡ Fazer Chamada no Modelo", type="primary", width="stretch")

if enviar:
    payload = {
        "gender": "Female", "SeniorCitizen": senior, "Partner": "No", "Dependents": "No",
        "tenure": int(tenure), "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": bak,
        "DeviceProtection": dev, "TechSupport": tech, "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total_ch) if total_ch > 0 else float(tenure * monthly),
    }
    try:
        r = score(payload)
        st.success("Inferência executada com sucesso pelo classificador!")
        r1, r2, r3 = st.columns(3)
        r1.metric("Classificação do Modelo", "Risco de Churn (1)" if r["churn_prediction"] else "Retenção Estável (0)")
        r2.metric("Probabilidade Calculada", f"{r['churn_probability'] * 100:.2f}%")
        r3.metric("Nível de Risco", r["risk_level"])
        if db_ok:
            save_prediction(payload, r, f"{info['model_name']} v{info['version']} ({info['run_name']})")
            st.caption("Inferência gravada na tabela `predictions` do PostgreSQL.")
    except ValueError as exc:
        st.error(f"Dado fora do contrato do modelo: {exc}")

st.divider()

# BLOCO 3 -------------------------------------------------------------------
st.subheader("🗂️ 3. Histórico de Inferências (PostgreSQL)")
if db_ok:
    with get_engine().connect() as conn:
        hist = pd.read_sql(text("SELECT created_at, churn_prediction, churn_probability, risk_level, "
                                "model_version FROM predictions ORDER BY created_at DESC LIMIT 20"), conn)
    if len(hist):
        st.dataframe(hist, width="stretch")
    else:
        st.info("Nenhuma inferência gravada ainda.")
