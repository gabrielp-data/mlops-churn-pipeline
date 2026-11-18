"""Painel de consulta e nova inferência; toda previsão vem da API."""

import os
from urllib.parse import quote

import requests
import streamlit as st

from churn.config import settings

st.set_page_config(page_title="Churn | Visão de clientes", page_icon="◉", layout="wide")
API_URL = settings.api_url.rstrip("/")
BATCH_MODE = os.environ.get("CHURN_SERVING_MODE", "online") == "batch"
st.markdown(
    """
<style>
.block-container { max-width: 1150px; padding-top: 2.5rem; }
[data-testid="stMetric"] { background: #f0f5f4; border-radius: 12px; padding: 16px; }
</style>
""",
    unsafe_allow_html=True,
)
st.caption("CUSTOMER INTELLIGENCE  /  RETENÇÃO")
st.title("Cada cliente conta.")
st.write("Consulte sua base e acompanhe a probabilidade de cancelamento.")


def api_request(method, path, payload=None):
    try:
        response = requests.request(
            method, f"{API_URL}{path}", json=payload, timeout=15
        )
    except requests.RequestException:
        st.error("Não foi possível conectar ao serviço. Tente novamente em instantes.")
        return None
    if response.status_code == 404:
        st.warning("Cliente não encontrado. Confira o código informado.")
        return None
    if response.status_code == 422:
        st.warning("Confira os dados: há um valor fora das opções aceitas.")
        return None
    if not response.ok:
        st.error("Serviço temporariamente indisponível.")
        return None
    try:
        return response.json()
    except ValueError:
        st.error("O serviço retornou uma resposta inválida.")
        return None


def show_result(result):
    cols = st.columns(3)
    cols[0].metric(
        "Probabilidade de cancelamento", f"{result['churn_probability']:.1%}"
    )
    cols[1].metric("Faixa de risco", result["risk_level"])
    cols[2].metric(
        "Previsão", "Cancelamento" if result["churn_prediction"] else "Permanência"
    )
    st.caption(
        "Faixas: baixo < 30% · médio de 30% a 60% · alto ≥ 60%. Limite de classificação: 50%."
    )
    if result.get("actual_churn") is not None:
        st.write(
            "Resultado observado:",
            "Cancelou" if result["actual_churn"] == "Yes" else "Permaneceu",
        )
    st.caption(f"Versão do modelo: {result['model_version']}")
    if "customer" in result:
        with st.expander("Dados do cliente"):
            st.json(result["customer"])


with st.sidebar:
    st.subheader("Sobre o modelo")
    if st.button("Consultar versão"):
        info = api_request("GET", "/model-info")
        if info:
            st.write(f"Versão {info['version']}")
            if "test_roc_auc" in info.get("metrics", {}):
                st.metric("ROC-AUC no teste", f"{info['metrics']['test_roc_auc']:.3f}")
            st.caption(f"Atualizado em {info['exported_at'][:10]}")
    st.divider()
    st.write(
        "A previsão apoia a priorização de contatos. O resultado observado pode ser diferente."
    )

tabs = st.tabs(
    ["Consultar cliente"] if BATCH_MODE else ["Consultar cliente", "Nova previsão"]
)
with tabs[0]:
    st.subheader("Encontre um cliente")
    with st.form("lookup"):
        customer_id = st.text_input("Código do cliente", placeholder="Ex.: 7590-VHVEG")
        lookup = st.form_submit_button("Consultar", type="primary")
    if lookup:
        if not customer_id.strip():
            st.warning("Informe o código do cliente.")
        else:
            result = api_request(
                "GET", f"/results/{quote(customer_id.strip(), safe='')}"
            )
            if result:
                show_result(result)

if not BATCH_MODE:
    with tabs[1]:
        st.subheader("Conheça o risco de um novo perfil")
        st.write("Preencha os dados atuais para calcular uma previsão.")
        with st.form("prediction"):
            left, middle, right = st.columns(3)
            with left:
                st.markdown("**Perfil e contrato**")
                gender = st.selectbox(
                    "Gênero",
                    ["Female", "Male"],
                    format_func=lambda x: {"Female": "Feminino", "Male": "Masculino"}[
                        x
                    ],
                )
                senior = st.selectbox(
                    "Pessoa idosa", [0, 1], format_func=lambda x: "Sim" if x else "Não"
                )
                partner = st.checkbox("Possui companheiro(a)")
                dependents = st.checkbox("Possui dependentes")
                tenure = st.number_input(
                    "Tempo como cliente (meses)", min_value=0, max_value=72, value=12
                )
                contract = st.selectbox(
                    "Contrato",
                    ["Month-to-month", "One year", "Two year"],
                    format_func=lambda x: {
                        "Month-to-month": "Mensal",
                        "One year": "Um ano",
                        "Two year": "Dois anos",
                    }[x],
                )
            with middle:
                st.markdown("**Serviços contratados**")
                phone = st.checkbox("Telefonia", value=True)
                multiple = st.checkbox("Múltiplas linhas")
                internet = st.selectbox(
                    "Internet",
                    ["DSL", "Fiber optic", "No"],
                    format_func=lambda x: {
                        "DSL": "DSL",
                        "Fiber optic": "Fibra óptica",
                        "No": "Sem internet",
                    }[x],
                )
                security = st.checkbox("Segurança online")
                backup = st.checkbox("Backup online")
                protection = st.checkbox("Proteção de dispositivos")
                support = st.checkbox("Suporte técnico")
                tv = st.checkbox("Streaming de TV")
                movies = st.checkbox("Streaming de filmes")
            with right:
                st.markdown("**Cobrança**")
                monthly = st.number_input(
                    "Mensalidade", min_value=0.0, value=75.0, step=5.0
                )
                total = st.number_input(
                    "Total já cobrado", min_value=0.0, value=900.0, step=10.0
                )
                paperless = st.checkbox("Fatura digital", value=True)
                payment = st.selectbox(
                    "Pagamento",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    format_func=lambda x: {
                        "Electronic check": "Cheque eletrônico",
                        "Mailed check": "Cheque por correio",
                        "Bank transfer (automatic)": "Débito automático",
                        "Credit card (automatic)": "Cartão automático",
                    }[x],
                )
            submitted = st.form_submit_button("Calcular previsão", type="primary")
        if submitted:

            def yes_no(value):
                return "Yes" if value else "No"

            payload = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": yes_no(partner),
                "Dependents": yes_no(dependents),
                "tenure": tenure,
                "PhoneService": yes_no(phone),
                "MultipleLines": yes_no(multiple) if phone else "No phone service",
                "InternetService": internet,
                "Contract": contract,
                "PaperlessBilling": yes_no(paperless),
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
            for field, selected in {
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
            }.items():
                payload[field] = (
                    yes_no(selected) if internet != "No" else "No internet service"
                )
            result = api_request("POST", "/predict", payload)
            if result:
                show_result(result)
