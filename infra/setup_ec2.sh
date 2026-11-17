#!/usr/bin/env bash
# Deploy da camada de aplicação (EC2 churn-app-host): baixa a imagem do ECR e sobe
# o container unificado (FastAPI + Streamlit) apontando para o PostgreSQL.
#
# Pré-requisitos: a EC2 tem a IAM Role EC2-ECR-ReadRole (lê o ECR sem chaves fixas).
#
# Uso (no terminal da churn-app-host):
#   1. Crie ~/churn.env com as variáveis do banco (modelo em .env.example),
#      DB_HOST = IP PRIVADO da churn-db-host.
#   2. bash setup_ec2.sh
set -euo pipefail

REGION="${REGION:-us-east-2}"
REPO_NAME="${REPO_NAME:-app-churn}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENV_FILE="${ENV_FILE:-$HOME/churn.env}"

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERRO: ${ENV_FILE} não encontrado (veja o cabeçalho deste script)."
    exit 1
fi

echo "=== 1. Instalando Docker e utilitários ==="
sudo apt-get update -y
sudo apt-get install -y docker.io unzip curl
sudo systemctl enable --now docker

echo "=== 2. Instalando AWS CLI v2 ==="
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

echo "=== 3. Autenticando no ECR via IAM Role (credenciais temporárias STS) ==="
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"
aws ecr get-login-password --region "${REGION}" | sudo docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "=== 4. Baixando e iniciando o container (FastAPI + Streamlit) ==="
sudo docker pull "${IMAGE_URI}"
sudo docker stop churn_app 2>/dev/null || true
sudo docker rm churn_app 2>/dev/null || true
sudo docker run -d -p 8501:8501 --name churn_app --restart unless-stopped \
    --env-file "${ENV_FILE}" "${IMAGE_URI}"

echo "=== 5. Verificando modelo e conexão com o banco ==="
sleep 10
sudo docker exec churn_app python -c \
  "import json, urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10)), indent=2))" || true

echo "=== Deploy concluído. Painel em http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP público>'):8501 ==="
