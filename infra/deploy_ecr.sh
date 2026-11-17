#!/usr/bin/env bash
# Build local da imagem da aplicação + push para o Amazon ECR (Linux/macOS/WSL).
# No Windows use infra/deploy_ecr.ps1.
#
# Uso (na raiz do projeto):
#   AWS_PROFILE=churn bash infra/deploy_ecr.sh
#   REGION=us-east-2 REPO_NAME=app-churn bash infra/deploy_ecr.sh
set -euo pipefail

REGION="${REGION:-us-east-2}"
REPO_NAME="${REPO_NAME:-app-churn}"

[ -f data/churn.csv ] || { echo "ERRO: data/churn.csv ausente. Rode 'uv run dvc pull' antes."; exit 1; }

echo "--> Obtendo Account ID da AWS..."
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR_URI}/${REPO_NAME}:latest"

echo "--> 0. Exportando o @champion do MLflow Model Registry..."
uv run python -m churn.registry --export

echo "--> 1. Garantindo o repositório ${REPO_NAME} no ECR..."
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO_NAME" >/dev/null

echo "--> 2. Autenticando o Docker no ECR (${REGION})..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"

echo "--> 3. Build da imagem (Dockerfile.app)..."
docker build -f Dockerfile.app -t "${REPO_NAME}:latest" .
docker tag "${REPO_NAME}:latest" "$IMAGE"

echo "--> 4. Push para o ECR..."
docker push "$IMAGE"

echo
echo "Publicação concluída com sucesso! URI: $IMAGE"
