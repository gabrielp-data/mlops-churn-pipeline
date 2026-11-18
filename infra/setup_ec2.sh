#!/usr/bin/env bash
# Pré-requisitos: Docker, AWS CLI e IAM Role com leitura do ECR.
set -euo pipefail
: "${IMAGE_URI:?Informe IMAGE_URI com tag de versão ou digest do ECR}"
: "${AWS_REGION:?Informe AWS_REGION}"
: "${ENV_FILE:?Informe ENV_FILE com CHURN_DATABASE_URL}"
[[ -f "$ENV_FILE" ]] || { echo "Arquivo de ambiente ausente." >&2; exit 1; }
registry="${IMAGE_URI%%/*}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$registry"
docker pull "$IMAGE_URI"
previous_image="$(docker inspect --format '{{.Config.Image}}' churn-app 2>/dev/null || true)"
if docker container inspect churn-app >/dev/null 2>&1; then
  docker stop churn-app
  docker rm churn-app
fi
docker run -d --name churn-app --restart unless-stopped \
  --env-file "$ENV_FILE" -p 8501:8501 -p 127.0.0.1:8000:8000 "$IMAGE_URI"
for attempt in $(seq 1 36); do
  status="$(docker inspect --format '{{.State.Health.Status}}' churn-app)"
  if [[ "$status" == "healthy" ]]; then
    echo "Aplicação saudável. Imagem anterior para rollback: $previous_image"
    exit 0
  fi
  sleep 5
done
echo "Health check falhou. Consulte docker logs churn-app. Imagem anterior: $previous_image" >&2
exit 1
