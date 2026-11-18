#!/usr/bin/env bash
# CI/ambiente de teste dedicado; não executa ações em outros projetos Compose.
set -euo pipefail
: "${COMPOSE_PROJECT_NAME:?Defina um nome exclusivo para o ambiente de teste}"
: "${POSTGRES_PASSWORD:?Defina a senha do PostgreSQL de teste}"
: "${CHURN_DATABASE_URL:?Defina a URL usando o host db}"

docker compose config --quiet
docker compose build app
docker compose --profile database up -d --wait --wait-timeout 120 db
docker compose run --rm app python -m churn.db --input /app/data/telco-public.csv
count="$(docker compose exec -T db psql -U churn -d churn -Atc 'SELECT count(*) FROM customers')"
[[ "$count" == "7043" ]] || { echo "Quantidade inesperada de clientes: $count"; exit 1; }

mkdir -p .tools
if docker compose run --rm app python -m churn.db --input /app/data/telco-public.csv >.tools/duplicate-import.log 2>&1; then
  echo "Carga duplicada deveria falhar sem alterar o banco." >&2
  exit 1
fi
count="$(docker compose exec -T db psql -U churn -d churn -Atc 'SELECT count(*) FROM customers')"
[[ "$count" == "7043" ]]

docker compose up -d --wait --wait-timeout 180 app
uv run python scripts/verify_deployment.py

# Uma falha do banco não pode interromper a inferência de novos clientes.
docker compose stop db
uv run python scripts/verify_deployment.py --database-down
docker compose --profile database up -d --wait --wait-timeout 120 db
uv run python scripts/verify_deployment.py

docker compose --profile batch up -d --build --wait --wait-timeout 180 lookup
uv run python scripts/verify_deployment.py --batch-url http://127.0.0.1:8001 --batch-panel-url http://127.0.0.1:8502
echo "Docker, PostgreSQL, API, painel, recuperação do banco e consulta de lote: OK"
