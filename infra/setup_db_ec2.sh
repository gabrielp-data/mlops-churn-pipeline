#!/usr/bin/env bash
# Executar na EC2 de dados; não remove containers nem volumes existentes.
set -euo pipefail
: "${DB_ENV_FILE:?Informe arquivo contendo POSTGRES_DB, POSTGRES_USER e POSTGRES_PASSWORD}"
: "${PRIVATE_IP:?Informe o IP privado da EC2 de dados}"
[[ -f "$DB_ENV_FILE" ]] || { echo "Arquivo de ambiente ausente." >&2; exit 1; }
docker volume create churn-pgdata
docker run -d --name churn-postgres --restart unless-stopped \
  --env-file "$DB_ENV_FILE" -p "$PRIVATE_IP:5432:5432" \
  -v churn-pgdata:/var/lib/postgresql/data \
  --health-cmd='pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  --health-interval=10s --health-retries=6 postgres:16
echo "PostgreSQL iniciado. Importe o CSV com python -m churn.db pela EC2 de aplicação."
