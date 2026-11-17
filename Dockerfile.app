# Imagem da aplicação (AWS): FastAPI + Streamlit no mesmo container, via Supervisord.
# Serve o @champion do MLflow Model Registry, exportado antes do build:
#   uv run python -m churn.registry --export   (feito automaticamente por infra/deploy_ecr.*)
#
#   docker build -f Dockerfile.app -t app-churn:latest .
#   docker run -p 8501:8501 --env-file churn.env app-churn:latest              # v2 (padrão)
#   docker run -p 8501:8501 -e API_MODULE=churn.api_lookup \
#              -e FRONTEND=src/churn/app_churn_v1.py app-churn:latest           # v1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    API_MODULE=churn.api_v2 \
    FRONTEND=src/churn/app_churn_v2.py

RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# mesmas versões de bibliotecas do treino: o uv.lock é o mesmo que gerou o @champion
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev

COPY supervisord.conf /etc/supervisor/conf.d/churn.conf
# remove BOM/CRLF caso o arquivo tenha sido editado no Windows
RUN sed -i '1s/^\xef\xbb\xbf//; s/\r$//' /etc/supervisor/conf.d/churn.conf

COPY artifacts/champion/ ./artifacts/champion/
COPY data/churn.csv ./data/churn.csv

# app-churn-v1: resultados pré-calculados pelo @champion para todos os clientes (parquet)
RUN python -m churn.batch_scoring

# v2: credenciais do PostgreSQL entram só em tempo de execução (docker run --env-file)
EXPOSE 8501
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/churn.conf"]
