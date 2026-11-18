FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.12-slim AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" MLFLOW_DISABLE_AGENT_HINT=1 CHURN_BIND_HOST=0.0.0.0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

FROM base AS training
CMD ["python", "-m", "churn.model"]

FROM base AS serving
# Gerado por python -m churn.export_champion antes do build.
COPY models/champion /app/models/champion
# Verifica as versões do pacote exportado contra o ambiente travado.
RUN uv pip install --python /app/.venv/bin/python -r models/champion/requirements.txt \
    && uv pip check --python /app/.venv/bin/python
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health',timeout=4); urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=4)"
CMD ["python", "-m", "churn.serve"]
