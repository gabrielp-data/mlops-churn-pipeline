import os

# MLflow 3.x detecta o uv.lock e tenta exportar as dependencias com o uv; quando
# o ambiente nao e o do uv.lock, o autolog falha em silencio e nao salva o modelo.
os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")
