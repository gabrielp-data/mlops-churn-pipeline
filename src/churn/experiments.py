"""Grid de experimentos rastreado com MLflow.

Uso: ``uv run python -m churn.experiments``

Roda 9 combinações (n_estimators × max_depth). Cada run registra params,
métricas, linhagem (git + DVC) e o **modelo** — é ele que o Registry promove
a ``@champion`` depois. Compare os runs com ``uv run mlflow ui``.
"""
from __future__ import annotations

import mlflow

from churn.config import Settings, settings
from churn.evaluate import evaluate
from churn.model import EXPERIMENT, build_pipeline, load_split, log_pipeline

GRID_N_ESTIMATORS = (100, 200, 500)
GRID_MAX_DEPTH = (None, 8, 16)


def run_grid(cfg: Settings = settings) -> None:
    x_train, x_test, y_train, y_test = load_split(cfg)

    mlflow.set_experiment(EXPERIMENT)
    for n in GRID_N_ESTIMATORS:
        for depth in GRID_MAX_DEPTH:
            with mlflow.start_run(run_name=f"rf-n{n}-d{depth}"):
                mlflow.log_params({"n_estimators": n, "max_depth": depth})
                pipe = build_pipeline(cfg, n_estimators=n, max_depth=depth).fit(x_train, y_train)
                mlflow.log_metrics(evaluate(pipe, x_test, y_test))
                log_pipeline(pipe, x_train, cfg)


def find_best() -> None:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        order_by=["metrics.roc_auc DESC"],
    )
    best = runs.iloc[0]
    print(
        f"Melhor run: {best['run_id'][:8]}  "
        f"n={best['params.n_estimators']}  "
        f"depth={best['params.max_depth']}  "
        f"roc_auc={float(best['metrics.roc_auc']):.4f}"
    )


if __name__ == "__main__":
    run_grid()
    find_best()
