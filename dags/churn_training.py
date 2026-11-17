"""DAG de retreino do modelo de churn — Encontros 9–11.

    validate_data → train_candidate → quality_gate → decide_promotion
                                                     ├─ promote_champion
                                                     └─ register_challenger

* Enc. 9  — DAG, tarefas e dependências (TaskFlow API).
* Enc. 10 — XCom (run_id/métricas entre tarefas), branching champion × challenger,
            retries e quality gate.
* Enc. 11 — Airflow em container (``docker compose up airflow``), agendamento
            semanal e ``catchup=False``.

A lógica vive em ``churn.pipeline`` (Python puro, testado com pytest); o DAG só
descreve *ordem*, *agendamento* e *política de falha*.
"""
from __future__ import annotations

from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

from churn import pipeline

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="churn_training",
    description="Valida dados, treina candidato e decide champion × challenger.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churn", "mlops"],
)
def churn_training():
    @task
    def validate_data() -> dict:
        return pipeline.validate_data()

    @task
    def train_candidate(_report: dict) -> dict:
        return pipeline.train_candidate()

    @task(retries=0)  # reprovar no gate é decisão, não falha transitória
    def quality_gate(candidate: dict) -> dict:
        return pipeline.quality_gate(candidate)

    @task.branch
    def decide_promotion(candidate: dict) -> str:
        return pipeline.decide_promotion(candidate)

    @task(task_id=pipeline.PROMOTE)
    def promote_champion(candidate: dict) -> int:
        return pipeline.promote_champion(candidate)

    @task(task_id=pipeline.KEEP)
    def register_challenger(candidate: dict) -> int:
        return pipeline.register_challenger(candidate)

    candidate = quality_gate(train_candidate(validate_data()))
    decide_promotion(candidate) >> [promote_champion(candidate), register_challenger(candidate)]


churn_training()
