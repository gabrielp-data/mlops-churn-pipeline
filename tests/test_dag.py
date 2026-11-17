"""Integridade do DAG — só roda se o Airflow estiver instalado (`uv sync --group airflow`)."""
from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS), include_examples=False)


def test_sem_erros_de_import(dagbag):
    assert dagbag.import_errors == {}


def test_estrutura_do_dag(dagbag):
    dag = dagbag.get_dag("churn_training")
    assert dag is not None
    assert dag.catchup is False
    downstream = {t.task_id: t.downstream_task_ids for t in dag.tasks}
    assert downstream["validate_data"] == {"train_candidate"}
    assert downstream["train_candidate"] == {"quality_gate"}
    assert {"promote_champion", "register_challenger"} <= downstream["decide_promotion"]
