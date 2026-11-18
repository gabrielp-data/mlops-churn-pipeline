import json

from sqlalchemy import text


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_empty_history_does_not_render_internal_documentation(dashboard):
    assert any("Nenhuma inferência" in x.value for x in dashboard.info)
    assert not dashboard.get("doc_string")


def test_lookup_returns_real_score(dashboard):
    button(dashboard, "Buscar Cliente").click().run()
    assert not dashboard.exception
    assert {m.label: m.value for m in dashboard.metric}["Probabilidade"] == "54.39%"
    assert dashboard.dataframe[0].value.iloc[0]["customerID"] == "7590-VHVEG"


def test_lookup_missing_customer(dashboard):
    dashboard.text_input[0].set_value("missing").run()
    button(dashboard, "Buscar Cliente").click().run()
    assert not dashboard.exception
    assert any("não encontrado" in x.value for x in dashboard.warning)


def test_prediction_is_persisted_and_rerun_does_not_duplicate(dashboard, database):
    button(dashboard, "⚡ Fazer Chamada no Modelo").click().run()
    assert not dashboard.exception
    assert {m.label: m.value for m in dashboard.metric}[
        "Probabilidade Calculada"
    ] == "67.66%"
    assert not dashboard.get("doc_string")
    with database[0].connect() as conn:
        row = conn.execute(text("SELECT * FROM predictions")).mappings().one()
        assert row["churn_probability"] == 0.6766
        assert json.loads(row["payload"])["tenure"] == 6
        assert "rf-n500-d8" in row["model_version"]
    assert len(dashboard.dataframe[-1].value) == 1
    dashboard.run()
    assert not dashboard.exception
    with database[0].connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM predictions")).scalar_one() == 1
