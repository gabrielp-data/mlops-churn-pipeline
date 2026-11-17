from churn.data import coerce_total_charges, drop_identifier
from churn.profiling import profile
from churn.schema_demo import break_contract, failure_cases


def test_profile_resume_o_dataset(raw_frame):
    summary = profile(raw_frame)
    assert summary["rows"] == 4
    assert summary["columns"] == 21
    assert summary["duplicated_ids"] == 0
    assert summary["total_charges_blank"] == 1
    assert summary["target_distribution"] == {"No": 50.0, "Yes": 50.0}


def test_quebra_proposital_e_barrada_pelo_contrato(raw_frame):
    bad = break_contract(raw_frame)
    assert bad["customerID"].duplicated().any()
    cases = failure_cases(coerce_total_charges(drop_identifier(bad, "customerID")))
    assert set(cases["column"]) == {"tenure", "MonthlyCharges", "Contract"}
    assert "Vitalicio" in set(cases["failure_case"])


def test_dado_limpo_nao_tem_falhas(raw_frame):
    cases = failure_cases(coerce_total_charges(drop_identifier(raw_frame, "customerID")))
    assert cases.empty
