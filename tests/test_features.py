import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data import load_clean_data
from src.features import features_for_inference


def test_packaged_model_scores_known_public_customer(customer, model_bundle):
    model, contract = model_bundle
    features = features_for_inference(pd.DataFrame([customer]), contract)
    assert list(features.columns) == contract["feature_columns"]
    assert round(float(model.predict_proba(features)[0, 1]), 4) == 0.5439


def test_unknown_category_is_rejected(customer, model_bundle):
    customer["Contract"] = "Vitalicio"
    with pytest.raises(ValueError, match="Contract"):
        features_for_inference(pd.DataFrame([customer]), model_bundle[1])


def test_new_customer_with_blank_total_can_be_scored(customer, model_bundle):
    customer.update(tenure=0, TotalCharges=" ")
    model, contract = model_bundle
    features = features_for_inference(pd.DataFrame([customer]), contract)
    assert features["TotalCharges"].iloc[0] == 0
    assert 0 <= model.predict_proba(features)[0, 1] <= 1


@pytest.mark.parametrize(
    "field,value", [("tenure", 999), ("MonthlyCharges", -1), ("Contract", "Vitalicio")]
)
def test_data_contract_blocks_invalid_training_data(tmp_path, customer, field, value):
    customer[field] = value
    path = tmp_path / "invalid.csv"
    pd.DataFrame([customer]).to_csv(path, index=False)
    with pytest.raises(SchemaErrors):
        load_clean_data(path)


def test_data_contract_rejects_duplicate_ids(tmp_path, customer):
    path = tmp_path / "duplicates.csv"
    pd.DataFrame([customer, customer]).to_csv(path, index=False)
    with pytest.raises(SchemaErrors):
        load_clean_data(path)
