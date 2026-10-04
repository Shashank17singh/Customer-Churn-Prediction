import pandas as pd

from churn.config import FEATURES, SAMPLE_FILE
from churn.data import load_data, normalise, split_xy, validate_batch


def test_load_keeps_every_customer_and_fills_blank_totals():
    df = load_data()
    assert len(df) == 7043
    assert df["TotalCharges"].notna().all()
    new_customers = df[df["tenure"] == 0]
    assert len(new_customers) == 11 and (new_customers["TotalCharges"] == 0).all()


def test_labels_are_not_collapsed_to_one_class():
    """Regression test for the original notebook, where every label became 0."""
    _, y = split_xy(load_data())
    assert set(y.unique()) == {0, 1}
    assert 0.25 < y.mean() < 0.28


def test_normalise_encodes_yes_no_and_no_service_values():
    row = load_data().iloc[[0]].copy()
    row["OnlineSecurity"] = "No internet service"
    row["MultipleLines"] = "No phone service"
    row["Partner"] = "Yes"
    row["gender"] = "Female"
    out = normalise(row)
    assert out.loc[row.index[0], "OnlineSecurity"] == 0
    assert out.loc[row.index[0], "MultipleLines"] == 0
    assert out.loc[row.index[0], "Partner"] == 1
    assert out.loc[row.index[0], "gender"] == 1


def test_normalise_fills_blank_total_charges_from_tenure():
    row = load_data().iloc[[0]].copy()
    row["TotalCharges"] = " "
    row["tenure"], row["MonthlyCharges"] = 10, 20.0
    assert normalise(row)["TotalCharges"].iloc[0] == 200.0


def test_validate_batch_accepts_the_bundled_sample():
    errors, warnings = validate_batch(pd.read_csv(SAMPLE_FILE))
    assert errors == [] and warnings == []


def test_validate_batch_reports_missing_columns():
    errors, _ = validate_batch(pd.read_csv(SAMPLE_FILE).drop(columns=["Contract", "tenure"]))
    assert errors and "Contract" in errors[0] and "tenure" in errors[0]


def test_validate_batch_reports_bad_numbers_and_unknown_categories():
    df = pd.read_csv(SAMPLE_FILE)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(str)
    df.loc[0, "MonthlyCharges"] = "abc"
    df.loc[1, "Contract"] = "Weekly"
    errors, warnings = validate_batch(df)
    assert any("MonthlyCharges" in e for e in errors)
    assert any("Contract" in w for w in warnings)


def test_validate_batch_rejects_empty_file():
    errors, _ = validate_batch(pd.DataFrame(columns=FEATURES))
    assert errors
