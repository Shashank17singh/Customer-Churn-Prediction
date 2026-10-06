"""
Tests for local explanation logic, Altair chart generation, and the CLI predict command.
Ensures charts build correctly and CLI handles file I/O properly.
"""
import pandas as pd
import pytest

from churn import charts
from churn.cli import main
from churn.config import FEATURES, SAMPLE_FILE
from churn.data import load_data
from churn.explain import customer_drivers, reference_values
from churn.model import feature_importance, threshold_table


@pytest.fixture(scope="module")
def risky_customer():
    return pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 170.0,
            }
        ]
    )[FEATURES]


def test_drivers_cover_all_features_sorted_by_impact(model, risky_customer):
    drivers = customer_drivers(model.pipeline, risky_customer, reference_values(model.X_train))
    assert len(drivers) == len(FEATURES)
    effects = drivers["Effect on churn probability"].abs()
    assert effects.is_monotonic_decreasing


def test_long_contract_lowers_risk_relative_to_typical(model, risky_customer):
    """Month-to-month is the typical contract, so a two-year contract should read as risk-lowering."""
    customer = risky_customer.assign(Contract="Two year")
    drivers = customer_drivers(model.pipeline, customer, reference_values(model.X_train)).set_index("Feature")
    assert drivers.loc["Contract", "Effect on churn probability"] < 0


def test_reference_values_are_typical(model):
    ref = reference_values(model.X_train)
    assert ref["Contract"] == "Month-to-month" and ref["SeniorCitizen"] == 0


def test_all_charts_build_valid_specs(model, risky_customer):
    df = load_data()
    df["churned"] = (df["Churn"] == "Yes").astype(int)
    drivers = customer_drivers(model.pipeline, risky_customer, reference_values(model.X_train))
    scored = pd.DataFrame({"churn_probability": model.proba_test})
    built = [
        charts.drivers_chart(drivers),
        charts.rate_chart(df, "Contract"),
        charts.hist_chart(df, "tenure"),
        charts.prob_hist(scored, 0.5),
        charts.roc_chart(model.roc_points(), 0.83),
        charts.threshold_chart(threshold_table(model.y_test, model.proba_test), 0.5),
        charts.importance_chart(feature_importance(model, repeats=2)),
        charts.confusion_chart(model.metrics_at(0.5)),
    ]
    for chart in built:
        assert chart.to_dict()


def test_cli_predict_writes_sorted_scores(tmp_path, capsys):
    out = tmp_path / "scored.csv"
    main(["predict", str(SAMPLE_FILE), "--out", str(out)])
    scored = pd.read_csv(out)
    assert len(scored) == 200 and scored["churn_probability"].is_monotonic_decreasing


def test_cli_predict_rejects_bad_file(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        main(["predict", str(bad)])
