"""
Streamlit UI tests using AppTest.
Validates tab rendering, form interaction, batch scoring, and cross-validation triggers.
"""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from churn.config import SAMPLE_FILE

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


def probability(at: AppTest) -> str:
    return next(m.value for m in at.metric if m.label == "Churn probability")


def test_app_renders_all_tabs_without_errors():
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict", "Batch scoring", "Data explorer", "Model performance", "About"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers"] == "7,043" and metrics["ROC-AUC"].startswith("0.8")


def test_presets_move_the_prediction_in_the_right_direction():
    at = run_app()
    risky = int(probability(at).strip("%<"))
    at.selectbox(key="preset").select("Loyal long-term customer").run()
    assert not at.exception
    loyal = probability(at)
    assert loyal == "<1%" or int(loyal.strip("%")) < risky
    assert at.success  # predicted to stay


def test_no_internet_hides_addons_and_still_predicts():
    at = run_app()
    at.selectbox(key="in_internet").select("No").run()
    assert not at.exception
    assert not any(s.key == "in_OnlineSecurity" for s in at.selectbox)


def test_lower_threshold_flags_more_customers():
    at = run_app()
    at.selectbox(key="preset").select("Loyal long-term customer").run()
    assert at.success
    at.slider[0].set_value(0.05).run()  # decision threshold
    assert not at.exception


def test_batch_scoring_of_bundled_sample():
    at = run_app()
    next(b for b in at.button if b.label == "Score the bundled sample").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers scored"] == "200"


def test_batch_with_missing_columns_shows_an_error():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["batch_df"] = pd.read_csv(SAMPLE_FILE).drop(columns=["Contract"])
    at.session_state["batch_name"] = "broken.csv"
    at.run()
    assert not at.exception
    assert any("Missing required column" in e.value for e in at.error)


def test_cross_validation_button():
    at = run_app()
    next(b for b in at.button if b.label == "Run 5-fold cross-validation").click().run()
    assert not at.exception
    assert any(len(df.value) == 5 for df in at.dataframe if "Metric" in df.value.columns)
