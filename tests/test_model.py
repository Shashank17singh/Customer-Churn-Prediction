"""
Tests for model training, metrics, and scoring.
Verifies determinism, data splits, and model performance.
"""
import numpy as np

from churn.config import NUMERIC
from churn.data import load_data, normalise
from churn.model import (
    compare_models,
    feature_importance,
    metrics_at,
    risk_band,
    save,
    score_frame,
    threshold_table,
    train,
)


def test_model_beats_chance_by_a_wide_margin(model):
    assert model.summary["roc_auc"] > 0.80
    assert model.summary["n_features"] == 26


def test_model_predicts_both_classes_at_default_threshold(model):
    m = model.metrics_at(0.5)
    assert m["tp"] > 0 and m["tn"] > 0
    assert m["recall"] > 0.35 and m["precision"] > 0.5


def test_no_row_is_in_both_train_and_test(model):
    assert set(model.X_train.index).isdisjoint(model.X_test.index)
    assert abs(model.y_train.mean() - model.y_test.mean()) < 0.01  # stratified


def test_scaler_is_fitted_on_training_data_only(model):
    scaler = model.pipeline.named_steps["prep"].named_transformers_["num"]
    train_max = normalise(model.X_train)[NUMERIC].max().to_numpy(dtype=float)
    assert np.allclose(scaler.data_max_, train_max)


def test_training_is_deterministic(model):
    again = train()
    assert np.allclose(again.proba_test, model.proba_test)


def test_unknown_category_does_not_crash(model):
    row = model.X_test.iloc[[0]].copy()
    row["InternetService"] = "Satellite"
    assert 0 <= model.predict_proba(row)[0] <= 1


def test_metrics_at_matches_hand_calculation():
    y = [1, 1, 0, 0, 1]
    p = [0.9, 0.4, 0.6, 0.1, 0.8]
    m = metrics_at(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == 2 / 3 and m["recall"] == 2 / 3 and m["accuracy"] == 3 / 5


def test_lower_threshold_raises_recall(model):
    assert model.metrics_at(0.3)["recall"] > model.metrics_at(0.7)["recall"]
    table = threshold_table(model.y_test, model.proba_test)
    assert table["recall"].is_monotonic_decreasing


def test_risk_bands():
    assert [risk_band(p) for p in (0.1, 0.45, 0.8)] == ["Low", "Medium", "High"]


def test_score_frame_adds_columns_and_preserves_rows(model):
    df = load_data().head(25)
    scored = score_frame(model, df, threshold=0.5)
    assert len(scored) == 25
    assert {"churn_probability", "predicted_churn", "risk_band"} <= set(scored.columns)
    assert scored["churn_probability"].between(0, 1).all()


def test_baseline_comparison_and_importance(model):
    table = compare_models(model)
    assert table["Model"].tolist()[0].startswith("ANN") and len(table) == 4
    importance = feature_importance(model, repeats=2)
    assert importance["Feature"].iloc[0] == "Contract"  # strongest signal in this dataset


def test_save_writes_model_and_metrics(model, tmp_path):
    path = save(model, tmp_path)
    assert path.exists() and (tmp_path / "metrics.json").exists()
