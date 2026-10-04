"""Model pipeline, training, evaluation and batch scoring.

The model is a small feed-forward ANN (26 -> 15 -> 1, ReLU, Adam, early stopping), the same
architecture as the original notebook, implemented with scikit-learn's ``MLPClassifier`` so
the app installs quickly on hosted platforms (no TensorFlow).

Differences from the notebook that matter for correctness:

* labels are encoded properly (the notebook's ``replace`` calls were never assigned, which
  turned every label into 0 and produced a meaningless 100% accuracy);
* the scaler is fitted on the training split only (no leakage);
* early stopping uses an internal validation split, not the test set;
* the split is stratified because only ~27% of customers churn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from churn.config import (
    BATCH_SIZE,
    BINARY,
    CATEGORICAL,
    HIDDEN_LAYERS,
    HIGH_RISK_FROM,
    LOW_RISK_BELOW,
    MODEL_DIR,
    NUMERIC,
    OTHER,
    RANDOM_STATE,
    TEST_SIZE,
)
from churn.data import load_data, normalise, split_xy

# --- pipeline ---------------------------------------------------------------------


def build_pipeline(classifier=None) -> Pipeline:
    """Raw customer rows -> encoded features -> classifier (the ANN unless one is given)."""
    if classifier is None:
        classifier = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            batch_size=BATCH_SIZE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
    prep = ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("pass", "passthrough", BINARY + OTHER),
        ]
    )
    return Pipeline([("normalise", FunctionTransformer(normalise)), ("prep", prep), ("clf", classifier)])


# --- metrics ----------------------------------------------------------------------


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-dependent metrics for the churn class."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float((tp + tn) / len(pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    return pd.DataFrame([metrics_at(y_true, proba, t) for t in thresholds])[["threshold", "precision", "recall", "f1", "accuracy"]]


def risk_band(p: float) -> str:
    return "Low" if p < LOW_RISK_BELOW else "High" if p >= HIGH_RISK_FROM else "Medium"


# --- trained model ----------------------------------------------------------------


@dataclass
class TrainedModel:
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    proba_test: np.ndarray
    summary: dict = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df)[:, 1]

    def metrics_at(self, threshold: float = 0.5) -> dict:
        return metrics_at(self.y_test, self.proba_test, threshold)

    @property
    def majority_baseline(self) -> float:
        """Accuracy of always predicting 'no churn' on the test split."""
        return float(1 - self.y_test.mean())

    def roc_points(self) -> pd.DataFrame:
        fpr, tpr, _ = roc_curve(self.y_test, self.proba_test)
        return pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})

    def pr_points(self) -> pd.DataFrame:
        precision, recall, _ = precision_recall_curve(self.y_test, self.proba_test)
        return pd.DataFrame({"Recall": recall, "Precision": precision})


def train(df: pd.DataFrame | None = None) -> TrainedModel:
    """Train the ANN on a stratified 80/20 split."""
    df = load_data() if df is None else df
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = build_pipeline().fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    clf = pipeline.named_steps["clf"]
    summary = {
        "customers": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "epochs": int(clf.n_iter_),
        "n_features": int(pipeline.named_steps["prep"].get_feature_names_out().shape[0]),
    }
    return TrainedModel(pipeline, X_train, X_test, y_train, y_test, proba, summary)


# --- extra evaluation (used by the app's performance tab) --------------------------


def compare_models(model: TrainedModel) -> pd.DataFrame:
    """ANN versus classic baselines on the same split and the same preprocessing."""
    candidates = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    rows = [{"Model": "ANN (26-15-1)", **_row(model.y_test, model.proba_test)}]
    for name, clf in candidates.items():
        proba = build_pipeline(clf).fit(model.X_train, model.y_train).predict_proba(model.X_test)[:, 1]
        rows.append({"Model": name, **_row(model.y_test, proba)})
    return pd.DataFrame(rows)


def _row(y_true, proba) -> dict:
    m = metrics_at(y_true, proba, 0.5)
    return {
        "ROC-AUC": round(roc_auc_score(y_true, proba), 3),
        "Precision": round(m["precision"], 3),
        "Recall": round(m["recall"], 3),
        "F1": round(m["f1"], 3),
        "Accuracy": round(m["accuracy"], 3),
    }


def feature_importance(model: TrainedModel, repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on the raw columns: how much ROC-AUC drops when a column is shuffled."""
    result = permutation_importance(
        model.pipeline, model.X_test, model.y_test, scoring="roc_auc",
        n_repeats=repeats, random_state=RANDOM_STATE, n_jobs=1,
    )
    table = pd.DataFrame(
        {"Feature": model.X_test.columns, "Importance": result.importances_mean, "Std": result.importances_std}
    )
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


def cross_validate_ann(df: pd.DataFrame | None = None, folds: int = 5) -> pd.DataFrame:
    """Stratified k-fold cross-validation of the full pipeline."""
    X, y = split_xy(load_data() if df is None else df)
    scores = cross_validate(
        build_pipeline(), X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE),
        scoring={"ROC-AUC": "roc_auc", "Precision": "precision", "Recall": "recall", "F1": "f1", "Accuracy": "accuracy"},
    )
    rows = {k.replace("test_", ""): v for k, v in scores.items() if k.startswith("test_")}
    return pd.DataFrame({"Metric": list(rows), "Mean": [v.mean() for v in rows.values()], "Std": [v.std() for v in rows.values()]})


# --- scoring ----------------------------------------------------------------------


def score_frame(model: TrainedModel, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Return ``df`` plus churn probability, predicted label and risk band."""
    scored = df.copy()
    proba = model.predict_proba(scored)
    scored["churn_probability"] = np.round(proba, 4)
    scored["predicted_churn"] = np.where(proba >= threshold, "Yes", "No")
    scored["risk_band"] = [risk_band(p) for p in proba]
    return scored


# --- persistence (optional; the app trains in memory) -------------------------------


def save(model: TrainedModel, directory: Path = MODEL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, directory / "churn_model.joblib")
    (directory / "metrics.json").write_text(
        json.dumps({**model.summary, **model.metrics_at(0.5)}, indent=2), encoding="utf-8"
    )
    return directory / "churn_model.joblib"
