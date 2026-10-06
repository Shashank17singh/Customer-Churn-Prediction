"""
Implements local explanation of predictions by computing the marginal effect of replacing
individual features with their reference (median/mode) values.
"""
from __future__ import annotations

import pandas as pd

from churn.config import FEATURES, NUMERIC


def reference_values(X_train: pd.DataFrame) -> dict:
    """
    Computes a 'typical' profile from the training set.
    Uses median for numeric features and mode for categoricals to act as 
    the baseline for marginal effect computation.
    """
    ref = {}
    for col in FEATURES:
        ref[col] = float(X_train[col].median()) if col in NUMERIC else X_train[col].mode().iloc[0]
    return ref


def customer_drivers(pipeline, customer: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """
    Approximates local feature importance by swapping each feature with its 
    reference value and observing the change in predicted probability.
    """
    base = float(pipeline.predict_proba(customer)[0, 1])
    variants = []
    for col in FEATURES:
        v = customer.copy()
        v[col] = reference[col]
        variants.append(v)
    swapped = pd.concat(variants, ignore_index=True)
    proba = pipeline.predict_proba(swapped)[:, 1]

    rows = [
        {
            "Feature": col,
            "Customer value": customer.iloc[0][col],
            "Typical value": reference[col],
            "Effect on churn probability": base - p,
        }
        for col, p in zip(FEATURES, proba)
    ]
    table = pd.DataFrame(rows)
    table["Customer value"] = table["Customer value"].astype(str)
    table["Typical value"] = table["Typical value"].astype(str)
    order = table["Effect on churn probability"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)
