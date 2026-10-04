"""Per-customer explanations by occlusion.

For each input, swap this customer's value for the "typical" value (most common category or
median number in the training data) and see how the churn probability moves. A positive
change means the customer's value pushes risk *up* relative to a typical customer.

This is a transparent approximation, not SHAP: features are swapped one at a time, so it
ignores interactions between them.
"""

from __future__ import annotations

import pandas as pd

from churn.config import FEATURES, NUMERIC


def reference_values(X_train: pd.DataFrame) -> dict:
    ref = {}
    for col in FEATURES:
        ref[col] = float(X_train[col].median()) if col in NUMERIC else X_train[col].mode().iloc[0]
    return ref


def customer_drivers(pipeline, customer: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Rank the features of a one-row ``customer`` frame by their effect on churn probability."""
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
