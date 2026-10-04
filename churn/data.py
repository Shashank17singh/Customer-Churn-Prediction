"""Loading, cleaning and encoding the telecom churn data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import (
    BINARY,
    CATEGORICAL,
    CHOICES,
    DATA_FILE,
    FEATURES,
    ID_COLUMN,
    NUMERIC,
    TARGET,
)


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """Read the CSV and return the raw frame with ``TotalCharges`` as a number.

    Eleven brand-new customers (tenure 0) have a blank ``TotalCharges``. Nothing has been
    billed yet, so those rows are kept with a total of 0 instead of being dropped.
    """
    df = pd.read_csv(path)
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = total.fillna(df["tenure"] * df["MonthlyCharges"])
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features (raw, un-encoded) and a 0/1 target."""
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df[FEATURES].copy(), y


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw customer rows for the model. Pure function, safe for training and inference.

    * ``TotalCharges`` is coerced to a number; blanks become ``tenure * MonthlyCharges``.
    * Yes/No style columns become 1/0; "No internet service" and "No phone service" are "No".
    * ``gender`` becomes 1 for Female, 0 otherwise (same convention as the original notebook).
    * Unknown categories stay as text and are ignored by the one-hot encoder.
    """
    out = df[FEATURES].copy()

    for col in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    blank = out["TotalCharges"].isna()
    out.loc[blank, "TotalCharges"] = out.loc[blank, "tenure"] * out.loc[blank, "MonthlyCharges"]

    for col in BINARY:
        out[col] = (out[col].astype(str).str.strip().str.lower() == "yes").astype(int)
    out["gender"] = (out["gender"].astype(str).str.strip().str.lower() == "female").astype(int)
    for col in CATEGORICAL:
        out[col] = out[col].astype(str).str.strip()
    return out


def validate_batch(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check an uploaded frame. Returns ``(errors, warnings)``; errors block scoring."""
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return errors, warnings
    if df.empty:
        errors.append("The file has no rows.")
        return errors, warnings

    for col in NUMERIC + ["SeniorCitizen"]:
        bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna() & (df[col].astype(str).str.strip() != "")
        if bad.any():
            errors.append(f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
    for col in ("tenure", "MonthlyCharges"):
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            errors.append(f"Column '{col}' has empty values; it is required.")

    for col, allowed in CHOICES.items():
        unknown = sorted(set(df[col].astype(str).str.strip()) - set(allowed))
        if unknown:
            warnings.append(f"Column '{col}' has unfamiliar value(s) {unknown[:3]}; the model treats them as 'no match'.")
    return errors, warnings


def describe(df: pd.DataFrame) -> dict:
    """Headline numbers for the explorer."""
    _, y = split_xy(df)
    return {
        "customers": len(df),
        "churn_rate": float(y.mean()),
        "avg_tenure": float(df["tenure"].mean()),
        "avg_monthly": float(df["MonthlyCharges"].mean()),
        "has_id": ID_COLUMN in df.columns,
    }
