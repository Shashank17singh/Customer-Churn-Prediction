"""
Data loading, preprocessing, and validation logic.
Cleans inputs and splits features from targets.
"""
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
    """
    Loads raw customer data and imputes missing lifetime charges.
    Missing 'TotalCharges' occurs for brand new customers (tenure=0),
    so we fallback to tenure * MonthlyCharges.
    """
    df = pd.read_csv(path)
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = total.fillna(df["tenure"] * df["MonthlyCharges"])
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df[FEATURES].copy(), y


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forces numeric and boolean columns into predictable types.
    Pandas object columns can silently carry mixed types, so we explicitly
    cast to numeric or 1/0 integer representations for the ML pipeline.
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
    _, y = split_xy(df)
    return {
        "customers": len(df),
        "churn_rate": float(y.mean()),
        "avg_tenure": float(df["tenure"].mean()),
        "avg_monthly": float(df["MonthlyCharges"].mean()),
        "has_id": ID_COLUMN in df.columns,
    }
