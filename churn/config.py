"""Paths, column groups and model settings in one place."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", ROOT / "data"))
DATA_FILE = DATA_DIR / "customer_churn.csv"
SAMPLE_FILE = DATA_DIR / "sample_customers.csv"
MODEL_DIR = Path(os.getenv("CHURN_MODEL_DIR", ROOT / "models"))

TARGET = "Churn"
ID_COLUMN = "customerID"

# Raw input columns, grouped by how they are encoded.
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]  # Yes -> 1, anything else ("No", "No internet service", "No phone service") -> 0
CATEGORICAL = ["InternetService", "Contract", "PaymentMethod"]  # one-hot encoded
OTHER = ["gender", "SeniorCitizen"]  # gender: Female -> 1; SeniorCitizen already 0/1
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Allowed values, used by the input form and for validating uploads.
CHOICES = {
    "gender": ["Female", "Male"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# --- Model (same architecture as the original notebook: 26 -> 15 -> 1) -----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIDDEN_LAYERS = (26, 15)
BATCH_SIZE = 32  # Keras default, as in the notebook

# --- Risk bands -------------------------------------------------------------------
LOW_RISK_BELOW = 0.30
HIGH_RISK_FROM = 0.60
