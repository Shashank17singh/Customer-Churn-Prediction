"""
Streamlit application for customer churn prediction.
Handles UI state, layout, and calling ML model inference routines.
Architecture note: Separates model logic (churn/model.py) from presentation layer to ensure testability.
"""
from __future__ import annotations

import pandas as pd  # type: ignore
import streamlit as st

from churn import APP_NAME, __version__, charts
from churn.config import (
    CHOICES,
    FEATURES,
    HIGH_RISK_FROM,
    INTERNET_ADDONS,
    LOW_RISK_BELOW,
    NUMERIC,
    SAMPLE_FILE,
)
from churn.data import describe, load_data, validate_batch
from churn.explain import customer_drivers, reference_values
from churn.model import (
    TrainedModel,
    compare_models,
    cross_validate_ann,
    feature_importance,
    metrics_at,
    risk_band,
    score_frame,
    threshold_table,
    train,
)

st.set_page_config(page_title=APP_NAME, layout="wide")

RISK_BADGE = {"Low": ":green-badge[Low risk]", "Medium": ":orange-badge[Medium risk]", "High": ":red-badge[High risk]"}


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner="Training the ANN (a few seconds on first load)...")
def get_model() -> TrainedModel:
    return train(get_data())


@st.cache_resource(show_spinner=False)
def get_reference() -> dict:
    return reference_values(get_model().X_train)


@st.cache_data(show_spinner="Training baseline models...")
def get_comparison() -> pd.DataFrame:
    return compare_models(get_model())


@st.cache_data(show_spinner="Computing permutation importance...")
def get_importance() -> pd.DataFrame:
    return feature_importance(get_model())


@st.cache_data(show_spinner="Running 5-fold cross-validation...")
def get_cv() -> pd.DataFrame:
    return cross_validate_ann(get_data())


def pct(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}%}"


def pct_prob(p: float) -> str:
    return "<1%" if p < 0.005 else pct(p)


DEFAULTS: dict[str, object] = dict(
    gender="Male",
    senior=0,
    partner="No",
    dependents="No",
    tenure=2,
    phone="Yes",
    lines="No",
    internet="Fiber optic",
    contract="Month-to-month",
    paperless="Yes",
    payment="Electronic check",
    monthly=85.0,
    OnlineSecurity="No",
    OnlineBackup="No",
    DeviceProtection="No",
    TechSupport="No",
    StreamingTV="Yes",
    StreamingMovies="Yes",
)
PRESETS: dict[str, dict[str, object] | None] = {
    "Custom": None,
    "Loyal long-term customer": dict(
        gender="Female",
        senior=0,
        partner="Yes",
        dependents="Yes",
        tenure=60,
        phone="Yes",
        lines="Yes",
        internet="DSL",
        contract="Two year",
        paperless="No",
        payment="Credit card (automatic)",
        monthly=65.0,
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
    ),
    "New month-to-month fiber customer": DEFAULTS,
}
FORM_KEYS = {
    "gender": "in_gender",
    "senior": "in_senior",
    "partner": "in_partner",
    "dependents": "in_dependents",
    "tenure": "in_tenure",
    "phone": "in_phone",
    "lines": "in_lines",
    "internet": "in_internet",
    "contract": "in_contract",
    "paperless": "in_paperless",
    "payment": "in_payment",
    "monthly": "in_monthly",
    **{a: f"in_{a}" for a in INTERNET_ADDONS},
}


def init_form() -> None:
    for name, key in FORM_KEYS.items():
        st.session_state.setdefault(key, DEFAULTS[name])
    st.session_state.setdefault("in_override", False)


def apply_preset() -> None:
    preset = PRESETS[st.session_state["preset"]]
    if preset:
        for name, key in FORM_KEYS.items():
            st.session_state[key] = preset[name]
        st.session_state["in_override"] = False


def sidebar() -> float:
    with st.sidebar:
        st.title(APP_NAME)
        st.caption("ANN-based telecom churn prediction")

        threshold = st.slider(
            "Decision threshold",
            0.05,
            0.95,
            0.50,
            0.05,
            help="A customer is predicted to churn when their probability reaches this value. "
            "Lower it to catch more churners (higher recall) at the cost of more false alarms.",
        )
        st.caption(f"Risk bands: low below {pct(LOW_RISK_BELOW)}, high from {pct(HIGH_RISK_FROM)}.")

        stats = describe(get_data())
        st.divider()
        st.metric("Customers in dataset", f"{stats['customers']:,}")
        st.metric("Overall churn rate", pct(stats["churn_rate"], 1))
        st.divider()
        st.caption(
            "Predictions are statistical estimates from a public telecom sample. "
            "Use them to prioritise outreach, not to make automated decisions about individuals."
        )
        st.caption(f"v{__version__}")
    return threshold


def tab_predict(threshold: float) -> None:
    model, reference = get_model(), get_reference()
    init_form()

    st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=apply_preset)
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Customer profile")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", CHOICES["gender"], key=FORM_KEYS["gender"])
        senior = c2.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No", key=FORM_KEYS["senior"])
        partner = c1.selectbox("Has partner", ["Yes", "No"], key=FORM_KEYS["partner"])
        dependents = c2.selectbox("Has dependents", ["Yes", "No"], key=FORM_KEYS["dependents"])

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, key=FORM_KEYS["tenure"])
        contract = st.selectbox("Contract", CHOICES["Contract"], key=FORM_KEYS["contract"])
        c3, c4 = st.columns(2)
        paperless = c3.selectbox("Paperless billing", ["Yes", "No"], key=FORM_KEYS["paperless"])
        payment = c4.selectbox("Payment method", CHOICES["PaymentMethod"], key=FORM_KEYS["payment"])
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, step=0.25, key=FORM_KEYS["monthly"])
        override = st.checkbox(
            "Enter total charges manually",
            key="in_override",
            help="Otherwise total charges are estimated as tenure x monthly charges.",
        )
        total = (
            st.number_input("Total charges ($)", 0.0, 10000.0, float(round(tenure * monthly, 2)), 10.0)
            if override
            else round(tenure * monthly, 2)
        )
        if not override:
            st.caption(f"Total charges estimated at ${total:,.2f} (tenure x monthly).")

        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"], key=FORM_KEYS["phone"])
        lines = st.selectbox("Multiple lines", ["Yes", "No"], key=FORM_KEYS["lines"]) if phone == "Yes" else "No"
        internet = st.selectbox("Internet service", CHOICES["InternetService"], key=FORM_KEYS["internet"])
        addons = {}
        if internet == "No":
            st.caption("No internet service, so all internet add-ons are 'No'.")
            addons = dict.fromkeys(INTERNET_ADDONS, "No")
        else:
            cols = st.columns(2)
            for i, name in enumerate(INTERNET_ADDONS):
                label = {
                    "OnlineSecurity": "Online security",
                    "OnlineBackup": "Online backup",
                    "DeviceProtection": "Device protection",
                    "TechSupport": "Tech support",
                    "StreamingTV": "Streaming TV",
                    "StreamingMovies": "Streaming movies",
                }[name]
                addons[name] = cols[i % 2].selectbox(label, ["Yes", "No"], key=FORM_KEYS[name])

    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": lines,
                "InternetService": internet,
                **addons,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )[FEATURES]
    probability = float(model.predict_proba(customer)[0])
    band = risk_band(probability)

    with right:
        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", pct_prob(probability))
        m2.markdown("**Risk level**")
        m2.markdown(RISK_BADGE[band])
        st.progress(probability)
        if probability >= threshold:
            st.error(
                f"Predicted to **churn** (probability {pct_prob(probability)} is at or above the {pct(threshold)} threshold)."
            )
        else:
            st.success(f"Predicted to **stay** (probability {pct_prob(probability)} is below the {pct(threshold)} threshold).")

        st.subheader("What is driving this prediction?")
        drivers = customer_drivers(model.pipeline, customer, reference)
        st.altair_chart(charts.drivers_chart(drivers), width="stretch")
        st.caption(
            "Each bar shows how much the probability changes if this value were replaced by the typical customer's. "
            "Red raises risk, green lowers it. Features are changed one at a time, so this is an approximation."
        )
        with st.expander("Details"):
            table = drivers.copy()
            table["Effect on churn probability"] = (table["Effect on churn probability"] * 100).round(1)
            st.dataframe(table.rename(columns={"Effect on churn probability": "Effect (pp)"}), hide_index=True, width="stretch")


def tab_batch(threshold: float) -> None:
    model = get_model()
    st.markdown(
        "Upload a CSV of customers to score them all at once. The columns must match the training data "
        "(`customerID` is optional and `Churn`, if present, is used only to measure accuracy)."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    uploaded = c1.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if c2.button("Score the bundled sample", width="stretch"):
        st.session_state["batch_df"] = pd.read_csv(SAMPLE_FILE)
        st.session_state["batch_name"] = "bundled sample (200 customers)"
    c3.download_button("Download template", SAMPLE_FILE.read_bytes(), "customer_template.csv", "text/csv", width="stretch")

    if uploaded is not None:
        try:
            st.session_state["batch_df"] = pd.read_csv(uploaded)
            st.session_state["batch_name"] = uploaded.name
        except Exception as exc:  # unreadable or not a CSV
            st.error(f"Could not read that file: {exc}")
            return

    df = st.session_state.get("batch_df")
    if df is None:
        st.info("Upload a CSV or score the bundled sample to get started.")
        return

    errors, warnings = validate_batch(df)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    scored = score_frame(model, df, threshold).sort_values("churn_probability", ascending=False).reset_index(drop=True)
    st.subheader(f"Results: {st.session_state['batch_name']}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{len(scored):,}")
    k2.metric("Predicted to churn", f"{(scored['predicted_churn'] == 'Yes').sum():,}")
    k3.metric("High risk", f"{(scored['risk_band'] == 'High').sum():,}")
    k4.metric("Average probability", pct(scored["churn_probability"].mean(), 1))

    st.altair_chart(charts.prob_hist(scored, threshold), width="stretch")

    if "Churn" in scored.columns:
        truth = (scored["Churn"].astype(str).str.lower() == "yes").astype(int)
        m = metrics_at(truth, scored["churn_probability"], threshold)
        st.info(
            f"The file contains actual outcomes. At this threshold: accuracy {pct(m['accuracy'], 1)}, precision {pct(m['precision'], 1)}, "
            f"recall {pct(m['recall'], 1)}. If these customers were part of the training data the numbers will look optimistic."
        )

    bands = st.multiselect("Show risk bands", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    view = scored[scored["risk_band"].isin(bands)]
    lead = [c for c in ["customerID", "churn_probability", "predicted_churn", "risk_band"] if c in view.columns]
    st.dataframe(
        view[lead + [c for c in view.columns if c not in lead]],
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent", min_value=0, max_value=1)
        },
    )
    st.download_button(
        "Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "scored_customers.csv", "text/csv", type="primary"
    )


def tab_explore() -> None:
    df = get_data().copy()
    df["churned"] = (df["Churn"] == "Yes").astype(int)
    stats = describe(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{stats['customers']:,}")
    k2.metric("Churn rate", pct(stats["churn_rate"], 1))
    k3.metric("Average tenure", f"{stats['avg_tenure']:.1f} months")
    k4.metric("Average monthly charge", f"${stats['avg_monthly']:.2f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Churn rate by category")
        categorical = [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
            "OnlineSecurity",
            "PaperlessBilling",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "gender",
        ]
        column = st.selectbox("Feature", categorical)
        st.altair_chart(charts.rate_chart(df, column), width="stretch")
        st.caption("The dashed red line is the overall churn rate.")
    with right:
        st.subheader("Distribution by outcome")
        numeric = st.selectbox("Numeric feature", NUMERIC)
        st.altair_chart(charts.hist_chart(df, numeric), width="stretch")
        st.caption("Overlaid histograms: customers who churned versus those who stayed.")

    with st.expander("Preview the data"):
        st.dataframe(get_data().head(200), hide_index=True, width="stretch")


def tab_performance(threshold: float) -> None:
    model = get_model()
    m = model.metrics_at(threshold)
    s = model.summary

    st.markdown(
        f"Evaluated on **{s['test_rows']:,} held-out customers** (20% stratified split) the model never saw during training."
    )
    k = st.columns(6)
    k[0].metric("Accuracy", pct(m["accuracy"], 1))
    k[1].metric("Precision", pct(m["precision"], 1), help="Of customers flagged as churners, the share who really churned.")
    k[2].metric("Recall", pct(m["recall"], 1), help="Of customers who really churned, the share we caught.")
    k[3].metric("F1", pct(m["f1"], 1))
    k[4].metric("ROC-AUC", f"{s['roc_auc']:.3f}", help="Threshold-independent ranking quality (0.5 = random, 1.0 = perfect).")
    k[5].metric("PR-AUC", f"{s['pr_auc']:.3f}")
    st.caption(
        f"At threshold {pct(threshold)}. Always predicting 'no churn' would already score {pct(model.majority_baseline, 1)} accuracy, "
        "so look at precision and recall for the churn class, not accuracy alone."
    )

    c1, c2 = st.columns(2, gap="large")
    c1.altair_chart(charts.confusion_chart(m), width="stretch")
    c2.altair_chart(charts.roc_chart(model.roc_points(), s["roc_auc"]), width="stretch")

    st.altair_chart(charts.threshold_chart(threshold_table(model.y_test, model.proba_test), threshold), width="stretch")
    st.caption(
        "Moving the decision threshold trades precision for recall. Pick it from the cost of a missed churner versus a wasted retention offer."
    )

    st.subheader("What matters most?")
    st.altair_chart(charts.importance_chart(get_importance()), width="stretch")
    st.caption("Permutation importance on the test set: how much ROC-AUC falls when a column's values are shuffled.")

    st.subheader("Compared with simpler models")
    st.dataframe(get_comparison(), hide_index=True, width="stretch")
    st.caption(
        "Same split, same preprocessing, threshold 0.5. On tabular data like this, simple models are often as good as a neural network."
    )

    st.subheader("Robustness")
    if st.button("Run 5-fold cross-validation"):
        st.session_state["cv"] = get_cv()
    if "cv" in st.session_state:
        cv = st.session_state["cv"].copy()
        cv["Result"] = [f"{mean:.3f} ± {std:.3f}" for mean, std in zip(cv["Mean"], cv["Std"])]
        st.dataframe(cv[["Metric", "Result"]], hide_index=True, width="stretch")
        st.caption("Mean ± standard deviation across 5 stratified folds on the full dataset.")


def tab_about() -> None:
    st.markdown(
        f"""
**{APP_NAME}** predicts which telecom customers are likely to leave, so retention teams can focus on the right people.

**Model**

- Feed-forward neural network: 26 inputs → 26 → 15 → 1 (ReLU, sigmoid output), trained with Adam and early stopping.
- Inputs: 19 customer attributes. Numeric columns are min-max scaled, categorical columns one-hot encoded, Yes/No columns mapped to 1/0.
- 7,043 customers, about 27% churn. Stratified 80/20 split; the scaler is fitted on the training split only.

**Data dictionary**

| Column | Meaning |
|---|---|
| `tenure` | Months as a customer |
| `Contract` | Month-to-month, one year or two year |
| `InternetService` | DSL, fiber optic or none |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | Add-on services |
| `StreamingTV`, `StreamingMovies` | Streaming add-ons |
| `PaperlessBilling`, `PaymentMethod` | Billing preferences |
| `MonthlyCharges`, `TotalCharges` | Current monthly bill and lifetime billed amount |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographics |
| `Churn` | Whether the customer left (target, training only) |

**Good practice**

- Probabilities rank customers; they do not explain *why* someone will leave. The driver chart is a model-based approximation, not a causal claim.
- `gender` and `SeniorCitizen` are model inputs because they are in the dataset. Review whether using them is appropriate before acting on predictions in a real business.
- Check the **Model performance** tab: recall for churners is moderate at the default threshold. Tune the threshold to your retention budget.
        """
    )


def main() -> None:
    get_model()  # train once up front, with a spinner
    threshold = sidebar()
    st.title("Customer Churn Prediction")
    st.caption("Estimate churn risk for one customer or a whole file, and see how the model performs.")

    predict, batch, explore, performance, about = st.tabs(
        ["Predict", "Batch scoring", "Data explorer", "Model performance", "About"]
    )
    with predict:
        tab_predict(threshold)
    with batch:
        tab_batch(threshold)
    with explore:
        tab_explore()
    with performance:
        tab_performance(threshold)
    with about:
        tab_about()


main()
