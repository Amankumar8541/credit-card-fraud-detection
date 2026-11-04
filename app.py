
import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")
st.caption("Suspicious Transaction Detection Platform | ML-powered fraud analysis")

MODEL_PATH = "artifacts/fraud_model.pkl"
DEMO_PATH = "artifacts/demo_transactions.csv"
METRICS_PATH = "artifacts/metrics.json"

if not all(os.path.exists(p) for p in [
    MODEL_PATH, DEMO_PATH, METRICS_PATH
]):
    st.error("Model files not found. Please run: python train.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

model = load_model()
demo = load_demo()
metrics = load_metrics()

# Dashboard metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Demo Transactions", len(demo))
c2.metric("Fraud Cases in Demo", int(demo["Class"].sum()))
c3.metric("Precision", f'{metrics["precision"]:.2%}')
c4.metric("Recall", f'{metrics["recall"]:.2%}')

st.divider()

tab1, tab2, tab3 = st.tabs([
    "Transaction Detection",
    "Batch Prediction",
    "Model Performance"
])

with tab1:
    st.subheader("Analyze a Transaction")

    st.info(
        "Choose a transaction from the real test dataset "
        "to demonstrate fraud detection."
    )

    transaction_id = st.selectbox(
        "Select Transaction ID",
        range(len(demo))
    )

    row = demo.iloc[[transaction_id]]
    actual = int(row["Class"].iloc[0])

    X = row.drop(columns=["Class"])

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    st.write(f"Transaction Amount: **€{X['Amount'].iloc[0]:.2f}**")

    left, right = st.columns(2)

    left.metric("Fraud Probability", f"{probability:.2%}")
    right.metric("Model Prediction",
                 "FRAUD" if prediction == 1 else "LEGITIMATE")

    if prediction == 1:
        st.error("⚠️ Suspicious transaction detected!")
    else:
        st.success("✅ Transaction classified as legitimate.")

    if actual == 1:
        st.warning("Actual dataset label: Fraud")
    else:
        st.info("Actual dataset label: Legitimate")

    st.caption(
        "This is an offline demonstration using held-out "
        "historical transactions, not a live bank connection."
    )

with tab2:
    st.subheader("Batch Transaction Analysis")

    st.write(
        "Upload a CSV containing Time, V1-V28 and Amount columns. "
        "Do not include the Class column."
    )

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            expected = list(demo.drop(columns=["Class"]).columns)

            if set(batch.columns) != set(expected):
                st.error("CSV columns do not match the required schema.")
            else:
                batch = batch[expected]

                if batch.isnull().any().any():
                    st.error("CSV contains missing values.")
                elif not all(
                    pd.api.types.is_numeric_dtype(batch[c])
                    for c in expected
                ):
                    st.error("All transaction features must be numeric.")
                else:
                    predictions = model.predict(batch)
                    probabilities = model.predict_proba(batch)[:, 1]

                    results = batch[["Amount"]].copy()
                    results["Prediction"] = [
                        "Fraud" if p == 1 else "Legitimate"
                        for p in predictions
                    ]
                    results["Fraud Probability"] = probabilities

                    st.dataframe(results, use_container_width=True)

                    st.metric(
                        "Suspicious Transactions",
                        int((predictions == 1).sum())
                    )

                    st.download_button(
                        "Download Results",
                        results.to_csv(index=False),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Could not process CSV: {e}")

with tab3:
    st.subheader("Model Evaluation")

    a, b, c = st.columns(3)

    a.metric("Precision", f'{metrics["precision"]:.2%}')
    b.metric("Recall", f'{metrics["recall"]:.2%}')
    c.metric("F1 Score", f'{metrics["f1_score"]:.2%}')

    st.metric("PR-AUC", f'{metrics["pr_auc"]:.4f}')

    st.write("Confusion Matrix")

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(cm, use_container_width=True)

    st.caption(
        "Metrics are calculated on the held-out test set. "
        "PR-AUC is particularly useful for imbalanced fraud data."
    )

st.divider()
st.caption("Academic MLOps Prototype | Credit Card Fraud Detection")
