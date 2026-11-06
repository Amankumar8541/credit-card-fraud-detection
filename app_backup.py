
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
DEMO_PATH = BASE_DIR / "artifacts" / "demo_transactions.csv"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

# ---------------- LOAD MODEL AND DATA ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_demo_data():
    return pd.read_csv(DEMO_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return {}


try:
    model = load_model()
    demo = load_demo_data()
    metrics = load_metrics()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.info(
        "Please check that fraud_model.pkl and "
        "demo_transactions.csv exist inside the artifacts folder."
    )
    st.stop()


# ---------------- HELPER FUNCTIONS ----------------

def get_predictions(data):
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    return predictions, probabilities


def show_prediction(prediction, probability):
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )


def validate_csv(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            return None, "The uploaded CSV file is empty."

        # Allow an optional actual Class column
        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        missing_columns = [
            col for col in FEATURE_COLUMNS
            if col not in data.columns
        ]

        extra_columns = [
            col for col in data.columns
            if col not in FEATURE_COLUMNS
        ]

        if missing_columns:
            return None, (
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        if extra_columns:
            return None, (
                "Unexpected columns found: "
                + ", ".join(extra_columns)
            )

        data = data[FEATURE_COLUMNS].copy()

        for col in FEATURE_COLUMNS:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        if data.isnull().values.any():
            return None, (
                "CSV contains empty or non-numeric values. "
                "Please correct them."
            )

        if not np.isfinite(data.to_numpy()).all():
            return None, (
                "CSV contains infinite or invalid numeric values."
            )

        return data, None

    except Exception as e:
        return None, f"Unable to read CSV: {e}"


# ---------------- HEADER ----------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    An ML-powered platform to identify suspicious credit card
    transactions using Logistic Regression.
    
    Upload your transaction dataset, analyze fraud predictions,
    and download the results.
    """
)

st.divider()

# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Detection",
    "📂 Batch Prediction",
    "📊 Model Performance"
])


# ====================================================
# TAB 1: TRANSACTION DETECTION
# ====================================================

with tab1:

    st.subheader("Single Transaction Detection")

    st.write(
        "Select a sample transaction from the test dataset "
        "to check whether it is fraudulent or legitimate."
    )

    transaction_index = st.selectbox(
        "Select Transaction",
        options=list(range(len(demo))),
        format_func=lambda x: f"Transaction {x + 1}"
    )

    selected_transaction = demo.iloc[[transaction_index]]

    input_data = selected_transaction[FEATURE_COLUMNS]

    with st.expander("View Transaction Details"):
        st.dataframe(
            input_data,
            width="stretch"
        )

    if st.button(
        "Analyze Transaction",
        type="primary",
        key="single_prediction"
    ):

        prediction, probability = get_predictions(input_data)

        st.subheader("Prediction Result")

        show_prediction(
            int(prediction[0]),
            float(probability[0])
        )

        if "Class" in selected_transaction.columns:

            actual = int(
                selected_transaction["Class"].iloc[0]
            )

            st.write(
                "**Actual Dataset Label:**",
                "Fraud" if actual == 1 else "Legitimate"
            )


# ====================================================
# TAB 2: BATCH PREDICTION
# ====================================================

with tab2:

    st.subheader("Upload Your Own Transaction Dataset")

    st.write(
        "Upload a CSV containing the required transaction "
        "features. The existing trained model will analyze "
        "your uploaded transactions."
    )

    st.info(
        "Required columns: Time, V1 to V28, Amount. "
        "The Class column is optional. This model uses the "
        "anonymized features of the Credit Card Fraud dataset."
    )

    # Download sample CSV

    sample_csv = demo[FEATURE_COLUMNS].head(10).to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_data, error = validate_csv(uploaded_file)

        if error:
            st.error(error)

        else:

            st.success(
                f"CSV uploaded successfully! "
                f"{len(batch_data)} transactions found."
            )

            with st.expander("Preview Uploaded Dataset"):
                st.dataframe(
                    batch_data.head(10),
                    width="stretch"
                )

            if st.button(
                "🚀 Analyze Transactions",
                type="primary",
                key="batch_prediction"
            ):

                with st.spinner("Analyzing transactions..."):

                    predictions, probabilities = get_predictions(
                        batch_data
                    )

                    results = batch_data.copy()

                    results["Prediction"] = [
                        "Fraud" if p == 1 else "Legitimate"
                        for p in predictions
                    ]

                    results["Fraud Probability (%)"] = (
                        probabilities * 100
                    ).round(2)

                    total_transactions = len(results)

                    fraud_count = int(
                        np.sum(predictions == 1)
                    )

                    legitimate_count = int(
                        np.sum(predictions == 0)
                    )

                    fraud_percentage = (
                        fraud_count / total_transactions
                    ) * 100

                    st.subheader("Analysis Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Total Transactions",
                        total_transactions
                    )

                    col2.metric(
                        "Fraud Detected",
                        fraud_count
                    )

                    col3.metric(
                        "Legitimate",
                        legitimate_count
                    )

                    col4.metric(
                        "Fraud Percentage",
                        f"{fraud_percentage:.2f}%"
                    )

                    st.divider()

                    st.subheader("Transaction Distribution")

                    chart_data = pd.DataFrame({
                        "Transaction Type": [
                            "Fraud",
                            "Legitimate"
                        ],
                        "Count": [
                            fraud_count,
                            legitimate_count
                        ]
                    })

                    st.bar_chart(
                        chart_data.set_index("Transaction Type"),
                        width="stretch"
                    )

                    st.divider()

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        width="stretch"
                    )

                    # Download prediction report

                    output_csv = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction Report",
                        data=output_csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )


# ====================================================
# TAB 3: MODEL PERFORMANCE
# ====================================================

with tab3:

    st.subheader("Model Performance Dashboard")

    st.write(
        "Performance metrics of the trained "
        "Credit Card Fraud Detection model."
    )

    if metrics:

        metric_columns = st.columns(4)

        preferred_metrics = [
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1"),
            ("PR-AUC", "pr_auc")
        ]

        for column, (label, key) in zip(
            metric_columns,
            preferred_metrics
        ):

            value = metrics.get(key)

            if value is not None:
                column.metric(
                    label,
                    f"{float(value):.4f}"
                )

        # Confusion matrix if available

        confusion_matrix = metrics.get("confusion_matrix")

        if confusion_matrix is not None:

            st.divider()
            st.subheader("Confusion Matrix")

            if (
                isinstance(confusion_matrix, list)
                and len(confusion_matrix) == 2
            ):

                cm_df = pd.DataFrame(
                    confusion_matrix,
                    index=[
                        "Actual Legitimate",
                        "Actual Fraud"
                    ],
                    columns=[
                        "Predicted Legitimate",
                        "Predicted Fraud"
                    ]
                )

                st.dataframe(
                    cm_df,
                    width="stretch"
                )

        with st.expander("View Complete Metrics"):
            st.json(metrics)

    else:

        st.warning(
            "Model metrics file was not found or is empty."
        )

    st.divider()

    st.subheader("About the Model")

    st.markdown(
        """
        - **Algorithm:** Logistic Regression
        - **Dataset:** Credit Card Fraud Detection Dataset
        - **Preprocessing:** StandardScaler
        - **Class Imbalance Handling:** Balanced Class Weights
        - **Model Storage:** Joblib
        - **Deployment:** Streamlit Community Cloud
        """
    )

    st.caption(
        "Note: This is an academic prototype. "
        "It predicts uploaded transaction data using an existing "
        "trained model. It does not connect to live banking systems "
        "or automatically retrain the model."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Credit Card Fraud Detection System | "
    "Python • Scikit-learn • Streamlit"
)
