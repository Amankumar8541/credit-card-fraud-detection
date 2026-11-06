import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import os
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
DEMO_PATH = BASE_DIR / "artifacts" / "demo_transactions.csv"

# Local API default. For Streamlit Cloud, set FRAUDGUARD_API_URL
# to your publicly deployed FastAPI service URL.
API_URL = st.secrets.get(
    "FRAUDGUARD_API_URL",
    os.getenv("FRAUDGUARD_API_URL", "http://127.0.0.1:8002")
).rstrip("/")

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# =====================================================
# DARK FINTECH DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0b1020;
    color: #e8edf7;
}
[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.9);
}
[data-testid="stSidebar"] {
    background: #10182b;
    border-right: 1px solid #26334d;
}
[data-testid="stSidebar"] * {
    color: #e8edf7;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
.hero {
    background: linear-gradient(120deg, #172641, #111a30);
    border: 1px solid #293b5d;
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 24px;
}
.hero-label {
    color: #9db4d9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin: 8px 0;
}
.hero p {
    color: #aab9d2;
    margin: 0;
    font-size: 14px;
}
.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #f1f5ff;
    margin: 20px 0 14px 0;
}
.metric-card {
    background: #141f35;
    border: 1px solid #293954;
    border-radius: 15px;
    padding: 19px;
    min-height: 118px;
}
.metric-label {
    color: #9aabc7;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}
.metric-value {
    color: #f5f7ff;
    font-size: 28px;
    font-weight: 800;
}
.metric-note {
    color: #8093b2;
    font-size: 11px;
    margin-top: 5px;
}
.status-fraud {
    background: #351b2a;
    border: 1px solid #87364f;
    color: #ff718b;
    padding: 14px 17px;
    border-radius: 12px;
    font-weight: 700;
}
.status-safe {
    background: #123329;
    border: 1px solid #24684f;
    color: #61dfa7;
    padding: 14px 17px;
    border-radius: 12px;
    font-weight: 700;
}
.info-card {
    background: #141f35;
    border: 1px solid #293954;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}
.muted {
    color: #9aabc7;
    font-size: 13px;
}
div.stButton > button {
    background: #e84b62;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 700;
    min-height: 43px;
}
div.stButton > button:hover {
    background: #ff6378;
    color: white;
    border: none;
}
div.stDownloadButton > button {
    background: #1d3557;
    color: white;
    border: 1px solid #42628d;
    border-radius: 9px;
    font-weight: 600;
}
div.stDownloadButton > button:hover {
    background: #294b78;
    color: white;
    border-color: #5b83b5;
}
[data-testid="stFileUploader"] {
    background: #141f35;
    border: 1px dashed #42628d;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stMetric"] {
    background: #141f35;
    border: 1px solid #293954;
    border-radius: 12px;
    padding: 15px;
}
[data-testid="stMetricLabel"] {
    color: #9aabc7;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
hr {
    border-color: #293954;
}
.sidebar-brand {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    padding: 10px 0 2px 0;
}
.sidebar-sub {
    font-size: 11px;
    color: #91a4c4;
    letter-spacing: 1px;
    margin-bottom: 25px;
}
.footer {
    color: #7183a2;
    font-size: 12px;
    text-align: center;
    padding-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD PROJECT FILES
# =====================================================

@st.cache_data
def load_demo_data():
    return pd.read_csv(DEMO_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

try:
    demo = load_demo_data()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.info(
        "Check that demo_transactions.csv and metrics.json "
        "exist inside the artifacts folder."
    )
    st.stop()

# =====================================================
# API HELPER FUNCTIONS
# =====================================================

def get_predictions(data):
    """Send one or more transaction rows to the FastAPI backend."""
    if data.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    records = data[FEATURE_COLUMNS].to_dict(orient="records")

    try:
        if len(records) == 1:
            response = requests.post(
                f"{API_URL}/predict",
                json=records[0],
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            prediction = 1 if result["prediction"] == "Fraud" else 0
            probability = float(result["fraud_probability"])

            return (
                np.array([prediction], dtype=int),
                np.array([probability], dtype=float)
            )

        response = requests.post(
            f"{API_URL}/predict/batch",
            json=records,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        predictions = np.array([
            1 if item["prediction"] == "Fraud" else 0
            for item in result["results"]
        ], dtype=int)

        probabilities = np.array([
            float(item["fraud_probability"])
            for item in result["results"]
        ], dtype=float)

        return predictions, probabilities

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FraudGuard API. Start FastAPI in another "
            "terminal using: python3 -m uvicorn api:app --reload --port 8002"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The API request timed out. Try a smaller CSV file.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        st.stop()
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Unexpected API response: {e}")
        st.stop()

def validate_csv(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            return None, "The uploaded CSV file is empty."

        # Class column is optional and is not sent to the prediction API.
        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
        extra = [col for col in data.columns if col not in FEATURE_COLUMNS]

        if missing:
            return None, "Missing columns: " + ", ".join(missing)

        if extra:
            return None, "Unexpected columns: " + ", ".join(extra)

        data = data[FEATURE_COLUMNS].copy()

        for col in FEATURE_COLUMNS:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        if data.isnull().values.any():
            return None, "CSV contains empty or non-numeric values."

        if not np.isfinite(data.to_numpy()).all():
            return None, "CSV contains invalid or infinite values."

        return data, None

    except Exception as e:
        return None, f"Unable to read CSV: {e}"

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_prediction(prediction, probability):
    if prediction == 1:
        st.markdown(
            '<div class="status-fraud">⚠️ FRAUDULENT TRANSACTION DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-safe">✓ TRANSACTION CLASSIFIED AS LEGITIMATE</div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.metric("Fraud Probability", f"{probability * 100:.2f}%")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🛡️ FraudGuard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-sub">INTELLIGENT FRAUD MONITORING</div>',
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Transaction Detection",
            "📂 Batch Analysis",
            "📊 Model Performance"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**API STATUS**")

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        health_data = health_response.json()
        if health_data.get("status") == "healthy":
            st.markdown(
                '<div class="status-safe">● FraudGuard API Connected</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("API responded, but health status is not healthy.")
    except requests.exceptions.RequestException:
        st.markdown(
            '<div class="status-fraud">● API Disconnected</div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Backend: FastAPI")
    st.caption("Frontend: Streamlit")
    st.caption("Academic MLOps Prototype")

# =====================================================
# COMMON HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Intelligent Transaction Security</div>
        <h1>Credit Card Fraud Detection</h1>
        <p>Monitor transactions, identify suspicious activity
        and analyze fraud predictions with machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":
    st.markdown(
        '<div class="section-title">Overview Dashboard</div>',
        unsafe_allow_html=True
    )

    demo_features = demo[FEATURE_COLUMNS]

    with st.spinner("Fetching demo predictions from FraudGuard API..."):
        demo_predictions, demo_probabilities = get_predictions(demo_features)

    total_demo = len(demo)
    fraud_demo = int(np.sum(demo_predictions == 1))
    legitimate_demo = int(np.sum(demo_predictions == 0))

    fraud_rate = fraud_demo / total_demo * 100 if total_demo else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("DEMO TRANSACTIONS", f"{total_demo:,}", "Sample test records")
    with c2:
        metric_card("FRAUD DETECTED", f"{fraud_demo:,}", "Model predictions")
    with c3:
        metric_card("LEGITIMATE", f"{legitimate_demo:,}", "Model predictions")
    with c4:
        metric_card("DEMO FRAUD RATE", f"{fraud_rate:.2f}%", "Predicted fraud share")

    st.markdown(
        '<div class="section-title">Transaction Distribution</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Transaction Type": ["Fraud", "Legitimate"],
        "Count": [fraud_demo, legitimate_demo]
    })
    st.bar_chart(chart_data.set_index("Transaction Type"), width="stretch")

    st.markdown(
        '<div class="section-title">Model Snapshot</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{float(metrics.get('precision', 0)) * 100:.2f}%")
    m2.metric("Recall", f"{float(metrics.get('recall', 0)) * 100:.2f}%")
    m3.metric("F1 Score", f"{float(metrics.get('f1_score', 0)) * 100:.2f}%")

    st.info(
        "Demo statistics are predictions on the sample transactions. "
        "They are not confirmed real-world fraud cases."
    )

# =====================================================
# SINGLE TRANSACTION DETECTION
# =====================================================

elif page == "🔍 Transaction Detection":
    st.markdown(
        '<div class="section-title">Single Transaction Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Select a historical sample transaction '
        'to check its predicted fraud risk.</div>',
        unsafe_allow_html=True
    )

    transaction_index = st.selectbox(
        "Select Transaction ID",
        options=list(range(len(demo))),
        format_func=lambda x: f"Transaction {x}"
    )

    selected = demo.iloc[[transaction_index]]
    input_data = selected[FEATURE_COLUMNS]
    amount = float(selected["Amount"].iloc[0])

    a, b = st.columns(2)
    with a:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="muted">TRANSACTION ID</div>
                <h2>{transaction_index}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    with b:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="muted">TRANSACTION AMOUNT</div>
                <h2>{amount:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("View Transaction Features"):
        st.dataframe(input_data, width="stretch")

    if st.button("🔍 Analyze Transaction", key="single_prediction"):
        with st.spinner("Sending transaction to FraudGuard API..."):
            predictions, probabilities = get_predictions(input_data)

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True
        )
        show_prediction(prediction, probability)

        if "Class" in selected.columns:
            actual = int(selected["Class"].iloc[0])
            st.info(
                "Actual dataset label: "
                + ("Fraud" if actual == 1 else "Legitimate")
            )

    st.caption(
        "This is an offline demonstration using historical test "
        "transactions, not a live bank connection."
    )

# =====================================================
# BATCH ANALYSIS
# =====================================================

elif page == "📂 Batch Analysis":
    st.markdown(
        '<div class="section-title">Batch Transaction Analysis</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Upload a CSV file containing transaction features. "
        "The existing trained model will analyze the uploaded data through the API."
    )
    st.info(
        "Required columns: Time, V1 to V28, Amount. "
        "Class is optional. This model requires the anonymized "
        "features used by the Credit Card Fraud dataset."
    )

    sample_csv = demo[FEATURE_COLUMNS].head(10).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Sample CSV",
        data=sample_csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if uploaded_file is not None:
        batch_data, error = validate_csv(uploaded_file)

        if error:
            st.error(error)
        else:
            st.success(
                f"CSV uploaded successfully! {len(batch_data)} transactions found."
            )

            with st.expander("Preview Uploaded Dataset"):
                st.dataframe(batch_data.head(10), width="stretch")

            if st.button("🚀 Analyze Transactions", key="batch_prediction"):
                with st.spinner("Analyzing transactions through FraudGuard API..."):
                    predictions, probabilities = get_predictions(batch_data)

                    results = batch_data.copy()
                    results["Prediction"] = [
                        "Fraud" if p == 1 else "Legitimate"
                        for p in predictions
                    ]
                    results["Fraud Probability (%)"] = (
                        probabilities * 100
                    ).round(2)

                    total = len(results)
                    fraud_count = int(np.sum(predictions == 1))
                    legitimate_count = int(np.sum(predictions == 0))
                    fraud_percentage = (
                        fraud_count / total * 100 if total else 0
                    )

                st.markdown(
                    '<div class="section-title">Analysis Summary</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("TOTAL TRANSACTIONS", f"{total:,}")
                with c2:
                    metric_card("FRAUD DETECTED", f"{fraud_count:,}")
                with c3:
                    metric_card("LEGITIMATE", f"{legitimate_count:,}")
                with c4:
                    metric_card("FRAUD PERCENTAGE", f"{fraud_percentage:.2f}%")

                st.markdown(
                    '<div class="section-title">Transaction Distribution</div>',
                    unsafe_allow_html=True
                )
                distribution = pd.DataFrame({
                    "Transaction Type": ["Fraud", "Legitimate"],
                    "Count": [fraud_count, legitimate_count]
                })
                st.bar_chart(
                    distribution.set_index("Transaction Type"),
                    width="stretch"
                )

                st.markdown(
                    '<div class="section-title">Prediction Results</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(results, width="stretch", height=400)

                output_csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Prediction Report",
                    data=output_csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Fraud labels are model predictions, "
                    "not confirmed fraud findings."
                )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "📊 Model Performance":
    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    if metrics:
        precision = float(metrics.get("precision", 0))
        recall = float(metrics.get("recall", 0))
        f1 = float(metrics.get("f1_score", 0))
        pr_auc = float(metrics.get("pr_auc", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{precision * 100:.2f}%")
        c2.metric("Recall", f"{recall * 100:.2f}%")
        c3.metric("F1 Score", f"{f1 * 100:.2f}%")
        c4.metric("PR-AUC", f"{pr_auc:.4f}")

        confusion_matrix = metrics.get("confusion_matrix")

        if confusion_matrix is not None:
            st.markdown(
                '<div class="section-title">Confusion Matrix</div>',
                unsafe_allow_html=True
            )

            if isinstance(confusion_matrix, list) and len(confusion_matrix) == 2:
                cm_df = pd.DataFrame(
                    confusion_matrix,
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=["Predicted Legitimate", "Predicted Fraud"]
                )
                st.dataframe(cm_df, width="stretch")

        with st.expander("View Complete Model Metrics"):
            st.json(metrics)
    else:
        st.warning("Metrics file not found or contains no metrics.")

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-card">
        <b>Algorithm:</b> Logistic Regression<br><br>
        <b>Preprocessing:</b> StandardScaler<br><br>
        <b>Class Imbalance:</b> Balanced Class Weights<br><br>
        <b>Model Storage:</b> Joblib (loaded by FastAPI backend)<br><br>
        <b>Frontend:</b> Streamlit<br><br>
        <b>Backend:</b> FastAPI
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "The model has high recall but low precision. "
        "It can detect many fraud cases but may also flag "
        "legitimate transactions as suspicious."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        FraudGuard • Credit Card Fraud Detection<br>
        Python · Scikit-learn · FastAPI · Streamlit
        <br><br>
        Academic prototype — not connected to live banking systems.
    </div>
    """,
    unsafe_allow_html=True
)
