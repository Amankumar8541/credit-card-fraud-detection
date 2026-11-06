
from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create API
app = FastAPI(
    title="FraudGuard API",
    description="Credit Card Fraud Detection System",
    version="1.0.0"
)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.pkl"

model = joblib.load(MODEL_PATH)

# Required features
FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# Input validation
class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# Prediction function
def get_prediction(transaction):

    data = transaction.model_dump()

    df = pd.DataFrame([data], columns=FEATURES)

    prediction = int(model.predict(df)[0])

    classes = list(model.classes_)
    fraud_index = classes.index(1)

    probability = float(
        model.predict_proba(df)[0][fraud_index]
    )

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(probability, 4)
    }


# Home API
@app.get("/")
def home():
    return {
        "message": "Welcome to FraudGuard API",
        "status": "running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Single prediction
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        return {
            "status": "success",
            **get_prediction(transaction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch prediction
@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    try:
        results = []

        for index, transaction in enumerate(transactions):
            results.append({
                "transaction_id": index,
                **get_prediction(transaction)
            })

        fraud_count = sum(
            item["prediction"] == "Fraud"
            for item in results
        )

        return {
            "status": "success",
            "total_transactions": len(results),
            "fraud_count": fraud_count,
            "legitimate_count": len(results) - fraud_count,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
