# 🛡️ FraudGuard – Intelligent Credit Card Fraud Detection System

An ML-powered fraud detection platform that identifies potentially fraudulent credit card transactions using Logistic Regression, FastAPI, and Streamlit.

<p align="center">
  <a href="https://credit-card-fraud-detection-pjpvqnn3goslg2yotacb7.streamlit.app/">🚀 Live Dashboard</a> •
  <a href="https://credit-card-fraud-detection-a3nd.onrender.com/">⚡ Live API</a> •
  <a href="https://credit-card-fraud-detection-a3nd.onrender.com/docs">📖 API Documentation</a>
</p>

---

## 📌 Project Overview

FraudGuard is a machine learning-based web application designed to detect potentially fraudulent credit card transactions.

It uses a trained Logistic Regression model to classify transactions as **Fraudulent** or **Legitimate**.

The platform combines a Streamlit frontend with a FastAPI backend, allowing users to analyze individual transactions, upload CSV datasets, perform batch predictions, view model evaluation metrics, and download prediction reports.

The backend exposes REST API endpoints so that other applications and platforms can integrate with the fraud detection model.

## ✨ Features

- 📊 Interactive dark-themed dashboard
- 🔍 Individual transaction fraud detection
- 📁 Upload custom transaction CSV files
- ⚡ Batch transaction prediction through FastAPI
- 🔗 REST API integration
- 📈 Fraud and legitimate transaction statistics
- 📥 Download prediction reports
- 📉 Model evaluation and confusion matrix
- 📌 Precision, Recall, F1 Score and PR-AUC metrics
- ☁️ Cloud deployment using Streamlit Community Cloud and Render
- ❤️ API health monitoring

## 🌐 Live Deployment

| Component | Platform | Link |
|---|---|---|
| Frontend Dashboard | Streamlit Cloud | [Open Dashboard](https://credit-card-fraud-detection-pjpvqnn3goslg2yotacb7.streamlit.app/) |
| Backend API | Render | [Open API](https://credit-card-fraud-detection-a3nd.onrender.com/) |
| API Documentation | FastAPI Swagger UI | [Explore Endpoints](https://credit-card-fraud-detection-a3nd.onrender.com/docs) |
| Health Check | FastAPI | [Check API Status](https://credit-card-fraud-detection-a3nd.onrender.com/health) |

## 🏗️ System Architecture

FraudGuard follows a frontend-backend architecture in which Streamlit communicates with the FastAPI backend through HTTP requests.

```text
                USER
                  |
                  v
        STREAMLIT FRONTEND
         (Dashboard / UI)
                  |
                  | HTTP Requests
                  v
           FASTAPI BACKEND
             (REST API)
                  |
          +-------+-------+
          |               |
          v               v
     Single Prediction  Batch Prediction
          |               |
          +-------+-------+
                  |
                  v
          TRAINED ML MODEL
         (Logistic Regression)
                  |
                  v
          PREDICTION RESULTS
                  |
                  v
        STREAMLIT DASHBOARD
```

**Deployment:**
- Frontend: Streamlit Community Cloud
- Backend: Render Web Service
- Model: Scikit-learn Pipeline stored using Joblib

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Logistic Regression | Classification model |
| StandardScaler | Feature scaling |
| Joblib | Model persistence |
| FastAPI | Backend REST API |
| Uvicorn | ASGI server |
| Pydantic | Request data validation |
| Requests | HTTP communication |
| Streamlit | Frontend web application |
| Git & GitHub | Version control |
| Render | Backend deployment |
| Streamlit Community Cloud | Frontend deployment |

## 🤖 Machine Learning Model

- **Algorithm:** Logistic Regression
- **Dataset:** Credit Card Fraud Detection Dataset (ULB)
- **Class imbalance handling:** `class_weight="balanced"`
- **Feature scaling:** StandardScaler
- **Model pipeline:** Scikit-learn Pipeline
- **Model storage:** Joblib (`fraud_model.pkl`)

The model uses 30 input features:
- `Time`
- `V1` to `V28`
- `Amount`

The output classifies a transaction as Fraud or Legitimate and returns its estimated fraud probability.

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Precision | 6.10% |
| Recall | 91.84% |
| F1 Score | 11.44% |
| PR-AUC | 0.7190 |

The model achieves high recall, detecting most fraud cases in the test dataset. However, its precision is low, meaning it also flags many legitimate transactions as potentially fraudulent.

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 55,478 | 1,386 |
| Actual Fraud | 8 | 90 |

## 🔌 API Endpoints

The FastAPI backend provides the following REST endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict a single transaction |
| POST | `/predict/batch` | Predict multiple transactions |

### 1. API Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Single Transaction Prediction

```http
POST /predict
```

Accepts one transaction containing `Time`, `V1`–`V28`, and `Amount`.

Example response:

```json
{
  "status": "success",
  "prediction": "Legitimate",
  "fraud_probability": 0.0182
}
```

### 3. Batch Transaction Prediction

```http
POST /predict/batch
```

Accepts a JSON array of transaction records.

Example response:

```json
{
  "status": "success",
  "total_transactions": 2,
  "fraud_count": 1,
  "legitimate_count": 1,
  "results": [
    {
      "transaction_id": 0,
      "prediction": "Fraud",
      "fraud_probability": 0.85
    },
    {
      "transaction_id": 1,
      "prediction": "Legitimate",
      "fraud_probability": 0.02
    }
  ]
}
```

*The example values are illustrative.*

### Interactive API Documentation

Explore and test the endpoints using FastAPI Swagger UI:

👉 https://credit-card-fraud-detection-a3nd.onrender.com/docs

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── artifacts/
│   ├── fraud_model.pkl
│   ├── metrics.json
│   └── demo_transactions.csv
│
├── data/
│   └── creditcard.csv
│
├── api.py
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Amankumar8541/credit-card-fraud-detection.git
```

### 2. Navigate to the project

```bash
cd credit-card-fraud-detection
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

Mac/Linux:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the FastAPI backend

Open the first terminal:

```bash
python3 -m uvicorn api:app --reload --port 8002
```

FastAPI will run at:

```text
http://127.0.0.1:8002
```

Swagger documentation:

```text
http://127.0.0.1:8002/docs
```

### 7. Start the Streamlit frontend

Open a second terminal:

```bash
python3 -m streamlit run app.py
```

The Streamlit application will open in your browser.

**Important:** Keep both terminals running while using the application locally.

## ☁️ Deployment Configuration

### Frontend – Streamlit Community Cloud

The Streamlit application is deployed through GitHub integration.

Configure the following secret in Streamlit Cloud:

```toml
FRAUDGUARD_API_URL = "https://credit-card-fraud-detection-a3nd.onrender.com"
```

The application uses this URL to communicate with the deployed FastAPI backend.

### Backend – Render

The FastAPI backend is deployed as a Render Web Service.

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

The backend loads the trained model from:

```text
artifacts/fraud_model.pkl
```

**Note:** Render's free instance may spin down after inactivity. The first request after inactivity can take additional time.

## 📁 Dataset

Dataset: Credit Card Fraud Detection Dataset (ULB)

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized transaction features, including Time, V1–V28, Amount, and Class.

**Note:** Batch prediction requires the same anonymized feature format. Ordinary bank statements or UPI transaction exports are not directly supported.

## ⚠️ Limitations

- This is an academic machine learning prototype.
- It is not connected to live banking or payment systems.
- Uploaded datasets are analyzed using the existing trained model; they do not automatically retrain it.
- Predictions are not confirmed real-world fraud cases.
- The model may generate false positives.
- The model's low precision means additional investigation may be required for flagged transactions.
- Render's free service may experience cold starts after inactivity.

## 🔮 Future Improvements

- Integrate MLflow for experiment tracking and model versioning.
- Add automated model monitoring and data drift detection.
- Implement automated retraining pipelines.
- Add authentication and API access control.
- Improve model precision and reduce false positives.
- Introduce real-time transaction streaming integration.
- Add automated CI/CD workflows.

## 👨‍💻 Author

**Aman Kumar Gupta**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/Amankumar8541

---

⭐ If you find this project interesting, consider giving it a star!
