# 🛡️ FraudGuard – Credit Card Fraud Detection System

An ML-powered platform to identify suspicious credit card transactions using Logistic Regression.

🔗 **Live Demo:** https://credit-card-fraud-detection-pjpvqnn3goslg2yotacb7.streamlit.app/

---

## 📌 Project Overview

FraudGuard is a machine learning-based web application designed to detect potentially fraudulent credit card transactions.

It uses a trained Logistic Regression model to classify transactions as **Fraudulent** or **Legitimate**.

The application provides an interactive dashboard where users can analyze individual transactions, upload transaction datasets, view prediction summaries, and download prediction reports.

## ✨ Features

- 📊 Interactive dark-themed dashboard
- 🔍 Individual transaction fraud detection
- 📁 Upload custom transaction CSV files
- ⚡ Batch transaction prediction
- 📈 Fraud and legitimate transaction statistics
- 📥 Download prediction reports
- 📉 Model evaluation and confusion matrix
- 📌 Precision, Recall, F1 Score and PR-AUC metrics

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Logistic Regression | Classification model |
| Joblib | Model persistence |
| Streamlit | Web application |
| Git & GitHub | Version control |
| Streamlit Community Cloud | Deployment |

## 🤖 Machine Learning Model

- Algorithm: Logistic Regression
- Dataset: Credit Card Fraud Detection Dataset
- Class imbalance handling: `class_weight="balanced"`
- Feature scaling: StandardScaler
- Model pipeline: Scikit-learn Pipeline

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
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Amankuamr8541/credit-card-fraud-detection.git
```

### 2. Navigate to the project

```bash
cd credit-card-fraud-detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
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

### 6. Run the application

```bash
python -m streamlit run app.py
```

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

## 👨‍💻 Author

**Aman Kumar Gupta**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/Amankumar8541

---

⭐ If you find this project interesting, consider giving it a star!
