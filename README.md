
# Credit Card Fraud Detection System 🛡️

An ML-powered Suspicious Transaction Detection Platform that
classifies credit card transactions as Fraudulent or Legitimate.

## Project Overview

This project uses Machine Learning to identify potentially
fraudulent credit card transactions using historical transaction data.

## Features

- Credit card fraud classification
- Fraud probability prediction
- Interactive Streamlit dashboard
- Batch CSV transaction prediction
- Model evaluation and performance metrics
- Saved ML model using Joblib

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Logistic Regression
- Streamlit
- Joblib
- Git and GitHub

## Dataset

ULB Credit Card Fraud Detection Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv and place it inside the data folder.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd Credit-Card-Fraud-Detection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Run the Application

```bash
streamlit run app.py
```

## Project Architecture

Dataset → Preprocessing → Model Training →
Model Evaluation → Model Saving → Streamlit Dashboard

## MLOps Concepts

- Reproducible model training
- Model artifact management
- Evaluation metrics
- Batch inference
- Git-based version control

## Important Note

This is an academic machine learning and MLOps prototype.
It is not connected to a real banking system.

## Author

Your Name
B.Tech Computer Science and Engineering
