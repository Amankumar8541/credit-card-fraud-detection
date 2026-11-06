
import os
import json
import joblib
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

os.makedirs("artifacts", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)
print("Fraud cases:", df["Class"].sum())

X = df.drop("Class", axis=1)
y = df["Class"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

print("Evaluating model...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "pr_auc": float(average_precision_score(
        y_test, probabilities
    )),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist()
}

joblib.dump(model, "artifacts/fraud_model.pkl")

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Create a small stratified demo dataset
demo_X, _, demo_y, _ = train_test_split(
    X_test,
    y_test,
    train_size=min(500, len(X_test)),
    random_state=42,
    stratify=y_test
)

demo = demo_X.copy()
demo["Class"] = demo_y.values
demo.to_csv("artifacts/demo_transactions.csv", index=False)

print("\nTraining completed!")
print(json.dumps(metrics, indent=4))
print("\nModel saved successfully!")
