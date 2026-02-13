import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# =========================
# 1. Load Dataset
# =========================
print("📂 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset shape:", df.shape)
print("Fraud cases:", sum(y == 1))
print("Genuine cases:", sum(y == 0))


# =========================
# 2. Scale Amount
# =========================
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])


# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 4. Train Logistic Regression
# =========================
print("\n🤖 Training Logistic Regression...")

model = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")


# =========================
# 5. Save & Load Model
# =========================
joblib.dump(model, "ml/final_fraud_model.pkl")
print("✅ Model saved")

loaded_model = joblib.load("ml/final_fraud_model.pkl")
print("✅ Model loaded")


# =========================
# 6. Evaluate Model
# =========================
print("\n📊 Evaluating model on test data...")

y_pred = loaded_model.predict(X_test)
y_prob = loaded_model.predict_proba(X_test)[:, 1]

print("\n🔢 CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\n🎯 ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# =========================
# 7. Threshold Tuning Demo
# =========================
print("\n⚙️ Threshold Tuning Demonstration")

threshold = 0.3  # You can change this value

y_custom = (y_prob >= threshold).astype(int)

print(f"\nConfusion Matrix with Threshold = {threshold}")
print(confusion_matrix(y_test, y_custom))


import numpy as np

print("\n🔍 AUTOMATIC SAMPLE DEMONSTRATION")

def decide(prob):
    if prob >= 0.7:
        return "🚨 FRAUD (High Risk)"
    elif prob >= 0.3:
        return "⚠️ SUSPICIOUS (Medium Risk)"
    else:
        return "✅ GENUINE (Low Risk)"


# -------------------------------
# 1️⃣ Show 5 Random Genuine Samples
# -------------------------------
print("\n🟢 Random Genuine Transactions:\n")

genuine_indices = np.where(y_test == 0)[0]
random_genuine = np.random.choice(genuine_indices, 5, replace=False)

for idx in random_genuine:
    sample = X_test.iloc[[idx]]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"Transaction Index: {idx}")
    print("Fraud Probability:", round(prob, 5))
    print("Decision:", decide(prob))
    print("-" * 40)


# -------------------------------
# 2️⃣ Show 5 Random Fraud Samples
# -------------------------------
print("\n🔴 Random Fraud Transactions:\n")

fraud_indices = np.where(y_test == 1)[0]
random_fraud = np.random.choice(fraud_indices, 5, replace=False)

for idx in random_fraud:
    sample = X_test.iloc[[idx]]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"Transaction Index: {idx}")
    print("Fraud Probability:", round(prob, 5))
    print("Decision:", decide(prob))
    print("-" * 40)
