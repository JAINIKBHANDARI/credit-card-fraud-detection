"""
main.py — Training Pipeline + Evaluation
==========================================
Trains LR and RF models, then evaluates all three:
    1. Logistic Regression  (standalone)
    2. Random Forest        (standalone)
    3. Hybrid               (LR + RF combined)

Fixed threshold = 0.8 for all models.

Run:
    python ml/main.py

Author  : FraudShield Project
Purpose : Academic submission
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from model import (
    split_and_scale,
    apply_smote,
    train_logistic_regression,
    train_random_forest,
    predict_batch,
    hybrid_predict_batch,
    hybrid_predict,
    LR_THRESHOLD,
    RF_THRESHOLD,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
OUT_DIR   = BASE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helper — print evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(label: str, y_test, y_pred, y_prob=None):
    """
    Print a clean evaluation block for any model.

    Args:
        label  : model name (e.g. "Logistic Regression")
        y_test : true labels
        y_pred : predicted labels (0/1)
        y_prob : predicted probabilities (for ROC-AUC, optional)
    """
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  ← of all flagged, how many are real fraud")
    print(f"  Recall    : {rec:.4f}  ← of all real fraud, how many were caught")
    print(f"  F1 Score  : {f1:.4f}  ← harmonic mean of precision and recall")

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC   : {auc:.4f}  ← overall discrimination power")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':20s}  Predicted Genuine  Predicted Fraud")
    print(f"  {'Actual Genuine':20s}  {cm[0,0]:>17}  {cm[0,1]:>15}  (FP = {cm[0,1]})")
    print(f"  {'Actual Fraud':20s}  {cm[1,0]:>17}  {cm[1,1]:>15}  (FN = {cm[1,0]})")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — print model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(y_test, lr_pred, rf_pred, hybrid_pred, lr_prob, rf_prob, h_prob):
    """Print a side-by-side comparison of all three models."""

    def get_metrics(y_pred, y_prob=None):
        return {
            "precision" : precision_score(y_test, y_pred, zero_division=0),
            "recall"    : recall_score(y_test, y_pred, zero_division=0),
            "f1"        : f1_score(y_test, y_pred, zero_division=0),
            "auc"       : roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
        }

    lr  = get_metrics(lr_pred,     lr_prob)
    rf  = get_metrics(rf_pred,     rf_prob)
    hyb = get_metrics(hybrid_pred, h_prob)

    print(f"\n{'='*60}")
    print(f"  Model Comparison (LR threshold={LR_THRESHOLD}, RF threshold={RF_THRESHOLD})")
    print(f"{'='*60}")
    print(f"  {'Metric':12s}  {'Log. Regression':>18}  {'Random Forest':>14}  {'Hybrid':>8}")
    print(f"  {'-'*56}")
    for key in ["precision", "recall", "f1", "auc"]:
        label = key.upper() if key == "auc" else key.capitalize()
        print(f"  {label:12s}  {lr[key]:>18.4f}  {rf[key]:>14.4f}  {hyb[key]:>8.4f}")




# ─────────────────────────────────────────────────────────────────────────────
# Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(lr_model, rf_model, scaler):
    joblib.dump(lr_model, os.path.join(OUT_DIR, "final_fraud_model.pkl"))
    joblib.dump(rf_model, os.path.join(OUT_DIR, "random_forest.pkl"))
    joblib.dump(scaler,   os.path.join(OUT_DIR, "scaler.pkl"))
    print(f"\n💾 Saved artefacts to ml/")
    print(f"   final_fraud_model.pkl  (Logistic Regression)")
    print(f"   random_forest.pkl      (Random Forest)")
    print(f"   scaler.pkl             (StandardScaler)")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape      : {df.shape}")
    print(f"   Fraud      : {df['Class'].sum()}")
    print(f"   Genuine    : {(df['Class']==0).sum()}")
    print(f"   Fraud rate : {df['Class'].mean()*100:.3f}%")

    # ── 2. Split + scale ──────────────────────────────────────────────────────
    print("\n✂️  Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    print(f"   Train size : {len(X_train)}")
    print(f"   Test size  : {len(X_test)}")

    # ── 3. Apply SMOTE ────────────────────────────────────────────────────────
    print("\n⚖️  Applying SMOTE (strategy=0.1)...")
    X_res, y_res = apply_smote(X_train, y_train, sampling_strategy=0.1)
    print(f"   Before : {np.bincount(y_train)}")
    print(f"   After  : {np.bincount(y_res)}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\n🤖 Training Logistic Regression...")
    lr_model = train_logistic_regression(X_res, y_res)
    print("   ✅ Done")

    print("\n🌳 Training Random Forest (200 trees)...")
    rf_model = train_random_forest(X_res, y_res)
    print("   ✅ Done")

    # ── 5. Get predictions from all 3 models ──────────────────────────────────
    lr_pred, lr_prob           = predict_batch(lr_model, X_test, threshold=LR_THRESHOLD)
    rf_pred, rf_prob           = predict_batch(rf_model, X_test, threshold=RF_THRESHOLD)
    h_pred, h_prob, risk_levels = hybrid_predict_batch(lr_model, rf_model, X_test)

    # ── 6. Evaluate each model ────────────────────────────────────────────────
    print_metrics("Logistic Regression", y_test, lr_pred, lr_prob)
    print_metrics("Random Forest",       y_test, rf_pred, rf_prob)
    print_metrics("Hybrid (LR + RF)",    y_test, h_pred,  h_prob)

    # ── 7. Side-by-side comparison ────────────────────────────────────────────
    print_comparison(y_test, lr_pred, rf_pred, h_pred, lr_prob, rf_prob, h_prob)

    # ── 8. Hybrid Risk Breakdown ──────────────────────────────────────────────
    risk_series = pd.Series(risk_levels)
    print(f"\n  Risk Level Distribution (Hybrid):")
    print(f"  Low    : {(risk_series=='Low').sum():>6}  (Genuine)")
    print(f"  Medium : {(risk_series=='Medium').sum():>6}  (Crossed Soft Threshold)")
    print(f"  High   : {(risk_series=='High').sum():>6}  (High Confidence > 0.7)")

    print(f"\n  Hybrid Risk Confusion Matrix:")
    risk_df = pd.DataFrame({
        "Actual": np.where(y_test == 1, "Fraud", "Genuine"),
        "Predicted Risk": risk_levels
    })
    cross_tab = pd.crosstab(risk_df["Actual"], risk_df["Predicted Risk"])
    for col in ["Low", "Medium", "High"]:
        if col not in cross_tab.columns:
            cross_tab[col] = 0
    cross_tab = cross_tab[["Low", "Medium", "High"]]
    
    formatted_ct = cross_tab.to_string(index_names=False)
    for line in formatted_ct.split('\n'):
        print(f"  {line}")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    save_artefacts(lr_model, rf_model, scaler)

    print("\n🎉 Training complete!")
