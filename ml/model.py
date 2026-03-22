"""
model.py — Model Definitions
==============================
Models:
    1. Logistic Regression (LR)
    2. Random Forest       (RF)

Hybrid Logic:
    Both LR and RF predict fraud  → High Risk  / Fraud
    Only RF predicts fraud         → Medium Risk / Fraud  (RF is trusted more)
    Neither predicts fraud         → Low Risk   / Genuine

Fixed threshold = 0.8 for both models.

Author  : FraudShield Project
Purpose : Academic submission
"""

import numpy as np
import pandas as pd

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# WHY different thresholds?
# LR outputs probabilities near 0 or 1 — 0.5 works fine
# RF outputs probabilities between 0.3–0.6 even for real fraud
# because it averages votes across 200 trees — 0.4 is the right cutoff
LR_THRESHOLD = 0.5
RF_THRESHOLD = 0.4

THRESHOLD = RF_THRESHOLD  # kept for backward compatibility


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    """
    Split dataset into train/test sets and scale the Amount column.

    Steps:
        1. Separate features (X) and target (y)
        2. Stratified 80/20 train-test split
        3. Fit StandardScaler on training data only (prevents data leakage)
        4. Apply scaler to both train and test sets

    Args:
        df : raw creditcard DataFrame

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split preserves the fraud/genuine ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y,
    )

    # Scale Amount — fit ONLY on training data
    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"]  = scaler.transform(X_test[["Amount"]])

    return X_train, X_test, y_train, y_test, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                sampling_strategy: float = 0.1):
    """
    Oversample the minority (fraud) class using SMOTE.

    Applied ONLY on training data — never on test data.
    sampling_strategy=0.1 means fraud becomes 10% of training set.
    This is intentionally low to avoid overfitting and keep precision high.

    Args:
        X_train           : training features
        y_train           : training labels
        sampling_strategy : ratio of minority to majority class after resampling

    Returns:
        X_resampled, y_resampled
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Training
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train: pd.DataFrame,
                               y_train: pd.Series) -> LogisticRegression:
    """
    Train Logistic Regression model.

    C=0.1  → strong regularization, reduces false positives from SMOTE.
    lbfgs  → efficient solver for small-to-medium datasets.

    Args:
        X_train : SMOTE-resampled training features
        y_train : SMOTE-resampled training labels

    Returns:
        Fitted LogisticRegression model
    """
    model = LogisticRegression(
        C            = 0.1,
        max_iter     = 5000,
        solver       = "lbfgs",
        random_state = 42,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series) -> RandomForestClassifier:
    """
    Train Random Forest model.

    Random Forest is the primary/trusted model in this hybrid because:
        - Ensemble of 200 trees reduces individual tree errors
        - Handles class imbalance better than Logistic Regression
        - Non-linear decision boundaries capture complex fraud patterns
        - class_weight='balanced' adjusts for fraud rarity automatically

    Args:
        X_train : SMOTE-resampled training features
        y_train : SMOTE-resampled training labels

    Returns:
        Fitted RandomForestClassifier model
    """
    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = 15,
        # class_weight = "balanced",  # Removed to avoid double-dipping with SMOTE
        random_state = 42,
        n_jobs       = 2,
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Prediction — Single Row (used by API and Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_predict(lr_model, rf_model, X, lr_weight: float = 0.3, rf_weight: float = 0.7) -> dict:
    """
    Combine LR and RF predictions into a single hybrid decision using Soft Voting.

    Hybrid Logic (Soft Voting):
        Computes a weighted average of probabilities.
        Thresholds are also weighted to determine the final prediction.

    Args:
        lr_model : fitted LogisticRegression
        rf_model : fitted RandomForestClassifier
        X        : single-row scaled DataFrame
        lr_weight: weight given to Logistic Regression model
        rf_weight: weight given to Random Forest model

    Returns:
        dict with prediction details
    """
    lr_prob  = float(lr_model.predict_proba(X)[0][1])
    rf_prob  = float(rf_model.predict_proba(X)[0][1])

    hybrid_prob = (lr_weight * lr_prob) + (rf_weight * rf_prob)
    hybrid_threshold = (lr_weight * LR_THRESHOLD) + (rf_weight * RF_THRESHOLD)

    lr_flag  = lr_prob >= LR_THRESHOLD
    rf_flag  = rf_prob >= RF_THRESHOLD

    # Hybrid decision rules
    if hybrid_prob >= 0.7:
        prediction = "Fraud"
        risk_level = "High"
    elif hybrid_prob >= hybrid_threshold:
        prediction = "Fraud"
        risk_level = "Medium"
    else:
        prediction = "Genuine"
        risk_level = "Low"

    return {
        "prediction"        : prediction,
        "risk_level"        : risk_level,
        "hybrid_probability": round(hybrid_prob, 5),
        "lr_probability"    : round(lr_prob, 5),
        "rf_probability"    : round(rf_prob, 5),
        "lr_flag"           : bool(lr_flag),
        "rf_flag"           : bool(rf_flag),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Prediction — Full Test Set (used by main.py for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(model, X_test: pd.DataFrame, **kwargs) -> tuple:
    """
    Run a single model on the full test set.
    Returns predictions and probabilities for metric calculation.

    Args:
        model  : fitted LR or RF model
        X_test : scaled test features

    Returns:
        y_pred (0/1 array), y_prob (float array)
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    # use RF_THRESHOLD (0.4) as default — caller can override
    threshold = kwargs.get('threshold', THRESHOLD)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def hybrid_predict_batch(lr_model, rf_model, X_test: pd.DataFrame, lr_weight: float = 0.3, rf_weight: float = 0.7) -> tuple:
    """
    Run hybrid prediction on the full test set using Soft Voting.
    Used by main.py to evaluate hybrid model metrics.

    Args:
        lr_model : fitted LogisticRegression
        rf_model : fitted RandomForestClassifier
        X_test   : scaled test features
        lr_weight: weight given to Logistic Regression model
        rf_weight: weight given to Random Forest model

    Returns:
        y_pred      : 0/1 fraud labels
        hybrid_probs: Combined hybrid probabilities
        risk_levels : list of "Low" / "Medium" / "High"
    """
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    hybrid_probs = (lr_weight * lr_probs) + (rf_weight * rf_probs)
    hybrid_threshold = (lr_weight * LR_THRESHOLD) + (rf_weight * RF_THRESHOLD)

    y_pred      = []
    risk_levels = []

    for h_prob in hybrid_probs:
        if h_prob >= 0.7:
            y_pred.append(1)
            risk_levels.append("High")
        elif h_prob >= hybrid_threshold:
            y_pred.append(1)
            risk_levels.append("Medium")
        else:
            y_pred.append(0)
            risk_levels.append("Low")

    return np.array(y_pred), hybrid_probs, risk_levels
