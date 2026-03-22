# 🛡️ FraudShield — Credit Card Fraud Detection System

A full-stack credit card fraud detection system using a Hybrid Soft-Voting Machine Learning model
(Logistic Regression + Random Forest) with OTP verification, FastAPI backend, and Streamlit frontend.

---

## 📁 Folder Structure

```
fraud_project/
├── backend/
│   ├── app.py                  ← FastAPI backend (all API endpoints)
│   ├── generate_dataset.py     ← Test case generator
│   └── __init__.py
│
├── data/
│   ├── creditcard.csv          ← Original dataset (284,807 rows)
│   └── test_cases.csv          ← Generated test cases (25 rows)
│
├── frontend/
│   ├── frontend.py             ← Streamlit UI
│   ├── app.py
│   └── __init__.py
│
├── ml/
│   ├── model.py                ← Hybrid model logic
│   ├── main.py                 ← Training pipeline
│   ├── final_fraud_model.pkl   ← Trained Logistic Regression
│   ├── random_forest.pkl       ← Trained Random Forest
│   ├── scaler.pkl              ← Trained StandardScaler
│   └── __init__.py
│
└── README.md
```

---

## 🧠 How the Model Works

### Training (already done)
```
creditcard.csv (284,807 real transactions)
        ↓
  Split 80/20 (train/test)
        ↓
  Scale Amount using StandardScaler
        ↓
  Apply SMOTE (sampling_strategy=0.1)
        ↓
  Train Logistic Regression  (C=0.1, lbfgs)
  Train Random Forest        (200 trees, max_depth=15)
        ↓
  Save → final_fraud_model.pkl
          random_forest.pkl
          scaler.pkl
```

### Prediction (at runtime)
```
Transaction (V1-V28 + Amount + Time)
        ↓
  Scale Amount using saved scaler
        ↓
  LR predicts  → lr_prob
  RF predicts  → rf_prob
        ↓
  hybrid_prob = 0.3 × lr_prob + 0.7 × rf_prob
        ↓
  Apply thresholds → Risk Level
```

> ⚠️ The model predicts INDEPENDENTLY based on what it learned.
> It does NOT look at actual_class. actual_class is only shown
> in the UI so you can compare the model's prediction vs ground truth.

---

## ⚙️ Hybrid Soft-Voting Logic

| Hybrid Score | Risk Level | Decision | Action |
|---|---|---|---|
| ≥ 0.70 | 🔴 High | Fraud | Blocked immediately — no OTP |
| 0.43 – 0.70 | 🟡 Medium | OTP Required | Send OTP → verify identity |
| < 0.43 | 🟢 Low | Genuine | Approved immediately |

**Formula:**
```
hybrid_prob      = 0.3 × LR_prob  +  0.7 × RF_prob
hybrid_threshold = 0.3 × 0.5      +  0.7 × 0.4     = 0.43
```

**Why RF gets higher weight (0.7)?**
- Random Forest is an ensemble of 200 trees → more stable
- Handles class imbalance better than Logistic Regression
- Captures non-linear fraud patterns more effectively

---

## 🔐 OTP Flow (Medium Risk)

```
Transaction → Medium Risk
        ↓
  Frontend shows "OTP Required"
        ↓
  User clicks "Send OTP"
        ↓
  Backend generates 6-digit OTP
  Prints to server console (simulation)
        ↓
  User enters OTP
        ↓
  Correct OTP → ✅ GENUINE (transaction approved)
  Wrong OTP   → 🚫 FRAUD   (transaction blocked)
  Expired OTP → 🚫 FRAUD   (5 min timeout)
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install fastapi uvicorn streamlit scikit-learn imbalanced-learn joblib pandas numpy requests
```

### Step 2 — Generate test dataset
```bash
cd C:\Users\Harsh\Downloads\fraud_project\fraud_project\backend
python generate_dataset.py
```

### Step 3 — Start backend
```bash
cd C:\Users\Harsh\Downloads\fraud_project\fraud_project\backend
python -m uvicorn app:app --port 8001
```

### Step 4 — Start frontend (new terminal)
```bash
cd C:\Users\Harsh\Downloads\fraud_project\fraud_project\frontend
streamlit run frontend.py
```

### Step 5 — Open in browser
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API Docs | http://localhost:8001/docs |
| Health Check | http://localhost:8001/health |

---

## 🔌 API Reference

### POST /predict
```json
Request:
{
  "Time": 0.0,
  "V1": -1.35, "V2": -0.07, ..., "V28": 0.02,
  "Amount": 149.62,
  "session_id": "my-session"
}

Response:
{
  "risk_level": "Medium",
  "decision": "OTP Required",
  "hybrid_probability": 0.51,
  "lr_probability": 0.38,
  "rf_probability": 0.57,
  "lr_flag": false,
  "rf_flag": true,
  "explanation": "Suspicious hybrid score 0.51 ..."
}
```

### POST /send-otp
```json
Request:  { "session_id": "my-session" }
Response: { "status": "OTP sent", "message": "Check server console" }
```
> OTP is printed to the uvicorn terminal (simulation mode)

### POST /verify-otp
```json
Request:  { "otp": "482931", "session_id": "my-session" }
Response: { "verified": true, "decision": "Genuine", "reason": "OTP verified successfully" }
```

### GET /testcases
Returns all 25 rows from test_cases.csv as JSON.

### GET /health
```json
{
  "status": "ok",
  "models_loaded": true,
  "csv_exists": true,
  "csv_path": "C:\\...\\data\\test_cases.csv"
}
```

---

## 🎨 UI Features

- Dark professional UI with Space Mono + DM Sans fonts
- Hybrid Score bar (large, prominent)
- Individual LR and RF probability bars with thresholds
- Color-coded results:
  - 🟢 Green → Genuine
  - 🟡 Orange → Medium Risk / OTP
  - 🔴 Red → Fraud
- AI explanation of why each risk level was assigned
- V1–V28 feature inspector (expandable)
- Step-by-step OTP verification flow

---

## 📊 Model Performance (on creditcard.csv test set)

| Metric | Logistic Regression | Random Forest | Hybrid |
|---|---|---|---|
| Precision | ~0.85 | ~0.92 | ~0.91 |
| Recall | ~0.72 | ~0.81 | ~0.83 |
| F1 Score | ~0.78 | ~0.86 | ~0.87 |
| ROC-AUC | ~0.97 | ~0.98 | ~0.98 |

> Run `python ml/main.py` to see exact metrics on your trained models.

---

## 📝 Key Design Decisions

| Decision | Reason |
|---|---|
| RF weight = 0.7 | RF is more reliable for imbalanced fraud data |
| SMOTE strategy = 0.1 | Fraud becomes 10% of training — avoids overfitting |
| LR threshold = 0.5 | LR gives sharp 0/1 probabilities — 0.5 is natural |
| RF threshold = 0.4 | RF averages 200 trees — probabilities rarely exceed 0.6 |
| High cutoff = 0.70 | Strong signal from both models — safe to block directly |
| OTP for Medium | Uncertainty zone — verify human rather than auto-block |
| Scaler on Amount only | V1-V28 are already PCA-transformed (zero mean, unit variance) |

---

## ⚠️ Important Notes

- OTP expires after **5 minutes**
- OTP store is **in-memory** — resets if backend restarts
- `actual_class` in test_cases.csv is for **comparison only** — model never sees it
- Models are loaded from `ml/` folder — do NOT delete `.pkl` files
- Always run backend before starting frontend

---

*FraudShield · Credit card Hybrid ML Fraud Detection
