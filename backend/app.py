"""
app.py — FraudShield Backend (FastAPI)
=======================================
REST API for Credit Card Fraud Detection using Hybrid Soft-Voting ML Model.

Hybrid Logic (matches model.py exactly):
    hybrid_prob = 0.3 * LR_prob + 0.7 * RF_prob
    hybrid_threshold = 0.3 * 0.5 + 0.7 * 0.4 = 0.43

    hybrid_prob >= 0.70        → High Risk  / Fraud   (blocked, no OTP)
    hybrid_prob >= 0.43        → Medium Risk / OTP Required
    hybrid_prob <  0.43        → Low Risk   / Genuine

Author  : FraudShield Project
Purpose : Academic submission
"""

import os
import random
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraudshield")

# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudShield API",
    description="Hybrid Soft-Voting ML fraud detection with OTP verification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ──────────────────────────────────────────────────────────────

LR_THRESHOLD        = 0.5
RF_THRESHOLD        = 0.4
LR_WEIGHT           = 0.3
RF_WEIGHT           = 0.7
HIGH_RISK_THRESHOLD = 0.70
HYBRID_THRESHOLD    = (LR_WEIGHT * LR_THRESHOLD) + (RF_WEIGHT * RF_THRESHOLD)  # 0.43
OTP_TTL_SECONDS     = 300

# ─── Paths — HARDCODED ──────────────────────────────────────────────────────

PROJECT_ROOT = r"C:\Users\Harsh\Downloads\fraud_project\fraud_project"
DATA_CSV     = os.path.join(PROJECT_ROOT, "data", "test_cases.csv")
LR_PKL       = os.path.join(PROJECT_ROOT, "ml",   "final_fraud_model.pkl")
RF_PKL       = os.path.join(PROJECT_ROOT, "ml",   "random_forest.pkl")
SC_PKL       = os.path.join(PROJECT_ROOT, "ml",   "scaler.pkl")

# ─── OTP Store ──────────────────────────────────────────────────────────────

otp_store: dict = {}

# ─── Model Loading ──────────────────────────────────────────────────────────

lr_model = rf_model = scaler = None


def load_models():
    global lr_model, rf_model, scaler

    log.info(f"Looking for models at:")
    log.info(f"  LR  : {LR_PKL}")
    log.info(f"  RF  : {RF_PKL}")
    log.info(f"  SC  : {SC_PKL}")

    if all(os.path.exists(p) for p in [LR_PKL, RF_PKL, SC_PKL]):
        lr_model = joblib.load(LR_PKL)
        rf_model = joblib.load(RF_PKL)
        scaler   = joblib.load(SC_PKL)
        log.info("✅ All models loaded successfully")
    else:
        missing = [p for p in [LR_PKL, RF_PKL, SC_PKL] if not os.path.exists(p)]
        log.warning(f"⚠️  Missing model files: {missing}")
        log.warning("⚠️  Running in simulation mode")


load_models()

# ─── Schemas ────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    Time: float = 0.0   # ← required by model (trained with Time column)
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    session_id: Optional[str] = "default"


class OTPRequest(BaseModel):
    session_id: Optional[str] = "default"


class OTPVerify(BaseModel):
    otp: str
    session_id: Optional[str] = "default"


# ─── Hybrid Logic ───────────────────────────────────────────────────────────

def run_hybrid(transaction: dict) -> dict:
    """
    Soft-Voting hybrid prediction.
    Feature order: Time + V1-V28 + Amount (matches training data exactly)
    """
    # Include Time column — model was trained with it
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    row = pd.DataFrame([{col: transaction.get(col, 0.0) for col in feature_cols}])

    # Scale Amount only
    if scaler is not None:
        row["Amount"] = scaler.transform(row[["Amount"]])

    if lr_model is None or rf_model is None:
        lr_prob = random.uniform(0.1, 0.9)
        rf_prob = random.uniform(0.1, 0.9)
        log.warning("Simulation mode — random probabilities")
    else:
        lr_prob = float(lr_model.predict_proba(row)[0][1])
        rf_prob = float(rf_model.predict_proba(row)[0][1])

    # Soft-voting weighted combination
    hybrid_prob = (LR_WEIGHT * lr_prob) + (RF_WEIGHT * rf_prob)
    lr_flag     = lr_prob >= LR_THRESHOLD
    rf_flag     = rf_prob >= RF_THRESHOLD

    if hybrid_prob >= HIGH_RISK_THRESHOLD:
        risk_level  = "High"
        decision    = "Fraud"
        explanation = (
            f"Both models agree: hybrid score {hybrid_prob:.3f} ≥ {HIGH_RISK_THRESHOLD}. "
            f"LR={lr_prob:.3f}, RF={rf_prob:.3f}. Transaction blocked — no OTP."
        )
    elif hybrid_prob >= HYBRID_THRESHOLD:
        risk_level  = "Medium"
        decision    = "OTP Required"
        explanation = (
            f"Suspicious hybrid score {hybrid_prob:.3f} (≥ {HYBRID_THRESHOLD:.3f}). "
            f"LR={lr_prob:.3f}, RF={rf_prob:.3f}. OTP verification required."
        )
    else:
        risk_level  = "Low"
        decision    = "Genuine"
        explanation = (
            f"Low hybrid score {hybrid_prob:.3f} (< {HYBRID_THRESHOLD:.3f}). "
            f"LR={lr_prob:.3f}, RF={rf_prob:.3f}. Transaction approved."
        )

    return {
        "risk_level"        : risk_level,
        "decision"          : decision,
        "hybrid_probability": round(hybrid_prob, 5),
        "lr_probability"    : round(lr_prob, 5),
        "rf_probability"    : round(rf_prob, 5),
        "lr_flag"           : bool(lr_flag),
        "rf_flag"           : bool(rf_flag),
        "explanation"       : explanation,
        "thresholds"        : {
            "lr"    : LR_THRESHOLD,
            "rf"    : RF_THRESHOLD,
            "hybrid": round(HYBRID_THRESHOLD, 4),
            "high"  : HIGH_RISK_THRESHOLD,
        },
    }


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FraudShield API running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "models_loaded" : lr_model is not None,
        "csv_exists"    : os.path.exists(DATA_CSV),
        "csv_path"      : DATA_CSV,
    }


@app.post("/predict")
def predict(tx: TransactionInput):
    log.info(f"[{tx.session_id}] /predict | Amount={tx.Amount:.2f}")
    t0     = time.time()
    result = run_hybrid(tx.dict(exclude={"session_id"}))

    if result["risk_level"] == "Medium":
        otp_store[tx.session_id] = {
            "transaction": tx.dict(),
            "otp"        : None,
            "expires_at" : time.time() + OTP_TTL_SECONDS,
            "result"     : result,
        }

    elapsed = (time.time() - t0) * 1000
    log.info(
        f"[{tx.session_id}] Risk={result['risk_level']} | "
        f"Hybrid={result['hybrid_probability']:.3f} | {elapsed:.1f}ms"
    )
    return result


@app.post("/send-otp")
def send_otp(req: OTPRequest):
    sid = req.session_id
    if sid not in otp_store:
        raise HTTPException(status_code=400,
            detail="No pending medium-risk transaction. Run /predict first.")

    otp = str(random.randint(100000, 999999))
    otp_store[sid]["otp"]        = otp
    otp_store[sid]["expires_at"] = time.time() + OTP_TTL_SECONDS

    log.info("")
    log.info(f"  ┌{'─'*42}┐")
    log.info(f"  │  📱  OTP for session [{sid}]")
    log.info(f"  │  >>>   {otp}   <<<")
    log.info(f"  └{'─'*42}┘")
    log.info("")

    return {"status": "OTP sent", "message": "Check server console", "session_id": sid}


@app.post("/verify-otp")
def verify_otp(req: OTPVerify):
    sid = req.session_id
    if sid not in otp_store or otp_store[sid].get("otp") is None:
        raise HTTPException(status_code=400,
            detail="No active OTP. Call /send-otp first.")

    entry = otp_store[sid]

    if time.time() > entry["expires_at"]:
        del otp_store[sid]
        return {
            "verified"          : False,
            "decision"          : "Fraud",
            "reason"            : "OTP expired",
            "risk_level"        : "Medium",
            "hybrid_probability": entry["result"]["hybrid_probability"],
            "lr_probability"    : entry["result"]["lr_probability"],
            "rf_probability"    : entry["result"]["rf_probability"],
        }

    correct = entry["otp"] == req.otp.strip()
    result  = entry["result"]
    del otp_store[sid]

    if correct:
        log.info(f"[{sid}] ✅ OTP correct → Genuine")
        return {
            "verified"          : True,
            "decision"          : "Genuine",
            "reason"            : "OTP verified successfully",
            "risk_level"        : "Medium",
            "hybrid_probability": result["hybrid_probability"],
            "lr_probability"    : result["lr_probability"],
            "rf_probability"    : result["rf_probability"],
        }
    else:
        log.warning(f"[{sid}] ❌ Wrong OTP → Fraud")
        return {
            "verified"          : False,
            "decision"          : "Fraud",
            "reason"            : "Incorrect OTP entered",
            "risk_level"        : "Medium",
            "hybrid_probability": result["hybrid_probability"],
            "lr_probability"    : result["lr_probability"],
            "rf_probability"    : result["rf_probability"],
        }


@app.get("/testcases")
def get_testcases():
    log.info(f"Loading CSV from: {DATA_CSV}")

    if not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=404,
            detail=f"File not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    log.info(f"✅ Loaded {len(df)} test cases")
    return df.to_dict(orient="records")
