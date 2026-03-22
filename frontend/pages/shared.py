"""
shared.py — Shared constants, state init, CSS, API helper
Used by all pages in the FraudShield multi-page app.
"""

import requests
import streamlit as st

# ─── Backend ─────────────────────────────────────────────────────────────────
BACKEND    = "http://localhost:8001"
SESSION_ID = "streamlit-session"

# ─── Thresholds ──────────────────────────────────────────────────────────────
LR_WEIGHT        = 0.3
RF_WEIGHT        = 0.7
LR_THRESHOLD     = 0.5
RF_THRESHOLD     = 0.4
HYBRID_THRESHOLD = round((LR_WEIGHT * LR_THRESHOLD) + (RF_WEIGHT * RF_THRESHOLD), 4)
HIGH_THRESHOLD   = 0.70


# ─── Session state init ───────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "predict_result" : None,
        "otp_sent"       : False,
        "final_decision" : None,
        "tx_history"     : [],
        "current_case_id": None,
        "current_actual" : None,
        "current_amount" : None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Global CSS ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #080d18; color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #0a0f1c; border-right: 1px solid #1e293b; }

    .page-title {
        font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700;
        color:#e2e8f0; margin-bottom:0.2rem;
    }
    .page-sub { color:#475569; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; }
    .card { background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:1.3rem; margin-bottom:0.9rem; }
    .card-label { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#334155; margin-bottom:0.6rem; }
    .badge { display:inline-block; padding:0.35rem 1rem; border-radius:999px; font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
    .badge-low    { background:#052e16; color:#34d399; border:1px solid #065f46; }
    .badge-medium { background:#431407; color:#fb923c; border:1px solid #7c2d12; }
    .badge-high   { background:#3f0a0a; color:#f87171; border:1px solid #7f1d1d; }
    .decision { border-radius:14px; padding:1.6rem; text-align:center; margin-top:0.8rem; }
    .d-genuine { background:linear-gradient(135deg,#052e16,#064e3b); border:1px solid #065f46; }
    .d-fraud   { background:linear-gradient(135deg,#1c0505,#450a0a); border:1px solid #7f1d1d; }
    .d-otp     { background:linear-gradient(135deg,#1a0e00,#431407); border:1px solid #7c2d12; }
    .d-icon    { font-size:2.8rem; margin-bottom:0.4rem; }
    .d-label   { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; }
    .d-sub     { color:#94a3b8; font-size:0.85rem; margin-top:0.3rem; }
    .prob-bar-bg   { background:#1e293b; border-radius:999px; height:8px; margin-top:0.5rem; }
    .prob-bar-fill { height:8px; border-radius:999px; }
    .explanation { background:#0c1422; border-left:3px solid #1e3a5f; border-radius:0 8px 8px 0; padding:0.8rem 1rem; font-size:0.86rem; color:#7dd3fc; margin-top:0.8rem; font-style:italic; }
    .threshold-row { display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:0.6rem; }
    .t-chip { background:#1e293b; border-radius:8px; padding:0.4rem 0.75rem; font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; }
    .stat-chip { display:inline-block; padding:0.5rem 1rem; border-radius:10px; font-family:'Space Mono',monospace; font-size:0.8rem; margin-right:0.5rem; margin-bottom:0.5rem; }
    hr { border-color:#1e293b; }
    .stButton > button { border-radius:10px; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.5px; }
    </style>
    """, unsafe_allow_html=True)


# ─── API helper ───────────────────────────────────────────────────────────────
def api(method, path, **kwargs):
    try:
        r = getattr(requests, method)(f"{BACKEND}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to backend. Run: `python -m uvicorn app:app --port 8001`"
    except requests.exceptions.HTTPError as e:
        return None, f"API {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


# ─── Probability bar ──────────────────────────────────────────────────────────
def prob_bar(label, value, threshold, color):
    pct        = int(value * 100)
    tpct       = int(threshold * 100)
    flagged    = value >= threshold
    flag_icon  = "⚑" if flagged else "✓"
    flag_color = "#f87171" if flagged else "#34d399"
    st.markdown(f"""
    <div style="margin-bottom:0.9rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:0.82rem;color:#94a3b8">{label}</span>
            <span style="font-family:'Space Mono',monospace;font-size:1rem;color:{color}">{pct}%
                <span style="font-size:0.72rem;color:{flag_color};margin-left:4px">{flag_icon} (thr={tpct}%)</span>
            </span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Add to history ───────────────────────────────────────────────────────────
def add_to_history(case_id, amount, actual_class, result, final_decision=None):
    risk = result["risk_level"]

    if risk == "Low":
        outcome, outcome_tag = "Genuine", "genuine"
    elif risk == "High":
        outcome, outcome_tag = "Fraud", "fraud"
    elif risk == "Medium":
        if final_decision == "Genuine":
            outcome, outcome_tag = "OTP ✓ Genuine", "otp_ok"
        elif final_decision == "Fraud":
            outcome, outcome_tag = "OTP ✗ Fraud", "otp_fail"
        else:
            outcome, outcome_tag = "OTP Pending", "pending"
    else:
        outcome, outcome_tag = "Unknown", "pending"

    entry = {
        "case_id"         : case_id,
        "amount"          : amount,
        "actual"          : "Fraud" if actual_class == 1 else "Genuine",
        "risk"            : risk,
        "hybrid_prob"     : result["hybrid_probability"],
        "lr_prob"         : result["lr_probability"],
        "rf_prob"         : result["rf_probability"],
        "hybrid_pct"      : f"{int(result['hybrid_probability']*100)}%",
        "lr_pct"          : f"{int(result['lr_probability']*100)}%",
        "rf_pct"          : f"{int(result['rf_probability']*100)}%",
        "outcome"         : outcome,
        "outcome_tag"     : outcome_tag,
        "correct"         : (
            (outcome in ["Genuine", "OTP ✓ Genuine"] and actual_class == 0) or
            (outcome in ["Fraud",   "OTP ✗ Fraud"]   and actual_class == 1)
        ),
    }

    for i, h in enumerate(st.session_state.tx_history):
        if h["case_id"] == case_id:
            st.session_state.tx_history[i] = entry
            return

    st.session_state.tx_history.append(entry)
