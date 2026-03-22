"""
frontend.py — FraudShield Home Page
=====================================
Run:
    streamlit run frontend/frontend.py

Pages:
    1_Checker.py   — Transaction fraud checker + OTP flow
    2_Analytics.py — Model Comparison, Confusion Matrix, Risk Pie, ROC Curve

Author  : FraudShield Project
Purpose : Academic submission
"""

import streamlit as st

st.set_page_config(
    page_title            = "FraudShield",
    page_icon             = "🛡️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080d18; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0a0f1c; border-right: 1px solid #1e293b; }
hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 2rem">
    <div style="font-family:'Space Mono',monospace;font-size:3.5rem;font-weight:700;
                background:linear-gradient(135deg,#38bdf8,#818cf8,#34d399);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:0.5rem">
        🛡️ FraudShield
    </div>
    <div style="color:#475569;font-size:0.85rem;letter-spacing:3px;text-transform:uppercase">
        Hybrid Credit Card Fraud Detection System · LR 30% + RF 70%
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:16px;
                padding:2rem;text-align:center;height:220px;
                display:flex;flex-direction:column;justify-content:center">
        <div style="font-size:3rem;margin-bottom:0.8rem">🔍</div>
        <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#38bdf8;
                    font-weight:700;margin-bottom:0.6rem">TRANSACTION CHECKER</div>
        <div style="color:#64748b;font-size:0.85rem;line-height:1.6">
            Select a real transaction from creditcard.csv,<br>
            run the hybrid model, and go through the<br>
            OTP verification flow for medium-risk cases.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:16px;
                padding:2rem;text-align:center;height:220px;
                display:flex;flex-direction:column;justify-content:center">
        <div style="font-size:3rem;margin-bottom:0.8rem">📊</div>
        <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#34d399;
                    font-weight:700;margin-bottom:0.6rem">ANALYTICS DASHBOARD</div>
        <div style="color:#64748b;font-size:0.85rem;line-height:1.6">
            Model Comparison Bar · Confusion Matrix<br>
            Risk Distribution Pie · ROC Curve<br>
            Computed on the full test set.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Use the **sidebar** to navigate between pages.")

# ─── Model info strip ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;padding:0.5rem 0">
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        LR weight = <span style="color:#60a5fa">0.3</span>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        RF weight = <span style="color:#a78bfa">0.7</span>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        Hybrid threshold = <span style="color:#fb923c">0.43</span>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        High risk cutoff = <span style="color:#f87171">0.70</span>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        SMOTE strategy = <span style="color:#34d399">0.1</span>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:0.6rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#475569">
        RF trees = <span style="color:#34d399">200</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.72rem;
            font-family:'Space Mono',monospace;letter-spacing:1px;padding-top:1.5rem">
    FRAUDSHIELD · Hybrid ML Fraud Detection · Academic Project
</div>
""", unsafe_allow_html=True)
