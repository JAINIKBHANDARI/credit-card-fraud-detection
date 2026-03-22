"""
1_Checker.py — Transaction Fraud Checker
==========================================
Page 1 of FraudShield multi-page app.
Select a transaction, run hybrid model, OTP flow.
"""

import time
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (
    init_state, inject_css, api, prob_bar, add_to_history,
    SESSION_ID, LR_WEIGHT, RF_WEIGHT, LR_THRESHOLD, RF_THRESHOLD,
    HYBRID_THRESHOLD, HIGH_THRESHOLD,
)

st.set_page_config(page_title="Checker · FraudShield", page_icon="🔍", layout="wide")
inject_css()
init_state()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
    <div class="page-title">🔍 Transaction Checker</div>
    <div class="page-sub">Select a transaction · Run hybrid model · OTP verification</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Layout ──────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1.1, 1], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div class="card-label">📋 Select Transaction</div>', unsafe_allow_html=True)

    with st.spinner("Loading test cases..."):
        cases, err = api("get", "/testcases")

    if err:
        st.error(err)
        st.stop()

    df = pd.DataFrame(cases)

    labels = [
        f"{r['case_id']}  │  ₹{r['Amount']:.2f}  │  "
        f"{'🚫 Fraud' if r['actual_class'] == 1 else '✅ Genuine'}"
        for _, r in df.iterrows()
    ]
    idx = st.selectbox("Choose a transaction:", range(len(labels)),
                        format_func=lambda i: labels[i])
    row = df.iloc[idx]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
<div style="font-size:0.8rem;color:#94a3b8">Case ID</div>
<div style="font-size:1.2rem;font-weight:600">{row["case_id"]}</div>
""", unsafe_allow_html=True)
    c2.metric("Amount",  f"₹{row['Amount']:.2f}")
    c3.metric("Actual",  "Fraud" if row["actual_class"] == 1 else "Genuine")

    with st.expander("V1–V28 features"):
        v_cols = [f"V{i}" for i in range(1, 29)]
        st.dataframe(row[v_cols].to_frame("Value").T.round(4), use_container_width=True)

    st.markdown("---")

    st.markdown(f"""
    <div class="card">
        <div class="card-label">Soft-Voting Weights & Thresholds</div>
        <div class="threshold-row">
            <span class="t-chip">LR weight = {LR_WEIGHT}</span>
            <span class="t-chip">RF weight = {RF_WEIGHT}</span>
            <span class="t-chip">Hybrid thr = {HYBRID_THRESHOLD}</span>
            <span class="t-chip">High thr = {HIGH_THRESHOLD}</span>
        </div>
        <div style="margin-top:0.6rem;font-size:0.78rem;color:#475569">
            hybrid_prob = {LR_WEIGHT}×LR + {RF_WEIGHT}×RF
        </div>
    </div>
    """, unsafe_allow_html=True)

    check = st.button("🔍  Check Transaction", type="primary", use_container_width=True)

    if check:
        st.session_state.predict_result  = None
        st.session_state.otp_sent        = False
        st.session_state.final_decision  = None
        st.session_state.current_case_id = row["case_id"]
        st.session_state.current_actual  = int(row["actual_class"])
        st.session_state.current_amount  = float(row["Amount"])

        with st.spinner("Running soft-voting model..."):
            time.sleep(0.35)
            payload = {"Time": float(row.get("Time", 0.0))}
            for col in [f"V{i}" for i in range(1, 29)]:
                payload[col] = float(row[col])
            payload["Amount"]     = float(row["Amount"])
            payload["session_id"] = SESSION_ID
            result, err = api("post", "/predict", json=payload)

        if err:
            st.error(err)
        else:
            st.session_state.predict_result = result
            add_to_history(
                case_id      = row["case_id"],
                amount       = float(row["Amount"]),
                actual_class = int(row["actual_class"]),
                result       = result,
                final_decision = None,
            )

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_r:
    res = st.session_state.predict_result

    if res is None:
        st.markdown("""
        <div style="height:320px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;color:#1e293b;text-align:center;">
            <div style="font-size:3.5rem">🛡️</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                        letter-spacing:2px;margin-top:1rem;color:#334155">
                SELECT A TRANSACTION AND CLICK CHECK
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk      = res["risk_level"]
        badge_cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[risk]

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem">
            <span class="card-label" style="margin:0">Result</span>
            <span class="badge {badge_cls}">{risk} Risk</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Probabilities</div>', unsafe_allow_html=True)

        h_pct   = int(res["hybrid_probability"] * 100)
        h_color = "#f87171" if risk == "High" else ("#fb923c" if risk == "Medium" else "#34d399")

        st.markdown(f"""
        <div style="margin-bottom:1.2rem">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
                <span style="font-size:0.9rem;color:#94a3b8;font-weight:600">
                    Hybrid Score
                    <span style="font-size:0.72rem;color:#475569">(0.3×LR + 0.7×RF)</span>
                </span>
                <span style="font-family:'Space Mono',monospace;font-size:1.6rem;
                             font-weight:700;color:{h_color}">{h_pct}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{h_pct}%;background:{h_color}"></div>
            </div>
            <div style="font-size:0.72rem;color:#475569;margin-top:0.3rem">
                Medium ≥ {int(HYBRID_THRESHOLD*100)}% · High ≥ {int(HIGH_THRESHOLD*100)}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        prob_bar("Logistic Regression (30%)", res["lr_probability"], LR_THRESHOLD, "#60a5fa")
        prob_bar("Random Forest (70%)",       res["rf_probability"], RF_THRESHOLD, "#a78bfa")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="explanation">💬 {res["explanation"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Decision panels ──────────────────────────────────────────────────
        if risk == "Low":
            st.markdown("""
            <div class="decision d-genuine">
                <div class="d-icon">✅</div>
                <div class="d-label" style="color:#34d399">GENUINE</div>
                <div class="d-sub">Transaction approved · Low hybrid score</div>
            </div>""", unsafe_allow_html=True)

        elif risk == "High":
            st.markdown("""
            <div class="decision d-fraud">
                <div class="d-icon">🚫</div>
                <div class="d-label" style="color:#f87171">FRAUD BLOCKED</div>
                <div class="d-sub">High hybrid score · Blocked immediately</div>
            </div>""", unsafe_allow_html=True)

        elif risk == "Medium":
            if st.session_state.final_decision is not None:
                fd = st.session_state.final_decision
                add_to_history(
                    case_id       = st.session_state.current_case_id,
                    amount        = st.session_state.current_amount,
                    actual_class  = st.session_state.current_actual,
                    result        = res,
                    final_decision= fd,
                )
                if fd == "Genuine":
                    st.markdown("""
                    <div class="decision d-genuine">
                        <div class="d-icon">✅</div>
                        <div class="d-label" style="color:#34d399">GENUINE</div>
                        <div class="d-sub">OTP verified · Transaction approved</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="decision d-fraud">
                        <div class="d-icon">🚫</div>
                        <div class="d-label" style="color:#f87171">FRAUD</div>
                        <div class="d-sub">OTP failed · Transaction blocked</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="decision d-otp">
                    <div class="d-icon">⚠️</div>
                    <div class="d-label" style="color:#fb923c">OTP REQUIRED</div>
                    <div class="d-sub">Suspicious hybrid score · Verify identity</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if not st.session_state.otp_sent:
                    if st.button("📱  Send OTP to Cardholder", use_container_width=True):
                        with st.spinner("Generating OTP..."):
                            time.sleep(0.4)
                            _, err = api("post", "/send-otp",
                                         json={"session_id": SESSION_ID})
                        if err:
                            st.error(err)
                        else:
                            st.session_state.otp_sent = True
                            st.success("✅ OTP sent! Check the backend console.")
                            st.rerun()

                if st.session_state.otp_sent:
                    st.info("📋 OTP printed to server console (simulation mode).")
                    otp_input = st.text_input("Enter OTP:", max_chars=6,
                                               placeholder="6-digit code")
                    if st.button("🔐  Verify OTP", type="primary", use_container_width=True):
                        if not otp_input or len(otp_input.strip()) != 6:
                            st.warning("Please enter a valid 6-digit OTP.")
                        else:
                            with st.spinner("Verifying..."):
                                time.sleep(0.3)
                                vr, err = api("post", "/verify-otp",
                                              json={"otp": otp_input,
                                                    "session_id": SESSION_ID})
                            if err:
                                st.error(err)
                            else:
                                st.session_state.final_decision = vr["decision"]
                                st.rerun()

# ─── Prediction History ──────────────────────────────────────────────────────
history = st.session_state.tx_history

if history:
    st.markdown("---")
    st.markdown("## 📜 Prediction History")

    total    = len(history)
    genuine  = sum(1 for h in history if h["outcome"] in ["Genuine", "OTP ✓ Genuine"])
    fraud    = sum(1 for h in history if h["outcome"] in ["Fraud",   "OTP ✗ Fraud"])
    pending  = sum(1 for h in history if "Pending" in h["outcome"])
    correct  = sum(1 for h in history if h["correct"])
    accuracy = f"{correct/total*100:.0f}%" if total > 0 else "—"

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("🔢 Total",       total)
    s2.metric("✅ Genuine",     genuine)
    s3.metric("🚫 Fraud",       fraud)
    s4.metric("⏳ OTP Pending", pending)
    s5.metric("🎯 Accuracy",    accuracy)

    st.markdown("<br>", unsafe_allow_html=True)

    risk_icon    = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
    actual_icon  = {"Genuine": "✅ Genuine", "Fraud": "🚫 Fraud"}
    outcome_icon = {
        "Genuine"      : "✅ Genuine",
        "Fraud"        : "🚫 Fraud",
        "OTP ✓ Genuine": "✅ OTP Verified",
        "OTP ✗ Fraud"  : "🚫 OTP Failed",
        "OTP Pending"  : "⏳ OTP Pending",
    }

    rows = []
    for h in reversed(history):
        rows.append({
            "Case ID"        : h["case_id"],
            "Amount"         : f"₹{h['amount']:.2f}",
            "Actual Class"   : actual_icon.get(h["actual"], h["actual"]),
            "Risk Level"     : risk_icon.get(h["risk"], h["risk"]),
            "Hybrid Score"   : h["hybrid_pct"],
            "LR Score"       : h["lr_pct"],
            "RF Score"       : h["rf_pct"],
            "Final Decision" : outcome_icon.get(h["outcome"], h["outcome"]),
            "Correct?"       : "✅ Yes" if h["correct"] else "❌ No",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        if st.button("🗑️  Clear History", use_container_width=True):
            st.session_state.tx_history = []
            st.rerun()

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.75rem;
            font-family:'Space Mono',monospace;letter-spacing:1px">
    FRAUDSHIELD · Soft Voting: 0.3×LR + 0.7×RF · Academic Project
</div>""", unsafe_allow_html=True)
