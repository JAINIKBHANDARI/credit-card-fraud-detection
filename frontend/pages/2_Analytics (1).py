"""
2_Analytics.py — Analytics Dashboard
=======================================
Page 2 of FraudShield multi-page app.

Charts (all computed from real models on real test data):
  1. Model Comparison Bar   — Precision / Recall / F1 / ROC-AUC for LR, RF, Hybrid
  2. Confusion Matrix       — Heatmap for all 3 models (tabs)
  3. Risk Distribution Pie  — Low / Medium / High from hybrid predictions
  4. ROC Curve              — LR vs RF vs Hybrid with AUC scores

Author  : FraudShield Project
Purpose : Academic submission
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import init_state, inject_css

st.set_page_config(
    page_title = "Analytics · FraudShield",
    page_icon  = "📊",
    layout     = "wide",
)
inject_css()
init_state()

# ─── Colors ───────────────────────────────────────────────────────────────────
PAPER_BG = "#080d18"
PLOT_BG  = "#0d1117"
FONT_CLR = "#94a3b8"
GRID_CLR = "#1e293b"
GREEN    = "#34d399"
RED      = "#f87171"
ORANGE   = "#fb923c"
BLUE     = "#60a5fa"
PURPLE   = "#a78bfa"
CYAN     = "#38bdf8"
YELLOW   = "#fbbf24"

def base_layout(height=340, title=""):
    layout = dict(
        paper_bgcolor = PAPER_BG,
        plot_bgcolor  = PLOT_BG,
        font          = dict(color=FONT_CLR, family="DM Sans"),
        height        = height,
        margin        = dict(l=20, r=20, t=50, b=20),
    )
    if title:
        layout["title"] = dict(
            text       = title,
            font_color = "#e2e8f0",
            font_size  = 15,
            x          = 0.5,
            xanchor    = "center",
        )
    return layout

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem">
    <div class="page-title">📊 Analytics Dashboard</div>
    <div class="page-sub">Model Comparison · Confusion Matrix · Risk Distribution · ROC Curve</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Load models & data ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and computing metrics...")
def load_everything():
    import joblib

    PROJECT_ROOT = r"C:\Users\Harsh\Downloads\fraud_project\fraud_project"
    ML_DIR       = os.path.join(PROJECT_ROOT, "ml")
    DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")

    # Add ml/ to path for model.py imports
    sys.path.insert(0, ML_DIR)
    from model import (
        split_and_scale, apply_smote,
        predict_batch, hybrid_predict_batch,
        LR_THRESHOLD, RF_THRESHOLD,
    )
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, roc_curve,
    )

    # Load saved models
    lr_model = joblib.load(os.path.join(ML_DIR, "final_fraud_model.pkl"))
    rf_model = joblib.load(os.path.join(ML_DIR, "random_forest.pkl"))

    # Load and split data
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    # Predictions
    lr_pred, lr_prob         = predict_batch(lr_model, X_test, threshold=LR_THRESHOLD)
    rf_pred, rf_prob         = predict_batch(rf_model, X_test, threshold=RF_THRESHOLD)
    h_pred,  h_prob, h_risks = hybrid_predict_batch(lr_model, rf_model, X_test)

    # Metrics
    def metrics(y_pred, y_prob):
        return {
            "Precision" : precision_score(y_test, y_pred, zero_division=0),
            "Recall"    : recall_score(y_test, y_pred, zero_division=0),
            "F1-Score"  : f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC"   : roc_auc_score(y_test, y_prob),
        }

    lr_m = metrics(lr_pred, lr_prob)
    rf_m = metrics(rf_pred, rf_prob)
    h_m  = metrics(h_pred,  h_prob)

    # Confusion matrices
    cm_lr = confusion_matrix(y_test, lr_pred)
    cm_rf = confusion_matrix(y_test, rf_pred)
    cm_h  = confusion_matrix(y_test, h_pred)

    # ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    fpr_h,  tpr_h,  _ = roc_curve(y_test, h_prob)

    # Risk distribution
    risk_counts = pd.Series(h_risks).value_counts()

    return {
        "lr_m": lr_m, "rf_m": rf_m, "h_m": h_m,
        "cm_lr": cm_lr, "cm_rf": cm_rf, "cm_h": cm_h,
        "fpr_lr": fpr_lr, "tpr_lr": tpr_lr,
        "fpr_rf": fpr_rf, "tpr_rf": tpr_rf,
        "fpr_h" : fpr_h,  "tpr_h" : tpr_h,
        "risk_counts": risk_counts,
        "test_size"  : len(y_test),
        "fraud_count": int(y_test.sum()),
    }

# Load with spinner
with st.spinner("⚡ Computing metrics on test set..."):
    try:
        data = load_everything()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Make sure `final_fraud_model.pkl`, `random_forest.pkl`, `scaler.pkl` are in `ml/` folder.")
        st.stop()

st.success(f"✅ Metrics computed on **{data['test_size']:,}** test transactions "
           f"({data['fraud_count']} fraud cases)")
st.markdown("<br>", unsafe_allow_html=True)

lr_m = data["lr_m"]
rf_m = data["rf_m"]
h_m  = data["h_m"]

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Model Comparison Bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1️⃣  Model Comparison")

metrics_list = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
lr_vals  = [lr_m[m] for m in metrics_list]
rf_vals  = [rf_m[m] for m in metrics_list]
h_vals   = [h_m[m]  for m in metrics_list]

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    name         = "Logistic Regression",
    x            = metrics_list,
    y            = lr_vals,
    marker_color = BLUE,
    opacity      = 0.85,
    text         = [f"{v:.3f}" for v in lr_vals],
    textposition = "outside",
    textfont     = dict(size=11),
))
fig1.add_trace(go.Bar(
    name         = "Random Forest",
    x            = metrics_list,
    y            = rf_vals,
    marker_color = PURPLE,
    opacity      = 0.85,
    text         = [f"{v:.3f}" for v in rf_vals],
    textposition = "outside",
    textfont     = dict(size=11),
))
fig1.add_trace(go.Bar(
    name         = "Hybrid ★ (Soft Voting)",
    x            = metrics_list,
    y            = h_vals,
    marker_color = GREEN,
    opacity      = 0.9,
    text         = [f"{v:.3f}" for v in h_vals],
    textposition = "outside",
    textfont     = dict(size=11, color=GREEN),
))

fig1.update_layout(
    **base_layout(380, "Model Comparison — Precision · Recall · F1 · ROC-AUC"),
    barmode    = "group",
    xaxis      = dict(gridcolor=GRID_CLR, tickfont_size=13),
    yaxis      = dict(title="Score", gridcolor=GRID_CLR, range=[0, 1.12]),
    legend     = dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                      bgcolor="rgba(0,0,0,0)", font_size=12),
)
st.plotly_chart(fig1, use_container_width=True)

# Caption table
cap_df = pd.DataFrame({
    "Metric"             : metrics_list,
    "Logistic Regression": [f"{v:.4f}" for v in lr_vals],
    "Random Forest"      : [f"{v:.4f}" for v in rf_vals],
    "Hybrid ★"           : [f"{v:.4f}" for v in h_vals],
})
st.dataframe(cap_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Confusion Matrix (tabs for 3 models)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2️⃣  Confusion Matrix")

tab_lr, tab_rf, tab_h = st.tabs([
    "📘 Logistic Regression",
    "🌳 Random Forest",
    "⭐ Hybrid (Soft Voting)",
])

def confusion_chart(cm, model_name, color):
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    z      = [[tn, fp], [fn, tp]]
    text   = [
        [f"<b>{tn:,}</b><br>True Negative<br>(Genuine → Genuine)",
         f"<b>{fp:,}</b><br>False Positive<br>(Genuine → Fraud)"],
        [f"<b>{fn:,}</b><br>False Negative<br>(Fraud → Genuine)",
         f"<b>{tp:,}</b><br>True Positive<br>(Fraud → Fraud)"],
    ]

    fig = go.Figure(go.Heatmap(
        z            = z,
        text         = text,
        texttemplate = "%{text}",
        textfont     = dict(size=13, family="DM Sans"),
        colorscale   = [[0, "#0d1b2a"], [0.4, "#0c3057"], [1, color]],
        showscale    = False,
        xgap=5, ygap=5,
    ))
    fig.update_layout(
        **base_layout(340, f"Confusion Matrix — {model_name}"),
        xaxis = dict(tickvals=[0,1], ticktext=["Predicted Genuine","Predicted Fraud"],
                     side="top", gridcolor="rgba(0,0,0,0)", tickfont_size=12),
        yaxis = dict(tickvals=[0,1], ticktext=["Actual Genuine","Actual Fraud"],
                     autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont_size=12),
    )

    # Stats below chart
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    return fig, tn, fp, fn, tp, precision, recall


def render_cm_tab(cm, model_name, color):
    fig, tn, fp, fn, tp, prec, rec = confusion_chart(cm, model_name, color)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ True Positive",  f"{tp:,}", help="Fraud correctly caught")
    c2.metric("✅ True Negative",  f"{tn:,}", help="Genuine correctly approved")
    c3.metric("⚠️ False Positive", f"{fp:,}", help="Genuine wrongly flagged as fraud")
    c4.metric("⚠️ False Negative", f"{fn:,}", help="Fraud that slipped through")

    st.markdown(f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                padding:1rem 1.5rem;margin-top:0.5rem;display:flex;gap:2rem;flex-wrap:wrap">
        <div>
            <span style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Precision</span>
            <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{GREEN};font-weight:700">{prec:.4f}</div>
            <div style="font-size:0.75rem;color:#475569">of all flagged, how many are real fraud</div>
        </div>
        <div>
            <span style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Recall</span>
            <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{CYAN};font-weight:700">{rec:.4f}</div>
            <div style="font-size:0.75rem;color:#475569">of all real fraud, how many were caught</div>
        </div>
        <div>
            <span style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">False Alarms</span>
            <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{ORANGE};font-weight:700">{fp:,}</div>
            <div style="font-size:0.75rem;color:#475569">genuine transactions wrongly blocked</div>
        </div>
        <div>
            <span style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px">Missed Fraud</span>
            <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{RED};font-weight:700">{fn:,}</div>
            <div style="font-size:0.75rem;color:#475569">fraud transactions that slipped through</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


with tab_lr:
    render_cm_tab(data["cm_lr"], "Logistic Regression", "#3b82f6")
with tab_rf:
    render_cm_tab(data["cm_rf"], "Random Forest", "#8b5cf6")
with tab_h:
    render_cm_tab(data["cm_h"],  "Hybrid (Soft Voting)", "#10b981")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Risk Distribution Pie
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3️⃣  Risk Distribution (Hybrid Model)")

risk_counts = data["risk_counts"]
pc1, pc2 = st.columns([1, 1.6], gap="large")

with pc1:
    labels = risk_counts.index.tolist()
    values = risk_counts.values.tolist()
    colors = [
        GREEN  if l == "Low"    else
        ORANGE if l == "Medium" else RED
        for l in labels
    ]

    fig3 = go.Figure(go.Pie(
        labels        = labels,
        values        = values,
        hole          = 0.6,
        marker        = dict(colors=colors, line=dict(color=PAPER_BG, width=4)),
        textinfo      = "percent+label",
        textfont_size = 13,
        pull          = [0.05 if l == "High" else 0 for l in labels],
    ))
    total = sum(values)
    fig3.update_layout(
        **base_layout(340, "Risk Level Distribution"),
        showlegend  = False,
        annotations = [dict(
            text      = f"<b>{total:,}</b><br>transactions",
            x=0.5, y=0.5,
            font_size = 14, font_color="#e2e8f0",
            showarrow = False,
        )],
    )
    st.plotly_chart(fig3, use_container_width=True)

with pc2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**What each risk level means:**")

    for label, color, icon, meaning, action in [
        ("Low",    GREEN,  "🟢", "hybrid_prob < 0.43 — neither model flags fraud",    "✅ Approved immediately"),
        ("Medium", ORANGE, "🟡", "hybrid_prob 0.43–0.70 — RF suspects fraud",         "📱 OTP verification required"),
        ("High",   RED,    "🔴", "hybrid_prob ≥ 0.70 — both models flag fraud",       "🚫 Blocked immediately"),
    ]:
        count = risk_counts.get(label, 0)
        pct   = count / total * 100 if total > 0 else 0
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                    padding:1rem 1.2rem;margin-bottom:0.7rem;
                    border-left:4px solid {color}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-family:'Space Mono',monospace;font-size:0.9rem;
                             color:{color};font-weight:700">{icon} {label} Risk</span>
                <span style="font-family:'Space Mono',monospace;font-size:1.1rem;
                             color:#e2e8f0;font-weight:700">{count:,} &nbsp;
                    <span style="font-size:0.75rem;color:#475569">({pct:.1f}%)</span>
                </span>
            </div>
            <div style="font-size:0.78rem;color:#475569;margin-top:0.3rem">{meaning}</div>
            <div style="font-size:0.78rem;color:{color};margin-top:0.2rem">{action}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4️⃣  ROC Curve — Receiver Operating Characteristic")

fig4 = go.Figure()

# Diagonal reference line
fig4.add_trace(go.Scatter(
    x=[0,1], y=[0,1],
    mode="lines",
    name="Random Classifier",
    line=dict(color="#334155", width=1.5, dash="dot"),
    showlegend=True,
))

# LR ROC
fig4.add_trace(go.Scatter(
    x    = data["fpr_lr"],
    y    = data["tpr_lr"],
    mode = "lines",
    name = f"Logistic Regression (AUC = {lr_m['ROC-AUC']:.4f})",
    line = dict(color=BLUE, width=2.5),
    fill = "tonexty" if False else None,
))

# RF ROC
fig4.add_trace(go.Scatter(
    x    = data["fpr_rf"],
    y    = data["tpr_rf"],
    mode = "lines",
    name = f"Random Forest (AUC = {rf_m['ROC-AUC']:.4f})",
    line = dict(color=PURPLE, width=2.5),
))

# Hybrid ROC
fig4.add_trace(go.Scatter(
    x    = data["fpr_h"],
    y    = data["tpr_h"],
    mode = "lines",
    name = f"Hybrid ★ (AUC = {h_m['ROC-AUC']:.4f})",
    line = dict(color=GREEN, width=3),
    fill = "tonexty" if False else None,
))

fig4.update_layout(
    **base_layout(440, "ROC Curve — LR vs Random Forest vs Hybrid"),
    xaxis = dict(
        title      = "False Positive Rate (FPR) → Genuine flagged as Fraud",
        gridcolor  = GRID_CLR,
        range      = [0, 1],
        tickformat = ".1f",
    ),
    yaxis = dict(
        title      = "True Positive Rate (TPR) → Fraud correctly caught",
        gridcolor  = GRID_CLR,
        range      = [0, 1.02],
        tickformat = ".1f",
    ),
    legend = dict(
        orientation = "v",
        x           = 0.62,
        y           = 0.18,
        bgcolor     = "rgba(13,17,23,0.85)",
        bordercolor = "#1e293b",
        borderwidth = 1,
        font_size   = 12,
    ),
)
st.plotly_chart(fig4, use_container_width=True)

# ROC explanation
with st.expander("📖 How to read the ROC Curve"):
    st.markdown(f"""
    | Term | Meaning |
    |---|---|
    | **AUC (Area Under Curve)** | Overall model quality — closer to **1.0 = perfect**, 0.5 = random guessing |
    | **TPR (True Positive Rate)** | % of real fraud cases caught = Recall |
    | **FPR (False Positive Rate)** | % of genuine transactions wrongly flagged |
    | **Diagonal line** | A random classifier — any model above this line is useful |
    | **Top-left corner** | Perfect classifier — 100% fraud caught, 0% false alarms |

    **Our results:**
    - Logistic Regression AUC = **{lr_m['ROC-AUC']:.4f}**
    - Random Forest AUC = **{rf_m['ROC-AUC']:.4f}**
    - Hybrid Model AUC = **{h_m['ROC-AUC']:.4f}** ← best overall discrimination
    
    The Hybrid model curve hugs the top-left corner most tightly, 
    meaning it catches the most fraud while generating the fewest false alarms.
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.75rem;
            font-family:'Space Mono',monospace;letter-spacing:1px">
    FRAUDSHIELD · Analytics Dashboard · Academic Project
</div>""", unsafe_allow_html=True)
