import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load Trained Model (Joblib)
# -----------------------------
model = joblib.load("final_fraud_model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it is Fraud or Genuine.")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Transaction Features")

features = []

# Time (first column in dataset)
time = st.number_input("Time", value=0.0)
features.append(time)

# V1 to V28
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, format="%.6f")
    features.append(val)

# Amount (last column)
amount = st.number_input("Amount", value=0.0)
features.append(amount)

# Convert to numpy array
input_data = np.array(features).reshape(1, -1)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Transaction"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Fraud Probability: {probability:.5f}")

    if prediction == 1:
        st.error("⚠ FRAUD Transaction Detected (High Risk)")
    else:
        st.success("✅ Genuine Transaction (Low Risk)")
