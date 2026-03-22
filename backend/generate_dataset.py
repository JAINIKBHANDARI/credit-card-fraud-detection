"""
generate_dataset.py — Synthetic Test Case Generator
=====================================================
Generates 25 synthetic transactions matching creditcard.csv format.
Includes Time column (required by trained model).
Saves to data/test_cases.csv

Run from backend folder:
    python generate_dataset.py

Author  : FraudShield Project
Purpose : Academic submission
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

GENUINE_PARAMS = {
    "V1":  (0.00, 1.95), "V2":  (0.00, 1.65), "V3":  (0.00, 1.52),
    "V4":  (0.00, 1.42), "V5":  (0.00, 1.43), "V6":  (0.00, 1.33),
    "V7":  (0.00, 1.24), "V8":  (0.00, 1.19), "V9":  (0.00, 1.10),
    "V10": (0.00, 1.09), "V11": (0.00, 1.02), "V12": (0.00, 1.00),
    "V13": (0.00, 1.00), "V14": (0.00, 0.96), "V15": (0.00, 0.91),
    "V16": (0.00, 0.87), "V17": (0.00, 0.85), "V18": (0.00, 0.84),
    "V19": (0.00, 0.81), "V20": (0.00, 0.77), "V21": (0.00, 0.73),
    "V22": (0.00, 0.73), "V23": (0.00, 0.62), "V24": (0.00, 0.60),
    "V25": (0.00, 0.52), "V26": (0.00, 0.48), "V27": (0.00, 0.40),
    "V28": (0.00, 0.33),
    "Amount": (88.0, 250.0),
}

FRAUD_PARAMS = {
    "V1":  (-4.77, 3.16), "V2":  (3.94, 4.29), "V3":  (-7.03, 4.73),
    "V4":  (4.40, 2.90),  "V5":  (-3.15, 4.39), "V6":  (-1.40, 2.54),
    "V7":  (-5.57, 5.57), "V8":  (0.57, 3.74),  "V9":  (-2.58, 2.51),
    "V10": (-4.55, 3.18), "V11": (4.02, 2.83),  "V12": (-7.08, 5.47),
    "V13": (-0.08, 1.12), "V14": (-7.63, 4.95), "V15": (0.18, 1.00),
    "V16": (-4.13, 3.40), "V17": (-8.35, 6.78), "V18": (-2.41, 2.47),
    "V19": (0.73, 1.94),  "V20": (0.58, 3.63),  "V21": (0.64, 2.36),
    "V22": (0.05, 0.82),  "V23": (-0.14, 1.31), "V24": (-0.14, 0.64),
    "V25": (-0.11, 0.67), "V26": (-0.13, 0.63), "V27": (0.30, 1.13),
    "V28": (0.13, 0.62),
    "Amount": (122.0, 256.0),
}


def generate_rows(params, n, label):
    data = {}
    for feat, (mean, std) in params.items():
        if feat == "Amount":
            data[feat] = np.abs(np.random.normal(mean, std, n)).round(2)
        else:
            data[feat] = np.random.normal(mean, std, n).round(6)
    # Time: simulate seconds elapsed (0 to 172800 = 48 hours)
    data["Time"]         = np.random.uniform(0, 172800, n).round(0)
    data["actual_class"] = label
    return pd.DataFrame(data)


def main():
    n_genuine = 17
    n_fraud   = 8

    df = pd.concat([
        generate_rows(GENUINE_PARAMS, n_genuine, 0),
        generate_rows(FRAUD_PARAMS,   n_fraud,   1),
    ], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "case_id", [f"TXN-{i+1:03d}" for i in range(len(df))])

    # Save to data/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path   = os.path.join(data_dir, "test_cases.csv")

    df.to_csv(out_path, index=False)

    print(f"✅ Generated {len(df)} test cases → {os.path.abspath(out_path)}")
    print(f"   Genuine : {n_genuine}")
    print(f"   Fraud   : {n_fraud}")
    print(f"\nPreview:")
    print(df[["case_id", "Amount", "Time", "actual_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
