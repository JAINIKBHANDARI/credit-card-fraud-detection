import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from data_loader import load_data
from preprocess import preprocess_data, apply_smote
from train_models import train_logistic_regression
from model_persistence import save_model, load_model

# Load data
df = load_data()

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Apply SMOTE
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Train final model
model = train_logistic_regression(X_train_smote, y_train_smote)

# Save model
save_model(model, "ml/final_fraud_model.pkl")
print("✅ Model saved")

# Load model
loaded_model = load_model("ml/final_fraud_model.pkl")
print("✅ Model loaded")

# Predict one transaction
# Genuine behavior
# Genuine behavior
genuine_sample = X_test[y_test == 0].iloc[[0]].copy()
genuine_sample["Amount"] = 1000

# Fraud behavior
fraud_sample = X_test[y_test == 1].iloc[[0]].copy()
fraud_sample["Amount"] = 5000


def take_action(prob):
    if prob > 0.7:
        return "🚨 High risk → Transaction BLOCKED"
    elif prob > 0.3:
        return "⚠️ Medium risk → OTP verification"
    else:
        return "✅ Low risk → Transaction ALLOWED"


# Predict genuine transaction
prob_genuine = loaded_model.predict_proba(genuine_sample)[0][1]
print("\nGenuine Transaction")
print("Fraud Probability:", prob_genuine)
print(take_action(prob_genuine))


# Predict fraud transaction
prob_fraud = loaded_model.predict_proba(fraud_sample)[0][1]
print("\nFraud-like Transaction")
print("Fraud Probability:", prob_fraud)
print(take_action(prob_fraud))
