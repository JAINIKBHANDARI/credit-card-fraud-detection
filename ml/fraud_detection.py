# -------------------------------
# 1. Import required libraries
# -------------------------------

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


# -------------------------------
# 2. Load the dataset
# -------------------------------

# Load Kaggle Credit Card Fraud dataset
df = pd.read_csv("data/creditcard.csv")

# Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum())


# -------------------------------
# 3. Separate features and target
# -------------------------------

# X -> input features
# y -> target variable (Class: 0 = Genuine, 1 = Fraud)
X = df.drop('Class', axis=1)
y = df['Class']


# -------------------------------
# 4. Feature Scaling
# -------------------------------

# Scale the 'Amount' column to normalize values
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])


# -------------------------------
# 5. Train-Test Split
# -------------------------------

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# =========================================================
# 6. Logistic Regression WITHOUT SMOTE (Baseline Model)
# =========================================================

# Create Logistic Regression model
lr_no_smote = LogisticRegression(max_iter=3000)

# Train the model on original (imbalanced) data
lr_no_smote.fit(X_train, y_train)

# Make predictions
y_pred_no_smote = lr_no_smote.predict(X_test)

# Evaluation results
print("\nConfusion Matrix (WITHOUT SMOTE):")
print(confusion_matrix(y_test, y_pred_no_smote))

print("\nClassification Report (WITHOUT SMOTE):")
print(classification_report(y_test, y_pred_no_smote))


# =========================================================
# 7. Apply SMOTE to Handle Class Imbalance
# =========================================================

# Apply SMOTE only on training data to avoid data leakage
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution before SMOTE:")
print(y_train.value_counts())

print("\nClass distribution after SMOTE:")
print(y_train_resampled.value_counts())


# =========================================================
# 8. Logistic Regression WITH SMOTE (Improved Model)
# =========================================================

# Train Logistic Regression on balanced data
lr_model = LogisticRegression(max_iter=3000)
lr_model.fit(X_train_resampled, y_train_resampled)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Evaluation results
print("\nConfusion Matrix (WITH SMOTE):")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report (WITH SMOTE):")
print(classification_report(y_test, y_pred_lr))
