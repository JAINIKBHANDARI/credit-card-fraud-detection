import pandas as pd

# Load dataset
df = pd.read_csv("../data/creditcard.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df['Class'].value_counts())
# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nDataset Description:")
print(df.describe())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())


