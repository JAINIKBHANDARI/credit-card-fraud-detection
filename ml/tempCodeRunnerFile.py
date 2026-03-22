def load_data(path: str) -> pd.DataFrame:
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"   Shape: {df.shape}  |  Fraud: {df['Class'].sum()}  |  Genuine: {(df['Class']==0).sum()}")
    return df