import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROP_COLS = ["customerID"]  # ignore identifiers

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()

    # Drop irrelevant IDs if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "Churn" not in df.columns:
        raise KeyError("Column 'Churn' not found in dataset")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Scale numerical
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns, scaler
