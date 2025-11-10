import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def main():
    csv_path = "data/raw/customer_churn.csv"

    print("PWD:", os.getcwd())
    print("Looking for:", os.path.abspath(csv_path))
    print("Exists:", os.path.exists(csv_path))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Churn" not in df.columns:
        raise KeyError("Column 'Churn' not found in dataset")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Training complete | Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
