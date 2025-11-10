import joblib
import pandas as pd

def predict(input_dict):
    saved = joblib.load("models/model.joblib")
    model = saved["model"]
    cols = saved["columns"]
    scaler = saved["scaler"]

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df, drop_first=True)

    # align columns
    for c in cols:
        if c not in df:
            df[c] = 0

    df = df[cols]
    df_scaled = scaler.transform(df)

    return model.predict(df_scaled)[0]
