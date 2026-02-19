#library imports
from fastapi import FastAPI
import joblib
import numpy as np

# create app object
app = FastAPI(title="PF Auto Prediction API")

# Load once at startup
model = joblib.load("Anuja.pkl")
df = joblib.load("Anuja_df_model.pkl")

features = [
    'IR','IY','IB',
    'Avg_I',
    'VRN','VYN','VBN',
    'AVG_VLL',
    'KW','Freq',
    'Avg_PF',
    'PF_lag_1','PF_lag_5','PF_roll_mean_5'
]


def pf_status(pf):
    if pf >= 0.95:
        return "Normal"
    elif pf >= 0.90:
        return "Warning"
    else:
        return "Critical"


@app.get("/")
def auto_prediction():

    latest = df.iloc[-1]
    X = np.array([latest[features]])

    predicted_pf = float(model.predict(X)[0])
    current_pf = float(latest["Avg_PF"])

    trend = (
        "Improving" if predicted_pf > current_pf
        else "Dropping" if predicted_pf < current_pf
        else "Stable"
    )

    return {
        "current_pf": current_pf,
        "predicted_future_pf_15min": predicted_pf,
        "status": pf_status(predicted_pf),
        "trend": trend,
        "difference": round(predicted_pf - current_pf, 4)
    }
