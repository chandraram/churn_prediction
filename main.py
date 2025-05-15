from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load saved model and transformers
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

app = FastAPI()

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Transform input
    contract_encoded = encoder.transform([data.Contract])[0]
    features = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges, contract_encoded]])
    scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "probability": round(probability, 2)
    }
