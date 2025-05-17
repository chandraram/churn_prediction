# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# # Load saved model and transformers
# model = joblib.load("model/churn_model.pkl")
# scaler = joblib.load("model/scaler.pkl")
# encoder = joblib.load("model/encoder.pkl")

# app = FastAPI()

# class CustomerData(BaseModel):
#     tenure: int
#     MonthlyCharges: float
#     TotalCharges: float
#     Contract: str

# @app.post("/predict/model")
# def predict_churn(data: CustomerData):
#     # Transform input
#     contract_encoded = encoder.transform([data.Contract])[0]
#     features = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges, contract_encoded]])
#     scaled = scaler.transform(features)
    
#     # Predict
#     prediction = model.predict(scaled)[0]
#     probability = model.predict_proba(scaled)[0][1]

#     return {
#         "churn_prediction": "Yes" if prediction == 1 else "No",
#         "probability": round(probability, 2)
#     }

# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import boto3
# import json
# import uuid
# import datetime

# # Load saved model and transformers
# model = joblib.load("model/churn_model.pkl")
# scaler = joblib.load("model/scaler.pkl")
# encoder = joblib.load("model/encoder.pkl")

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize S3 client (uses AWS credentials from env or config)
# s3 = boto3.client("s3")
# BUCKET_NAME = "churn79327"

# class CustomerData(BaseModel):
#     tenure: int
#     MonthlyCharges: float
#     TotalCharges: float
#     Contract: str

# @app.post("/predict/model")
# def predict_churn(data: CustomerData):
#     # Transform input
#     contract_encoded = encoder.transform([data.Contract])[0]
#     features = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges, contract_encoded]])
#     scaled = scaler.transform(features)

#     # Predict
#     prediction = model.predict(scaled)[0]
#     probability = model.predict_proba(scaled)[0][1]

#     # Prepare log entry
#     timestamp = datetime.datetime.utcnow().isoformat()
#     log_data = {
#         "timestamp": timestamp,
#         "input": data.dict(),
#         "prediction": "Yes" if prediction == 1 else "No",
#         "probability": round(probability, 2)
#     }

#     # Save to S3
#     log_file_name = f"logs/{uuid.uuid4()}.json"
#     s3.put_object(
#         Bucket=BUCKET_NAME,
#         Key=log_file_name,
#         Body=json.dumps(log_data),
#         ContentType='application/json'
#     )

#     # Return response
#     return {
#         "churn_prediction": log_data["prediction"],
#         "probability": log_data["probability"]
#     }



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import boto3
import json
import uuid
import datetime
import os

# Load saved model and transformers
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

app = FastAPI()

# Get S3 bucket name from env or default
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "churn79327")

# Initialize S3 client
s3 = boto3.client("s3")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str

@app.post("/predict/model")
def predict_churn(data: CustomerData):
    contract_encoded = encoder.transform([data.Contract])[0]
    features = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges, contract_encoded]])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    timestamp = datetime.datetime.utcnow().isoformat()
    log_data = {
        "timestamp": timestamp,
        "input": data.dict(),
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(probability, 2)
    }

    log_file_name = f"logs/{uuid.uuid4()}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=log_file_name,
        Body=json.dumps(log_data),
        ContentType='application/json'
    )

    return {
        "churn_prediction": log_data["prediction"],
        "probability": log_data["probability"]
    }
