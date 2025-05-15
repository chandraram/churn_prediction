from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_churn():
    payload = {
        "tenure": 24,
        "MonthlyCharges": 75.25,
        "TotalCharges": 1800.50,
        "Contract": "Month-to-month"
    }

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "churn_prediction" in data
    assert "probability" in data
    assert data["churn_prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
