from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_missing_required_fields():
    response = client.post("/predict", json=dict())

    assert response.status_code == 422

    errors = response.json()["detail"]

    missing_fields = [
        error["loc"][-1]
        for error in errors
    ]

    assert "age" in missing_fields
    assert "contract" in missing_fields

    
def test_predict_rejects_partial_customer_data():
    response = client.post(
        "/predict",
        json=dict(age=45, contract="Month-to-Month")
    )

    assert response.status_code == 422


def test_predict_accepts_complete_customer_data():
    customer = {
        "gender": "Female",
        "age": 45,
        "married": "No",
        "number_of_dependents": 0,
        "city": "Los Angeles",
        "zip_code": 90001,
        "number_of_referrals": 0,
        "tenure_in_months": 12,
        "offer": "No Offer",
        "phone_service": "Yes",
        "avg_monthly_long_distance_charges": 20.0,
        "multiple_lines": "No",
        "internet_service": "Yes",
        "internet_type": "Fiber Optic",
        "avg_monthly_gb_download": 25.0,
        "online_security": "No",
        "online_backup": "No",
        "device_protection_plan": "No",
        "premium_tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "streaming_music": "No",
        "unlimited_data": "Yes",
        "contract": "Month-to-Month",
        "paperless_billing": "Yes",
        "payment_method": "Credit Card",
        "monthly_charge": 85.0,
        "total_charges": 1000.0,
        "total_refunds": 0.0,
        "total_extra_data_charges": 0.0,
        "total_long_distance_charges": 200.0,
        "total_revenue": 1200.0,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_label"] in ["High Risk", "Low Risk"]
