
from fastapi.testclient import TestClient

from api.main import app, CustomerInput


# Create FastAPI test client
client = TestClient(app)


# --------------------------------------------------
# Test 1: Reject completely empty requests
# --------------------------------------------------

def test_predict_missing_required_fields():
    response = client.post(
        "/predict",
        json=dict()
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    missing_fields = [
        error["loc"][-1]
        for error in errors
    ]

    assert "age" in missing_fields
    assert "contract" in missing_fields


# --------------------------------------------------
# Test 2: Reject incomplete customer data
# --------------------------------------------------

def test_predict_rejects_partial_customer_data():
    response = client.post(
        "/predict",
        json={
            "age": 45,
            "contract": "Month-to-Month"
        }
    )

    assert response.status_code == 422


# --------------------------------------------------
# Test 3: Accept valid 25-feature customer input
# --------------------------------------------------

def test_predict_accepts_complete_customer_data():

    customer = {
        "gender": "Female",
        "age": 45,
        "married": "No",
        "number_of_dependents": 0,
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
    }

    # Verify the new input contract
    assert len(customer) == 25
    assert len(CustomerInput.model_fields) == 25

    # Send prediction request
    response = client.post(
        "/predict",
        json=customer
    )

    # Verify HTTP status
    assert response.status_code == 200

    result = response.json()

    # Verify response structure
    assert "churn_probability" in result
    assert "churn_probability_percent" in result
    assert "prediction" in result
    assert "risk_label" in result

    # Verify prediction values
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_label"] in [
        "High Risk",
        "Low Risk"
    ]

    # Verify consistency between prediction and label
    expected_label = (
        "High Risk"
        if result["prediction"] == 1
        else "Low Risk"
    )

    assert result["risk_label"] == expected_label

def test_predict_rejects_extra_features():
    response = client.post(
        "/predict",
        json={
            "age": 45,
            "contract": "Month-to-Month",
            "total_revenue": 1200.0,
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["type"] == "extra_forbidden"
        and error["loc"] == ["body", "total_revenue"]
        for error in errors
    )
