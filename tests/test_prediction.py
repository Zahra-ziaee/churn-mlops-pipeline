import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_churn


def test_predict_churn_output_format():
    sample_customer = {
        "Gender": "Female",
        "Age": 45,
        "Married": "No",
        "Number of Dependents": 0,
        "City": "Los Angeles",
        "Zip Code": 90001,
        "Number of Referrals": 0,
        "Tenure in Months": 12,
        "Offer": "No Offer",
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 20.0,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 25.0,
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-Month",
        "Paperless Billing": "Yes",
        "Payment Method": "Credit Card",
        "Monthly Charge": 85.0,
        "Total Charges": 1000.0,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0.0,
        "Total Long Distance Charges": 200.0,
        "Total Revenue": 1200.0,
    }

    result = predict_churn(sample_customer)

    assert "churn_probability" in result
    assert "churn_probability_percent" in result
    assert "prediction" in result
    assert "risk_label" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_label"] in ["High Risk", "Low Risk"]