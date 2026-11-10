
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="A FastAPI service for predicting telecom customer churn risk.",
    version="1.0.0",
)


# --------------------------------------------------
# Customer Input Schema
# --------------------------------------------------

class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(...)
    age: int = Field(..., ge=18, le=100)
    married: Literal["Yes", "No"] = Field(...)
    number_of_dependents: int = Field(..., ge=0)

    city: str = Field(...)
    zip_code: int = Field(...)

    number_of_referrals: int = Field(..., ge=0)
    tenure_in_months: int = Field(..., ge=0)

    offer: str = Field(...)
    phone_service: Literal["Yes", "No"] = Field(...)
    avg_monthly_long_distance_charges: float = Field(..., ge=0)
    multiple_lines: str = Field(...)

    internet_service: Literal["Yes", "No"] = Field(...)
    internet_type: str = Field(...)
    avg_monthly_gb_download: float = Field(..., ge=0)

    online_security: str = Field(...)
    online_backup: str = Field(...)
    device_protection_plan: str = Field(...)
    premium_tech_support: str = Field(...)

    streaming_tv: str = Field(...)
    streaming_movies: str = Field(...)
    streaming_music: str = Field(...)
    unlimited_data: str = Field(...)

    contract: str = Field(...)
    paperless_billing: Literal["Yes", "No"] = Field(...)
    payment_method: str = Field(...)

    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    total_refunds: float = Field(..., ge=0)
    total_extra_data_charges: float = Field(..., ge=0)
    total_long_distance_charges: float = Field(..., ge=0)
    total_revenue: float = Field(..., ge=0)


# --------------------------------------------------
# Convert API Input to Model Input
# --------------------------------------------------

def convert_api_input_to_model_input(customer: CustomerInput) -> dict:
    """
    Convert snake_case API fields to the original
    dataset column names expected by the trained model.
    """
    return {
        "Gender": customer.gender,
        "Age": customer.age,
        "Married": customer.married,
        "Number of Dependents": customer.number_of_dependents,
        "City": customer.city,
        "Zip Code": customer.zip_code,
        "Number of Referrals": customer.number_of_referrals,
        "Tenure in Months": customer.tenure_in_months,
        "Offer": customer.offer,
        "Phone Service": customer.phone_service,
        "Avg Monthly Long Distance Charges": customer.avg_monthly_long_distance_charges,
        "Multiple Lines": customer.multiple_lines,
        "Internet Service": customer.internet_service,
        "Internet Type": customer.internet_type,
        "Avg Monthly GB Download": customer.avg_monthly_gb_download,
        "Online Security": customer.online_security,
        "Online Backup": customer.online_backup,
        "Device Protection Plan": customer.device_protection_plan,
        "Premium Tech Support": customer.premium_tech_support,
        "Streaming TV": customer.streaming_tv,
        "Streaming Movies": customer.streaming_movies,
        "Streaming Music": customer.streaming_music,
        "Unlimited Data": customer.unlimited_data,
        "Contract": customer.contract,
        "Paperless Billing": customer.paperless_billing,
        "Payment Method": customer.payment_method,
        "Monthly Charge": customer.monthly_charge,
        "Total Charges": customer.total_charges,
        "Total Refunds": customer.total_refunds,
        "Total Extra Data Charges": customer.total_extra_data_charges,
        "Total Long Distance Charges": customer.total_long_distance_charges,
        "Total Revenue": customer.total_revenue,
    }


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
    }


# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Gradient Boosting churn model",
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerInput):
    """
    Validate customer input, transform column names,
    and return churn prediction results.
    """
    customer_data = convert_api_input_to_model_input(customer)

    prediction_result = predict_churn(customer_data)

    return prediction_result
