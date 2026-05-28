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


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(default="Female")
    age: int = Field(default=45, ge=18, le=100)
    married: Literal["Yes", "No"] = Field(default="No")
    number_of_dependents: int = Field(default=0, ge=0)
    city: str = Field(default="Los Angeles")
    zip_code: int = Field(default=90001)
    number_of_referrals: int = Field(default=0, ge=0)
    tenure_in_months: int = Field(default=12, ge=0)
    offer: str = Field(default="No Offer")
    phone_service: Literal["Yes", "No"] = Field(default="Yes")
    avg_monthly_long_distance_charges: float = Field(default=20.0, ge=0)
    multiple_lines: str = Field(default="No")
    internet_service: Literal["Yes", "No"] = Field(default="Yes")
    internet_type: str = Field(default="Fiber Optic")
    avg_monthly_gb_download: float = Field(default=25.0, ge=0)
    online_security: str = Field(default="No")
    online_backup: str = Field(default="No")
    device_protection_plan: str = Field(default="No")
    premium_tech_support: str = Field(default="No")
    streaming_tv: str = Field(default="No")
    streaming_movies: str = Field(default="No")
    streaming_music: str = Field(default="No")
    unlimited_data: str = Field(default="Yes")
    contract: str = Field(default="Month-to-Month")
    paperless_billing: Literal["Yes", "No"] = Field(default="Yes")
    payment_method: str = Field(default="Credit Card")
    monthly_charge: float = Field(default=85.0, ge=0)
    total_charges: float = Field(default=1000.0, ge=0)
    total_refunds: float = Field(default=0.0, ge=0)
    total_extra_data_charges: float = Field(default=0.0, ge=0)
    total_long_distance_charges: float = Field(default=200.0, ge=0)
    total_revenue: float = Field(default=1200.0, ge=0)


def convert_api_input_to_model_input(customer: CustomerInput) -> dict:
    """
    Convert snake_case API input into original dataset column names expected by the model.
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


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Gradient Boosting churn model",
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    customer_data = convert_api_input_to_model_input(customer)
    prediction_result = predict_churn(customer_data)

    return prediction_result