
import sys
from pathlib import Path

import streamlit as st
from pydantic import ValidationError


# --------------------------------------------------
# Project path and imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import (
    CustomerInput,
    convert_api_input_to_model_input,
)
from src.config import MODEL_FILE
from src.predict import predict_churn


# --------------------------------------------------
# Customer input form - 25 features
# --------------------------------------------------

def build_customer_input():
    st.subheader("Customer Information")

    st.info(
        "The form contains example values. Review and "
        "replace them with actual customer information "
        "before requesting a prediction."
    )

    with st.form("churn_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Customer Profile")

            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            age = st.slider("Age", 18, 100, 45)

            married = st.selectbox(
                "Married", ["No", "Yes"]
            )

            number_of_dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                value=0,
                step=1,
            )

            number_of_referrals = st.number_input(
                "Number of Referrals",
                min_value=0,
                value=0,
                step=1,
            )

            tenure_in_months = st.number_input(
                "Tenure in Months",
                min_value=0,
                value=12,
                step=1,
            )

            offer = st.selectbox(
                "Offer",
                [
                    "No Offer",
                    "Offer A",
                    "Offer B",
                    "Offer C",
                    "Offer D",
                    "Offer E",
                ],
            )

            st.markdown("### Phone Services")

            phone_service = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            avg_monthly_long_distance_charges = (
                st.number_input(
                    "Avg Monthly Long Distance Charges",
                    min_value=0.0,
                    value=20.0,
                )
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No Phone Service"],
            )

            st.markdown("### Billing and Contract")

            contract = st.selectbox(
                "Contract",
                ["Month-to-Month", "One Year", "Two Year"],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Credit Card",
                    "Bank Withdrawal",
                    "Mailed Check",
                ],
            )

        with col2:
            st.markdown("### Internet Services")

            internet_service = st.selectbox(
                "Internet Service", ["Yes", "No"]
            )

            internet_type = st.selectbox(
                "Internet Type",
                [
                    "Fiber Optic",
                    "DSL",
                    "Cable",
                    "No Internet Service",
                ],
            )

            avg_monthly_gb_download = st.number_input(
                "Avg Monthly GB Download",
                min_value=0.0,
                value=25.0,
            )

            service_options = [
                "No",
                "Yes",
                "No Internet Service",
            ]

            online_security = st.selectbox(
                "Online Security", service_options
            )

            online_backup = st.selectbox(
                "Online Backup", service_options
            )

            device_protection_plan = st.selectbox(
                "Device Protection Plan", service_options
            )

            premium_tech_support = st.selectbox(
                "Premium Tech Support", service_options
            )

            st.markdown("### Streaming Services")

            streaming_tv = st.selectbox(
                "Streaming TV", service_options
            )

            streaming_movies = st.selectbox(
                "Streaming Movies", service_options
            )

            streaming_music = st.selectbox(
                "Streaming Music", service_options
            )

            unlimited_data = st.selectbox(
                "Unlimited Data", service_options
            )

            st.markdown("### Monthly Charges")

            monthly_charge = st.number_input(
                "Monthly Charge",
                min_value=0.0,
                value=85.0,
            )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            type="primary",
        )

    customer_data = {
        "gender": gender,
        "age": age,
        "married": married,
        "number_of_dependents": number_of_dependents,
        "number_of_referrals": number_of_referrals,
        "tenure_in_months": tenure_in_months,
        "offer": offer,
        "phone_service": phone_service,
        "avg_monthly_long_distance_charges":
            avg_monthly_long_distance_charges,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "internet_type": internet_type,
        "avg_monthly_gb_download":
            avg_monthly_gb_download,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection_plan": device_protection_plan,
        "premium_tech_support": premium_tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "streaming_music": streaming_music,
        "unlimited_data": unlimited_data,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charge": monthly_charge,
    }

    return submitted, customer_data


# --------------------------------------------------
# Display prediction results
# --------------------------------------------------

def show_risk_result(result):
    churn_probability = result["churn_probability"]
    churn_percent = result["churn_probability_percent"]
    prediction = result["prediction"]

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Churn Probability",
        value=churn_percent,
    )

    if prediction == 1:
        st.warning(
            "The model predicts that this customer "
            "belongs to the churn class."
        )
    else:
        st.success(
            "The model predicts that this customer "
            "belongs to the non-churn class."
        )

    st.progress(churn_probability)

    st.caption(
        "The classification uses the model's default "
        "decision threshold. The estimated probability "
        "has not yet been independently calibrated."
    )

    st.markdown("### Raw Prediction Output")
    st.json(result)


# --------------------------------------------------
# Main Streamlit application
# --------------------------------------------------

def main():
    st.set_page_config(
        page_title="Churn MLOps Prediction App",
        layout="wide",
    )

    st.title("Customer Churn Prediction")

    st.write(
        "An interactive application for estimating "
        "telecom customer churn risk using a trained "
        "Gradient Boosting model."
    )

    st.divider()

    st.subheader("Model Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Gradient Boosting")
    col2.metric("Input Features", "25")

    col3.metric(
        "Model Artifact",
        "Available" if MODEL_FILE.is_file()
        else "Missing",
    )

    st.caption(
        "Performance metrics for the reduced 25-feature "
        "model are pending final evaluation. Metrics "
        "from the earlier 32-feature model are not "
        "displayed here."
    )

    st.divider()

    submitted, customer_data = build_customer_input()

    if submitted:
        try:
            # Validate using the same schema as FastAPI
            customer = CustomerInput.model_validate(
                customer_data
            )

            # Convert API field names to model columns
            model_input = (
                convert_api_input_to_model_input(
                    customer
                )
            )

            # Verify the expected input size
            if len(model_input) != 25:
                st.error(
                    "Input mismatch: expected 25 features."
                )
                return

            # Call the saved prediction pipeline
            result = predict_churn(model_input)

            st.divider()
            show_risk_result(result)

        except ValidationError as error:
            st.error("Invalid customer information.")
            st.write(error.errors())

        except FileNotFoundError:
            st.error(
                "The trained model file was not found."
            )
            st.code(
                "python train_reduced_model.py"
            )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    main()
