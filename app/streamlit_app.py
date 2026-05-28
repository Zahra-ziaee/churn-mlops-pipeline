import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_churn


def build_customer_input() -> dict:
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 100, 45)
        married = st.selectbox("Married", ["No", "Yes"])
        number_of_dependents = st.slider("Number of Dependents", 0, 10, 0)
        city = st.text_input("City", "Los Angeles")
        zip_code = st.number_input("Zip Code", min_value=0, value=90001)
        number_of_referrals = st.slider("Number of Referrals", 0, 20, 0)
        tenure_in_months = st.slider("Tenure in Months", 0, 72, 12)

    with col2:
        contract = st.selectbox(
            "Contract",
            ["Month-to-Month", "One Year", "Two Year"],
        )
        internet_type = st.selectbox(
            "Internet Type",
            ["Fiber Optic", "DSL", "Cable", "No Internet Service"],
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No Internet Service"],
        )
        premium_tech_support = st.selectbox(
            "Premium Tech Support",
            ["No", "Yes", "No Internet Service"],
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Credit Card", "Bank Withdrawal", "Mailed Check"],
        )
        monthly_charge = st.slider("Monthly Charge", 0.0, 150.0, 85.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
        total_revenue = st.number_input("Total Revenue", min_value=0.0, value=1200.0)

    customer_data = {
        "Gender": gender,
        "Age": age,
        "Married": married,
        "Number of Dependents": number_of_dependents,
        "City": city,
        "Zip Code": zip_code,
        "Number of Referrals": number_of_referrals,
        "Tenure in Months": tenure_in_months,
        "Offer": "No Offer",
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 20.0,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": internet_type,
        "Avg Monthly GB Download": 25.0,
        "Online Security": online_security,
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": premium_tech_support,
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": contract,
        "Paperless Billing": "Yes",
        "Payment Method": payment_method,
        "Monthly Charge": monthly_charge,
        "Total Charges": total_charges,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0.0,
        "Total Long Distance Charges": 200.0,
        "Total Revenue": total_revenue,
    }

    return customer_data


def show_risk_result(churn_probability: float, churn_percent: str) -> None:
    if churn_probability >= 0.70:
        st.error(f"High churn risk detected. Probability: {churn_percent}")
        st.markdown("### Business Interpretation")
        st.write(
            "This customer should be prioritized for retention actions such as "
            "discounts, contract upgrade offers, or support outreach."
        )

    elif churn_probability >= 0.40:
        st.warning(f"Moderate churn risk detected. Probability: {churn_percent}")
        st.markdown("### Business Interpretation")
        st.write(
            "This customer has moderate churn risk and may benefit from targeted engagement."
        )

    else:
        st.success(f"Low churn risk detected. Probability: {churn_percent}")
        st.markdown("### Business Interpretation")
        st.write(
            "This customer currently appears relatively stable based on the model."
        )


def main():
    st.set_page_config(
        page_title="Churn MLOps Prediction App",
        layout="wide",
    )

    st.title("🚀 Customer Churn MLOps Prediction App")

    st.write(
        "This app uses a trained and saved Gradient Boosting model to predict "
        "telecom customer churn risk. The model is loaded from the saved artifact "
        "`models/churn_model.joblib`."
    )

    st.divider()

    st.subheader("Model Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "84.60%")
    col2.metric("Precision", "73.57%")
    col3.metric("Recall", "65.51%")
    col4.metric("F1-score", "69.31%")
    col5.metric("ROC-AUC", "91.15%")

    st.divider()

    customer_data = build_customer_input()

    st.divider()

    if st.button("Predict Churn Risk"):
        try:
            result = predict_churn(customer_data)

            churn_probability = result["churn_probability"]
            churn_percent = result["churn_probability_percent"]

            st.subheader("Prediction Result")

            show_risk_result(
                churn_probability=churn_probability,
                churn_percent=churn_percent,
            )

            st.progress(churn_probability)

            st.markdown("### Raw Prediction Output")
            st.json(result)

        except FileNotFoundError:
            st.error("Model file not found.")
            st.write("Please train the model first by running:")
            st.code("python main.py")


if __name__ == "__main__":
    main()