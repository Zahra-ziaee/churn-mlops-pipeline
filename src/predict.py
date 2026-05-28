import pandas as pd

from src.train import load_model


def create_customer_dataframe(customer_data: dict) -> pd.DataFrame:
    """
    Convert input dictionary into a one-row DataFrame compatible with the trained model.
    """
    return pd.DataFrame([customer_data])


def predict_churn(customer_data: dict) -> dict:
    """
    Load trained model and return churn prediction result.
    """
    model = load_model()

    input_df = create_customer_dataframe(customer_data)

    churn_probability = float(model.predict_proba(input_df)[0][1])
    churn_prediction = int(model.predict(input_df)[0])

    risk_label = "High Risk" if churn_prediction == 1 else "Low Risk"

    return {
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percent": f"{churn_probability * 100:.2f}%",
        "prediction": churn_prediction,
        "risk_label": risk_label,
    }