
import pandas as pd

from src.config import DECISION_THRESHOLD
from src.train import load_model


def predict_churn(customer_data: dict) -> dict:
    """
    Predict customer churn using the trained model
    and the configured business decision threshold.
    """

    # Load the trained 25-feature model
    model = load_model()

    # Convert customer data to a DataFrame
    input_df = pd.DataFrame([customer_data])

    # Predict probability of churn (class 1)
    churn_probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    # Apply configurable decision threshold
    prediction = int(
        churn_probability >= DECISION_THRESHOLD
    )

    # Assign risk label
    risk_label = (
        "High Risk"
        if prediction == 1
        else "Low Risk"
    )

    # Return prediction results
    return {
        "churn_probability": churn_probability,
        "churn_probability_percent": round(
            churn_probability * 100, 2
        ),
        "prediction": prediction,
        "risk_label": risk_label,
    }
