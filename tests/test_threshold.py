
import numpy as np
import pytest

import src.predict as predict_module


@pytest.mark.parametrize(
    "probability, expected_prediction, expected_risk",
    [
        (0.20, 0, "Low Risk"),
        (0.29, 0, "Low Risk"),
        (0.30, 1, "High Risk"),
        (0.35, 1, "High Risk"),
        (0.60, 1, "High Risk"),
    ],
)
def test_decision_threshold(
    monkeypatch,
    probability,
    expected_prediction,
    expected_risk,
):
    # Fake model with a controlled probability
    class FakeModel:
        def predict_proba(self, X):
            return np.array([
                [1 - probability, probability]
            ])

    # Replace real model loading for this test
    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda: FakeModel(),
    )

    customer_data = {
        "Age": 45,
        "Contract": "Month-to-Month",
    }

    result = predict_module.predict_churn(
        customer_data
    )

    assert result["prediction"] == expected_prediction
    assert result["risk_label"] == expected_risk
    assert result["churn_probability"] == pytest.approx(
        probability
    )
