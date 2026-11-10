
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.train import build_model_pipeline


def main():
    print("=" * 60)
    print("Churn Model - Decision Threshold Analysis")
    print("=" * 60)

    # 1. Load and prepare the 25-feature dataset
    raw_df = load_raw_data()
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    # 2. Use training data only
    X_train, _, y_train, _ = train_test_split_data(X, y)

    # 3. Stratified cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    pipeline = build_model_pipeline(X_train)

    # 4. Generate out-of-fold predicted probabilities
    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # 5. Thresholds to compare
    thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    results = []

    # 6. Evaluate each threshold
    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "True Negatives": tn,
            "Customers Flagged": int(predictions.sum()),
        })

    # 7. Display comparison table
    results_df = pd.DataFrame(results)

    print("\nTHRESHOLD COMPARISON")
    print(results_df.round(4).to_string(index=False))

    print("\nExperiment completed.")
    print("No saved model or original metrics were modified.")


if __name__ == "__main__":
    main()
