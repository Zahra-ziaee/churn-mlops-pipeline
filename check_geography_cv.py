
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.train import build_model_pipeline


# -----------------------------------------
# Feature groups
# -----------------------------------------

FINANCIAL_FEATURES = [
    "Total Revenue",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]

GEOGRAPHY_FEATURES = [
    "City",
    "Zip Code",
]


# -----------------------------------------
# Main experiment
# -----------------------------------------

def main():
    print("=" * 65)
    print("Churn Model - Feature Ablation with Cross-Validation")
    print("=" * 65)

    # 1. Load and preprocess the dataset
    raw_df = load_raw_data()
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    # 2. Keep the original test set separate
    X_train, _, y_train, _ = train_test_split_data(X, y)

    # 3. Define four feature configurations
    variants = {
        "All Features": X_train,

        "Without Geography": X_train.drop(
            columns=GEOGRAPHY_FEATURES
        ),

        "Without Financial": X_train.drop(
            columns=FINANCIAL_FEATURES
        ),

        "Without Both": X_train.drop(
            columns=GEOGRAPHY_FEATURES + FINANCIAL_FEATURES
        ),
    }

    # 4. Stratified 3-fold cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    results = []

    # 5. Evaluate each feature configuration
    for name, features in variants.items():
        print(f"\nEvaluating: {name}")
        print(f"Number of features: {features.shape[1]}")

        # Fresh pipeline for each configuration
        pipeline = build_model_pipeline(features)

        scores = cross_validate(
            estimator=pipeline,
            X=features,
            y=y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=1,
            error_score="raise",
        )

        results.append({
            "Model": name,
            "Features": features.shape[1],
            "ROC-AUC Mean": scores["test_roc_auc"].mean(),
            "ROC-AUC Std": scores["test_roc_auc"].std(),
            "Recall Mean": scores["test_recall"].mean(),
            "F1 Mean": scores["test_f1"].mean(),
        })

    # 6. Display results
    result_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 65)

    print(result_df.round(4).to_string(index=False))

    # 7. Compare against the full feature set
    baseline = result_df.iloc[0]

    print("\n" + "=" * 65)
    print("CHANGES COMPARED WITH ALL FEATURES")
    print("=" * 65)

    for _, row in result_df.iloc[1:].iterrows():
        print(f"\n{row['Model']}:")
        print(
            "ROC-AUC change: "
            f"{row['ROC-AUC Mean'] - baseline['ROC-AUC Mean']:+.4f}"
        )
        print(
            "Recall change: "
            f"{row['Recall Mean'] - baseline['Recall Mean']:+.4f}"
        )
        print(
            "F1 change: "
            f"{row['F1 Mean'] - baseline['F1 Mean']:+.4f}"
        )

    print("\nExperiment completed successfully.")
    print("The saved model and original metrics were not modified.")


if __name__ == "__main__":
    main()
