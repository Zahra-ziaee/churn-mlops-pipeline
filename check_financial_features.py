
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.train import build_model_pipeline


# Financial features to investigate
FINANCIAL_FEATURES = [
    "Total Revenue",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]


def evaluate_variant(name, X_train, X_test, y_train, y_test):
    print(f"\nTraining: {name}")

    # Build a fresh model and preprocessing pipeline
    pipeline = build_model_pipeline(X_train)

    # Fit only on training data
    pipeline.fit(X_train, y_train)

    # Predict on the same test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Features": X_train.shape[1],
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }


def main():
    print("=" * 60)
    print("Financial Features Ablation Study")
    print("=" * 60)

    # 1. Load and prepare data
    raw_df = load_raw_data()
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    # 2. Use exactly the same train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y
    )

    # 3. Check that all financial columns exist
    missing = [
        col for col in FINANCIAL_FEATURES
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing financial columns: {missing}"
        )

    # 4. Model with all features
    result_full = evaluate_variant(
        "All Features",
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 5. Remove cumulative financial features
    X_train_reduced = X_train.drop(
        columns=FINANCIAL_FEATURES
    )
    X_test_reduced = X_test.drop(
        columns=FINANCIAL_FEATURES
    )

    result_reduced = evaluate_variant(
        "Without Financial Features",
        X_train_reduced,
        X_test_reduced,
        y_train,
        y_test,
    )

    # 6. Compare results
    results = pd.DataFrame([
        result_full,
        result_reduced,
    ])

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results.round(4).to_string(index=False))

    print("\nChanges after removing financial features:")

    for metric in ["ROC-AUC", "Recall", "F1"]:
        difference = (
            result_reduced[metric] - result_full[metric]
        )
        print(f"{metric}: {difference:+.4f}")

    print("\nExperiment finished. No model files were overwritten.")


if __name__ == "__main__":
    main()
