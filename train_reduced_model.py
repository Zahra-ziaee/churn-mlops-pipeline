
from pathlib import Path

import joblib

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.train import build_model_pipeline


def main():
    print("=" * 60)
    print("Training Reduced Churn Model - 25 Features")
    print("=" * 60)

    # Load and prepare data
    raw_df = load_raw_data()
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    # Keep test data separate
    X_train, _, y_train, _ = train_test_split_data(X, y)

    # Build and train the reduced model
    model = build_model_pipeline(X_train)

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Save under a new filename
    output_path = Path("models/churn_model_25features.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    print("\nTraining completed.")
    print("Model saved to:", output_path)
    print("Input feature count:", len(model.feature_names_in_))
    print("Training rows:", len(X_train))


if __name__ == "__main__":
    main()
