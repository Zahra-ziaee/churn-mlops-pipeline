from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    save_processed_data,
    split_features_target,
    train_test_split_data,
)
from src.train import train_churn_model, save_model
from src.utils import print_metrics, save_metrics


def main():
    print("=" * 60)
    print("Churn MLOps Pipeline - Training")
    print("=" * 60)

    raw_df = load_raw_data()

    print("\nRaw data shape:")
    print(raw_df.shape)

    processed_df = prepare_churn_dataset(raw_df)

    print("\nProcessed data shape:")
    print(processed_df.shape)

    print("\nTarget distribution:")
    print(processed_df["Churn"].value_counts(normalize=True))

    save_processed_data(processed_df)

    X, y = split_features_target(processed_df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("\nTrain/Test split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    model_pipeline, metrics = train_churn_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print_metrics(metrics)
    save_metrics(metrics)
    save_model(model_pipeline)


if __name__ == "__main__":
    main()