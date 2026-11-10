
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    LOCATION_COLUMNS_TO_DROP,
    REDUCED_MODEL_DROP_COLUMNS,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    TARGET_BINARY_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
)


# --------------------------------------------------
# 1. Create binary churn target
# --------------------------------------------------

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert customer status into a binary target.

    Churned = 1
    Stayed / Joined = 0
    """
    df = df.copy()

    df[TARGET_BINARY_COLUMN] = df[TARGET_COLUMN].apply(
        lambda value: 1 if value == "Churned" else 0
    )

    return df


# --------------------------------------------------
# 2. Handle missing values
# --------------------------------------------------

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill selected missing values using domain-based rules.
    """
    df = df.copy()

    categorical_fill_values = {
        "Offer": "No Offer",
        "Multiple Lines": "No Phone Service",
        "Internet Type": "No Internet Service",
        "Online Security": "No Internet Service",
        "Online Backup": "No Internet Service",
        "Device Protection Plan": "No Internet Service",
        "Premium Tech Support": "No Internet Service",
        "Streaming TV": "No Internet Service",
        "Streaming Movies": "No Internet Service",
        "Streaming Music": "No Internet Service",
        "Unlimited Data": "No Internet Service",
    }

    for column, fill_value in categorical_fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)

    numeric_fill_values = {
        "Avg Monthly Long Distance Charges": 0,
        "Avg Monthly GB Download": 0,
    }

    for column, fill_value in numeric_fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)

    return df


# --------------------------------------------------
# 3. Remove unnecessary and excluded features
# --------------------------------------------------

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifiers, target leakage columns,
    selected location features, and features
    excluded by the reduced-model experiments.
    """
    df = df.copy()

    columns_to_drop = (
        ID_COLUMNS
        + LEAKAGE_COLUMNS
        + LOCATION_COLUMNS_TO_DROP
        + REDUCED_MODEL_DROP_COLUMNS
        + [TARGET_COLUMN]
    )

    existing_columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    return df.drop(columns=existing_columns_to_drop)


# --------------------------------------------------
# 4. Prepare complete churn dataset
# --------------------------------------------------

def prepare_churn_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target, clean missing values,
    and remove excluded features.
    """
    df = create_binary_target(df)
    df = clean_missing_values(df)
    df = drop_unnecessary_columns(df)

    return df


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features (X) and target (y).
    """
    X = df.drop(columns=[TARGET_BINARY_COLUMN])
    y = df[TARGET_BINARY_COLUMN]

    return X, y


# --------------------------------------------------
# 6. Train/test split
# --------------------------------------------------

def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split data while preserving class proportions.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# --------------------------------------------------
# 7. Save processed dataset
# --------------------------------------------------

def save_processed_data(df: pd.DataFrame) -> None:
    """
    Save the processed dataset to CSV.
    """
    PROCESSED_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
    )

    print(
        f"Processed data saved to: {PROCESSED_DATA_FILE}"
    )
