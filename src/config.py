from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_FILE = RAW_DATA_DIR / "telecom_customer_churn.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_churn_data.csv"

MODEL_FILE = MODELS_DIR / "churn_model_25features.joblib"
METRICS_FILE = RESULTS_DIR / "metrics.csv"

TARGET_COLUMN = "Customer Status"
TARGET_BINARY_COLUMN = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMNS = [
    "Customer ID",
]

LEAKAGE_COLUMNS = [
    "Churn Category",
    "Churn Reason",
]

LOCATION_COLUMNS_TO_DROP = [
    "Latitude",
    "Longitude",
]

# Features excluded from the reduced 25-feature model
MODEL_FEATURES_TO_DROP = [
    "City",
    "Zip Code",
    "Total Revenue",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]

# Features excluded from the reduced 25-feature model
REDUCED_MODEL_DROP_COLUMNS = [
    "City",
    "Zip Code",
    "Total Revenue",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]

# Illustrative business threshold selected from training CV
DECISION_THRESHOLD = 0.30