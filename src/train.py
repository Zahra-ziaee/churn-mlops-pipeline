from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_FILE, RANDOM_STATE
from src.evaluation import evaluate_classifier


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_churn_model(
    X_train,
    y_train,
    X_test,
    y_test,
) -> Tuple[Pipeline, dict]:
    model_pipeline = build_model_pipeline(X_train)

    print("Training Gradient Boosting churn model...")
    model_pipeline.fit(X_train, y_train)
    print("Training finished.")

    metrics = evaluate_classifier(model_pipeline, X_test, y_test)

    return model_pipeline, metrics


def save_model(model_pipeline: Pipeline) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


def load_model() -> Pipeline:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run python main.py first."
        )

    return joblib.load(MODEL_FILE)