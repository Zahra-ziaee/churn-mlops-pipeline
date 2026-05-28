from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import METRICS_FILE


def save_metrics(metrics: Dict) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_save = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

    metrics_df = pd.DataFrame([metrics_to_save])
    metrics_df.to_csv(METRICS_FILE, index=False)

    print(f"Metrics saved to: {METRICS_FILE}")


def print_metrics(metrics: Dict) -> None:
    print("\nModel Evaluation Metrics")
    print("=" * 60)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])