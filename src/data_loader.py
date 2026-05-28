from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILE


def load_csv_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    last_error = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Could not read {file_path} with tried encodings: {encodings_to_try}",
    )


def load_raw_data() -> pd.DataFrame:
    return load_csv_file(RAW_DATA_FILE)