"""Loading and splitting the dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE

LABEL_CANDIDATES = ["v1", "label", "category", "class", "type"]
TEXT_CANDIDATES = ["v2", "message", "text", "sms", "content"]
LABEL_MAP = {"spam": 1, "ham": 0, "legitimate": 0, "legit": 0, "not spam": 0}


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:
            last_error = err
    raise last_error


def load_dataset(path=DATA_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Return a clean DataFrame with columns: label (0/1), text (str)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Copy your Kaggle file to data/spam.csv"
        )
    df = _read_csv_any_encoding(path)
    lowered = {c.lower().strip(): c for c in df.columns}

    label_col = next((lowered[c] for c in LABEL_CANDIDATES if c in lowered), None)
    text_col = next((lowered[c] for c in TEXT_CANDIDATES if c in lowered), None)
    if label_col is None or text_col is None:
        raise ValueError(
            f"Could not find label/text columns. Columns found: {list(df.columns)}"
        )

    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Found labels other than spam/ham. Check the label column.")
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()

    if drop_duplicates:
        # Duplicates across train/test would inflate scores (data leakage).
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Stratified split keeps the spam/ham ratio the same in train and test."""
    return train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
