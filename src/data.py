# src/data.py
from pathlib import Path
import pandas as pd
from .config import DATA_RAW


def load_train() -> pd.DataFrame:
    path = DATA_RAW / "train.csv"
    df = pd.read_csv(path)
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)
    return df


def load_test() -> pd.DataFrame:
    path = DATA_RAW / "test.csv"
    df = pd.read_csv(path)
    if "date_id" in df.columns:
        df = df.sort_values("date_id").reset_index(drop=True)
    return df