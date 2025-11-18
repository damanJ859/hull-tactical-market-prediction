import pandas as pd
from .config import DATA_RAW

def load_train():
    path = DATA_RAW / "train.csv"
    return pd.read_csv(path)

def load_mock_test():
    path = DATA_RAW / "test.csv"
    return pd.read_csv(path)