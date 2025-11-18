from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_train
from src.config import DATA_PROCESSED

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_train()

    # Example: pretend 'forward_returns' is the target, everything else numeric is feature
    target_col = "forward_returns"  # confirm from train.csv columns
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Very naive handling: drop non numeric columns
    X = X.select_dtypes(include="number").fillna(0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time series, no shuffle
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    mse = mean_squared_error(y_valid, preds)
    print(f"Validation MSE: {mse:.6f}")

if __name__ == "__main__":
    main()