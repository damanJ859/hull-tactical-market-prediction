"""
Improved baseline training script with a proper offline test split.

SPLIT LOGIC:
1) Take train.csv sorted by date_id
2) Reserve the LAST 10% of dates as OFFLINE TEST SET (never used for training)
3) Use remaining 90%:
    - 80% old dates → TRAIN
    - 20% later dates → VALIDATION

This prevents overfitting to the validation regime and gives us a real measure
of how well the model generalizes to unseen data.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_train
from src.config import DATA_PROCESSED


def split_offline_test(df, date_col="date_id", test_frac=0.10):
    """Hold out the last X% of dates as a final offline test set."""
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_test = max(1, int(n_dates * test_frac))

    test_dates = set(unique_dates[-n_test:])
    train_val_dates = set(unique_dates[:-n_test])

    mask_test = df[date_col].isin(test_dates)
    mask_train_val = df[date_col].isin(train_val_dates)

    return df[mask_train_val].copy(), df[mask_test].copy()


def time_split_train_val(df, date_col="date_id", valid_frac=0.20):
    """Split train_val into train/validation chronologically."""
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_valid = max(1, int(n_dates * valid_frac))

    valid_dates = set(unique_dates[-n_valid:])
    mask_valid = df[date_col].isin(valid_dates)
    mask_train = ~df[date_col].isin(valid_dates)

    return df[mask_train].copy(), df[mask_valid].copy()


def main():
    # 1. Load data
    df = load_train()  # sorted by date_id
    print("Full train.csv shape:", df.shape)

    target_col = "market_forward_excess_returns"
    date_col = "date_id"

    # 2. Split into (train+val) and offline test set
    train_val_df, offline_test_df = split_offline_test(df, date_col, test_frac=0.10)
    print("Train+Val shape:", train_val_df.shape)
    print("Offline Test shape:", offline_test_df.shape)

    # 3. Split train_val into train and val
    train_df, val_df = time_split_train_val(train_val_df, date_col, valid_frac=0.20)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    # 4. Extract features/targets
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_valid = val_df[feature_cols].copy()
    y_valid = val_df[target_col].values

    X_test = offline_test_df[feature_cols].copy()
    y_test = offline_test_df[target_col].values

    # 5. Missing value imputation (fit only on TRAIN)
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)
    X_test = X_test.fillna(medians)  # <- evaluate later

    # 6. Train model
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression",
        random_state=42,
        n_jobs=-1,
    )

    print("Training LightGBM...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
    )

    # 7. Evaluate on validation set
    preds_val = model.predict(X_valid)
    mse_val = mean_squared_error(y_valid, preds_val)
    print("\nValidation Results:")
    print("MSE:", mse_val)
    print("RMSE:", np.sqrt(mse_val))

    # 8. Evaluate on OFFLINE TEST (never seen during training)
    preds_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, preds_test)
    print("\nOFFLINE TEST Results (True Generalization Score):")
    print("MSE:", mse_test)
    print("RMSE:", np.sqrt(mse_test))

    # 9. Save model
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "medians": medians,
    }

    out_path = models_dir / "lgbm_baseline.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nSaved model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()