"""
Train an XGBoost model with basic lag features for the
Hull Tactical - Market Prediction competition.

SPLIT LOGIC:
1) Take train.csv sorted by date_id
2) Reserve the LAST 10% of dates as OFFLINE TEST SET (never used for training)
3) Use remaining 90%:
    - 80% old dates → TRAIN
    - 20% later dates → VALIDATION

- Target  : market_forward_excess_returns
- Features: all numeric columns (incl. new lagged_* features),
           excluding the target itself.
- Model   : XGBRegressor
- Output  : data/processed/models/xgb_lagged.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_train
from src.features import add_basic_lag_features
from src.config import DATA_PROCESSED


def split_offline_test(df: pd.DataFrame,
                       date_col: str = "date_id",
                       test_frac: float = 0.10):
    """
    Hold out the last X% of unique dates as an offline test set.
    """
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_test = max(1, int(n_dates * test_frac))

    test_dates = set(unique_dates[-n_test:])
    train_val_dates = set(unique_dates[:-n_test])

    mask_test = df[date_col].isin(test_dates)
    mask_train_val = df[date_col].isin(train_val_dates)

    return df[mask_train_val].copy(), df[mask_test].copy()


def time_split_train_val(df: pd.DataFrame,
                         date_col: str = "date_id",
                         valid_frac: float = 0.20):
    """
    Split train_val into train and validation chronologically.
    """
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_valid = max(1, int(n_dates * valid_frac))

    valid_dates = set(unique_dates[-n_valid:])
    mask_valid = df[date_col].isin(valid_dates)
    mask_train = ~df[date_col].isin(valid_dates)

    return df[mask_train].copy(), df[mask_valid].copy()


def main():
    # ------------------------------------------------------------------
    # 1. Load and enrich training data
    # ------------------------------------------------------------------
    df = load_train()  # already sorted by date_id in src.data
    print("Original train shape:", df.shape)

    # Add lagged_forward_returns, lagged_risk_free_rate,
    # lagged_market_forward_excess_returns
    df = add_basic_lag_features(df, date_col="date_id")
    print("With lag features shape:", df.shape)

    target_col = "market_forward_excess_returns"
    date_col = "date_id"

    assert target_col in df.columns, f"{target_col} missing in train.csv"
    assert date_col in df.columns, f"{date_col} missing in train.csv"

    # ------------------------------------------------------------------
    # 2. Split into (train+val) and offline test
    # ------------------------------------------------------------------
    train_val_df, offline_test_df = split_offline_test(
        df, date_col=date_col, test_frac=0.10
    )
    print("Train+Val shape:", train_val_df.shape)
    print("Offline Test shape:", offline_test_df.shape)

    # ------------------------------------------------------------------
    # 3. Split train_val into train and validation
    # ------------------------------------------------------------------
    train_df, val_df = time_split_train_val(
        train_val_df, date_col=date_col, valid_frac=0.20
    )
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    # ------------------------------------------------------------------
    # 4. Build feature matrices
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_valid = val_df[feature_cols].copy()
    y_valid = val_df[target_col].values

    X_test = offline_test_df[feature_cols].copy()
    y_test = offline_test_df[target_col].values

    # ------------------------------------------------------------------
    # 5. Handle missing values (medians from train only)
    # ------------------------------------------------------------------
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)
    X_test = X_test.fillna(medians)

    # ------------------------------------------------------------------
    # 6. Define and train XGBoost model
    # ------------------------------------------------------------------
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",   # fast on CPU
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )

    print("Training XGBoost model with lag features...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=200,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate on validation
    # ------------------------------------------------------------------
    preds_val = model.predict(X_valid)
    mse_val = mean_squared_error(y_valid, preds_val)
    rmse_val = np.sqrt(mse_val)

    print("\nValidation Results:")
    print("XGBoost Val MSE (raw):", mse_val)
    print(f"XGBoost Val MSE (scientific): {mse_val:.8e}")
    print(f"XGBoost Val RMSE: {rmse_val:.8e}")

    # ------------------------------------------------------------------
    # 8. Evaluate on OFFLINE TEST (true generalization)
    # ------------------------------------------------------------------
    preds_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, preds_test)
    rmse_test = np.sqrt(mse_test)

    print("\nOFFLINE TEST Results (True Generalization Score):")
    print("XGBoost Test MSE (raw):", mse_test)
    print(f"XGBoost Test MSE (scientific): {mse_test:.8e}")
    print(f"XGBoost Test RMSE: {rmse_test:.8e}")

    # ------------------------------------------------------------------
    # 9. Save model + metadata
    # ------------------------------------------------------------------
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "medians": medians,
    }

    out_path = models_dir / "xgb_lagged.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nSaved XGBoost model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
    