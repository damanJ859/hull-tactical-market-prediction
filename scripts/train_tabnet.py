"""
Train a TabNet regressor for Hull Tactical Market Prediction.

SPLIT LOGIC:
1) Take train.csv sorted by date_id
2) Reserve the LAST 10% of dates as OFFLINE TEST SET (never used for training)
3) Use remaining 90%:
    - 80% old dates → TRAIN
    - 20% later dates → VALIDATION

TabNet learns feature masks and nonlinear interactions very well,
which often helps when the signal-to-noise ratio is small.

Output:
- data/processed/models/tabnet_regressor.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from pytorch_tabnet.tab_model import TabNetRegressor

from src.data import load_train
from src.features import add_basic_lag_features
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
    """Split train_val into train and validation chronologically."""
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_valid = max(1, int(n_dates * valid_frac))

    valid_dates = set(unique_dates[-n_valid:])
    mask_valid = df[date_col].isin(valid_dates)
    mask_train = ~df[date_col].isin(valid_dates)

    return df[mask_train].copy(), df[mask_valid].copy()


def main():
    # -----------------------------------------------------------
    # 1. Load and enrich data
    # -----------------------------------------------------------
    df = load_train()                     # sorted by date_id in src.data
    df = add_basic_lag_features(df)       # add lagged_* features
    print("Train shape with lag features:", df.shape)

    target_col = "market_forward_excess_returns"
    date_col = "date_id"

    # -----------------------------------------------------------
    # 2. Split into (train+val) and offline test
    # -----------------------------------------------------------
    train_val_df, offline_test_df = split_offline_test(
        df, date_col=date_col, test_frac=0.10
    )
    print("Train+Val shape:", train_val_df.shape)
    print("Offline Test shape:", offline_test_df.shape)

    # -----------------------------------------------------------
    # 3. Split train_val into train and validation
    # -----------------------------------------------------------
    train_df, val_df = time_split_train_val(
        train_val_df, date_col=date_col, valid_frac=0.20
    )
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    # -----------------------------------------------------------
    # 4. Feature selection
    # -----------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    # Targets
    y_full = df[target_col].values
    TARGET_SCALE = 1e4
    y_scaled_full = y_full * TARGET_SCALE

    # Indexes mapped back to original df.index
    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = offline_test_df.index

    # Features
    X_train = df.loc[train_idx, feature_cols].copy()
    X_valid = df.loc[val_idx, feature_cols].copy()
    X_test = df.loc[test_idx, feature_cols].copy()

    # TabNet expects targets as 2D arrays: (n_samples, n_outputs)
    y_train = y_scaled_full[train_idx].reshape(-1, 1)
    y_valid = y_scaled_full[val_idx].reshape(-1, 1)
    y_test = y_scaled_full[test_idx].reshape(-1, 1)

    # -----------------------------------------------------------
    # 5. Handle missing & scale features
    # -----------------------------------------------------------
    # For TabNet we can fill NAs with 0 then standardize features
    X_train = X_train.fillna(0).values
    X_valid = X_valid.fillna(0).values
    X_test = X_test.fillna(0).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # -----------------------------------------------------------
    # 6. Define TabNet model
    # -----------------------------------------------------------
    model = TabNetRegressor(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        seed=42,
        lambda_sparse=1e-4,
    )

    # -----------------------------------------------------------
    # 7. Train on train, evaluate on validation
    # -----------------------------------------------------------
    print("Training TabNet...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=200,
        patience=25,
        batch_size=1024,
        virtual_batch_size=128,
    )

    # -----------------------------------------------------------
    # 8. Evaluate on validation
    # -----------------------------------------------------------
    preds_val_scaled = model.predict(X_valid).reshape(-1)
    preds_val = preds_val_scaled / TARGET_SCALE
    y_valid_true = y_valid.reshape(-1) / TARGET_SCALE

    mse_val = mean_squared_error(y_valid_true, preds_val)
    rmse_val = np.sqrt(mse_val)

    print("\nValidation Results:")
    print("TabNet Val RMSE:", rmse_val)
    print("TabNet Val MSE:", mse_val)

    # -----------------------------------------------------------
    # 9. Evaluate on OFFLINE TEST (true generalization)
    # -----------------------------------------------------------
    preds_test_scaled = model.predict(X_test).reshape(-1)
    preds_test = preds_test_scaled / TARGET_SCALE
    y_test_true = y_test.reshape(-1) / TARGET_SCALE

    mse_test = mean_squared_error(y_test_true, preds_test)
    rmse_test = np.sqrt(mse_test)

    print("\nOFFLINE TEST Results (True Generalization Score):")
    print("TabNet Test RMSE:", rmse_test)
    print("TabNet Test MSE:", mse_test)

    # -----------------------------------------------------------
    # 10. Save everything
    # -----------------------------------------------------------
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "target_scale": TARGET_SCALE,
    }

    out_path = models_dir / "tabnet_regressor.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nSaved TabNet model to: {out_path}")


if __name__ == "__main__":
    main()
