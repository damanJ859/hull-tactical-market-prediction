"""
Train a TabNet regressor for Hull Tactical Market Prediction.

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


def make_time_split(df, date_col="date_id", valid_frac=0.2):
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_valid = max(1, int(n_dates * valid_frac))

    valid_dates = set(unique_dates[-n_valid:])
    train_mask = ~df[date_col].isin(valid_dates)
    valid_mask = df[date_col].isin(valid_dates)

    return df.index[train_mask], df.index[valid_mask]


def main():
    # -----------------------------------------------------------
    # 1. Load and enrich data
    # -----------------------------------------------------------
    df = load_train()
    df = add_basic_lag_features(df)
    print("Train shape with lag features:", df.shape)

    target_col = "market_forward_excess_returns"
    date_col = "date_id"

    y = df[target_col].values

    # Scaling target is extremely important for neural models
    TARGET_SCALE = 1e4
    y_scaled = y * TARGET_SCALE

    # -----------------------------------------------------------
    # 2. Feature selection
    # -----------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].copy()

    # -----------------------------------------------------------
    # 3. Time split
    # -----------------------------------------------------------
    train_idx, valid_idx = make_time_split(df, date_col, valid_frac=0.2)

    X_train = X.iloc[train_idx].fillna(0).values
    X_valid = X.iloc[valid_idx].fillna(0).values

    # TabNet expects targets as 2D arrays: (n_samples, n_outputs)
    y_train = y_scaled[train_idx].reshape(-1, 1)
    y_valid = y_scaled[valid_idx].reshape(-1, 1)

    # -----------------------------------------------------------
    # 4. Scale features
    # -----------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # -----------------------------------------------------------
    # 5. Define TabNet model
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
    # 6. Train
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
    # 7. Evaluate
    # -----------------------------------------------------------
    preds_scaled = model.predict(X_valid)           # shape (n_samples, 1)
    preds_scaled = preds_scaled.reshape(-1)         # -> (n_samples,)
    preds = preds_scaled / TARGET_SCALE

    y_valid_true = y_valid.reshape(-1) / TARGET_SCALE

    mse = mean_squared_error(y_valid_true, preds)
    rmse = np.sqrt(mse)

    print("TabNet RMSE:", rmse)
    print("TabNet MSE:", mse)

    # -----------------------------------------------------------
    # 8. Save everything
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
    print(f"Saved TabNet model to: {out_path}")


if __name__ == "__main__":
    main()
    