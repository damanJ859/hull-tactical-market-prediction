"""
Baseline training script for Hull Tactical - Market Prediction.

- Target  : market_forward_excess_returns
- Features: all numeric columns except the target
- Split   : time-ordered (last ~20% of dates used as validation)
- Model   : LightGBM regressor
- Output  : models/lgbm_baseline.pkl (model + feature list + medians)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_train
from src.config import DATA_PROCESSED


def make_time_split(df: pd.DataFrame,
                    date_col: str,
                    valid_frac: float = 0.2):
    """
    Split on unique dates to avoid leakage.

    Parameters
    ----------
    df : DataFrame sorted by date_col
    date_col : name of date column (e.g. 'date_id')
    valid_frac : fraction of dates to use for validation

    Returns
    -------
    train_idx, valid_idx : index arrays for train/valid rows
    """
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    n_valid = max(1, int(n_dates * valid_frac))

    valid_dates = set(unique_dates[-n_valid:])
    train_mask = ~df[date_col].isin(valid_dates)
    valid_mask = df[date_col].isin(valid_dates)

    return df.index[train_mask], df.index[valid_mask]


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_train()  # assumes src.data.load_train sorts by date_id
    print("Train shape:", df.shape)

    target_col = "market_forward_excess_returns"
    date_col = "date_id"

    assert target_col in df.columns, f"{target_col} not found in train.csv"
    assert date_col in df.columns, f"{date_col} not found in train.csv"

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    y = df[target_col]

    # Use all numeric columns except the target as features
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].copy()

    # Fill missing values with median of training data later, but we
    # need the split first to avoid leaking validation information.
    # ------------------------------------------------------------------
    # 3. Time-based train/validation split (no shuffle)
    # ------------------------------------------------------------------
    train_idx, valid_idx = make_time_split(df, date_col=date_col, valid_frac=0.2)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]

    # ------------------------------------------------------------------
    # 4. Simple missing-value handling (median per feature from train)
    # ------------------------------------------------------------------
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)

    # ------------------------------------------------------------------
    # 5. Define and train LightGBM baseline model
    # ------------------------------------------------------------------
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("Training LightGBM baseline...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
    )

    # ------------------------------------------------------------------
    # 6. Evaluate (MSE; note target is ~1e-3 so MSE is tiny)
    # ------------------------------------------------------------------
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)

    print("Validation MSE (raw):", mse)
    print(f"Validation MSE (scientific): {mse:.8e}")
    print(f"Validation RMSE: {np.sqrt(mse):.8f}")

    # ------------------------------------------------------------------
    # 7. Save model + metadata for inference
    # ------------------------------------------------------------------
    models_dir = DATA_PROCESSED / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "medians": medians,
    }

    out_path = models_dir / "lgbm_baseline.pkl"
    joblib.dump(artifact, out_path)
    print(f"Saved baseline model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()