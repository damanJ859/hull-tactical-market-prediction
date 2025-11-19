"""
Train an XGBoost model with basic lag features for the
Hull Tactical - Market Prediction competition.

- Target  : market_forward_excess_returns
- Features: all numeric columns (incl. new lagged_* features),
           excluding the target itself.
- Split   : time-based on date_id (last ~20% dates as validation)
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


def make_time_split(df: pd.DataFrame,
                    date_col: str,
                    valid_frac: float = 0.2):
    """
    Split on unique dates to avoid leakage.

    Parameters
    ----------
    df : DataFrame sorted by date_col
    date_col : str
        Name of the date column (e.g. 'date_id').
    valid_frac : float
        Fraction of unique dates to allocate to validation.

    Returns
    -------
    train_idx, valid_idx : index arrays
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
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    y = df[target_col]

    # Use all numeric columns except the target as features
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].copy()

    # ------------------------------------------------------------------
    # 3. Time-based split (no leakage)
    # ------------------------------------------------------------------
    train_idx, valid_idx = make_time_split(df, date_col=date_col, valid_frac=0.2)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]

    # ------------------------------------------------------------------
    # 4. Handle missing values (medians from train)
    # ------------------------------------------------------------------
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)

    # ------------------------------------------------------------------
    # 5. Define and train XGBoost model
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
        eval_metric="rmse",   # define eval metric here
    )

    print("Training XGBoost model with lag features...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=200,          # no eval_metric here
    )


    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    rmse = np.sqrt(mse)

    print("XGBoost Validation MSE (raw):", mse)
    print(f"XGBoost Validation MSE (scientific): {mse:.8e}")
    print(f"XGBoost Validation RMSE: {rmse:.8e}")

    # ------------------------------------------------------------------
    # 7. Save model + metadata
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
    print(f"Saved XGBoost model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()