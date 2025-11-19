# src/features.py

import pandas as pd


def add_basic_lag_features(
    df: pd.DataFrame,
    date_col: str = "date_id",
) -> pd.DataFrame:
    """
    Add lagged versions of key financial series to mimic what is
    provided in test.csv:

    - lagged_forward_returns
    - lagged_risk_free_rate
    - lagged_market_forward_excess_returns

    Parameters
    ----------
    df : DataFrame
        Must contain date_id, forward_returns, risk_free_rate,
        market_forward_excess_returns.
    date_col : str
        Name of the date column used for sorting.

    Returns
    -------
    DataFrame
        Copy of df with added lagged columns.
    """
    df = df.sort_values(date_col).reset_index(drop=True).copy()

    lag_specs = [
        ("forward_returns", "lagged_forward_returns"),
        ("risk_free_rate", "lagged_risk_free_rate"),
        ("market_forward_excess_returns", "lagged_market_forward_excess_returns"),
    ]

    for col, lag_name in lag_specs:
        if col in df.columns and lag_name not in df.columns:
            df[lag_name] = df[col].shift(1)

    return df