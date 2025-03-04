import pandas as pd
import datetime


def period_max_drawdown(
        asset: str,
        date_start: datetime.date,
        date_end: datetime.date,
        df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the drawdown between two dates.

    Parameters:
    ---
    asset:
        Asset in DataFrame for which to calculate the drawdown.

    date_start: datetime.date
        Period start date.

    date_end: datetime.date
        Period end date.

    df_ret: pd.DataFrame (change this parameter name)
        Contains equity curve of asset for which to calculate
        the drawdown.
    ---
    """
    col_name = f"equity_{asset}"
    query = "@date_start <= date <= @date_end"
    df = df_ret.query(query)[["date", col_name]].copy()
    df["drawdown"] = (df[col_name] / df[col_name].cummax()) - 1

    return df["drawdown"].min()
