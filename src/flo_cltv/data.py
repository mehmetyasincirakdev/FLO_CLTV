"""Input validation and customer data preparation."""

import numpy as np
import pandas as pd

from flo_cltv.config import (
    ORDER_COLUMNS,
    OUTLIER_COLUMNS,
    REQUIRED_COLUMNS,
    VALUE_COLUMNS,
)


def outlier_thresholds(dataframe: pd.DataFrame, variable: str) -> tuple[float, float]:
    """Return non-negative Tukey fences based on the 1st and 99th percentiles."""
    if variable not in dataframe.columns:
        raise ValueError(f"Required column not found: {variable}")

    lower_percentile = dataframe[variable].quantile(0.01)
    upper_percentile = dataframe[variable].quantile(0.99)
    interquantile_range = upper_percentile - lower_percentile
    lower_limit = max(0.0, lower_percentile - 1.5 * interquantile_range)
    upper_limit = upper_percentile + 1.5 * interquantile_range

    if variable in ORDER_COLUMNS:
        lower_limit = round(lower_limit)
        upper_limit = round(upper_limit)

    return lower_limit, upper_limit


def replace_with_thresholds(dataframe: pd.DataFrame, variable: str) -> None:
    """Winsorize a non-negative numeric column in place."""
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].clip(
        lower=lower_limit,
        upper=upper_limit,
    )


def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare source data without modifying the caller's frame."""
    if raw_data.empty:
        raise ValueError("Input dataset cannot be empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw_data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    dataframe = raw_data.copy()
    if dataframe["master_id"].isna().any():
        raise ValueError("'master_id' cannot contain missing values.")
    if dataframe["master_id"].duplicated().any():
        raise ValueError("'master_id' must identify exactly one row per customer.")

    date_columns = [column for column in dataframe.columns if column.endswith("_date")]
    for column in date_columns:
        dataframe[column] = pd.to_datetime(dataframe[column], errors="raise")

    for column in OUTLIER_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        if not np.isfinite(dataframe[column].to_numpy(dtype=float)).all():
            raise ValueError(f"'{column}' must contain finite numeric values.")
        if (dataframe[column] < 0).any():
            raise ValueError(f"'{column}' cannot contain negative values.")

    if dataframe[["first_order_date", "last_order_date"]].isna().any().any():
        raise ValueError("First and last order dates cannot be missing.")
    if (dataframe["first_order_date"] > dataframe["last_order_date"]).any():
        raise ValueError("'first_order_date' cannot be later than 'last_order_date'.")
    if any((dataframe[column] % 1 != 0).any() for column in ORDER_COLUMNS):
        raise ValueError("Order count columns must contain whole numbers.")

    for column in OUTLIER_COLUMNS:
        replace_with_thresholds(dataframe, column)

    dataframe["order_num_total"] = (
        dataframe[list(ORDER_COLUMNS)].sum(axis=1).round().astype(int)
    )
    dataframe["customer_value_total"] = dataframe[list(VALUE_COLUMNS)].sum(axis=1)
    return dataframe
