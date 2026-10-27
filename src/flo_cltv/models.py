"""BG/NBD and Gamma-Gamma customer lifetime value models."""

import datetime as dt

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

from flo_cltv.config import (
    BG_NBD_PENALIZER,
    DISCOUNT_RATE,
    GAMMA_GAMMA_PENALIZER,
    WEEKS_IN_SIX_MONTHS,
    WEEKS_IN_THREE_MONTHS,
)
from flo_cltv.data import prepare_data


def create_cltv_dataframe(
    raw_data: pd.DataFrame,
    analysis_date: str | dt.date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fit the CLTV models and return estimates for eligible repeat purchasers."""
    dataframe = prepare_data(raw_data)

    if analysis_date is None:
        analysis_date = dataframe["last_order_date"].max() + dt.timedelta(days=2)
    else:
        analysis_date = pd.Timestamp(analysis_date)

    latest_order_date = dataframe["last_order_date"].max()
    if analysis_date <= latest_order_date:
        raise ValueError(
            "Analysis date must be later than every customer's last order."
        )

    eligible = dataframe.loc[
        (dataframe["order_num_total"] > 1) & (dataframe["customer_value_total"] > 0)
    ].copy()
    if len(eligible) < 4:
        raise ValueError(
            "At least four customers with repeat purchases and positive spend "
            "are required to fit the models and create four segments."
        )

    cltv_dataframe = pd.DataFrame(index=eligible.index)
    cltv_dataframe["customer_id"] = eligible["master_id"]
    cltv_dataframe["recency_cltv_weekly"] = (
        eligible["last_order_date"] - eligible["first_order_date"]
    ).dt.days / 7
    cltv_dataframe["T_weekly"] = (
        analysis_date - eligible["first_order_date"]
    ).dt.days / 7
    cltv_dataframe["frequency"] = eligible["order_num_total"] - 1
    cltv_dataframe["monetary_cltv_avg"] = (
        eligible["customer_value_total"] / eligible["order_num_total"]
    )

    bgf = BetaGeoFitter(penalizer_coef=BG_NBD_PENALIZER)
    bgf.fit(
        cltv_dataframe["frequency"],
        cltv_dataframe["recency_cltv_weekly"],
        cltv_dataframe["T_weekly"],
    )
    cltv_dataframe["exp_sales_3_month"] = bgf.predict(
        WEEKS_IN_THREE_MONTHS,
        cltv_dataframe["frequency"],
        cltv_dataframe["recency_cltv_weekly"],
        cltv_dataframe["T_weekly"],
    )
    cltv_dataframe["exp_sales_6_month"] = bgf.predict(
        WEEKS_IN_SIX_MONTHS,
        cltv_dataframe["frequency"],
        cltv_dataframe["recency_cltv_weekly"],
        cltv_dataframe["T_weekly"],
    )

    ggf = GammaGammaFitter(penalizer_coef=GAMMA_GAMMA_PENALIZER)
    ggf.fit(cltv_dataframe["frequency"], cltv_dataframe["monetary_cltv_avg"])
    cltv_dataframe["exp_average_value"] = ggf.conditional_expected_average_profit(
        cltv_dataframe["frequency"],
        cltv_dataframe["monetary_cltv_avg"],
    )
    cltv_dataframe["cltv"] = ggf.customer_lifetime_value(
        bgf,
        cltv_dataframe["frequency"],
        cltv_dataframe["recency_cltv_weekly"],
        cltv_dataframe["T_weekly"],
        cltv_dataframe["monetary_cltv_avg"],
        time=WEEKS_IN_SIX_MONTHS,
        freq="W",
        discount_rate=DISCOUNT_RATE,
    )

    prediction_columns = (
        "exp_sales_3_month",
        "exp_sales_6_month",
        "exp_average_value",
        "cltv",
    )
    if not np.isfinite(cltv_dataframe[list(prediction_columns)].to_numpy()).all():
        raise RuntimeError(
            "The fitted models produced non-finite predictions. "
            "Check the input data and model fit."
        )

    cltv_std = cltv_dataframe["cltv"].std(ddof=0)
    cltv_dataframe["scaled_cltv"] = (
        (cltv_dataframe["cltv"] - cltv_dataframe["cltv"].mean()) / cltv_std
        if cltv_std
        else 0.0
    )
    cltv_dataframe["cltv_segment"] = pd.qcut(
        cltv_dataframe["cltv"].rank(method="first"),
        q=4,
        labels=["D", "C", "B", "A"],
    )
    return cltv_dataframe.reset_index(drop=True)
