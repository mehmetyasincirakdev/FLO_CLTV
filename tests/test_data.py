import pandas as pd
import pytest

from flo_cltv.data import outlier_thresholds, prepare_data, replace_with_thresholds


@pytest.fixture
def raw_data():
    return pd.DataFrame(
        {
            "master_id": ["customer-1", "customer-2", "customer-3", "customer-4"],
            "first_order_date": ["2020-01-01"] * 4,
            "last_order_date": ["2020-02-01"] * 4,
            "order_num_total_ever_online": [1, 2, 3, 4],
            "order_num_total_ever_offline": [1, 0, 1, 0],
            "customer_value_total_ever_offline": [20.0, 0.0, 10.0, 0.0],
            "customer_value_total_ever_online": [10.0, 30.0, 20.0, 40.0],
        }
    )


def test_prepare_data_creates_totals_without_mutating_input(raw_data):
    prepared = prepare_data(raw_data)

    assert "order_num_total" not in raw_data.columns
    assert prepared["order_num_total"].tolist() == [2, 2, 4, 4]
    assert prepared["customer_value_total"].tolist() == [30.0, 30.0, 30.0, 40.0]
    assert pd.api.types.is_datetime64_any_dtype(prepared["first_order_date"])


def test_missing_required_column_fails_with_column_name(raw_data):
    with pytest.raises(ValueError, match="master_id"):
        prepare_data(raw_data.drop(columns="master_id"))


def test_duplicate_customer_ids_are_rejected(raw_data):
    duplicate_ids = raw_data.copy()
    duplicate_ids.loc[1, "master_id"] = "customer-1"

    with pytest.raises(ValueError, match="exactly one row"):
        prepare_data(duplicate_ids)


def test_empty_dataset_is_rejected(raw_data):
    with pytest.raises(ValueError, match="cannot be empty"):
        prepare_data(raw_data.iloc[0:0])


def test_infinite_spend_is_rejected(raw_data):
    invalid_data = raw_data.copy()
    invalid_data.loc[0, "customer_value_total_ever_online"] = float("inf")

    with pytest.raises(ValueError, match="finite numeric values"):
        prepare_data(invalid_data)


def test_outlier_thresholds_are_non_negative_and_order_limits_are_integers(raw_data):
    lower, upper = outlier_thresholds(raw_data, "order_num_total_ever_online")

    assert lower >= 0
    assert lower == round(lower)
    assert upper == round(upper)


def test_replace_with_thresholds_caps_extreme_value():
    dataframe = pd.DataFrame({"spend": [*range(1, 101), 1_000_000]})
    _, upper = outlier_thresholds(dataframe, "spend")

    replace_with_thresholds(dataframe, "spend")

    assert dataframe["spend"].max() == upper
