import json

import numpy as np
import pandas as pd
import pytest

from flo_cltv.analysis import run_analysis
from flo_cltv.config import DEFAULT_INPUT
from flo_cltv.models import create_cltv_dataframe


@pytest.fixture(scope="module")
def source_data():
    return pd.read_csv(DEFAULT_INPUT)


def test_model_returns_finite_cltv_and_four_complete_segments(source_data):
    predictions = create_cltv_dataframe(source_data)

    assert np.isfinite(predictions["cltv"]).all()
    assert predictions["cltv_segment"].nunique() == 4
    assert predictions["cltv_segment"].notna().all()


def test_run_analysis_writes_complete_reports(tmp_path, source_data):
    input_path = tmp_path / "source.csv"
    source_data.to_csv(input_path, index=False)
    output_dir = tmp_path / "results"

    predictions = run_analysis(input_path=input_path, output_dir=output_dir)

    assert len(pd.read_csv(output_dir / "cltv_predictions.csv")) == len(predictions)
    assert len(pd.read_csv(output_dir / "top_customers.csv")) == 20
    summary = pd.read_csv(output_dir / "segment_summary.csv")
    assert summary["customer_count"].sum() == len(predictions)

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["analysis_date"] == "2021-06-01"
    assert metadata["customer_count"] == len(predictions)


def test_analysis_date_must_be_after_last_order(source_data):
    with pytest.raises(ValueError, match="later than every customer's last order"):
        create_cltv_dataframe(source_data, analysis_date="2021-05-30")


def test_run_analysis_reports_missing_date_column(tmp_path, source_data):
    input_path = tmp_path / "invalid.csv"
    source_data.drop(columns="last_order_date").to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="last_order_date"):
        run_analysis(input_path=input_path, output_dir=tmp_path / "results")
