"""End-to-end analysis orchestration."""

import datetime as dt
from pathlib import Path

import pandas as pd

from flo_cltv.config import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR
from flo_cltv.models import create_cltv_dataframe
from flo_cltv.reporting import write_analysis_outputs


def run_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 20,
    analysis_date: str | dt.date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the full analysis and write all generated reports."""
    if top_n < 1:
        raise ValueError("'top_n' must be greater than zero.")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    raw_data = pd.read_csv(input_path)
    if "last_order_date" not in raw_data.columns:
        raise ValueError("Missing required column: last_order_date")
    resolved_analysis_date = (
        pd.Timestamp(analysis_date)
        if analysis_date is not None
        else pd.to_datetime(raw_data["last_order_date"]).max() + dt.timedelta(days=2)
    )
    cltv_dataframe = create_cltv_dataframe(raw_data, resolved_analysis_date)
    write_analysis_outputs(
        cltv_dataframe=cltv_dataframe,
        output_dir=output_dir,
        input_path=input_path,
        analysis_date=resolved_analysis_date,
        top_n=top_n,
    )
    return cltv_dataframe
