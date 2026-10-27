"""Analysis summaries and output file generation."""

import json
from pathlib import Path

import pandas as pd

from flo_cltv.config import DISCOUNT_RATE, WEEKS_IN_SIX_MONTHS


def build_segment_summary(cltv_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer counts and predicted value by segment."""
    return (
        cltv_dataframe.groupby("cltv_segment", observed=True)
        .agg(
            customer_count=("customer_id", "count"),
            average_cltv=("cltv", "mean"),
            total_cltv=("cltv", "sum"),
            average_expected_sales_6_month=("exp_sales_6_month", "mean"),
        )
        .reset_index()
    )


def write_analysis_outputs(
    cltv_dataframe: pd.DataFrame,
    output_dir: str | Path,
    input_path: str | Path,
    analysis_date: pd.Timestamp,
    top_n: int,
) -> None:
    """Write predictions, ranking, segment summary, and run metadata."""
    output_dir = Path(output_dir)
    input_path = Path(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cltv_dataframe.to_csv(output_dir / "cltv_predictions.csv", index=False)
    cltv_dataframe.nlargest(top_n, "cltv").to_csv(
        output_dir / "top_customers.csv",
        index=False,
    )
    build_segment_summary(cltv_dataframe).to_csv(
        output_dir / "segment_summary.csv",
        index=False,
    )

    metadata = {
        "source_file": input_path.name,
        "analysis_date": analysis_date.date().isoformat(),
        "customer_count": int(len(cltv_dataframe)),
        "segment_count": int(cltv_dataframe["cltv_segment"].nunique()),
        "cltv_horizon_weeks": WEEKS_IN_SIX_MONTHS,
        "discount_rate": DISCOUNT_RATE,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
