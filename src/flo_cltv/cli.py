"""Command-line entry point."""

import argparse
import datetime as dt
from pathlib import Path

from flo_cltv.analysis import run_analysis
from flo_cltv.config import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Estimate customer lifetime value with BG/NBD and Gamma-Gamma."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of highest-CLTV customers to export (default: 20)",
    )
    parser.add_argument(
        "--analysis-date",
        type=dt.date.fromisoformat,
        help=(
            "Analysis cutoff date in YYYY-MM-DD format "
            "(default: latest order date + 2 days)"
        ),
    )
    return parser


def main() -> None:
    """Run the analysis from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        predictions = run_analysis(
            input_path=args.input,
            output_dir=args.output_dir,
            top_n=args.top_n,
            analysis_date=args.analysis_date,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Analysis complete: {len(predictions):,} repeat-purchase customers")
    print(f"Outputs written to: {args.output_dir.resolve()}")
