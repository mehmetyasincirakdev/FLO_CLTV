# FLO Customer Lifetime Value

Estimate customer lifetime value for FLO's omnichannel customers with BG/NBD
purchase-frequency and Gamma-Gamma monetary-value models.

> **Dataset note:** The CSV's source and redistribution license are not
> specified in the project files. Confirm that you are authorized to share the
> dataset and check any privacy or commercial-data restrictions before making
> this repository public.

## What it does

- Validates the input schema, dates, customer IDs, order counts, and spend.
- Caps extreme values using Tukey fences calculated from the 1st and 99th
  percentiles.
- Calculates weekly customer histories using repeat purchases (total orders
  minus the initial purchase).
- Predicts purchases over 13 and 26 weeks and estimates six-month CLTV.
- Assigns four quantile segments (`D` through `A`) and exports customer-level
  predictions, top customers, a segment summary, and run metadata.

The analysis date defaults to two days after the latest order in the dataset.
Customers with fewer than two orders or no positive spend are excluded from
model fitting. Predictions are estimates, not guaranteed revenue.

## Project structure

```text
.
├── .github/workflows/tests.yml
├── data/
│   └── flo_data_20k.csv
├── src/flo_cltv/
│   ├── analysis.py       # End-to-end workflow
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Paths and model parameters
│   ├── data.py           # Input validation and preparation
│   ├── models.py         # BG/NBD, Gamma-Gamma, and CLTV
│   └── reporting.py      # Segment summaries and output files
├── tests/
├── .gitignore
└── pyproject.toml
```

## Requirements and installation

Python **3.10–3.13** is supported. Create and activate a virtual environment,
then install the project and development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate`.

## Run

After installation, run the command from any directory:

```bash
flo-cltv
```

Alternatively, run the package from the project directory:

```bash
python -m flo_cltv
```

Override the input, output directory, top-customer count, or analysis cutoff:

```bash
flo-cltv --input data/flo_data_20k.csv --output-dir outputs --top-n 20 --analysis-date 2021-06-01
```

The default output directory is `outputs/`. The command generates:

| File | Contents |
| --- | --- |
| `cltv_predictions.csv` | Predictions, monetary values, standardized CLTV, and segment |
| `top_customers.csv` | Highest-CLTV customers (`--top-n`) |
| `segment_summary.csv` | Customer count and predicted-value aggregates by segment |
| `run_metadata.json` | Analysis date, cohort size, horizon, and discount rate |

Generated reports, local environments, caches, build artifacts, secrets, and
IDE/OS files are excluded from version control by `.gitignore`.

## Tests and lint

```bash
pytest
ruff check .
```

GitHub Actions runs tests and lint on Python 3.10 through 3.13 for pushes and
pull requests.

## Interpretation

Segments describe relative value within this dataset and cutoff, not fixed
business tiers. The `A` group can be evaluated for retention or loyalty offers;
the `D` group can be tested with lower-cost reactivation campaigns. Measure
incremental lift with a control group before scaling either action.
