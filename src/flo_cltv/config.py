from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "flo_data_20k.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

ORDER_COLUMNS = (
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
)
VALUE_COLUMNS = (
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online",
)
OUTLIER_COLUMNS = ORDER_COLUMNS + VALUE_COLUMNS
REQUIRED_COLUMNS = (
    "master_id",
    "first_order_date",
    "last_order_date",
    *OUTLIER_COLUMNS,
)

WEEKS_IN_THREE_MONTHS = 13
WEEKS_IN_SIX_MONTHS = 26
BG_NBD_PENALIZER = 0.01
GAMMA_GAMMA_PENALIZER = 0.01
DISCOUNT_RATE = 0.01
