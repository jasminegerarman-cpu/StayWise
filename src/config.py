"""Shared, documented input schema and project paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = ROOT / "models" / "staywise_pipeline.joblib"
REPORT_PATH = ROOT / "models" / "evaluation.json"
HISTORY_PATH = ROOT / "data" / "analysis_history.sqlite3"
TARGET = "is_canceled"  # 1 = canceled, 0 = not canceled

FEATURES = [
    "hotel", "lead_time", "arrival_date_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal",
    "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "deposit_type", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]
CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Public, original source dataset (TidyTuesday mirrors the 2019 published dataset).
DATA_URLS = [
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv",
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
    "https://huggingface.co/datasets/aki-008/hotel_data/resolve/main/hotel_bookings.csv",
]
