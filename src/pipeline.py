"""Dataset validation, preprocessing, training helpers, and deterministic predictions."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURES, CATEGORICAL, NUMERIC, TARGET


def load_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if len(df) < 5001:
        raise ValueError("Source needs more than 5,000 ORIGINAL records; no demo/synthetic data allowed.")
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").fillna("Unknown").astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    # Invalid source rows: no guests or no stay; no imputation masquerading as original records.
    df = df.loc[(df[["adults", "children", "babies"]].fillna(0).sum(axis=1) > 0)]
    df = df.loc[(df[["stays_in_weekend_nights", "stays_in_week_nights"]].fillna(0).sum(axis=1) > 0)]
    df = df.loc[(df["adr"].isna()) | (df["adr"] >= 0)]
    if len(df) < 5001:
        raise ValueError("Too few valid original records after cleaning")
    return df.reset_index(drop=True)


def make_preprocessor():
    # Preprocessing FIT ON TRAIN ONLY, via Pipeline, to avoid training/test leakage.
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def validate_booking(booking):
    errors = []
    parsed = {}
    for field in NUMERIC:
        v = booking.get(field)
        try:
            parsed[field] = float(v)
            if not np.isfinite(parsed[field]) or parsed[field] < 0:
                errors.append(f"{field.replace('_', ' ').title()} must be zero or higher.")
            elif field != "adr" and not parsed[field].is_integer():
                errors.append(f"{field.replace('_', ' ').title()} must be a whole number.")
        except (ValueError, TypeError):
            errors.append(f"{field.replace('_', ' ').title()} must be a valid number.")
    if not errors:
        if sum(parsed[k] for k in ("adults", "children", "babies")) <= 0:
            errors.append("Enter at least one guest.")
        if sum(parsed[k] for k in ("stays_in_weekend_nights", "stays_in_week_nights")) <= 0:
            errors.append("Stay must be at least one night.")
    for field in CATEGORICAL:
        value = booking.get(field)
        if value is None or pd.isna(value) or not str(value).strip():
            errors.append(f"Please specify {field.replace('_',' ')}.")
    return errors


def cancellation_probabilities(model, records):
    """Return P(target=1) based on explicit class ordering, never assume index 1."""
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    missing = set(FEATURES) - set(records.columns)
    if missing:
        raise ValueError(f"Missing booking fields: {sorted(missing)}")
    cls = list(model.classes_)
    if 1 not in cls:
        raise ValueError(f"Model does not have canceled class 1: {cls}")
    return np.asarray(model.predict_proba(records[FEATURES])[:, cls.index(1)], dtype=float)


def risk_bucket(prob):
    return "Low" if prob < .30 else "Moderate" if prob < .70 else "High"
