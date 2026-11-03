"""Session-only, anonymous demo reservation operations.

This is not a reservation, inventory, cancellation or payment system.
State belongs to a single Streamlit browser session and is intentionally not stored
in a shared database on publicly hosted demo deployments.
"""
from datetime import datetime, timezone
from uuid import uuid4

from .pipeline import validate_booking, risk_bucket
from .config import FEATURES

OUTCOMES = ("Awaiting arrival", "Guest canceled (staff-reported)", "Stay completed (staff-reported)")


def _locate(records, reservation_id):
    for r in records:
        if r["id"] == reservation_id:
            return r
    raise ValueError(f"Demo reservation not found: {reservation_id}")


def add_reservation(records, booking, *, reservation_id=None):
    """Add an anonymous reservation DEMO record; never create a hotel booking."""
    invalid = validate_booking(booking)
    if invalid:
        raise ValueError("Invalid reservation: " + "; ".join(invalid))
    if len(records) >= 100:
        raise ValueError("Demo session limit reached (100 entries). Export/delete records to continue.")
    rid = reservation_id or ("SW-" + uuid4().hex[:10].upper())
    if any(r["id"] == rid for r in records):
        raise ValueError("Duplicate demo reservation ID")
    record = {
        "id": rid,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "booking": {field: booking[field] for field in FEATURES},
        "outcome": OUTCOMES[0],
        "probability": None,
        "risk_level": None,
    }
    records.insert(0, record)
    return record


def set_prediction(records, reservation_id, probability):
    """Store prediction metadata; never alter staff-entered outcome."""
    p = float(probability)
    if not 0 <= p <= 1:
        raise ValueError("Cancellation score must be between 0 and 1")
    record = _locate(records, reservation_id)
    record["probability"] = p
    record["risk_level"] = risk_bucket(p)
    return record


def update_outcome(records, reservation_id, new_outcome):
    """Record a reported DEMO outcome manually, not a hotel cancellation."""
    if new_outcome not in OUTCOMES:
        raise ValueError("Unknown guest outcome")
    record = _locate(records, reservation_id)
    record["outcome"] = new_outcome
    return record


def delete_reservation(records, reservation_id):
    record = _locate(records, reservation_id)
    records.remove(record)
    return record


def reservation_summary(record):
    booking = record["booking"]
    return {
        "Demo ID": record["id"],
        "Recorded at": record["created_at"],
        "Hotel": booking["hotel"],
        "Arrival month": booking["arrival_date_month"],
        "Nights": int(booking["stays_in_weekend_nights"]) + int(booking["stays_in_week_nights"]),
        "Adults": int(booking["adults"]),
        "Reported outcome": record["outcome"],
        "Cancellation score": None if record["probability"] is None else f"{record['probability']:.1%}",
        "Risk level": record["risk_level"] or "Not analyzed",
    }
