"""Download original public dataset. No synthetic data, no silently generated samples."""
import csv
import os
import sys
import urllib.request
from pathlib import Path
from src.config import DATA_PATH, DATA_URLS


def valid_csv(path):
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not {"hotel", "is_canceled", "lead_time", "adr"}.issubset(header):
                return False
            # Verify full dataset size using CSV reader (safe for quoted fields).
            count = sum(1 for _ in reader)
            return count >= 100_000
    except (OSError, UnicodeError, StopIteration, csv.Error):
        return False


def download_dataset():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and valid_csv(DATA_PATH):
        print(f"[OK] Original dataset present: {DATA_PATH}")
        return DATA_PATH
    print("[INFO] Downloading original Hotel Booking Demand dataset (~16 MB). Internet required once.", flush=True)
    last_error = None
    for i, url in enumerate(DATA_URLS, 1):
        temp = DATA_PATH.with_suffix(".part")
        print(f"[INFO] Data source {i}/{len(DATA_URLS)}: {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StayWise-Academic-ML-Project/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response, open(temp, "wb") as outfile:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)
            if not valid_csv(temp):
                raise ValueError("Downloaded file is not the expected 100k+ hotel bookings CSV.")
            os.replace(temp, DATA_PATH)
            print("[OK] Verified dataset: more than 100,000 original rows.", flush=True)
            return DATA_PATH
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Source unavailable: {exc}", flush=True)
            temp.unlink(missing_ok=True)
    raise RuntimeError(
        "Could not download the original dataset. Connect to the internet and restart, "
        "or manually download hotels.csv from the TidyTuesday project and save it as "
        f"{DATA_PATH}. Last error: {last_error}"
    )


if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as e:
        print("\nERROR:", str(e), file=sys.stderr)
        sys.exit(1)
