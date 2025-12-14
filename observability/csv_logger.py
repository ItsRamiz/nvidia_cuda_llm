import csv
import os
from datetime import datetime
from threading import Lock

CSV_PATH = "observability/traces.csv"
_lock = Lock()

HEADER = [
    "timestamp",
    "run_id",
    "node",
    "elapsed_ms",
    "success",
]

def write_row(row: dict):
    """
    Append a single row to the CSV file.
    Thread-safe and process-safe enough for your current setup.
    """
    file_exists = os.path.exists(CSV_PATH)

    with _lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
