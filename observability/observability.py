import time
import logging
from datetime import datetime
from observability.csv_logger import write_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class Trace:
    def __init__(self, name: str, run_id: str):
        self.name = name
        self.run_id = run_id

    def __enter__(self):
        self.start = time.perf_counter()
        logging.info(f"START {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        success = exc_type is None

        logging.info(f"END {self.name} ({elapsed_ms:.2f} ms)")

        write_row({
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "node": self.name,
            "elapsed_ms": round(elapsed_ms, 2),
            "success": success,
        })

        return False
