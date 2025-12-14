import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class Trace:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        logging.info(f"START {self.name}")

    def __exit__(self, exc_type, exc, tb):
        elapsed = (time.perf_counter() - self.start) * 1000
        logging.info(f"END {self.name} ({elapsed:.2f} ms)")
