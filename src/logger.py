import logging
import sys


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stdout. Safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
