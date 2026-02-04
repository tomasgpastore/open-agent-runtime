from __future__ import annotations

import logging
import os


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"



def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # Hide verbose client request logs that can reveal endpoint details.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
