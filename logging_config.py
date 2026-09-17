import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:

    # ============================================================
    # Create logs directory
    # ============================================================

    os.makedirs("logs", exist_ok=True)

    log_file = os.path.join("logs", "app.log")


    # ============================================================
    # Log Format
    # ============================================================

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


    # ============================================================
    # Console Handler
    # ============================================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)


    # ============================================================
    # Rotating File Handler
    # ============================================================

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)


    # ============================================================
    # Root Logger
    # ============================================================

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers when Streamlit reloads
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)

    root_logger.addHandler(file_handler)


    # ============================================================
    # Suppress Noisy Libraries
    # ============================================================

    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.getLogger("google").setLevel(logging.WARNING)

    logging.getLogger("grpc").setLevel(logging.ERROR)

    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger("watchdog").setLevel(logging.WARNING)

    logging.getLogger("streamlit").setLevel(logging.WARNING)