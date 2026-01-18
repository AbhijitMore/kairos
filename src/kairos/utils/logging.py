import logging
import sys
from logging.handlers import RotatingFileHandler
import os


def setup_kairos_logger(name: str = "kairos", level: int = logging.INFO):
    """
    Standardized logging factory for KAIROS components.
    Provides consistent formatting and console/file output.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format: [TIMESTAMP] [LEVEL] [NAME] Message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional - if logs directory exists)
    if os.path.exists("logs"):
        file_handler = RotatingFileHandler(
            "logs/kairos.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_kairos_logger()
