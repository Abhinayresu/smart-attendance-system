import logging
import os
import sys
from pathlib import Path

"""
Logger Module
-------------
Centralized logging configuration for the entire application.
Ensures consistent format, levels, and file-based logging.
"""

def setup_logger(name: str = "AttendanceSystem", level: int = logging.INFO, log_file: str = "logs/app.log"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.
        level (int): Logging level (DEBUG, INFO, etc.).
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Ensure log directory exists
    log_dir = Path(log_file).parent
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Standard formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Default logger instance
logger = setup_logger()
