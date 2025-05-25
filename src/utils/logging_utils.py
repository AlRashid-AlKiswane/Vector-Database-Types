"""
logging_utils.py

Generalized logging utility for multi-service and multi-module Python projects.

This module provides standardized logging functions for different logging levels 
(info, debug, warning, error). It supports tagging logs with optional service 
names for clearer tracing in distributed or modular systems. Logs are formatted 
with timestamps and output to stdout and a log file.
"""

import logging
import sys
from typing import Optional
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Setup logger
logger = logging.getLogger("multi_service_logger")
logger.setLevel(logging.DEBUG)

# Formatter style
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(f"{ROOT_DIR}/logs/logs.log", mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def info(message: str, service: Optional[str] = None) -> None:
    """
    Log an informational message.

    Args:
        message (str): The message to log.
        service (Optional[str]): Optional name of the service or module logging the message.
    """
    if service:
        logger.info("[%s] %s", service, message)
    else:
        logger.info(message)


def debug(message: str, service: Optional[str] = None) -> None:
    """
    Log a debug-level message for diagnostic purposes.

    Args:
        message (str): The message to log.
        service (Optional[str]): Optional name of the service or module logging the message.
    """
    if service:
        logger.debug("[%s] %s", service, message)
    else:
        logger.debug(message)


def warning(message: str, service: Optional[str] = None) -> None:
    """
    Log a warning message indicating a potential issue.

    Args:
        message (str): The warning message to log.
        service (Optional[str]): Optional name of the service or module logging the message.
    """
    if service:
        logger.warning("[%s] %s", service, message)
    else:
        logger.warning(message)


def error(message: str, service: Optional[str] = None) -> None:
    """
    Log an error message indicating a failure or serious issue.

    Args:
        message (str): The error message to log.
        service (Optional[str]): Optional name of the service or module logging the message.
    """
    if service:
        logger.error("[%s] %s", service, message)
    else:
        logger.error(message)
