"""
error_module.py - Error handling for the Lego sorting application

This module provides a standardized approach to error handling across the
Lego sorting application, including custom exceptions and logging setup.
"""

import logging
from typing import Optional, Dict, Any, Union


# Configure application-wide logging
def setup_logging(log_file: str = "lego_sorting.log", console_level: int = logging.WARNING) -> None:
    """Set up logging configuration for the entire application.

    Args:
        log_file: Path to the log file
        console_level: Logging level for console output
                      (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50)
    """
    # Configure root logger with a null handler to avoid duplicate messages
    logging.getLogger().handlers = []

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')  # Simpler format for console

    # File handler - captures everything for debugging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler - only shows important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Initial log message
    logging.getLogger(__name__).info("Logging initialized")
    logging.getLogger(__name__).debug(f"Console logging level: {console_level}")

# Custom Exception Hierarchy
class LegoSortingError(Exception):
    """Base exception for all Lego sorting errors."""
    pass


class ConfigError(LegoSortingError):
    """Configuration-related errors."""
    pass


class CameraError(LegoSortingError):
    """Camera-related errors."""
    pass


class APIError(LegoSortingError):
    """API communication errors."""
    pass


class DetectorError(LegoSortingError):
    """Detection-related errors."""
    pass


class SortingError(LegoSortingError):
    """Sorting logic errors."""
    pass


# Helper functions for standardized error handling

def handle_error(error: Exception, logger: logging.Logger,
                 return_dict: bool = False, raise_error: bool = False) -> Optional[Dict[str, Any]]:
    """Process an error with consistent handling across the application.

    Args:
        error: The exception that occurred
        logger: Logger instance to log the error
        return_dict: Whether to return an error dictionary
        raise_error: Whether to re-raise the exception

    Returns:
        Error dictionary if return_dict is True, otherwise None

    Raises:
        The original exception if raise_error is True
    """
    # Log the error
    logger.error(f"{error.__class__.__name__}: {str(error)}")

    # Re-raise if requested
    if raise_error:
        raise

    # Return error dictionary if requested
    if return_dict:
        return {"error": str(error), "error_type": error.__class__.__name__}

    return None


def log_and_return(result: Union[Dict[str, Any], Exception],
                   logger: logging.Logger,
                   success_level: int = logging.DEBUG) -> Dict[str, Any]:
    """Log a result and standardize return format.

    This helper ensures that all functions return consistently formatted
    dictionaries, with error information when necessary.

    Args:
        result: Either a result dictionary or an exception
        logger: Logger instance to log results
        success_level: Logging level for successful operations

    Returns:
        Standardized result dictionary
    """
    if isinstance(result, Exception):
        return handle_error(result, logger, return_dict=True)

    logger.log(success_level, f"Operation succeeded: {result}")
    return result

