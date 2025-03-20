"""
error_module.py - Error handling for the Lego sorting application

This module provides a standardized approach to error handling across the
Lego sorting application, including custom exceptions and logging setup.
"""

import logging
from typing import Optional, Dict, Any, Union


# Configure application-wide logging
def setup_logging(log_file: str = "lego_sorting.log", console_level: int = logging.INFO) -> None:
    """Set up logging configuration for the entire application.

    Args:
        log_file: Path to the log file
        console_level: Logging level for console output
    """
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels in the file
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler captures everything for debugging
            logging.FileHandler(log_file),
            # Console handler shows only specified level and above
            logging.StreamHandler()
        ]
    )

    # Set console handler to the specified level
    logging.getLogger().handlers[1].setLevel(console_level)

    # Initial log message
    logging.getLogger(__name__).info("Logging initialized")


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


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Example of using custom exceptions
    try:
        config_value = None
        if config_value is None:
            raise ConfigError("Missing required configuration value")
    except LegoSortingError as e:
        result = handle_error(e, logger, return_dict=True)
        print(f"Returned: {result}")