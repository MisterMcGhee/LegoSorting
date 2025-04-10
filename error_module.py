"""
error_module.py - Error handling for the Lego sorting application

This module provides a standardized approach to error handling across the
Lego sorting application, including custom exceptions, thread-aware logging,
and synchronization for multi-threaded operations.
"""

import logging
import threading
import traceback
import time
import os
from typing import Optional, Dict, Any, Union, Callable


# Thread-local storage for context information
thread_context = threading.local()


# Configure application-wide logging with thread safety
def setup_logging(log_file: str = "lego_sorting.log", console_level: int = logging.INFO) -> None:
    """Set up logging configuration for the entire application.

    Args:
        log_file: Path to the log file
        console_level: Logging level for console output
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger with a formatter that includes thread name
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels in the file
        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
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
    logging.getLogger(__name__).info("Logging initialized with thread awareness")


def set_thread_context(**kwargs) -> None:
    """Set context information for the current thread.

    This allows adding additional context to log messages that is specific
    to the current thread or operation.

    Args:
        **kwargs: Key-value pairs to add to the thread context
    """
    for key, value in kwargs.items():
        setattr(thread_context, key, value)


def get_thread_context(key: str, default: Any = None) -> Any:
    """Get context information for the current thread.

    Args:
        key: The context key to retrieve
        default: Default value if key is not found

    Returns:
        The context value or default if not found
    """
    return getattr(thread_context, key, default)


class ThreadSafeLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds thread context to log messages."""

    def process(self, msg, kwargs):
        # Get thread name
        thread_name = threading.current_thread().name

        # Add thread context if available
        extra_context = {}
        for key in dir(thread_context):
            if not key.startswith('_'):
                try:
                    extra_context[key] = getattr(thread_context, key)
                except AttributeError:
                    pass

        # Format context info
        context_str = ""
        if extra_context:
            context_items = [f"{k}={v}" for k, v in extra_context.items()]
            context_str = f" [{', '.join(context_items)}]"

        return f"[{thread_name}]{context_str} {msg}", kwargs


def get_logger(name: str) -> ThreadSafeLoggerAdapter:
    """Get a thread-safe logger for the specified name.

    Args:
        name: Logger name, typically __name__

    Returns:
        ThreadSafeLoggerAdapter instance
    """
    logger = logging.getLogger(name)
    return ThreadSafeLoggerAdapter(logger, {})


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


class ThreadingError(LegoSortingError):
    """Threading-related errors."""
    pass


class TimeoutError(LegoSortingError):
    """Timeout-related errors."""
    pass


class ResourceBusyError(LegoSortingError):
    """Resource busy or locked errors."""
    pass


# Thread-safe error handling mutex
_error_mutex = threading.RLock()


# Helper functions for standardized error handling
def handle_error(error: Exception, logger: logging.Logger,
                 return_dict: bool = False, raise_error: bool = False,
                 thread_safe: bool = True) -> Optional[Dict[str, Any]]:
    """Process an error with consistent handling across the application.

    Args:
        error: The exception that occurred
        logger: Logger instance to log the error
        return_dict: Whether to return an error dictionary
        raise_error: Whether to re-raise the exception
        thread_safe: Whether to use thread-safe logging

    Returns:
        Error dictionary if return_dict is True, otherwise None

    Raises:
        The original exception if raise_error is True
    """
    # Get stack trace
    stack_trace = traceback.format_exc()

    # Get thread information
    thread = threading.current_thread()
    thread_info = f"Thread: {thread.name} (ID: {thread.ident})"

    # Format error message with thread information
    error_message = f"{error.__class__.__name__}: {str(error)}"

    # Use mutex to ensure log entries don't get interleaved
    if thread_safe:
        with _error_mutex:
            logger.error(f"{error_message} - {thread_info}")
            logger.debug(f"Stack trace:\n{stack_trace}")
    else:
        logger.error(f"{error_message} - {thread_info}")
        logger.debug(f"Stack trace:\n{stack_trace}")

    # Re-raise if requested
    if raise_error:
        raise

    # Return error dictionary if requested
    if return_dict:
        return {
            "error": str(error),
            "error_type": error.__class__.__name__,
            "thread": thread.name,
            "timestamp": time.time()
        }

    return None


def log_and_return(result: Union[Dict[str, Any], Exception],
                   logger: logging.Logger,
                   success_level: int = logging.DEBUG,
                   thread_safe: bool = True) -> Dict[str, Any]:
    """Log a result and standardize return format.

    This helper ensures that all functions return consistently formatted
    dictionaries, with error information when necessary.

    Args:
        result: Either a result dictionary or an exception
        logger: Logger instance to log results
        success_level: Logging level for successful operations
        thread_safe: Whether to use thread-safe logging

    Returns:
        Standardized result dictionary
    """
    if isinstance(result, Exception):
        return handle_error(result, logger, return_dict=True, thread_safe=thread_safe)

    # Add thread information to result
    if isinstance(result, dict) and "thread" not in result:
        thread = threading.current_thread()
        result["thread"] = thread.name

    # Log with mutex if thread-safe mode is enabled
    if thread_safe:
        with _error_mutex:
            logger.log(success_level, f"Operation succeeded: {result}")
    else:
        logger.log(success_level, f"Operation succeeded: {result}")

    return result


def retry_on_error(max_attempts: int = 3, delay: float = 1.0,
                  exceptions: tuple = (Exception,), logger: logging.Logger = None) -> Callable:
    """Decorator for retrying a function on specific exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to retry on
        logger: Logger instance to log retry attempts

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger

            # Use provided logger or get a default one
            if logger is None:
                logger = get_logger(func.__module__)

            attempt = 1
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    thread_name = threading.current_thread().name
                    if attempt < max_attempts:
                        logger.warning(
                            f"[{thread_name}] Attempt {attempt}/{max_attempts} failed with error: {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        attempt += 1
                    else:
                        logger.error(
                            f"[{thread_name}] All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
                        raise

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)

    # Example of using thread context
    set_thread_context(operation="initialization", component="error_module")

    # Example of using custom exceptions with thread-safe handling
    try:
        config_value = None
        if config_value is None:
            raise ConfigError("Missing required configuration value")
    except LegoSortingError as e:
        result = handle_error(e, logger, return_dict=True)
        print(f"Returned: {result}")

    # Example of retry decorator
    @retry_on_error(max_attempts=3, delay=0.1, logger=logger)
    def example_retry_function():
        set_thread_context(operation="example_retry")
        logger.info("Attempting operation...")
        raise APIError("Temporary API error")

    try:
        example_retry_function()
    except APIError:
        logger.info("Retry example completed with expected failure")