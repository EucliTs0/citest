"""Provide helper functions for logging."""
import logging
from datetime import timedelta
from time import time
from typing import Any, Callable

from flask import current_app


class RequestFormatter(logging.Formatter):
    """Create a custom request formatter."""

    def __init__(self) -> None:
        """
        Initialize constructor by updating the format message.
        """
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        super().__init__(fmt)


def get_logger(logger_name: str) -> logging.Logger:
    """Get Logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(RequestFormatter())
    logger.addHandler(logger_handler)

    return logger


def log_execution_time() -> Callable[..., Any]:
    """
    Decorate a function to log the execution time.
    """

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def _call(*args: Any, **kwargs: Any) -> Any:
            start = time()
            result = func(*args, **kwargs)
            end = time()
            current_app.logger.info(
                "Executed %s from %s in %ss",
                func.__name__,
                func.__module__,
                timedelta(seconds=end - start).total_seconds(),
            )
            return result

        return _call

    return _decorate
