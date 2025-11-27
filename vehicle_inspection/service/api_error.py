"""API error classes."""
from typing import Any


class ApiError(Exception):
    """Base class for all of our API errors."""

    def __init__(
        self, status_code: int, error_code: str, description: str
    ) -> None:
        """Create an ApiError.

        Positional Arguments:
            status_code: HTTP status code
            error_code: Application error code
            description: Human description of the error code

        """
        Exception.__init__(self)
        self.status_code = status_code
        self.error_code = error_code
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of all attributes."""
        return {"error": self.error_code, "description": self.description}


class ValidationError(ApiError):
    """Define a validation error (HTTP 400)."""

    def __init__(self, error_code: str, message: str) -> None:
        """Create a ValidationError.

        Positional Arguments:
            error_code: Application error code
            description: Human description of the error code

        """
        ApiError.__init__(self, 400, error_code, message)
