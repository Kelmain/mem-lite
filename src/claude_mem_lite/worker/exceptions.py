"""Custom exceptions for the worker pipeline."""


class RetryableError(Exception):
    """Error that should trigger retry with backoff."""


class NonRetryableError(Exception):
    """Error that should mark item as permanently failed."""
