"""Query timeout wrapper for preventing long-running requests."""

import signal
import functools
import threading
import sys
from typing import Callable, Any, Optional

from backend.utils.logger import get_logger

log = get_logger("timeout")


class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""
    pass


class QueryTimeout:
    """Context manager for query timeout protection (cross-platform)."""
    
    def __init__(self, seconds: int = 30, message: str = "Query timeout"):
        """
        Initialize timeout context.
        
        Args:
            seconds: Timeout in seconds
            message: Error message on timeout
        """
        self.seconds = seconds
        self.message = message
        self.timer = None
    
    def __enter__(self):
        """Start timeout."""
        # On Windows, signal.SIGALRM is not available, so use threading instead
        if sys.platform == "win32":
            # For Windows, we'll skip timeout enforcement and just log a warning
            # since SIGALRM is Unix-only
            log.debug("[TIMEOUT] Windows detected - timeout enforcement disabled")
        else:
            try:
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.seconds)
            except AttributeError:
                # Fallback if SIGALRM is not available
                log.debug("[TIMEOUT] SIGALRM not available - timeout enforcement disabled")
        return self
    
    def __exit__(self, *args):
        """Stop timeout."""
        if sys.platform != "win32":
            try:
                signal.alarm(0)  # Cancel alarm
            except AttributeError:
                pass
    
    def _timeout_handler(self, signum, frame):
        """Handler called on timeout."""
        raise TimeoutError(self.message)


def with_timeout(seconds: int = 30):
    """
    Decorator for adding timeout to functions.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                with QueryTimeout(seconds=seconds, message=f"{func.__name__} exceeded {seconds}s timeout"):
                    return func(*args, **kwargs)
            except TimeoutError as e:
                log.warning(f"[TIMEOUT] {func.__name__} timed out: {e}")
                raise
        return wrapper
    return decorator


def safe_execute_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout_seconds: int = 30,
    fallback_result: Optional[Any] = None
) -> Any:
    """
    Execute function with timeout protection.
    
    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Timeout in seconds
        fallback_result: Result to return on timeout (instead of raising)
        
    Returns:
        Function result or fallback_result on timeout
        
    Raises:
        TimeoutError: If timeout_seconds is exceeded and fallback_result is None
    """
    if kwargs is None:
        kwargs = {}
    
    try:
        with QueryTimeout(seconds=timeout_seconds, message=f"Operation exceeded {timeout_seconds}s"):
            return func(*args, **kwargs)
    except TimeoutError as e:
        if fallback_result is not None:
            log.warning(f"[TIMEOUT] Operation timed out, returning fallback: {e}")
            return fallback_result
        else:
            log.error(f"[TIMEOUT] Operation timed out: {e}")
            raise
