import functools
import inspect
from fastapi import HTTPException

from src.utils.logger import (
    api_logger,
)


# -------------------------
# safe_handler decorator (sync & async aware)
# -------------------------
def safe_handler(default_status: int = 500, default_detail: str = "Unexpected error"):
    """
    Decorator that:
    - logs HTTPException (so every intentional error is visible in logs)
      * logs as WARNING for 4xx and as ERROR (with stack) for 5xx
    - logs unexpected exceptions with full stack trace and converts them to HTTPException
    Supports both sync and async handlers.
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except HTTPException as he:
                    # Log all HTTPExceptions so developer can see them in logs
                    try:
                        status = getattr(he, "status_code", None)
                        detail = getattr(he, "detail", None)
                        if status and status >= 500:
                            api_logger.exception(
                                "HTTPException raised (status=%s): %s", status, detail
                            )
                        else:
                            api_logger.warning(
                                "HTTPException raised (status=%s): %s", status, detail
                            )
                    except Exception:
                        api_logger.exception(
                            "HTTPException raised (unable to extract status/detail): %s",
                            he,
                        )
                    raise
                except Exception as e:
                    api_logger.exception("Unhandled exception in handler: %s", e)
                    # Do not leak internal error details to the client; use default_detail
                    raise HTTPException(
                        status_code=default_status, detail=f"{default_detail}"
                    )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except HTTPException as he:
                    try:
                        status = getattr(he, "status_code", None)
                        detail = getattr(he, "detail", None)
                        if status and status >= 500:
                            api_logger.exception(
                                "HTTPException raised (status=%s): %s", status, detail
                            )
                        else:
                            api_logger.warning(
                                "HTTPException raised (status=%s): %s", status, detail
                            )
                    except Exception:
                        api_logger.exception(
                            "HTTPException raised (unable to extract status/detail): %s",
                            he,
                        )
                    raise
                except Exception as e:
                    api_logger.exception("Unhandled exception in handler: %s", e)
                    raise HTTPException(
                        status_code=default_status, detail=f"{default_detail}"
                    )

            return sync_wrapper

    return decorator
