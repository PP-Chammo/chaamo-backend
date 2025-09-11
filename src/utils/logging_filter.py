import logging


class HealthCheckFilter(logging.Filter):
    """
    Filters out access logs for health check endpoint to reduce noise.

    This filter is referenced by `logging_config.json` as
    `src.utils.logging_filter.HealthCheckFilter` and is applied to the
    `uvicorn.access` handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            # Uvicorn Access logs provide `request_line` like: "GET /health HTTP/1.1"
            request_line = getattr(record, "request_line", "")
            if isinstance(request_line, str) and "/health" in request_line:
                return False  # drop health check log lines
        except Exception:
            # On any unexpected shape, allow logging rather than break it
            return True
        return True
