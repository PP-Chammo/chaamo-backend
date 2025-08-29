"""
Beautiful logging system with colors, emojis, and structured formatting.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors and emojis for beautiful terminal output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    # Emojis for different log levels
    EMOJIS = {
        "DEBUG": "🔍",
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    # Component emojis
    COMPONENT_EMOJIS = {
        "worker": "⚙️",
        "scraper": "🔍",
        "api": "🌐",
        "database": "💾",
        "scheduler": "⏰",
        "paypal": "💰",
        "proxy": "🔄",
    }

    def format(self, record):
        # Get colors
        level_color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        bold = self.COLORS["BOLD"]
        dim = self.COLORS["DIM"]

        # Get emoji
        emoji = self.EMOJIS.get(record.levelname, "📝")

        # Component emoji (detect from module name)
        component_emoji = ""
        for component, comp_emoji in self.COMPONENT_EMOJIS.items():
            if component in record.name.lower():
                component_emoji = f"{comp_emoji} "
                break

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format level name (padded to 8 chars)
        level = f"{record.levelname:<8}"

        # Format logger name (shortened)
        logger_name = record.name.split(".")[-1][:12]

        # Create the formatted message
        formatted_msg = (
            f"{dim}[{timestamp}]{reset} "
            f"{emoji} {level_color}{bold}{level}{reset} "
            f"{dim}│{reset} "
            f"{component_emoji}{bold}{logger_name:<12}{reset} "
            f"{dim}│{reset} "
            f"{record.getMessage()}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted_msg += f"\n{self.formatException(record.exc_info)}"

        return formatted_msg


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup a beautiful logger with colors and emojis.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set beautiful formatter
    formatter = ColorFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a beautiful logger with colors and emojis (alias for setup_logger).

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    return setup_logger(name, level)


def log_api_request(
    logger: logging.Logger, method: str, endpoint: str, params: Optional[dict] = None
):
    """Log API request in a beautiful format."""
    params_str = f" {params}" if params else ""
    logger.info(f"🌐 {method} {endpoint}{params_str}")


def log_worker_task(
    logger: logging.Logger, task_id: str, action: str, details: Optional[str] = None
):
    """Log worker task in a beautiful format."""
    details_str = f" - {details}" if details else ""
    logger.info(f"⚙️ Task {task_id[:8]}... {action}{details_str}")


def log_scrape_progress(logger: logging.Logger, current: int, total: int, query: str):
    """Log scraping progress in a beautiful format."""
    percentage = (current / total * 100) if total > 0 else 0
    logger.info(f"🔍 Scraping [{current}/{total}] {percentage:.1f}% - {query}")


def log_database_operation(
    logger: logging.Logger, operation: str, count: int, table: str
):
    """Log database operation in a beautiful format."""
    logger.info(f"💾 {operation} {count} records to {table}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str):
    """Log error with context in a beautiful format."""
    logger.error(f"❌ {context}: {str(error)}")


# Create module-level loggers
api_logger = setup_logger("chaamo.api")
worker_logger = setup_logger("chaamo.worker")
scraper_logger = setup_logger("chaamo.scraper")
scheduler_logger = setup_logger("chaamo.scheduler")
