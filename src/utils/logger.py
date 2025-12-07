import json
import logging
from datetime import datetime
from typing import Any, Dict

from . import __name__ as pkg_name  # type: ignore
from src.config import get_settings


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Extra fields
        for key in ("request_id", "scope", "phase"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(name: str | None = None) -> logging.Logger:
    """Setup a structured logger for the application.

    Args:
        name: Optional logger name. Defaults to package name.

    Returns:
        Configured logger instance.
    """
    settings = get_settings()
    logger_name = name or pkg_name

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        # Avoid adding multiple handlers if called repeatedly
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger
