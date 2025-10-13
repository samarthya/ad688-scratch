"""
Logging utility for Tech Career Intelligence Platform.

Provides controlled logging that can be silenced in production/Quarto rendering.
"""

import logging
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "tech_career_intel", level: str = "WARNING") -> logging.Logger:
    """
    Get or create a logger instance with appropriate configuration.

    Args:
        name: Logger name (default: "tech_career_intel")
        level: Logging level (default: "WARNING" to suppress INFO in Quarto)
               Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    Returns:
        Configured logger instance

    Usage:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("Processing data...")  # Only shows if level=INFO
        >>> logger.warning("Missing data found")  # Always shows
    """
    global _logger

    if _logger is not None:
        return _logger

    # Create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if not _logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))

        # Formatter
        formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        _logger.addHandler(handler)

    return _logger


def silence_logger():
    """Silence all logging output (useful for Quarto rendering)."""
    global _logger
    if _logger is not None:
        _logger.setLevel(logging.CRITICAL + 1)  # Above CRITICAL = silent


def enable_debug():
    """Enable debug-level logging (useful for development)."""
    global _logger
    if _logger is not None:
        _logger.setLevel(logging.DEBUG)
        for handler in _logger.handlers:
            handler.setLevel(logging.DEBUG)


# Convenience function for quick silent mode
def set_silent_mode(silent: bool = True):
    """
    Enable or disable silent mode.

    Args:
        silent: If True, suppress all logging output
    """
    if silent:
        silence_logger()
    else:
        global _logger
        if _logger is not None:
            _logger.setLevel(logging.WARNING)

